import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
from models.encoders import SimpleEncoder
from transformers import AutoImageProcessor
from smc_dataloader import CustomImageDataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import subprocess
import shutil
import sys
import copy
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import random
from umap import UMAP
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]= "2"  # Set the GPU 0 to use
MODALITIES = ['broad', 'slit', 'scatter', 'blue']

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def supervised_contrastive_loss(z1, z2, labels, temperature=0.5):
    """
    Supervised Contrastive Loss (SupCon)
    Args:
        z1, z2: 두 모달리티의 임베딩 [B, D]
        labels: 각 샘플의 클래스 [B]
        temperature: 온도 파라미터
    """
    # 임베딩 정규화
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    features = torch.cat([z1, z2], dim=0)  # [2B, D]
    labels = torch.cat([labels, labels], dim=0)  # [2B]

    similarity_matrix = torch.matmul(features, features.T) / temperature  # [2B, 2B]
    # 자기 자신은 제외
    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

    # 같은 클래스끼리 positive mask 생성
    labels_ = labels.contiguous().view(-1, 1)
    positive_mask = torch.eq(labels_, labels_.T).float()
    positive_mask = positive_mask * (~mask)  # 자기 자신 제외

    # logit의 log-softmax
    logits = similarity_matrix
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    # 각 anchor별 positive만 평균
    mean_log_prob_pos = (positive_mask * log_prob).sum(1) / positive_mask.sum(1).clamp(min=1)

    # loss는 -mean(log_prob_pos)
    loss = -mean_log_prob_pos.mean()
    return loss

def make_result_dir(base_dir=None):
    if base_dir is None:
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = os.path.join('result', now)
    else:
        result_dir = base_dir
    os.makedirs(result_dir, exist_ok=True)
    image_dir = os.path.join(result_dir, 'image')
    os.makedirs(image_dir, exist_ok=True)
    # run_train.sh 파일 robust하게 복사
    sh_candidates = [
        os.environ.get('BASH_SOURCE', None),
        os.path.join(os.getcwd(), 'run_train.sh'),
        os.path.join(os.path.dirname(__file__), 'run_train.sh')
    ]
    for sh_path in sh_candidates:
        if sh_path and os.path.exists(sh_path):
            shutil.copy(sh_path, os.path.join(result_dir, 'run_train.sh'))
            break
    return result_dir, image_dir

class MultiModalExcelDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.length = min([len(ds) for ds in datasets.values()])
        # 클래스별 인덱스 매핑 생성
        self.class_to_indices = {}
        # broad 기준으로 클래스 추출 (모든 모달리티가 동일 클래스 집합이라고 가정)
        labels = datasets['broad'].data['grade'] if 'grade' in datasets['broad'].data.columns else datasets['broad'].data.iloc[:, 1]
        for cls in set(labels):
            self.class_to_indices[cls] = []
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)
        self.labels = list(labels)
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        # 각 모달리티별 (img, label) 튜플 반환
        return {k: ds[idx] for k, ds in self.datasets.items()}

import random
class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.class_to_indices = dataset.class_to_indices
        self.classes = list(self.class_to_indices.keys())
        self.num_classes = len(self.classes)
        # 한 배치에 클래스별 샘플 개수 (균등 분포)
        self.samples_per_class = batch_size // self.num_classes
        assert self.samples_per_class > 0, 'batch_size가 클래스 수보다 커야 합니다.'
    def __iter__(self):
        # 각 epoch마다 클래스별 인덱스 셔플
        class_indices = {cls: random.sample(indices, len(indices)) for cls, indices in self.class_to_indices.items()}
        class_ptr = {cls: 0 for cls in self.classes}
        batch = []
        while True:
            for cls in self.classes:
                start = class_ptr[cls]
                end = start + self.samples_per_class
                if end > len(class_indices[cls]):
                    if self.drop_last:
                        return
                    else:
                        # 남은 샘플이 부족하면 다시 셔플
                        class_indices[cls] = random.sample(self.class_to_indices[cls], len(self.class_to_indices[cls]))
                        class_ptr[cls] = 0
                        start = 0
                        end = self.samples_per_class
                batch.extend(class_indices[cls][start:end])
                class_ptr[cls] += self.samples_per_class
            if len(batch) == self.batch_size:
                yield batch
                batch = []
            else:
                break
    def __len__(self):
        min_class_len = min([len(v) for v in self.class_to_indices.values()])
        return (min_class_len * self.num_classes) // self.batch_size

def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)
    B = z1.size(0)
    labels = torch.arange(B, device=z1.device)
    labels = torch.cat([labels, labels])
    mask = torch.eye(2*B, dtype=torch.bool, device=z1.device)
    similarity_matrix = similarity_matrix / temperature
    similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)
    positives = torch.cat([torch.diag(similarity_matrix, B), torch.diag(similarity_matrix, -B)])
    negatives = similarity_matrix[~mask].view(2*B, -1)
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(2*B, dtype=torch.long, device=z1.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

def eval_encoder(encoder, loader, device):
    encoder.eval()
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch['broad'].to(device)
            # 라벨 추출 (CustomImageDataset에서 (img, label) 반환)
            # 여기서는 label이 없으므로, 필요시 수정
            # 예시: imgs, labels = batch['broad']
            # 실제 라벨이 필요하다면 CustomImageDataset에서 반환값을 (img, label)로 바꿔야 함
            # 아래는 placeholder
            # all_labels.extend(labels.cpu().numpy())
            feats = encoder(imgs)
            all_feats.append(feats.cpu().numpy())
    # 실제 라벨이 있다면 아래에서 지표 계산
    # return all_feats, all_labels
    return np.concatenate(all_feats, axis=0)

def eval_metrics(y_true, y_pred, average='macro'):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    sensitivity = recall
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] == 2:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity_list = []
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificity_list.append(spec)
        specificity = np.mean(specificity_list)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': acc, 'sensitivity': sensitivity, 'specificity': specificity}

def get_num_classes_from_dataset(dataset):
    # CustomImageDataset의 라벨 컬럼에서 클래스 개수 자동 추출
    labels = dataset.data['grade'] if 'grade' in dataset.data.columns else dataset.data.iloc[:, 1]
    return len(set(labels))

def visualize_features(encoders, loader, device, image_dir, epoch, loss, num_classes):
    # 모든 인코더를 평가 모드로 전환
    for enc in encoders.values():
        enc.eval()
    features = {m: [] for m in MODALITIES}
    labels = []
    with torch.no_grad():
        for batch in loader:
            imgs = {k: v[0].to(device) for k, v in batch.items()}
            lbls = batch['broad'][1].cpu().numpy()
            for m in MODALITIES:
                feats = encoders[m](imgs[m]).cpu().numpy()
                features[m].append(feats)
            labels.append(lbls)
    labels = np.concatenate(labels, axis=0)
    for m in MODALITIES:
        features[m] = np.concatenate(features[m], axis=0)
    all_feats = np.concatenate([features[m] for m in MODALITIES], axis=0)
    all_labels = np.tile(labels, len(MODALITIES))
    modality_idx = np.concatenate([[m]*len(labels) for m in MODALITIES])
    umap = UMAP(n_components=2, random_state=42)
    all_feats_2d = umap.fit_transform(all_feats)
    plt.figure(figsize=(10, 10))
    color_map = plt.cm.get_cmap('tab10', num_classes)
    marker_map = {'broad': 'o', 'slit': 's', 'scatter': '^', 'blue': 'x'}
    added_labels = set()
    for m in MODALITIES:
        for c in range(num_classes):
            idxs = (modality_idx == m) & (all_labels == c)
            label_name = f'{m}-{c}'
            if np.any(idxs):
                if label_name not in added_labels:
                    plt.scatter(
                        all_feats_2d[idxs, 0], all_feats_2d[idxs, 1],
                        c=[color_map(c)], marker=marker_map[m],
                        label=label_name, alpha=0.6, s=20
                    )
                    added_labels.add(label_name)
                else:
                    plt.scatter(
                        all_feats_2d[idxs, 0], all_feats_2d[idxs, 1],
                        c=[color_map(c)], marker=marker_map[m],
                        alpha=0.6, s=20
                    )
    plt.legend(fontsize=9, ncol=2)
    plt.title(f'Multimodal UMAP (Epoch {epoch}, Loss {loss:.4f})')
    plt.savefig(os.path.join(image_dir, f'multimodal_feature_epoch{epoch}_loss{loss:.4f}_umap.png'))
    plt.close()

def get_module(model):
    return model.module if hasattr(model, 'module') else model

def main():
    set_seed(42)
    parser = argparse.ArgumentParser()
    # broad
    parser.add_argument('--broad_train', type=str, required=True)
    parser.add_argument('--broad_val', type=str, required=True)
    parser.add_argument('--broad_test', type=str, required=True)
    # slit
    parser.add_argument('--slit_train', type=str, required=True)
    parser.add_argument('--slit_val', type=str, required=True)
    parser.add_argument('--slit_test', type=str, required=True)
    # scatter
    parser.add_argument('--scatter_train', type=str, required=True)
    parser.add_argument('--scatter_val', type=str, required=True)
    parser.add_argument('--scatter_test', type=str, required=True)
    # blue
    parser.add_argument('--blue_train', type=str, required=True)
    parser.add_argument('--blue_val', type=str, required=True)
    parser.add_argument('--blue_test', type=str, required=True)
    # 기타
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_classes', type=int, default=0, help='클래스 개수(0이면 자동 추출)')
    parser.add_argument('--broad_backbone', type=str, default='resnet18', help='broad 모달리티 백본')
    parser.add_argument('--slit_backbone', type=str, default='resnet18', help='slit 모달리티 백본')
    parser.add_argument('--scatter_backbone', type=str, default='resnet18', help='scatter 모달리티 백본')
    parser.add_argument('--blue_backbone', type=str, default='resnet18', help='blue 모달리티 백본')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--auto_finetune', action='store_true', help='대조학습 후 분류기 학습까지 자동 실행')
    parser.add_argument('--finetune_train_label', type=str, default=None, help='분류기 학습용 train 라벨 파일')
    parser.add_argument('--finetune_val_label', type=str, default=None, help='분류기 학습용 val 라벨 파일')
    parser.add_argument('--finetune_test_label', type=str, default=None, help='분류기 학습용 test 라벨 파일')
    parser.add_argument('--finetune_epochs', type=int, default=50, help='분류기 학습 에폭')
    parser.add_argument('--finetune_batch_size', type=int, default=32, help='분류기 학습 배치사이즈')
    parser.add_argument('--finetune_lr', type=float, default=1e-3, help='분류기 학습 러닝레이트')
    parser.add_argument('--finetune_patience', type=int, default=10, help='분류기 얼리스탑핑')
    parser.add_argument('--num_heads', type=int, default=2, help='ProjectionAttentionHead의 head 수')
    parser.add_argument('--out_dim', type=int, default=256, help='SimpleEncoder projection head 출력 차원')
    parser.add_argument('--max_steps', type=int, default=0, help='총 학습 스텝 수(0이면 에폭*배치수로 자동 설정)')
    parser.add_argument('--eval_interval', type=int, default=100, help='몇 스텝마다 평가/저장/early stopping 체크')
    parser.add_argument('--alpha', type=float, default=0.6, help='loss2 가중치(alpha)')
    parser.add_argument('--alpha2', type=float, default=0.0, help='loss3 가중치(alpha2)')
    args = parser.parse_args()

    # DDP 환경 초기화 및 device 설정 제거, device만 단순 지정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 타임스탬프 기준 최상위 폴더 생성
    result_dir, image_dir = make_result_dir()
    args.result_dir = result_dir
    # 데이터셋 및 모델 생성
    image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    train_datasets = {
        'broad': CustomImageDataset(args.broad_train, args.image_folder, image_processor),
        'slit': CustomImageDataset(args.slit_train, args.image_folder, image_processor, broad_mode=True, broad_csv_file=args.broad_train),
        'scatter': CustomImageDataset(args.scatter_train, args.image_folder, image_processor, broad_mode=True, broad_csv_file=args.broad_train),
        'blue': CustomImageDataset(args.blue_train, args.image_folder, image_processor, broad_mode=True, broad_csv_file=args.broad_train),
    }
    # broad를 scatter의 grade 시퀀스에 맞춰 정렬
    scatter_grade_seq = list(train_datasets['scatter'].data['grade'])
    grade_to_indices = {g: train_datasets['broad'].data[train_datasets['broad'].data['grade'] == g].index.tolist() for g in train_datasets['broad'].data['grade'].unique()}
    ptr = {g: 0 for g in grade_to_indices}
    new_indices = []
    for g in scatter_grade_seq:
        indices = grade_to_indices.get(g, [])
        if not indices:
            new_indices.append(train_datasets['broad'].data.index[0])
        else:
            idx = indices[ptr[g] % len(indices)]
            new_indices.append(idx)
            ptr[g] += 1
    train_datasets['broad'].data = train_datasets['broad'].data.loc[new_indices].reset_index(drop=True)
    if not hasattr(args, 'num_classes') or args.num_classes <= 0:
        args.num_classes = get_num_classes_from_dataset(train_datasets['broad'])
    val_datasets = {
        'broad': CustomImageDataset(args.broad_val, args.image_folder, image_processor, num_classes=args.num_classes),
        'slit': CustomImageDataset(args.slit_val, args.image_folder, image_processor, num_classes=args.num_classes, broad_mode=True, broad_csv_file=args.broad_val),
        'scatter': CustomImageDataset(args.scatter_val, args.image_folder, image_processor, num_classes=args.num_classes, broad_mode=True, broad_csv_file=args.broad_val),
        'blue': CustomImageDataset(args.blue_val, args.image_folder, image_processor, num_classes=args.num_classes, broad_mode=True, broad_csv_file=args.broad_val),
    }
    test_datasets = {
        'broad': CustomImageDataset(args.broad_test, args.image_folder, image_processor, num_classes=args.num_classes),
        'slit': CustomImageDataset(args.slit_test, args.image_folder, image_processor, num_classes=args.num_classes, broad_mode=True, broad_csv_file=args.broad_test),
        'scatter': CustomImageDataset(args.scatter_test, args.image_folder, image_processor, num_classes=args.num_classes, broad_mode=True, broad_csv_file=args.broad_test),
        'blue': CustomImageDataset(args.blue_test, args.image_folder, image_processor, num_classes=args.num_classes, broad_mode=True, broad_csv_file=args.broad_test),
    }
    train_dataset = MultiModalExcelDataset(train_datasets)
    val_dataset = MultiModalExcelDataset(val_datasets)
    test_dataset = MultiModalExcelDataset(test_datasets)
    train_sampler = CustomBatchSampler(train_dataset, batch_size=args.batch_size, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # encoder를 모달리티별로 생성
    # ProjectionHead 공유를 위해 feat_dim을 임시로 broad backbone에서 추출
    tmp_encoder = SimpleEncoder(backbone=args.broad_backbone, use_attention_head=True, num_heads=args.num_heads, out_dim=args.out_dim)
    shared_proj_head = tmp_encoder.proj_head2  # feat_dim, out_dim이 동일하다고 가정
    del tmp_encoder
    encoders = {
        'broad': SimpleEncoder(backbone=args.broad_backbone, use_attention_head=True, num_heads=args.num_heads, out_dim=args.out_dim, shared_proj_head=shared_proj_head).to(device),
        'slit': SimpleEncoder(backbone=args.slit_backbone, use_attention_head=True, num_heads=args.num_heads, out_dim=args.out_dim, shared_proj_head=shared_proj_head).to(device),
        'scatter': SimpleEncoder(backbone=args.scatter_backbone, use_attention_head=True, num_heads=args.num_heads, out_dim=args.out_dim, shared_proj_head=shared_proj_head).to(device),
        'blue': SimpleEncoder(backbone=args.blue_backbone, use_attention_head=True, num_heads=args.num_heads, out_dim=args.out_dim, shared_proj_head=shared_proj_head).to(device),
    }
    # DataParallel 지원 (멀티 GPU)
    if torch.cuda.device_count() > 1:
        for k in encoders:
            encoders[k] = nn.DataParallel(encoders[k])
    # facebook/dinov2-base 백본이면 feature extractor만 freeze (proj_head는 학습)
    for k, encoder in encoders.items():
        backbone_name = getattr(args, f'{k}_backbone')
        enc_ = encoder.module if hasattr(encoder, 'module') else encoder
        if backbone_name == "facebook/dinov2-base":
            if hasattr(enc_, 'is_hf') and enc_.is_hf:
                for param in enc_.hf_model.parameters():
                    param.requires_grad = False
            else:
                for param in enc_.feature_extractor.parameters():
                    param.requires_grad = False
    params = []
    for encoder in encoders.values():
        params += list(encoder.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)
    metrics_log_path = os.path.join(result_dir, 'metrics_log.json')
    metrics_log = []
    if args.max_steps <= 0:
        args.max_steps = args.epochs * len(train_loader)
    step = 0
    best_loss = float('inf')
    best_step = 0
    patience_counter = 0
    train_iter = iter(train_loader)
    total_loss = 0
    pbar = tqdm(total=args.max_steps, desc="Training Steps")
    # 0 iter에서 UMAP 시각화
    visualize_features(encoders, train_loader, device, image_dir, 0, 0.0, args.num_classes)

    # === 배치 라벨 예시 출력 ===
    print("=== 배치별 라벨 예시 (grade) ===")
    for i in range(2):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        for m in MODALITIES:
            labels = batch[m][1].cpu().numpy()
            print(f"Batch {i+1} {m} labels: {labels}")
    train_iter = iter(train_loader)  # 다시 iterator 초기화
    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        for enc in encoders.values():
            enc.train()
        imgs = {k: v[0].to(device) for k, v in batch.items()}
        labels = batch['broad'][1].to(device)
        feat_backbone = {m: get_module(encoders[m])._extract_feature(imgs[m]) for m in MODALITIES}
        feat_proj = {m: get_module(encoders[m]).extract_projected_feature(imgs[m]) for m in MODALITIES}
        loss = 0
        for m in MODALITIES:
            if m == 'broad':
                continue
            loss += nt_xent_loss(feat_proj['broad'], feat_proj[m])
        loss2 = 0
        for m in MODALITIES:
            if m == 'broad':
                continue
            loss2 += supervised_contrastive_loss(feat_backbone['broad'], feat_backbone[m], labels)
        loss = loss * args.alpha + loss2 * args.alpha2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        step += 1
        pbar.update(1)
        if step % args.eval_interval == 0 or step == args.max_steps:
            avg_loss = total_loss / args.eval_interval
            print(f"[Step {step}] Train Loss: {avg_loss:.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_step = step
                patience_counter = 0
                torch.save({k: get_module(encoders[k]).state_dict() for k in encoders}, os.path.join(result_dir, 'best_model.pth'))
                visualize_features(encoders, train_loader, device, image_dir, step, avg_loss, args.num_classes)
            else:
                patience_counter += 1
            metrics_log.append({'step': step, 'train_loss': avg_loss})
            with open(metrics_log_path, 'w') as f:
                json.dump(metrics_log, f, indent=2)
            scheduler.step(avg_loss)
            total_loss = 0
            if patience_counter >= args.patience:
                print(f"Early stopping at step {step}")
                break
            else:
                print(f"[EarlyStopping] {args.patience - patience_counter}번 남음")
    pbar.close()
    # === Test Best Model ===
    best_model_dict = torch.load(os.path.join(result_dir, 'best_model.pth'), map_location=device)
    for k in encoders:
        get_module(encoders[k]).load_state_dict(best_model_dict[k])
    for enc in encoders.values():
        enc.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            imgs = {k: v[0].to(device) for k, v in batch.items()}
            labels = batch['broad'][1].to(device)
            feats = encoders['broad'](imgs['broad'])
            pred = feats.argmax(dim=1) if feats.ndim > 1 and feats.shape[1] == args.num_classes else torch.zeros_like(labels)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    test_metrics = eval_metrics(y_true, y_pred)
    with open(os.path.join(result_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Test metrics: {test_metrics}")
    print(f"[Test] P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f} F1={test_metrics['f1']:.4f} Acc={test_metrics['accuracy']:.4f}")
    # === Confusion Matrix 저장 ===
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(args.num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([str(i) for i in range(args.num_classes)])
    ax.set_yticklabels([str(i) for i in range(args.num_classes)])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'))
    plt.close()
    # === 옵션에 따라 fine-tune 자동 실행 ===
    if args.auto_finetune:
        print("[INFO] 대조학습 종료 후 분류기 학습(fine-tune) 자동 실행!")
        finetune_result_dir = os.path.join(result_dir, "finetune")
        os.makedirs(finetune_result_dir, exist_ok=True)
        finetune_cmd = [
            'python3', 'fine_tune.py',
            '--data_root', args.image_folder,
            '--train_label', args.finetune_train_label or args.broad_train,
            '--val_label', args.finetune_val_label or args.broad_val,
            '--test_label', args.finetune_test_label or args.broad_test,
            '--broad_test', args.broad_test,
            '--slit_test', args.slit_test,
            '--scatter_test', args.scatter_test,
            '--blue_test', args.blue_test,
            '--epochs', str(args.finetune_epochs),
            '--batch_size', str(args.finetune_batch_size),
            '--lr', str(args.finetune_lr),
            '--num_classes', str(args.num_classes),
            '--pretrained_encoder', os.path.join(result_dir, 'best_model.pth'),
            '--patience', str(args.finetune_patience),
            '--backbone', args.broad_backbone,
            '--device', args.device,
            '--result_dir', finetune_result_dir,
            '--out_dim', str(args.out_dim),
        ]
        print('[INFO] fine_tune.py 실행 커맨드:', ' '.join(finetune_cmd))
        subprocess.run(finetune_cmd)
    # 대조학습 모델 메모리 해제
    del encoders
    torch.cuda.empty_cache()
    import gc; gc.collect()

if __name__ == '__main__':
    main() 