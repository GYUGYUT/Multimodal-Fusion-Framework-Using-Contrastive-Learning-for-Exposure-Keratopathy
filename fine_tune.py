import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from models.encoders import SimpleEncoder
from models.encoders import ProjectionHead
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
import torch, gc
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random
from umap import UMAP

class BroadDataset(Dataset):
    def __init__(self, label_file, image_folder, transform=None, ext='.jpg'):
        # 엑셀/CSV 자동 판별
        if label_file.endswith('.csv'):
            self.data = pd.read_csv(label_file)
        else:
            self.data = pd.read_excel(label_file)
        self.image_folder = image_folder
        self.transform = transform
        self.ext = ext

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # 파일명만 추출 (경로 무시, 확장자 추가 X)
        img_id = os.path.basename(str(row['Detailed ID']))
        label = int(row['grade'])
        img_path = os.path.join(self.image_folder, img_id+self.ext)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

class DynamicWeightedClassifier(nn.Module):
    def __init__(self, encoder, num_classes, hidden_dim=256, out_dim=256, freeze_fc=False, freeze_backbone=False, freeze_projection=False):
        super().__init__()
        self.encoder = encoder
        feat_dim = encoder.feat_dim
        # 백본 동결 여부 설정
        for param in self.encoder.feature_extractor.parameters():
            param.requires_grad = not freeze_backbone
        if hasattr(self.encoder, 'proj_head2'):
            for param in self.encoder.proj_head2.parameters():
                param.requires_grad = not freeze_projection
        self.fc_backbone = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.fc_proj = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # nn.Linear(hidden_dim, num_classes)
        self.final_fc = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)
        )
        
        
        if freeze_fc:
            for param in self.final_fc.parameters():
                param.requires_grad = False
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        f_backbone = self.encoder._extract_feature(x)
        f_proj = self.encoder.proj_head2(f_backbone)
        h1 = self.fc_backbone(f_backbone)
        h2 = self.fc_proj(f_proj)
        gate_input = torch.cat([h1, h2], dim=1)
        alpha = self.gate(gate_input)  # shape: [batch, 1]
        fused = alpha * h1 + (1 - alpha) * h2
        fused = self.sigmoid(fused)
        out = self.final_fc(fused)
        return out

def eval_metrics(y_true, y_pred, average='macro'):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    # Sensitivity(민감도)는 recall과 동일
    sensitivity = recall
    # Specificity(특이도) 계산
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] == 2:
        # 이진 분류
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # 다중 클래스: 각 클래스별로 specificity 계산 후 macro 평균
        specificity_list = []
        for i in range(cm.shape[0]):
            # i번 클래스 vs 나머지
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificity_list.append(spec)
        specificity = np.mean(specificity_list)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': acc, 'sensitivity': sensitivity, 'specificity': specificity}

def make_result_dir(base_dir, subdir=None):
    if subdir is not None:
        result_dir = os.path.join(base_dir, subdir)
    else:
        result_dir = base_dir
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def get_model_module(model):
    return model.module if hasattr(model, 'module') else model

def train_and_evaluate(args, freeze_backbone=False, freeze_projection=False):
    # 실험 폴더명 생성 (분기 단순화)
    backbone_status = 'frozen_backbone' if freeze_backbone else 'unfrozen_backbone'
    proj_status = 'frozen_proj' if freeze_projection else 'unfrozen_proj'
    fc_dir = os.path.join(args.result_dir, f'fc_{backbone_status}_{proj_status}')
    os.makedirs(fc_dir, exist_ok=True)
    metrics_log_path = os.path.join(fc_dir, 'metrics_log.json')
    metrics_log = []
    best_model_path = os.path.join(fc_dir, 'best_model.pth')
    best_metrics_path = os.path.join(fc_dir, 'best_metrics.json')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = BroadDataset(args.train_label, args.data_root, transform)
    val_set = BroadDataset(args.val_label, args.data_root, transform)
    test_set = BroadDataset(args.test_label, args.data_root, transform)
    if args.num_classes <= 0:
        args.num_classes = len(set([label for label in train_set.data['grade']]))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(args.pretrained_encoder, map_location=device)
    # ProjectionHead 별도 저장 구조 지원
    if isinstance(state_dict, dict) and 'projection_head' in state_dict and 'encoders' in state_dict:
        # feat_dim 추출을 위해 임시 인코더 생성
        tmp_encoder = SimpleEncoder(backbone=args.backbone, use_attention_head=True, out_dim=args.out_dim)
        feat_dim = tmp_encoder.feat_dim
        del tmp_encoder
        projection_head = ProjectionHead(feat_dim, out_dim=args.out_dim)
        projection_head.load_state_dict(state_dict['projection_head'])
        encoder = SimpleEncoder(
            backbone=args.backbone,
            use_attention_head=True,
            out_dim=args.out_dim,
            shared_proj_head=projection_head
        )
        encoder.load_state_dict(state_dict['encoders']['broad'], strict=False)
    elif isinstance(state_dict, dict) and 'broad' in state_dict:
        encoder = SimpleEncoder(backbone=args.backbone, use_attention_head=True, out_dim=args.out_dim)
        encoder.load_state_dict(state_dict['broad'])
    else:
        encoder = SimpleEncoder(backbone=args.backbone, use_attention_head=True, out_dim=args.out_dim)
        encoder.load_state_dict(state_dict)
    if args.backbone == "facebook/dinov2-base":
        model = DynamicWeightedClassifier(encoder, args.num_classes, hidden_dim=256, out_dim=args.out_dim, freeze_fc=args.freeze_fc, freeze_backbone=freeze_backbone, freeze_projection=freeze_projection).to(device)
    else:
        model = DynamicWeightedClassifier(encoder, args.num_classes, hidden_dim=256, out_dim=args.out_dim, freeze_fc=args.freeze_fc, freeze_backbone=freeze_backbone, freeze_projection=freeze_projection)
        model = nn.DataParallel(model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # === 스케줄러 선택 ===
    if args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=args.gamma, verbose=True)
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = None

    if args.max_steps <= 0:
        args.max_steps = args.epochs * len(train_loader)
    step = 0
    best_f1 = 0
    best_step = 0
    patience_counter = 0
    train_iter = iter(train_loader)
    total_loss = 0
    y_true_train, y_pred_train = [], []
    pbar = tqdm(total=args.max_steps, desc="Training Steps")
    while step < args.max_steps:
        try:
            imgs, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            imgs, labels = next(train_iter)
        imgs, labels = imgs.to(device), labels.to(device)
        model.train()
        out = model(imgs)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(1)
        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(pred.cpu().numpy())
        step += 1
        pbar.update(1)
        if step % args.eval_interval == 0 or step == args.max_steps:
            avg_loss = total_loss / args.eval_interval
            train_metrics = eval_metrics(np.array(y_true_train), np.array(y_pred_train))
            # === Validation ===
            model.eval()
            y_true, y_pred = [], []
            val_loss = 0
            with torch.no_grad():
                for imgs_val, labels_val in val_loader:
                    imgs_val, labels_val = imgs_val.to(device), labels_val.to(device)
                    out_val = model(imgs_val)
                    loss_val = criterion(out_val, labels_val)
                    val_loss += loss_val.item()
                    pred_val = out_val.argmax(1)
                    y_true.extend(labels_val.cpu().numpy())
                    y_pred.extend(pred_val.cpu().numpy())
            val_loss /= len(val_loader)
            metrics = eval_metrics(np.array(y_true), np.array(y_pred))
            print(f"[Step {step}] Train Loss: {avg_loss:.4f} | Train: P={train_metrics['precision']:.4f} R={train_metrics['recall']:.4f} F1={train_metrics['f1']:.4f} Acc={train_metrics['accuracy']:.4f} Sens={train_metrics['sensitivity']:.4f} Spec={train_metrics['specificity']:.4f}")
            print(f"[Step {step}] Val   Loss: {val_loss:.4f} | Val  : P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} Acc={metrics['accuracy']:.4f} Sens={metrics['sensitivity']:.4f} Spec={metrics['specificity']:.4f}")
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_step = step
                patience_counter = 0
                torch.save(get_model_module(model).state_dict(), best_model_path)
                with open(best_metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
            else:
                patience_counter += 1
            metrics_log.append({
                'step': step,
                'train_loss': avg_loss,
                'train_metrics': train_metrics,
                'val_loss': val_loss,
                'val_metrics': metrics
            })
            with open(metrics_log_path, 'w') as f:
                json.dump(metrics_log, f, indent=2)
            if args.lr_scheduler == 'plateau':
                scheduler.step(metrics['f1'])
            elif scheduler is not None:
                scheduler.step()
            total_loss = 0
            y_true_train, y_pred_train = [], []
            if patience_counter >= args.patience:
                print(f"Early stopping at step {step}")
                break
            else:
                print(f"[EarlyStopping] {args.patience - patience_counter}번 남음")
    pbar.close()
    print(f"Best step: {best_step}, Best F1: {best_f1}")
    print(f"Best model saved at: {best_model_path}")
    print(f"Best metrics saved at: {best_metrics_path}")

    # === Test Best Model ===
    get_model_module(model).load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    test_labels = {
        'broad': args.broad_test if args.broad_test is not None else args.test_label,
        'slit': args.slit_test,
        'scatter': args.scatter_test,
        'blue': args.blue_test
    }
    test_results = {}
    
    # === Feature Collection for all modalities ===
    all_features = []
    all_labels = []
    all_modalities = []
    markers = {'broad': 'o', 'slit': 's', 'scatter': '^', 'blue': 'D'}
    
    for test_modality, test_label in test_labels.items():
        if test_label is None:
            continue
        test_set = BroadDataset(test_label, args.data_root, transform)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        # 피쳐와 라벨 수집
        features = []
        labels = []
        y_true, y_pred = [], []
        test_loss = 0
        all_outs = []  # <--- 추가: 모든 배치의 예측값 누적
        
        with torch.no_grad():
            for imgs, batch_labels in test_loader:
                imgs = imgs.to(device)
                batch_labels = batch_labels.to(device)
                
                # 인코더의 출력(피쳐) 추출
                feat = get_model_module(model).encoder(imgs, return_feature_only=True)
                out = model(imgs)
                all_outs.append(out.detach().cpu().numpy())  # <--- 예측값 누적
                
                # 피쳐와 라벨 저장
                features.append(feat.cpu().numpy())
                labels.extend(batch_labels.cpu().numpy())
                
                # 기존 테스트 메트릭 계산용
                loss = criterion(out, batch_labels)
                test_loss += loss.item()
                pred = out.argmax(1)
                y_true.extend(batch_labels.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
        
        # 피쳐 배열 만들기
        features = np.concatenate(features, axis=0)
        labels = np.array(labels)
        all_outs = np.concatenate(all_outs, axis=0)  # <--- 전체 예측값 shape: (N, num_classes)
        
        # 개별 모달리티 t-SNE 시각화
        print(f"\nComputing UMAP for {test_modality}...")
        umap = UMAP(n_components=2, random_state=42)
        features_2d = umap.fit_transform(features)
        
        # 시각화
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, 
                            cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(f'UMAP Feature Distribution ({test_modality})')
        plt.xlabel('UMAP dimension 1')
        plt.ylabel('UMAP dimension 2')
        
        # 범례 추가
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=plt.cm.tab10(i / args.num_classes), 
                                    label=f'Class {i}', markersize=10)
                         for i in range(args.num_classes)]
        plt.legend(handles=legend_elements, title='Classes')
        
        # 저장
        plt.savefig(os.path.join(fc_dir, f'umap_feature_distribution_{test_modality}.png'))
        plt.close()
        
        # 전체 데이터에 추가 (통합 시각화용)
        all_features.append(features)
        all_labels.extend(labels)
        all_modalities.extend([test_modality] * len(labels))
        
        # 테스트 메트릭 계산 및 저장
        test_loss /= len(test_loader)
        test_metrics = eval_metrics(np.array(y_true), np.array(y_pred))
        # === AUROC 계산 및 ROC 커브 저장 ===
        try:
            y_true_arr = np.array(y_true)
            # 다중 클래스: one-hot 변환
            if args.num_classes > 2:
                y_true_bin = np.eye(args.num_classes)[y_true_arr]
                y_score = all_outs  # <--- 전체 예측값 사용
                auroc = roc_auc_score(y_true_bin, y_score, average='macro', multi_class='ovr')
                # ROC 커브 (각 클래스별)
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(args.num_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                plt.figure(figsize=(8, 6))
                for i in range(args.num_classes):
                    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC={roc_auc[i]:.2f})')
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve ({test_modality})')
                plt.legend()
                plt.savefig(os.path.join(fc_dir, f'roc_curve_{test_modality}.png'))
                plt.close()
            else:
                # 이진 분류
                y_score = all_outs[:, 1] if all_outs.shape[1] > 1 else all_outs[:, 0]
                auroc = roc_auc_score(y_true_arr, y_score)
                fpr, tpr, _ = roc_curve(y_true_arr, y_score)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'AUC={auroc:.2f}')
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve ({test_modality})')
                plt.legend()
                plt.savefig(os.path.join(fc_dir, f'roc_curve_{test_modality}.png'))
                plt.close()
            test_metrics['auroc'] = float(auroc)
        except Exception as e:
            print(f"[Warning] AUROC 계산/그림 저장 실패: {e}")
            test_metrics['auroc'] = None
        test_results[test_modality] = test_metrics
        with open(os.path.join(fc_dir, f'test_metrics_{test_modality}.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"[Test-{test_modality}] Loss: {test_loss:.4f} | P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f} F1={test_metrics['f1']:.4f} Acc={test_metrics['accuracy']:.4f} Sens={test_metrics['sensitivity']:.4f} Spec={test_metrics['specificity']:.4f}")
        
        # Confusion Matrix 저장
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix ({test_modality})')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.savefig(os.path.join(fc_dir, f'confusion_matrix_{test_modality}.png'))
        plt.close()
    
    # === Feature Visualization (all modalities) ===
    if len(all_features) > 0:
        # 모든 피쳐 합치기
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.array(all_labels)
        all_modalities = np.array(all_modalities)
        
        # t-SNE로 차원 축소
        print("\nComputing UMAP for all modalities...")
        umap = UMAP(n_components=2, random_state=42)
        embedding = umap.fit_transform(all_features)
        
        # 시각화
        plt.figure(figsize=(12, 8))
        
        # 각 클래스와 모달리티 조합에 대해 플롯
        for label in range(args.num_classes):
            for modality in test_labels.keys():
                if modality in all_modalities:  # 해당 모달리티가 있는 경우만
                    mask = (all_labels == label) & (all_modalities == modality)
                    if np.any(mask):  # 해당하는 데이터가 있는 경우만
                        plt.scatter(embedding[mask, 0], embedding[mask, 1],
                                  c=[plt.cm.tab10(label / args.num_classes)],
                                  marker=markers[modality],
                                  label=f'Class {label} ({modality})',
                                  alpha=0.6)
        
        plt.title('UMAP Feature Distribution (All Modalities)')
        plt.xlabel('UMAP dimension 1')
        plt.ylabel('UMAP dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(fc_dir, 'umap_feature_distribution_all.png'), bbox_inches='tight')
        plt.close()

    return model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--train_label', type=str, required=True)
    parser.add_argument('--val_label', type=str, required=True)
    parser.add_argument('--test_label', type=str, required=True)
    parser.add_argument('--broad_test', type=str, required=False, default=None)
    parser.add_argument('--slit_test', type=str, required=False, default=None)
    parser.add_argument('--scatter_test', type=str, required=False, default=None)
    parser.add_argument('--blue_test', type=str, required=False, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_classes', type=int, default=0, help='클래스 개수(0이면 자동 추출)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pretrained_encoder', type=str, required=True)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--result_dir', type=str, required=True, help='대조학습 결과 폴더(여기 하위에 fc/ 생성)')
    parser.add_argument('--lr_scheduler', type=str, default='plateau', choices=['plateau', 'step', 'cosine'], help='lr 스케줄러 종류 (plateau, step, cosine)')
    parser.add_argument('--step_size', type=int, default=10, help='StepLR step_size (step 스케줄러용)')
    parser.add_argument('--gamma', type=float, default=0.5, help='StepLR/plateau gamma (step/plateau 스케줄러용)')
    parser.add_argument('--freeze_fc', action='store_true', help='FC layer도 동결할지 여부')
    parser.add_argument('--freeze_backbone', action='store_true', help='백본 네트워크를 동결할지 여부')
    parser.add_argument('--out_dim', type=int, default=256, help='SimpleEncoder projection head 출력 차원')
    parser.add_argument('--max_steps', type=int, default=0, help='총 학습 스텝 수(0이면 에폭*배치수로 자동 설정)')
    parser.add_argument('--eval_interval', type=int, default=100, help='몇 스텝마다 평가/저장/early stopping 체크')
    args = parser.parse_args()

    # 타임스탬프 폴더를 새로 만들지 않고, 인자로 받은 result_dir을 base_result_dir로 사용
    base_result_dir = args.result_dir
    # backbone/projection head 동결/비동결 실험만 남김
    model1 = train_and_evaluate(args, freeze_backbone=True, freeze_projection=True)
    del model1; torch.cuda.empty_cache(); gc.collect()
    model2 = train_and_evaluate(args, freeze_backbone=False, freeze_projection=False)
    del model2; torch.cuda.empty_cache(); gc.collect()


if __name__ == '__main__':
    main() 