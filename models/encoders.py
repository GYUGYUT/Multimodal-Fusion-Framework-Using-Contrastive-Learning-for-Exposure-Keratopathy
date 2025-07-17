import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel

# backbone별 feature layer 이름 및 인덱스 정보
backbone_dict = {
    'resnet18': ('fc',),
    'resnet34': ('fc',),
    'resnet50': ('fc',),
    'resnet101': ('fc',),
    'resnet152': ('fc',),
    'efficientnet_b0': ('classifier', 1),
    'efficientnet_b1': ('classifier', 1),
    'efficientnet_b2': ('classifier', 1),
    'efficientnet_b3': ('classifier', 1),
    'efficientnet_b4': ('classifier', 1),
    'efficientnet_b5': ('classifier', 1),
    'efficientnet_b6': ('classifier', 1),
    'efficientnet_b7': ('classifier', 1),
    'vgg16': ('classifier', 6),
    'vgg19': ('classifier', 6),
    'densenet121': ('classifier',),
    'densenet161': ('classifier',),
    'densenet169': ('classifier',),
    'densenet201': ('classifier',),
    'mobilenet_v2': ('classifier', 1),
    'mobilenet_v3_large': ('classifier', 3),
    'mobilenet_v3_small': ('classifier', 3),
}

def get_backbone_and_dim(backbone_name, pretrained_model_name=None):
    # Huggingface 모델명 직접 입력(슬래시 포함) 또는 base 키워드 처리
    if '/' in backbone_name:
        pretrained_model_name = backbone_name
        if 'dinov2' in backbone_name:
            backbone_name = 'dinov2'
        elif 'fundusdrgrading' in backbone_name:
            backbone_name = 'fundusdrgrading'
        else:
            backbone_name = 'vit'
    if backbone_name == 'base':
        pretrained_model_name = pretrained_model_name or 'facebook/dinov2-base'
        backbone_name = 'dinov2'
    # Huggingface ViT류 모델 처리
    if backbone_name in ['dinov2', 'vit', 'fundusdrgrading']:
        if pretrained_model_name is None:
            raise ValueError('Huggingface 모델은 pretrained_model_name을 지정해야 합니다.')
        model = AutoModel.from_pretrained(pretrained_model_name)
        def feature_extractor(x):
            outputs = model(x)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output
            else:
                return outputs.last_hidden_state[:, 0]
        feat_dim = model.config.hidden_size
        return feature_extractor, feat_dim, model
    # torchvision 백본 처리
    if not hasattr(models, backbone_name):
        raise ValueError(f"torchvision에 없는 백본: {backbone_name}")
    base = getattr(models, backbone_name)(pretrained=True)
    if 'resnet' in backbone_name:
        feature_extractor = nn.Sequential(*list(base.children())[:-1], nn.AdaptiveAvgPool2d((1, 1)))
    elif 'efficientnet' in backbone_name:
        feature_extractor = nn.Sequential(base.features, nn.AdaptiveAvgPool2d((1, 1)))
    elif 'vgg' in backbone_name:
        feature_extractor = nn.Sequential(*list(base.features.children()), nn.AdaptiveAvgPool2d((1, 1)))
    elif 'densenet' in backbone_name:
        feature_extractor = nn.Sequential(base.features, nn.AdaptiveAvgPool2d((1, 1)))
    elif 'mobilenet' in backbone_name:
        feature_extractor = nn.Sequential(base.features, nn.AdaptiveAvgPool2d((1, 1)))
    else:
        raise NotImplementedError(f"지원하지 않는 백본: {backbone_name}")
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        feat = feature_extractor(dummy)
        feat_dim = feat.view(1, -1).shape[1]
    return feature_extractor, feat_dim, None

class ProjectionHead(nn.Module):
    def __init__(self, feat_dim, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class SimpleEncoder(nn.Module):
    def __init__(self, backbone='resnet18', pretrained_model_name=None, use_attention_head=False, num_heads=2, out_dim=None, shared_proj_head=None):
        if '/' in backbone:
            pretrained_model_name = backbone
            if 'dinov2' in backbone:
                backbone = 'dinov2'
            elif 'fundusdrgrading' in backbone:
                backbone = 'fundusdrgrading'
            else:
                backbone = 'vit'
        if backbone == 'base':
            pretrained_model_name = pretrained_model_name or 'facebook/dinov2-base'
            backbone = 'dinov2'
        super().__init__()
        self.backbone = backbone
        self.pretrained_model_name = pretrained_model_name
        self.is_hf = False
        self.hf_model = None
        self.feature_extractor, feat_dim, hf_model = get_backbone_and_dim(backbone, pretrained_model_name)
        if hf_model is not None:
            self.is_hf = True
            self.hf_model = hf_model
        self.feat_dim = feat_dim
        self.use_attention_head = use_attention_head
        self.num_heads = num_heads
        if out_dim is None:
            out_dim = feat_dim
        if shared_proj_head is not None:
            self.proj_head2 = shared_proj_head
        else:
            self.proj_head2 = ProjectionHead(feat_dim, out_dim=out_dim)

    def _extract_feature(self, x):
        if self.is_hf and self.pretrained_model_name == 'facebook/dinov2-base':
            device = next(self.hf_model.parameters()).device
            x = x.to(device)
            x = self.feature_extractor(x)
        elif self.is_hf:
            x = self.feature_extractor(x)
        else:
            x = self.feature_extractor(x)
            if len(x.shape) == 4:
                x = x.view(x.size(0), -1)
        return x

    def extract_projected_feature(self, x):
        return self.proj_head2(self._extract_feature(x))

    def forward(self, x, return_feature_only=False, key=None, value=None):
        if isinstance(x, dict) and all(k in x for k in ['broad', 'slit', 'scatter', 'blue']):
            feat_broad = self.extract_projected_feature(x['broad'])
            feat_slit = self.extract_projected_feature(x['slit'])
            feat_scatter = self.extract_projected_feature(x['scatter'])
            feat_blue = self.extract_projected_feature(x['blue'])
            key = torch.stack([feat_slit, feat_scatter, feat_blue], dim=1).mean(dim=1)
            value = key
            x = feat_broad
        else:
            x = self._extract_feature(x)
            x = self.proj_head2(x)
        self.feat = x
        return x 