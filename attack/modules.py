"""
Modules: ASIT (Adaptive Spatial-Intensity Transformer) and MS-DIFE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASIT(nn.Module):
    """Adaptive Spatial-Intensity Transformer with optimized initialization."""
    def __init__(self, in_channels: int = 1, img_size: int = 128,
                 max_rotation_deg: float = 20.0, max_translate_px: float = 15.0,
                 max_scale_delta: float = 0.2, max_gain_delta: float = 0.3,
                 max_bias_delta: float = 0.3, enable_rotation: bool = True,
                 enable_translation: bool = True, enable_scale: bool = True,
                 enable_gain: bool = True, enable_bias: bool = True):
        super().__init__()
        self.img_size = img_size
        self.max_rotation_deg = max_rotation_deg
        self.max_translate_px = max_translate_px
        self.max_scale_delta = max_scale_delta
        self.max_gain_delta = max_gain_delta
        self.max_bias_delta = max_bias_delta

        # Encoder: 3-layer CNN
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.SiLU(inplace=True),
        )

        # Compute flatten dimension
        with torch.no_grad():
            flatten_dim = self.encoder(torch.zeros(1, in_channels, img_size, img_size)).view(1, -1).size(1)

        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.SiLU()
        )

        # Heads with optional ablation
        self.head_rot = nn.Linear(128, 1) if enable_rotation else None
        self.head_trans = nn.Linear(128, 2) if enable_translation else None
        self.head_scale = nn.Linear(128, 1) if enable_scale else None
        self.head_gain = nn.Linear(128, 1) if enable_gain else None
        self.head_bias = nn.Linear(128, 1) if enable_bias else None

        self._init_weights()

    def _init_weights(self):
        """Initialize heads (zeros) and layers (Kaiming + BN identity)."""
        with torch.no_grad():
            for head in [self.head_rot, self.head_trans, self.head_scale, self.head_gain, self.head_bias]:
                if head is not None:
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    nn.init.constant_(m.weight, 1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.Linear) and m not in [self.head_rot, self.head_trans, self.head_scale, self.head_gain, self.head_bias]:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass to predict geometric and intensity parameters.

        Returns:
            theta: Dict of geometric params {rot, trans, scale}
            intensity: Dict of intensity params {gain, bias}
        """
        B = x.size(0)
        dev = x.device

        # Handle variable input sizes
        if x.size(-1) != self.img_size or x.size(-2) != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode='bilinear', align_corners=False)

        feat = self.encoder(x).view(B, -1)
        feat = self.fc(feat)

        # Bounded geometric outputs (with ablation support)
        theta = {}

        if self.head_rot is not None:
            theta['rot'] = torch.tanh(self.head_rot(feat)) * self.max_rotation_deg
        else:
            theta['rot'] = torch.zeros(B, 1, device=dev)

        if self.head_trans is not None:
            theta['trans'] = torch.tanh(self.head_trans(feat)) * self.max_translate_px
        else:
            theta['trans'] = torch.zeros(B, 2, device=dev)

        if self.head_scale is not None:
            theta['scale'] = 1.0 + torch.tanh(self.head_scale(feat)) * self.max_scale_delta
        else:
            theta['scale'] = torch.ones(B, 1, device=dev)

        # Bounded intensity outputs (with ablation support)
        intensity = {}

        if self.head_gain is not None:
            intensity['gain'] = 1.0 + torch.tanh(self.head_gain(feat)) * self.max_gain_delta
        else:
            intensity['gain'] = torch.ones(B, 1, device=dev)

        if self.head_bias is not None:
            intensity['bias'] = torch.tanh(self.head_bias(feat)) * self.max_bias_delta
        else:
            intensity['bias'] = torch.zeros(B, 1, device=dev)

        return theta, intensity


class MS_DIFE(nn.Module):
    """
    Multi-Scale Dual-Invariant Feature Extractor.
    Extracts features for Identity Loss (must be frozen during attack training).
    """
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()

        # Encoder (2-layer for lightweight feature extraction)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Channel Attention (SE-style)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 2, base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels // 2, base_channels * 2),
            nn.Sigmoid()
        )

        # Multi-scale pooling sizes
        self.pool_sizes = [1, 2, 4]
        self.out_channels = base_channels * 2

    def forward(self, x):
        B = x.size(0)
        feat = self.encoder(x)

        # Apply channel attention
        ca = self.channel_att(feat).view(B, self.out_channels, 1, 1)
        feat = feat * ca

        # Multi-scale feature extraction
        local_feats = []
        for ps in self.pool_sizes:
            pooled = F.adaptive_avg_pool2d(feat, ps)
            local_feats.append(pooled.view(B, -1))

        # Concatenate multi-scale features
        return torch.cat(local_feats, dim=1)
