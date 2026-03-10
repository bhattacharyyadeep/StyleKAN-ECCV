import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import copy
from scipy.special import comb as binom_coeff  # for Bernstein basis
import math


# ────────────────────────────────────────────────
#  MLP utilities (unchanged)
# ────────────────────────────────────────────────

class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=epsilon)

    def forward(self, x):
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1)
            x = self.ln(x)
            x = x.permute(0, 3, 1, 2)
            return x
        elif x.dim() == 3:
            x = x.permute(0, 2, 1)
            x = self.ln(x)
            x = x.permute(0, 2, 1)
            return x
        else:
            raise ValueError(f"LN only supports 3D or 4D inputs, got {x.dim()}D")


class Spatial_FC(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.fc.weight, gain=1e-8)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
            x_flat = self.fc(x_flat)
            x = x_flat.view(B, H, W, C).permute(0, 3, 1, 2)
        elif x.dim() == 3:
            B, C, N = x.shape
            x = x.permute(0, 2, 1)
            x = self.fc(x)
            x = x.permute(0, 2, 1)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        return x


class Temporal_FC(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        return x


class MLPblock(nn.Module):
    def __init__(self, dim, seq_len, use_norm=True, use_spatial_fc=True):
        super().__init__()
        if use_spatial_fc:
            self.fc = Spatial_FC(dim)
        else:
            self.fc = Temporal_FC(seq_len)

        self.norm = LN(dim) if use_norm else nn.Identity()

        nn.init.xavier_uniform_(self.fc.fc.weight, gain=1e-8)
        nn.init.zeros_(self.fc.fc.bias)

    def forward(self, x):
        shortcut = x
        x = self.fc(x)
        x = self.norm(x)
        return shortcut + x


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class StylizationBlock(nn.Module):
    def __init__(self, dim, num_p=3):
        super().__init__()
        self.emb_layers = nn.Sequential(nn.Linear(dim, 2 * dim))
        self.norm = LN(dim)
        self.norm_global = LN(dim)
        self.out_layers = nn.Sequential(zero_module(nn.Linear(dim, dim)))
        self.global_emb_layers = nn.Sequential(nn.Linear(dim, dim))

    def forward(self, x, x_global):
        B, C, H, W = x.shape
        N = H * W

        x_flat = x.flatten(2)

        x_global_in = x_global.permute(0, 2, 1)
        emb = self.emb_layers(x_global_in)
        scale, shift = torch.chunk(emb, 2, dim=-1)
        scale = scale.permute(0, 2, 1)
        shift = shift.permute(0, 2, 1)
        x_flat = x_flat * (1 + scale) + shift
        x = x_flat.view(B, C, H, W)

        x_perm = x.permute(0, 2, 3, 1)
        x_perm = self.out_layers(x_perm)
        x = x_perm.permute(0, 3, 1, 2)
        x = self.norm(x)

        x_global_in = x_global.permute(0, 2, 1)
        x_global_new = self.global_emb_layers(x_global_in)
        x_global_new = x_global_new.permute(0, 2, 1)
        x_global = x_global + x_global_new
        x_global = self.norm_global(x_global)

        return x, x_global


class TransMLP(nn.Module):
    def __init__(self, dim, num_layers=2, interaction_interval=2, p=3, use_norm=True):
        super().__init__()
        self.local_mlps = nn.Sequential(*[
            MLPblock(dim, seq_len=None, use_norm=use_norm, use_spatial_fc=True)
            for _ in range(num_layers)
        ])
        self.global_mlps = nn.Sequential(*[
            MLPblock(dim, seq_len=None, use_norm=use_norm, use_spatial_fc=True)
            for _ in range(num_layers // interaction_interval)
        ])
        self.stylization_blocks = nn.ModuleList([
            StylizationBlock(dim=dim, num_p=p)
            for _ in range(num_layers // interaction_interval)
        ])
        self.interaction_interval = interaction_interval

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x_global = x.flatten(2)

        global_step = 0
        for i, local_layer in enumerate(self.local_mlps):
            x = local_layer(x)
            if (i + 1) % self.interaction_interval == 0 and global_step < len(self.global_mlps):
                x_global = self.global_mlps[global_step](x_global)
                x_new, x_global_new = self.stylization_blocks[global_step](x, x_global)
                x = x + x_new
                x_global = x_global + x_global_new
                global_step += 1
        return x


class TemporalAdapter(nn.Module):
    def __init__(self, channels, num_layers=2, interaction_interval=2, p=3):
        super().__init__()
        self.mlp = TransMLP(
            dim=channels,
            num_layers=num_layers,
            interaction_interval=interaction_interval,
            p=p,
            use_norm=True
        )

    def forward(self, x, x_prev):
        if x_prev is None:
            return x
        delta = x - x_prev
        out = self.mlp(delta)
        gate = torch.sigmoid(out.mean(dim=1, keepdim=True))
        return x + gate * out


# ────────────────────────────────────────────────
#  Bernstein Polynomial basis (adapted from kan_bernstein.py)
# ────────────────────────────────────────────────

class BernsteinBasis(nn.Module):
    def __init__(self, order=5):
        super().__init__()
        self.order = order
        # One learnable coefficient per basis function
        self.coeffs = nn.Parameter(torch.randn(order + 1) * 0.02)

    def forward(self, x):
        # x expected in [0,1] → we clamp + normalize features if needed
        x = x.clamp(0.0, 1.0)
        basis = torch.zeros_like(x)
        for i in range(self.order + 1):
            binom = binom_coeff(self.order, i)
            term = binom * (x ** i) * ((1.0 - x) ** (self.order - i))
            basis += self.coeffs[i] * term
        return basis


class BernsteinKANHead(nn.Module):
    """
    Simple per-pixel output head using Bernstein polynomials + linear base.
    Input:  (B*N, in_features)
    Output: (B*N, 1)
    """
    def __init__(
        self,
        in_features,
        bernstein_order=5,
        scale_base=1.0,
        base_activation=nn.SiLU,
    ):
        super().__init__()
        self.in_features = in_features

        self.base_weight = nn.Parameter(torch.empty(1, in_features))
        self.bernstein = BernsteinBasis(order=bernstein_order)

        self.scale_base = scale_base
        self.base_activation = base_activation()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

    def forward(self, x):
        # x : (B*N, in_features)

        # Linear base term
        base_out = F.linear(self.base_activation(x), self.base_weight)     # (B*N, 1)

        # Bernstein polynomial term (scalar output per location)
        bern_out = self.bernstein(x.mean(dim=1, keepdim=True))            # (B*N, 1)
        # ^ using mean over features as a simple conditioning signal

        return base_out + bern_out


# ────────────────────────────────────────────────
#  DPT Head – now using Bernstein-based final layer
# ────────────────────────────────────────────────

class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        temporal_layers_per_scale=2,
        # Bernstein parameters
        bernstein_order=5,
        scale_base=1.0,
    ):
        super().__init__()
        self.use_bn = use_bn

        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels[i], kernel_size=1)
            for i in range(len(out_channels))
        ])

        self.temporal_adapters = nn.ModuleList([
            TemporalAdapter(
                out_channels[i],
                num_layers=temporal_layers_per_scale,
                interaction_interval=2,
                p=3
            )
            for i in range(len(out_channels))
        ])

        self.prev_feats = [None] * len(out_channels)

        self.scratch = nn.Module()
        self.scratch.layer1_rn = nn.Conv2d(out_channels[0], features, 3, padding=1)
        self.scratch.layer2_rn = nn.Conv2d(out_channels[1], features, 3, padding=1)
        self.scratch.layer3_rn = nn.Conv2d(out_channels[2], features, 3, padding=1)
        self.scratch.layer4_rn = nn.Conv2d(out_channels[3], features, 3, padding=1)

        self.scratch.refinenet4 = FeatureFusionBlock(features, use_bn)
        self.scratch.refinenet3 = FeatureFusionBlock(features, use_bn)
        self.scratch.refinenet2 = FeatureFusionBlock(features, use_bn)
        self.scratch.refinenet1 = FeatureFusionBlock(features, use_bn)

        # ─── Bernstein-based output head ────────
        self.output_head = BernsteinKANHead(
            in_features     = features,
            bernstein_order = bernstein_order,
            scale_base      = scale_base,
        )

    def reset_state(self):
        self.prev_feats = [None] * len(self.prev_feats)

    def forward(self, out_features, patch_h, patch_w):
        feats = []

        for i, x in enumerate(out_features):
            B, N, C = x.shape
            x = x.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
            x = self.projects[i](x)

            prev = self.prev_feats[i]
            if prev is not None and prev.shape[0] != x.shape[0]:
                prev = None

            x = self.temporal_adapters[i](x, prev)
            self.prev_feats[i] = x.detach()
            feats.append(x)

        layer1 = self.scratch.layer1_rn(feats[0])
        layer2 = self.scratch.layer2_rn(feats[1])
        layer3 = self.scratch.layer3_rn(feats[2])
        layer4 = self.scratch.layer4_rn(feats[3])

        path4 = self.scratch.refinenet4(layer4)
        path3 = self.scratch.refinenet3(path4, layer3)
        path2 = self.scratch.refinenet2(path3, layer2)
        path1 = self.scratch.refinenet1(path2, layer1)

        # ─── Bernstein head ───────────────────────────────────────
        B, C, H, W = path1.shape
        N = H * W

        # Flatten spatial → per-pixel samples
        x = path1.permute(0, 2, 3, 1).reshape(B * N, C)      # (B*N, features)
        x = self.output_head(x)                              # (B*N, 1)
        x = x.view(B, H, W, 1).permute(0, 3, 1, 2)           # (B, 1, H, W)

        return x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, use_bn):
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual=None):
        if residual is not None:
            residual = F.interpolate(residual, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = x + residual
        x = self.relu(x)
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        return x


# ────────────────────────────────────────────────
#  Main DPT model
# ────────────────────────────────────────────────

class DPT(nn.Module):
    def __init__(
        self,
        backbone,
        nclass=1,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        temporal_layers_per_scale=2,
        bernstein_order=5,
        scale_base=1.0,
    ):
        super().__init__()

        self.backbone = backbone

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.head = DPTHead(
            in_channels=backbone.embed_dim,
            features=features,
            use_bn=use_bn,
            out_channels=out_channels,
            temporal_layers_per_scale=temporal_layers_per_scale,
            bernstein_order=bernstein_order,
            scale_base=scale_base,
        )

    def forward(self, x):
        patch_size = self.backbone.patch_size
        H, W = x.shape[-2:]
        patch_h = H // patch_size
        patch_w = W // patch_size

        feats = self.backbone.get_intermediate_layers(
            x, n=4, reshape=False
        )

        out = self.head(feats, patch_h, patch_w)

        out = F.interpolate(
            out, size=(H, W), mode="bilinear", align_corners=False
        )

        return out
