import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# V1 — original model (kept for loading old checkpoints)
# ---------------------------------------------------------------------------

def conv3d_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, out_ch):
        super().__init__()
        self.w_g = nn.Conv3d(g_ch, out_ch, 1)
        self.w_x = nn.Conv3d(x_ch, out_ch, 1)
        self.psi = nn.Sequential(
            nn.Conv3d(out_ch, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = nn.functional.interpolate(g1, size=x1.shape[2:], mode="trilinear", align_corners=False)
        a = self.psi(g1 + x1)
        return x * a


class UNet3DAttn(nn.Module):
    def __init__(self, in_ch=4, num_classes=4, base=28):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        self.enc1 = conv3d_block(in_ch, c1)
        self.enc2 = conv3d_block(c1, c2)
        self.enc3 = conv3d_block(c2, c3)
        self.enc4 = conv3d_block(c3, c4)
        self.pool = nn.MaxPool3d(2)

        self.bottleneck = conv3d_block(c4, c4)

        self.attn3 = AttentionGate(c4, c3, c3 // 2)
        self.dec3 = conv3d_block(c4 + c3, c3)
        self.attn2 = AttentionGate(c3, c2, c2 // 2)
        self.dec2 = conv3d_block(c3 + c2, c2)
        self.attn1 = AttentionGate(c2, c1, c1 // 2)
        self.dec1 = conv3d_block(c2 + c1, c1)

        self.out = nn.Conv3d(c1, num_classes, 1)
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(e4)

        d3 = self.up(b)
        e3_attn = self.attn3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_attn], 1))

        d2 = self.up(d3)
        e2_attn = self.attn2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_attn], 1))

        d1 = self.up(d2)
        e1_attn = self.attn1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_attn], 1))

        return self.out(d1)


# ---------------------------------------------------------------------------
# V2 — upgraded architecture for SOTA performance
#
#   - ResConv3dBlock with residual connections + LeakyReLU + InstanceNorm
#   - 5 encoder levels (adds enc5 at 16x channels)
#   - Deep supervision at decoder levels 4, 3, 2 (only during training)
#   - Attention gates preserved at all 4 decoder levels
#   - Default base=32 (~30M params)
# ---------------------------------------------------------------------------

class ResConv3dBlock(nn.Module):
    """Two 3x3x3 convolutions with residual skip and LeakyReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_ch)
        self.act = nn.LeakyReLU(0.01, inplace=False)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        residual = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class _DSDBottleneck(nn.Module):
    """1x1x1 conv projecting feature maps to num_classes for self-distillation."""

    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, num_classes, 1)

    def forward(self, x):
        return F.softmax(self.proj(x), dim=1)


class UNet3DAttnV2(nn.Module):
    """Upgraded 3D U-Net with residual blocks, 5 encoder levels, deep supervision,
    gradient checkpointing, optional dual self-distillation (DSD), and optional
    case-level classification head on the bottleneck (GAP + linear)."""

    def __init__(self, in_ch=4, num_classes=4, base=32,
                 use_grad_ckpt=False, use_dsd=False, num_cls: int = 0):
        super().__init__()
        self.use_grad_ckpt = use_grad_ckpt
        self.use_dsd = use_dsd
        self.num_cls = int(num_cls)
        self.width_base = int(base)  # for checkpoint metadata (match inference)

        c1 = base
        c2 = base * 2
        c3 = base * 4
        c4 = base * 8
        c5 = min(base * 16, 320)

        # Encoder
        self.enc1 = ResConv3dBlock(in_ch, c1)
        self.enc2 = ResConv3dBlock(c1, c2)
        self.enc3 = ResConv3dBlock(c2, c3)
        self.enc4 = ResConv3dBlock(c3, c4)
        self.enc5 = ResConv3dBlock(c4, c5)
        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ResConv3dBlock(c5, c5)

        # Case-level classification (optional): pool bottleneck, linear -> num_cls logits
        if self.num_cls > 0:
            self.cls_head = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(c5, self.num_cls),
            )
        else:
            self.cls_head = None

        # Decoder
        self.up5 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.attn4 = AttentionGate(c5, c4, c4 // 2)
        self.dec4 = ResConv3dBlock(c5 + c4, c4)

        self.up4 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.attn3 = AttentionGate(c4, c3, c3 // 2)
        self.dec3 = ResConv3dBlock(c4 + c3, c3)

        self.up3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.attn2 = AttentionGate(c3, c2, c2 // 2)
        self.dec2 = ResConv3dBlock(c3 + c2, c2)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.attn1 = AttentionGate(c2, c1, c1 // 2)
        self.dec1 = ResConv3dBlock(c2 + c1, c1)

        # Output head
        self.out = nn.Conv3d(c1, num_classes, 1)

        # Deep supervision heads
        self.ds4 = nn.Conv3d(c4, num_classes, 1)
        self.ds3 = nn.Conv3d(c3, num_classes, 1)
        self.ds2 = nn.Conv3d(c2, num_classes, 1)

        # Dual self-distillation bottleneck projections
        if use_dsd:
            self.dsd_enc = nn.ModuleList([
                _DSDBottleneck(c2, num_classes),
                _DSDBottleneck(c3, num_classes),
                _DSDBottleneck(c4, num_classes),
                _DSDBottleneck(c5, num_classes),
            ])
            self.dsd_dec = nn.ModuleList([
                _DSDBottleneck(c1, num_classes),
                _DSDBottleneck(c2, num_classes),
                _DSDBottleneck(c3, num_classes),
                _DSDBottleneck(c4, num_classes),
            ])

    def _ckpt(self, fn, *args):
        if self.use_grad_ckpt and self.training:
            return grad_checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x):
        e1 = self._ckpt(self.enc1, x)
        e2 = self._ckpt(self.enc2, self.pool(e1))
        e3 = self._ckpt(self.enc3, self.pool(e2))
        e4 = self._ckpt(self.enc4, self.pool(e3))
        e5 = self._ckpt(self.enc5, self.pool(e4))

        b = self._ckpt(self.bottleneck, e5)

        d4 = self.up5(b)
        d4 = self._ckpt(self.dec4, torch.cat([d4, self.attn4(d4, e4)], 1))

        d3 = self.up4(d4)
        d3 = self._ckpt(self.dec3, torch.cat([d3, self.attn3(d3, e3)], 1))

        d2 = self.up3(d3)
        d2 = self._ckpt(self.dec2, torch.cat([d2, self.attn2(d2, e2)], 1))

        d1 = self.up2(d2)
        d1 = self._ckpt(self.dec1, torch.cat([d1, self.attn1(d1, e1)], 1))

        main_out = self.out(d1)

        cls_logits = None
        if self.cls_head is not None:
            cls_logits = self.cls_head(b)

        if self.training:
            ds = [self.ds4(d4), self.ds3(d3), self.ds2(d2)]

            if self.use_dsd:
                enc_sm = [
                    self.dsd_enc[0](e2),
                    self.dsd_enc[1](e3),
                    self.dsd_enc[2](e4),
                    self.dsd_enc[3](e5),
                ]
                dec_sm = [
                    self.dsd_dec[0](d1),
                    self.dsd_dec[1](d2),
                    self.dsd_dec[2](d3),
                    self.dsd_dec[3](d4),
                ]
                dsd_out = {"enc_softmax": enc_sm, "dec_softmax": dec_sm}
                if cls_logits is not None:
                    return main_out, ds, dsd_out, cls_logits
                return main_out, ds, dsd_out

            if cls_logits is not None:
                return main_out, ds, cls_logits
            return main_out, ds

        if cls_logits is not None:
            return main_out, cls_logits
        return main_out
