# -*- coding: utf-8 -*-
"""
CLIPUNet.py — Skin-lesion segmentation with a pretrained Swin Transformer encoder + UNet decoder.

This is a corrected, server-ready rewrite of a student's `clip_unet.py`
(originally a Colab notebook).  It deliberately reuses the SAME
server-proven plumbing as clipsam.py so it runs in the same environment:
  • torchvision Swin Transformer (ImageNet-1K pretrained weights)
  • native hierarchical Swin feature maps at 1/4, 1/8, 1/16, and 1/32 scale
  • the albumentations-v2 augmentation pipeline         — same kwargs as clipsam.py
  • the ISIC-2017 `_segmentation.png` mask convention
  • headless matplotlib (Agg) for figure dumping

Architecture
    Swin Transformer encoder  →  hierarchical multi-scale feature extraction
    UNet CNN decoder          →  4 upsampling stages + optional high-res raw-image skip
    Loss                      →  0.4·BCE + 0.4·Dice + 0.2·Boundary

Bugs fixed relative to the student's version
    1. Replaced flat ViT-B/16 block taps with Swin's native hierarchical
       feature pyramid, which is a better inductive match for UNet decoding.
    2. Boundary loss numerical crash.  `probs*edges + 1e-6` can exceed 1.0,
       which F.binary_cross_entropy rejects.  Reformulated as a clamped,
       edge-masked elementwise BCE.
    3. Combined loss is now actually used (the student trained with plain BCE).
    4. Validation no longer leaks augmentation.  Uses the dedicated ISIC
       validation split with val_aug (no random transforms), instead of a
       random_split of an augment=True training set.
    5. Removed all Colab-only code (!pip, google.colab.drive); added argparse
       and headless plotting so it runs on the server.
    6. Decoder is resolution-agnostic (derives stage sizes from the feature
       grid and the input size) rather than hardcoding 14/28/56/112/224.
    7. albumentations-v2-compatible augmentation (matches clipsam.py).
"""

import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.models import (
    Swin_B_Weights,
    Swin_S_Weights,
    Swin_T_Weights,
    swin_b,
    swin_s,
    swin_t,
)
from torch.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')          # headless — no display on the server
import matplotlib.pyplot as plt
import yaml

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG / PATHS  (kept identical to clipsam.py so the same server data works)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET  = '/mnt/data/users/av543f/datasets'

IMG_PATH               = os.path.join(DATASET, 'ISIC-2017_Training_Data')
MASK_PATH              = os.path.join(DATASET, 'ISIC-2017_Training_Part1_GroundTruth')
VALIDATION_IMAGES_PATH = os.path.join(DATASET, 'ISIC-2017_Validation_Data')
VALIDATION_MASK_PATH   = os.path.join(DATASET, 'ISIC-2017_Validation_Part1_GroundTruth')
TEST_IMAGES_PATH       = os.path.join(DATASET, 'ISIC-2017_Test_v2_Data')
TEST_MASK_PATH         = os.path.join(DATASET, 'ISIC-2017_Test_v2_Part1_GroundTruth')
OUTPUT_DIR             = os.path.join(BASE_DIR, 'outputs_clipunet')

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
IMAGE_SIZE = 336                      # Swin uses a 4× patch stem → 84×84 first stage at 336
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ─────────────────────────────────────────────────────────────────────────────
#  MASK PATH RESOLUTION  (ISIC-2017 convention, from clipsam.py)
# ─────────────────────────────────────────────────────────────────────────────
def find_mask_path(masks_dir, image_fn):
    """ISIC-2017:  ISIC_XXXX.jpg  →  ISIC_XXXX_segmentation.png"""
    base = os.path.splitext(image_fn)[0]
    for ext in ('.png', '.jpg', '.jpeg'):
        p = os.path.join(masks_dir, base + '_segmentation' + ext)
        # p = os.path.join(masks_dir, base + ext)
        if os.path.exists(p):
            return p
    return None

# ─────────────────────────────────────────────────────────────────────────────
#  AUGMENTATION  (albumentations v2 — identical kwargs to clipsam.py)
# ─────────────────────────────────────────────────────────────────────────────
def build_train_aug(size):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                 scale=(0.85, 1.15), rotate=(-45, 45), p=0.4),
        A.ElasticTransform(alpha=80, sigma=8, p=0.3),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.OpticalDistortion(distort_limit=0.3, p=0.5),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30,
                             val_shift_limit=20, p=0.2),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),
            A.MedianBlur(blur_limit=5, p=0.2),
            A.MotionBlur(p=0.3),
        ], p=0.2),
        A.GaussNoise(std_range=(0.012, 0.028), p=0.2),
        A.CoarseDropout(num_holes_range=(1, 8),
                        hole_height_range=(1, 28), hole_width_range=(1, 28),
                        fill=0, p=0.1),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

def build_val_aug(size):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

# ─────────────────────────────────────────────────────────────────────────────
#  DATASET  (binary mask, no captions — purely visual)
# ─────────────────────────────────────────────────────────────────────────────
class SkinDataset(Dataset):
    def __init__(self, images_dir, masks_dir, size, augment=False):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.transform  = build_train_aug(size) if augment else build_val_aug(size)

        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Image dir not found: {images_dir}")

        imgs = sorted(f for f in os.listdir(images_dir)
                      if f.lower().endswith(IMAGE_EXTENSIONS))
        valid, missing = [], 0
        for fn in imgs:
            if find_mask_path(masks_dir, fn):
                valid.append(fn)
            else:
                missing += 1
        self.filenames = valid
        print(f"[Dataset] {len(valid)} pairs  ←  {images_dir}"
              + (f"  ({missing} images skipped, no mask)" if missing else ""))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn      = self.filenames[idx]
        image   = np.array(Image.open(os.path.join(self.images_dir, fn)).convert('RGB'))
        mask_np = np.array(
            Image.open(find_mask_path(self.masks_dir, fn)).convert('L'),
            dtype=np.uint8)

        out    = self.transform(image=image, mask=mask_np)
        img_t  = out['image'].float()
        # Binarise GT — Resize interpolation can introduce soft edge values.
        mask_t = (out['mask'].float() / 255.0 > 0.5).float().unsqueeze(0)
        return img_t, mask_t, fn

# ─────────────────────────────────────────────────────────────────────────────
#  SWIN ENCODER  (torchvision Swin + pretrained ImageNet-1K weights)
# ─────────────────────────────────────────────────────────────────────────────
SWIN_SPECS = {
    'swin_t': {'builder': swin_t, 'weights': Swin_T_Weights.IMAGENET1K_V1,
               'channels': (96, 192, 384, 768)},
    'swin_s': {'builder': swin_s, 'weights': Swin_S_Weights.IMAGENET1K_V1,
               'channels': (96, 192, 384, 768)},
    'swin_b': {'builder': swin_b, 'weights': Swin_B_Weights.IMAGENET1K_V1,
               'channels': (128, 256, 512, 1024)},
}


def build_swin_backbone(model_name: str, pretrained: bool = True):
    if model_name not in SWIN_SPECS:
        raise ValueError(f"Unsupported Swin model '{model_name}'. Choose one of {sorted(SWIN_SPECS)}")
    spec = SWIN_SPECS[model_name]
    weights = spec['weights'] if pretrained else None
    model = spec['builder'](weights=weights)
    print(f"[INFO] Loaded {model_name} "
          f"({'ImageNet-1K pretrained' if pretrained else 'random init'})")
    return model, spec['channels']


class SwinEncoder(nn.Module):
    """
    Returns Swin's native hierarchical feature pyramid.

    torchvision's Swin features are NHWC tensors after each transformer stage.
    We collect stages 1, 3, 5, and 7 and convert them to NCHW maps:
      shallow   : 1/4  resolution
      early-mid : 1/8  resolution
      mid-deep  : 1/16 resolution
      deep      : 1/32 resolution
    This multi-resolution structure is a natural fit for UNet skip connections.
    """
    STAGE_INDICES = (1, 3, 5, 7)

    def __init__(self, swin_model, fine_tune_stages=2):
        super().__init__()
        self.features = swin_model.features
        fine_tune_stages = max(0, min(fine_tune_stages, len(self.STAGE_INDICES)))
        trainable_feature_indices = set()
        if fine_tune_stages > 0:
            first_trainable_stage = self.STAGE_INDICES[-fine_tune_stages]
            trainable_feature_indices.update(
                i for i in range(first_trainable_stage, len(self.features)))

        for idx, block in enumerate(self.features):
            requires_grad = idx in trainable_feature_indices
            for p in block.parameters():
                p.requires_grad = requires_grad

        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[SwinEncoder] trainable: {n:,}  "
              f"(last {fine_tune_stages} hierarchical stage(s))")

    @staticmethod
    def _nhwc_to_nchw(x):
        return x.permute(0, 3, 1, 2).contiguous().clamp(-100, 100)

    def forward(self, x):
        maps = []
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx in self.STAGE_INDICES:
                maps.append(self._nhwc_to_nchw(x))
        return maps

# ─────────────────────────────────────────────────────────────────────────────
#  UNET DECODER  (resolution-agnostic)
# ─────────────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Conv→BN→ReLU ×2, with optional spatial dropout before output.

    Dropout2d (channel-wise) regularises better than plain Dropout for
    conv feature maps; p>0 only on deep/wide stages — never near the head.
    """
    def __init__(self, in_c, out_c, p_drop=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_c,  out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        ]
        if p_drop > 0:
            layers.append(nn.Dropout2d(p_drop))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


class CLIPUNetDecoder(nn.Module):
    """
    4-stage UNet decoder with Swin hierarchical skip connections + optional
    high-res raw-image skip.

    Swin already provides feature maps at different resolutions, so the decoder
    follows the encoder hierarchy directly instead of upsampling same-grid ViT
    taps.  For a 336×336 input, Swin-T emits approximately 84², 42², 21², and
    11² maps; patch merging pads odd sizes as needed, and the decoder simply
    uses the actual tensor sizes at runtime.
    """
    def __init__(self, encoder_channels, dropout=0.2, use_raw_img_skip=False):
        super().__init__()
        self.use_raw_img_skip = use_raw_img_skip
        sh_ch, em_ch, md_ch, deep_ch = encoder_channels
        # Dropout schedule: heaviest at the wide bottleneck, tapering to 0 at
        # the head so fine boundary detail is not randomly zeroed near output.
        d0, d1, d2, d3 = dropout, dropout * 0.75, dropout * 0.5, 0.0
        self.proj_deep = nn.Sequential(   # deepest tap → bottleneck
            nn.Conv2d(deep_ch, 512, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.proj_md   = nn.Sequential(   # mid-deep tap
            nn.Conv2d(md_ch, 256, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.proj_em   = nn.Sequential(   # early-mid tap
            nn.Conv2d(em_ch, 128, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.proj_sh   = nn.Sequential(   # shallow tap
            nn.Conv2d(sh_ch, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # Optional high-res raw-image skip → recovers edge detail beyond Swin's
        # 4× patch-stem resolution. Disable it for encoder-feature-only ablations.
        if self.use_raw_img_skip:
            self.img_skip = nn.Sequential(
                nn.Conv2d(3,  32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            )
            stage3_in = 64 + 64 + 32
        else:
            self.img_skip = None
            stage3_in = 64 + 64

        self.stage0 = ConvBlock(512,       256, p_drop=d0)
        self.stage1 = ConvBlock(256 + 256, 128, p_drop=d1)
        self.stage2 = ConvBlock(128 + 128,  64, p_drop=d2)
        self.stage3 = ConvBlock(stage3_in,  32, p_drop=d3)
        self.head   = nn.Conv2d(32, 1, 1)

    @staticmethod
    def _up(x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, maps, raw_img=None):
        """
        maps    : [shallow, early-mid, mid-deep, deep] hierarchical Swin maps.
        raw_img : optional normalised input (B, 3, S, S). Required only when
                  ``use_raw_img_skip=True``; otherwise it is used only to infer
                  the final output size.
        """
        if self.use_raw_img_skip and raw_img is None:
            raise ValueError("raw_img must be provided when use_raw_img_skip=True")

        S = raw_img.shape[-1] if raw_img is not None else maps[0].shape[-1] * 4

        f_deep = self.proj_deep(maps[3])  # bottleneck, lowest resolution
        f_md   = self.proj_md(maps[2])
        f_em   = self.proj_em(maps[1])
        f_sh   = self.proj_sh(maps[0])

        sh_size = f_sh.shape[-2:]
        em_size = f_em.shape[-2:]
        md_size = f_md.shape[-2:]

        img_feat = None
        if self.use_raw_img_skip:
            img_feat = self.img_skip(self._up(raw_img, sh_size))

        x = self.stage0(f_deep)
        x = self.stage1(torch.cat([self._up(x, md_size), f_md], 1))
        x = self.stage2(torch.cat([self._up(x, em_size), f_em], 1))
        stage3_inputs = [self._up(x, sh_size), f_sh]
        if img_feat is not None:
            stage3_inputs.append(img_feat)
        x = self.stage3(torch.cat(stage3_inputs, 1))
        x = self._up(x, (S, S))
        return self.head(x)                                      # (B, 1, S, S)

# ─────────────────────────────────────────────────────────────────────────────
#  FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────
class CLIPUNet(nn.Module):
    def __init__(self, swin_model, encoder_channels, fine_tune_stages=2,
                 dropout=0.2, use_raw_img_skip=False):
        super().__init__()
        self.encoder = SwinEncoder(swin_model, fine_tune_stages=fine_tune_stages)
        self.decoder = CLIPUNetDecoder(encoder_channels=encoder_channels,
                                       dropout=dropout,
                                       use_raw_img_skip=use_raw_img_skip)

    def forward(self, x):
        maps = self.encoder(x)
        return self.decoder(maps, x)

# ─────────────────────────────────────────────────────────────────────────────
#  LOSSES  (BCE + Dice + Boundary)
# ─────────────────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).reshape(-1)
        tgts  = targets.reshape(-1)
        inter = (probs * tgts).sum()
        return 1.0 - (2.0 * inter + self.smooth) / \
                     (probs.sum() + tgts.sum() + self.smooth)


class BoundaryLoss(nn.Module):
    """
    Edge-weighted BCE.  A Laplacian extracts GT boundaries; BCE is then
    averaged ONLY over boundary pixels.  Uses binary_cross_entropy_with_logits
    which is AMP-safe (applies sigmoid internally).
    """
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[0., 1., 0.],
                               [1., -4., 1.],
                               [0., 1., 0.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel', kernel)

    def forward(self, logits, targets):
        kernel = self.kernel.to(device=logits.device, dtype=logits.dtype)
        tgt    = targets.to(dtype=logits.dtype)
        edges  = (F.conv2d(tgt, kernel, padding=1).abs() > 0).to(logits.dtype)
        # binary_cross_entropy_with_logits is AMP-safe and applies sigmoid internally
        bce    = F.binary_cross_entropy_with_logits(logits, tgt, reduction='none')
        return (bce * edges).sum() / (edges.sum() + 1e-6)


class BCEDiceBoundaryLoss(nn.Module):
    """0.4·BCE + 0.4·Dice + 0.2·Boundary"""
    def __init__(self):
        super().__init__()
        self.bce      = nn.BCEWithLogitsLoss()
        self.dice     = DiceLoss()
        self.boundary = BoundaryLoss()

    def forward(self, logits, targets):
        return (0.4 * self.bce(logits, targets)
              + 0.4 * self.dice(logits, targets)
              + 0.2 * self.boundary(logits, targets))

# ─────────────────────────────────────────────────────────────────────────────
#  METRICS  (from clipsam.py)
# ─────────────────────────────────────────────────────────────────────────────
def dice_score(pred_bin, gt_bin):
    inter = (pred_bin & gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum()
    return 1.0 if union == 0 else 2.0 * inter / union

def iou_score(pred_bin, gt_bin):
    inter = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    return 1.0 if union == 0 else inter / union

def batch_dice_iou(preds_probs, masks, threshold=0.5):
    p = (preds_probs.detach() > threshold).bool().cpu()
    g = (masks.detach() > 0.5).bool().cpu()
    ds, js = [], []
    for b in range(p.shape[0]):
        ds.append(dice_score(p[b, 0].numpy(), g[b, 0].numpy()))
        js.append(iou_score (p[b, 0].numpy(), g[b, 0].numpy()))
    return float(np.mean(ds)), float(np.mean(js))

# ─────────────────────────────────────────────────────────────────────────────
#  TRAIN / EVAL
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    tot_loss, all_d, all_j = 0.0, [], []
    grad_norms, skipped_batches = [], 0
    amp_scale_start = scaler.get_scale() if scaler else None

    for imgs, masks, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        with autocast('cuda', enabled=(device == 'cuda')):
            logits = model(imgs)
            loss   = criterion(logits, masks)

        if not torch.isfinite(loss):
            skipped_batches += 1
            print("  [WARN] non-finite loss — batch skipped")
            continue

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        grad_norms.append(float(grad_norm.detach().cpu())
                          if torch.is_tensor(grad_norm) else float(grad_norm))
        tot_loss += loss.item()
        d, j = batch_dice_iou(torch.sigmoid(logits), masks)
        all_d.append(d); all_j.append(j)

    valid_batches = max(len(loader) - skipped_batches, 1)
    diagnostics = {
        'skipped_batches': skipped_batches,
        'grad_norm_mean': float(np.mean(grad_norms or [0])),
        'grad_norm_max': float(np.max(grad_norms or [0])),
        'amp_scale_start': amp_scale_start,
        'amp_scale_end': scaler.get_scale() if scaler else None,
    }
    return tot_loss / valid_batches, float(np.mean(all_d or [0])), \
           float(np.mean(all_j or [0])), diagnostics


@torch.no_grad()
def evaluate_loop(model, images_dir, masks_dir, device, size,
                  threshold=0.5, split_name=""):
    """Per-image evaluation at the ORIGINAL mask resolution."""
    model.eval()
    tf = build_val_aug(size)
    files = sorted(f for f in os.listdir(images_dir)
                   if f.lower().endswith(IMAGE_EXTENSIONS))
    dices, ious = [], []
    for fn in files:
        mp = find_mask_path(masks_dir, fn)
        if mp is None:
            continue
        image = np.array(Image.open(os.path.join(images_dir, fn)).convert('RGB'))
        gt    = np.array(Image.open(mp).convert('L'), dtype=np.float32) / 255.0
        img_t = tf(image=image)['image'].float().unsqueeze(0).to(device)

        with autocast('cuda', enabled=(device == 'cuda')):
            logits = model(img_t)
        prob = torch.sigmoid(logits.float())
        prob = F.interpolate(prob, size=gt.shape, mode='bilinear',
                             align_corners=False)[0, 0].cpu().numpy()
        pred_bin = prob > threshold
        gt_bin   = gt > 0.5
        dices.append(dice_score(pred_bin, gt_bin))
        ious.append(iou_score(pred_bin, gt_bin))

    d = np.array(dices); j = np.array(ious)
    print("=" * 52)
    print(f"  {split_name}:  {len(d)} images")
    print(f"  Mean Dice : {d.mean():.4f} ± {d.std():.4f}")
    print(f"  Mean IoU  : {j.mean():.4f} ± {j.std():.4f}")
    print("=" * 52)
    return d, j


@torch.no_grad()
def save_visualisations(model, images_dir, masks_dir, device, size, out_dir,
                        threshold=0.5, max_n=20):
    """Headless side-by-side panels: Input | GT | Prediction."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    tf = build_val_aug(size)
    files = sorted(f for f in os.listdir(images_dir)
                   if f.lower().endswith(IMAGE_EXTENSIONS))[:max_n]
    for fn in files:
        mp = find_mask_path(masks_dir, fn)
        image = np.array(Image.open(os.path.join(images_dir, fn)).convert('RGB'))
        img_t = tf(image=image)['image'].float().unsqueeze(0).to(device)
        with autocast('cuda', enabled=(device == 'cuda')):
            logits = model(img_t)
        H, W = image.shape[:2]
        prob = F.interpolate(torch.sigmoid(logits.float()), size=(H, W),
                             mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
        pred = prob > threshold

        gt = (np.array(Image.open(mp).convert('L')) > 127) if mp else None
        ncol = 3 if gt is not None else 2
        fig, ax = plt.subplots(1, ncol, figsize=(5 * ncol, 5))
        ax[0].imshow(image);  ax[0].set_title('Input');      ax[0].axis('off')
        c = 1
        if gt is not None:
            ax[c].imshow(gt, cmap='gray'); ax[c].set_title('Ground Truth'); ax[c].axis('off'); c += 1
            d = dice_score(pred, gt); j = iou_score(pred, gt)
            ax[c].set_title(f'Pred  Dice {d:.3f} IoU {j:.3f}')
        else:
            ax[c].set_title('Prediction')
        ax[c].imshow(pred, cmap='gray'); ax[c].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, os.path.splitext(fn)[0] + '.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)
    print(f"[INFO] Saved {len(files)} visualisations → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING CURVES / RESULTS LOGGING  (mirrors clipsam.py outputs)
# ─────────────────────────────────────────────────────────────────────────────
def save_training_plots(history: dict, output_dir: str):
    """Save CLIPUNet training/validation curves to training_curves.png."""
    if not history.get('loss'):
        print("[INFO] No training history available — skipping training_curves.png")
        return

    epochs = list(range(1, len(history['loss']) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Swin-UNet Training Curves (ISIC-2017)',
                 fontsize=13, fontweight='bold')

    panels = [
        ('loss', None, 'Loss', 'tab:red'),
        ('train_dice', 'val_dice', 'Dice Coefficient', 'tab:blue'),
        ('train_iou', 'val_iou', 'IoU (Jaccard)', 'tab:green'),
    ]
    for ax, (train_key, val_key, title, color) in zip(axes, panels):
        ax.plot(epochs, history[train_key], color=color, linewidth=1.8,
                label='train' if val_key else title)
        ax.fill_between(epochs, history[train_key], alpha=0.12, color=color)
        if val_key and history.get(val_key):
            ax.plot(epochs, history[val_key], color='tab:orange', linewidth=1.8,
                    linestyle='--', label='validation')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Training curves saved → {out_path}")


def save_results_yaml(history: dict, best_dice: float, val_results: dict,
                      test_results: dict, args, output_dir: str,
                      best_epoch=None):
    """Write a structured results.yaml compatible with CLIPSAM-style analysis."""
    doc = {
        'model': 'SwinUNet',
        'checkpoint': {
            'best_epoch': best_epoch,
            'best_val_dice': round(float(best_dice), 6),
        },
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'image_size': args.image_size,
            'swin_model': args.swin_model,
            'pretrained': not args.no_pretrained,
            'fine_tune_stages': args.fine_tune_stages,
            'lr_encoder': args.lr_encoder,
            'lr_decoder': args.lr_dec,
            'lr_min': args.lr_min,
            'weight_decay': args.weight_decay,
            'dropout': args.dropout,
            'use_raw_img_skip': not args.no_raw_img_skip,
            'threshold': args.threshold,
            'train_with_val_data': args.train_with_val_data,
            'img_path': args.img_path,
            'mask_path': args.mask_path,
            'val_img': args.val_img,
            'val_mask': args.val_mask,
            'test_img': args.test_img,
            'test_mask': args.test_mask,
        },
        'history': {
            'epoch_loss': [round(float(v), 6) for v in history.get('loss', [])],
            'epoch_train_dice': [round(float(v), 6) for v in history.get('train_dice', [])],
            'epoch_train_iou': [round(float(v), 6) for v in history.get('train_iou', [])],
            'epoch_val_dice': [round(float(v), 6) for v in history.get('val_dice', [])],
            'epoch_val_iou': [round(float(v), 6) for v in history.get('val_iou', [])],
            'lr_encoder': [round(float(v), 10) for v in history.get('lr_encoder', [])],
            'lr_decoder': [round(float(v), 10) for v in history.get('lr_decoder', [])],
        },
        'validation': val_results,
        'test': test_results,
    }
    out_path = os.path.join(output_dir, 'results.yaml')
    with open(out_path, 'w') as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] Results YAML saved → {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
#  ARGS
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Swin-UNet ISIC-2017 segmentation")
    p.add_argument('--img-path',   default=IMG_PATH)
    p.add_argument('--mask-path',  default=MASK_PATH)
    p.add_argument('--val-img',    default=VALIDATION_IMAGES_PATH)
    p.add_argument('--val-mask',   default=VALIDATION_MASK_PATH)
    p.add_argument('--test-img',   default=TEST_IMAGES_PATH)
    p.add_argument('--test-mask',  default=TEST_MASK_PATH)
    p.add_argument('--output-dir', default=OUTPUT_DIR)
    p.add_argument('--epochs',      type=int,   default=40)
    p.add_argument('--batch-size',  type=int,   default=8)
    p.add_argument('--num-workers', type=int,   default=4)
    p.add_argument('--image-size', '--clip-size', dest='image_size',
                   type=int, default=IMAGE_SIZE,
                   help="Input resolution. Default 336 gives Swin feature maps at roughly 84/42/21/11; --clip-size is accepted as a backward-compatible alias.")
    p.add_argument('--swin-model', choices=sorted(SWIN_SPECS), default='swin_t',
                   help="torchvision Swin backbone to use with ImageNet-1K pretrained weights")
    p.add_argument('--no-pretrained', action='store_true',
                   help="Disable ImageNet-1K pretrained Swin weights (not recommended).")
    p.add_argument('--fine-tune-stages', type=int, default=2,
                   help="Number of final Swin hierarchical stages to fine-tune (0-4).")
    p.add_argument('--lr-encoder', '--lr-clip', dest='lr_encoder',
                   type=float, default=1e-5,
                   help="LR for the fine-tuned Swin encoder stages; --lr-clip is accepted as a backward-compatible alias")
    p.add_argument('--lr-dec',      type=float, default=2e-4,
                   help="LR for the UNet decoder (lowered from 3e-4 to curb "
                        "fast memorisation of the 2k-image train set)")
    p.add_argument('--weight-decay',type=float, default=1e-2,
                   help="AdamW weight decay on conv/linear weights only "
                        "(norms & biases are excluded). Raised from 1e-5.")
    p.add_argument('--dropout',     type=float, default=0.2,
                   help="Decoder Dropout2d rate at the bottleneck; tapers to "
                        "0 toward the head. Set 0 to disable.")
    p.add_argument('--no-raw-img-skip', action='store_true',
                   help="Disable the high-resolution raw-image skip in the UNet decoder for Swin-feature-only ablations.")
    p.add_argument('--lr-min',      type=float, default=1e-6,
                   help="Cosine schedule floor (eta_min)")
    p.add_argument('--threshold',   type=float, default=0.5)
    p.add_argument('--train-with-val-data', action='store_true',
                   help="Research-only: append the dedicated validation split to the training loader. Validation is still evaluated after training, so reported validation metrics are no longer held-out.")
    p.add_argument('--resume',      default=None)
    p.add_argument('--test-only',   action='store_true',
                   help="Skip training; load checkpoint and evaluate only")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Device : {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, 'clipunet_best.pth')
    last_path = os.path.join(args.output_dir, 'clipunet_last.pth')

    # ── Swin Transformer (torchvision ImageNet-1K pretrained weights) ─────────
    swin_model, encoder_channels = build_swin_backbone(
        args.swin_model, pretrained=not args.no_pretrained)
    model = CLIPUNet(swin_model, encoder_channels,
                     fine_tune_stages=args.fine_tune_stages,
                     dropout=args.dropout,
                     use_raw_img_skip=not args.no_raw_img_skip).to(device)

    # Shape sanity check
    with torch.no_grad():
        dummy = torch.randn(2, 3, args.image_size, args.image_size, device=device)
        out   = model(dummy)
        assert out.shape == (2, 1, args.image_size, args.image_size), \
            f"bad output shape {out.shape}"
    print(f"[INFO] Shape check OK — output {tuple(out.shape)}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Params: {total/1e6:.1f}M total | {trainable/1e6:.1f}M trainable")

    criterion = BCEDiceBoundaryLoss().to(device)
    scaler    = GradScaler('cuda') if device == 'cuda' else None

    # ── TEST-ONLY ─────────────────────────────────────────────────────────────
    if args.test_only:
        ckpt = args.resume or best_path
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"No checkpoint at {ckpt}")
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state['model'] if 'model' in state else state)
        print(f"[INFO] Loaded {ckpt}")
        val_d, val_j = evaluate_loop(model, args.val_img, args.val_mask, device,
                                     args.image_size, args.threshold, "Validation")
        val_results = {'mean_dice': round(float(val_d.mean()), 6),
                       'std_dice': round(float(val_d.std()), 6),
                       'mean_iou': round(float(val_j.mean()), 6),
                       'std_iou': round(float(val_j.std()), 6)}
        test_results = None
        if os.path.isdir(args.test_img):
            td, tj = evaluate_loop(model, args.test_img, args.test_mask, device,
                                   args.image_size, args.threshold, "Test")
            test_results = {'mean_dice': round(float(td.mean()), 6),
                            'std_dice': round(float(td.std()), 6),
                            'mean_iou': round(float(tj.mean()), 6),
                            'std_iou': round(float(tj.std()), 6)}
            save_visualisations(model, args.test_img, args.test_mask, device,
                                args.image_size,
                                os.path.join(args.output_dir, 'test_outputs'),
                                args.threshold)
        save_results_yaml(state.get('history', {}), state.get('best_dice', 0.0),
                          val_results, test_results, args, args.output_dir,
                          best_epoch=state.get('epoch'))
        return

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = SkinDataset(args.img_path, args.mask_path, args.image_size, augment=True)
    if args.train_with_val_data:
        val_train_ds = SkinDataset(args.val_img, args.val_mask, args.image_size,
                                   augment=True)
        train_ds = ConcatDataset([train_ds, val_train_ds])
        print("[WARN] Research mode enabled: validation split appended to "
              "training data. Validation metrics are no longer held-out.")
        print(f"[INFO] Combined train samples: {len(train_ds)}")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=True)

    # ── Optimiser: two LR groups (Swin encoder vs decoder) × decay/no-decay ───────────
    # Weight decay is applied ONLY to conv/linear weights. BatchNorm scales,
    # biases, and all 1-D params are excluded — decaying them shrinks the
    # effective normalisation/offset and hurts more than it helps.
    def split_decay(module):
        decay, no_decay = [], []
        for _, p in module.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
        return decay, no_decay

    enc_d, enc_nd = split_decay(model.encoder)
    dec_d,  dec_nd  = split_decay(model.decoder)
    param_groups = []
    if enc_d:
        param_groups.append({'params': enc_d, 'lr': args.lr_encoder,
                             'weight_decay': args.weight_decay,
                             'name': 'encoder'})
    if enc_nd:
        param_groups.append({'params': enc_nd, 'lr': args.lr_encoder,
                             'weight_decay': 0.0, 'name': 'encoder'})
    param_groups.extend([
        {'params': dec_d,  'lr': args.lr_dec, 'weight_decay': args.weight_decay,
         'name': 'decoder'},
        {'params': dec_nd, 'lr': args.lr_dec, 'weight_decay': 0.0,
         'name': 'decoder'},
    ])
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min)

    start_epoch, best_dice = 0, 0.0
    if args.resume and os.path.exists(args.resume):
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state.get('epoch', 0) + 1
        best_dice   = state.get('best_dice', 0.0)
        print(f"[INFO] Resumed @ epoch {start_epoch}  best Dice {best_dice:.4f}")

    print("=" * 65)
    print("  Encoder : %s  pretrained=%s  fine-tune-stages=%d  (lr=%.0e)"
          % (args.swin_model, not args.no_pretrained, args.fine_tune_stages,
             args.lr_encoder))
    print("  Decoder : UNet 4-stage + optional raw-image skip  (lr=%.0e, drop=%.2f)"
          % (args.lr_dec, args.dropout))
    print("  Reg     : weight_decay=%.0e (weights only)  cosine→%.0e"
          % (args.weight_decay, args.lr_min))
    print("  Loss    : 0.4·BCE + 0.4·Dice + 0.2·Boundary")
    print("  RawSkip : %s" % ("enabled" if not args.no_raw_img_skip else "disabled"))
    if args.train_with_val_data:
        print("  Val     : reused in training (research-only; not held out)")
    else:
        print("  Val     : dedicated ISIC validation split (no augmentation)")
    print("=" * 65)

    history = {
        'loss': [], 'train_dice': [], 'train_iou': [],
        'val_dice': [], 'val_iou': [], 'lr_encoder': [], 'lr_decoder': [],
    }
    best_epoch = None
    diag_path = os.path.join(args.output_dir, 'training_diagnostics.csv')
    diag_fields = [
        'epoch', 'loss', 'train_dice', 'train_iou', 'val_dice', 'val_iou',
        'lr_encoder', 'lr_decoder', 'grad_norm_mean', 'grad_norm_max',
        'skipped_batches', 'amp_scale_start', 'amp_scale_end', 'best_val_dice',
    ]
    with open(diag_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=diag_fields).writeheader()
    print(f"[INFO] Diagnostics  → {diag_path}")

    for epoch in range(start_epoch, args.epochs):
        tr_loss, tr_d, tr_j, train_diag = train_one_epoch(
            model, train_dl, criterion, optimizer, scaler, device)
        scheduler.step()

        val_d, val_j = evaluate_loop(model, args.val_img, args.val_mask, device,
                                     args.image_size, args.threshold, "Validation")
        mean_val = float(val_d.mean())
        mean_val_iou = float(val_j.mean())
        lr_encoder = next((g['lr'] for g in optimizer.param_groups
                           if g.get('name') == 'encoder'), 0.0)
        lr_dec = next(g['lr'] for g in optimizer.param_groups
                      if g.get('name') == 'decoder')

        history['loss'].append(tr_loss)
        history['train_dice'].append(tr_d)
        history['train_iou'].append(tr_j)
        history['val_dice'].append(mean_val)
        history['val_iou'].append(mean_val_iou)
        history['lr_encoder'].append(lr_encoder)
        history['lr_decoder'].append(lr_dec)

        print(f"Epoch [{epoch+1:02d}/{args.epochs}]  loss={tr_loss:.4f}  "
              f"train Dice={tr_d:.4f}  IoU={tr_j:.4f}  "
              f"val Dice={mean_val:.4f}  IoU={mean_val_iou:.4f}")

        if mean_val > best_dice:
            best_dice = mean_val
            best_epoch = epoch
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_dice': best_dice, 'history': history}, best_path)
            print(f"  ✅ Best saved  (Val Dice={best_dice:.4f})")

        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_dice': best_dice, 'history': history}, last_path)

        row = {
            'epoch': epoch + 1,
            'loss': round(float(tr_loss), 6),
            'train_dice': round(float(tr_d), 6),
            'train_iou': round(float(tr_j), 6),
            'val_dice': round(float(mean_val), 6),
            'val_iou': round(float(mean_val_iou), 6),
            'lr_encoder': lr_encoder,
            'lr_decoder': lr_dec,
            'grad_norm_mean': round(float(train_diag['grad_norm_mean']), 6),
            'grad_norm_max': round(float(train_diag['grad_norm_max']), 6),
            'skipped_batches': train_diag['skipped_batches'],
            'amp_scale_start': train_diag['amp_scale_start'],
            'amp_scale_end': train_diag['amp_scale_end'],
            'best_val_dice': round(float(best_dice), 6),
        }
        with open(diag_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=diag_fields).writerow(row)

    save_training_plots(history, args.output_dir)

    # ── Final test evaluation with the best checkpoint ────────────────────────
    print("\n[INFO] Loading best checkpoint for final evaluation …")
    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state['model'])
    best_epoch = best_state.get('epoch', best_epoch)

    val_d, val_j = evaluate_loop(model, args.val_img, args.val_mask, device,
                                 args.image_size, args.threshold, "Validation")
    val_results = {'mean_dice': round(float(val_d.mean()), 6),
                   'std_dice': round(float(val_d.std()), 6),
                   'mean_iou': round(float(val_j.mean()), 6),
                   'std_iou': round(float(val_j.std()), 6)}

    test_results = None
    if os.path.isdir(args.test_img):
        td, tj = evaluate_loop(model, args.test_img, args.test_mask, device,
                               args.image_size, args.threshold, "Test")
        test_results = {'mean_dice': round(float(td.mean()), 6),
                        'std_dice': round(float(td.std()), 6),
                        'mean_iou': round(float(tj.mean()), 6),
                        'std_iou': round(float(tj.std()), 6)}
        save_visualisations(model, args.test_img, args.test_mask, device,
                            args.image_size,
                            os.path.join(args.output_dir, 'test_outputs'),
                            args.threshold)

    save_results_yaml(history, best_dice, val_results, test_results,
                      args, args.output_dir, best_epoch=best_epoch)
    print(f"[DONE] Outputs → {args.output_dir}")
    print("  clipunet_best.pth      — best validation-Dice checkpoint")
    print("  clipunet_last.pth      — last epoch checkpoint")
    print("  training_curves.png    — loss / Dice / IoU plots")
    print("  training_diagnostics.csv")
    print("  results.yaml           — config, history, val/test metrics")


if __name__ == '__main__':
    main()
