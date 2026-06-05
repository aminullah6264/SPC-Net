# -*- coding: utf-8 -*-
"""
CLIPUNet.py — Skin-lesion segmentation with a CLIP ViT-B/16 encoder + UNet decoder.

This is a corrected, server-ready rewrite of a student's `clip_unet.py`
(originally a Colab notebook).  It deliberately reuses the SAME
server-proven plumbing as clipsam.py so it runs in the same environment:
  • open_clip ViT-B/16 (pretrained='openai')          — not HuggingFace transformers
  • hook-based multi-scale ViT feature extraction      — handles LND / NLD layouts
  • the albumentations-v2 augmentation pipeline         — same kwargs as clipsam.py
  • the ISIC-2017 `_segmentation.png` mask convention
  • headless matplotlib (Agg) for figure dumping

Architecture
    CLIP ViT-B/16 encoder  →  taps 4 transformer blocks (multi-scale)
    UNet CNN decoder       →  4 upsampling stages + optional high-res raw-image skip
    Loss                   →  0.4·BCE + 0.4·Dice + 0.2·Boundary

Bugs fixed relative to the student's version
    1. Tap/unfreeze mismatch.  The student unfroze encoder.layers[1,3,7,11]
       but read hidden_states[1,3,7,11] (= outputs of layers 0,2,6,10), so
       the layers being fine-tuned were NOT the ones whose features were used.
       Here a forward hook on resblocks[idx] captures the output of block idx,
       and the SAME blocks are unfrozen — taps and trainable layers align.
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
import open_clip
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
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
CLIP_SIZE = 336                       # ViT-B/16 @ 336 → 21×21 patch grid
CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

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
        A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

def build_val_aug(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
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
#  CLIP ENCODER  (open_clip ViT-B/16 + hook-based multi-scale taps)
# ─────────────────────────────────────────────────────────────────────────────
class CLIPEncoder(nn.Module):
    """
    Taps 4 transformer blocks of CLIP ViT-B/16 for multi-scale features.

    A forward hook on resblocks[idx] captures the OUTPUT of block idx, and the
    SAME blocks are unfrozen for fine-tuning — so taps and trainable layers are
    consistent (the student's version tapped and trained different layers).

    HOOK_INDICES span the depth: shallow → early-mid → mid-deep → deep.
    Every tap is at the ViT patch resolution (grid×grid, e.g. 21×21 @ 336).
    """
    HOOK_INDICES = [2, 5, 8, 11]   # 0-based block indices (out of 12)
    FEAT_DIM     = 768             # ViT-B hidden dimension

    def __init__(self, clip_visual, clip_size):
        super().__init__()
        self.clip_visual = clip_visual
        self.grid        = clip_size // 16          # 21 for 336×336 inputs
        self._cache: dict = {}

        # Register hooks
        self._handles = []
        for level, idx in enumerate(self.HOOK_INDICES):
            h = self.clip_visual.transformer.resblocks[idx].register_forward_hook(
                self._make_hook(level))
            self._handles.append(h)

        # Freeze everything, then unfreeze the tapped blocks + the patch/pos
        # embeddings (so the fine-tuned weights match the features we read).
        for p in self.clip_visual.parameters():
            p.requires_grad = False
        for idx in self.HOOK_INDICES:
            for p in self.clip_visual.transformer.resblocks[idx].parameters():
                p.requires_grad = True
        for attr in ('conv1', 'ln_pre'):
            mod = getattr(self.clip_visual, attr, None)
            if isinstance(mod, nn.Module):
                for p in mod.parameters():
                    p.requires_grad = True
        for attr in ('class_embedding', 'positional_embedding'):
            param = getattr(self.clip_visual, attr, None)
            if isinstance(param, nn.Parameter):
                param.requires_grad = True

        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[CLIPEncoder] trainable: {n:,}  (blocks {self.HOOK_INDICES} + embeddings)")

    def _make_hook(self, level):
        def hook(_m, _i, output):
            self._cache[level] = output
        return hook

    def _tokens_to_spatial(self, tokens):
        """Handle both open_clip token layouts → (B, C, grid, grid)."""
        num_tokens = self.grid * self.grid + 1   # patches + CLS
        if tokens.shape[0] == num_tokens:
            # LND: (seq, B, C)
            patches = tokens[1:]                          # (N, B, C)
            B, C    = patches.shape[1], patches.shape[2]
            return patches.permute(1, 2, 0).view(B, C, self.grid, self.grid)
        # NLD: (B, seq, C)
        patches = tokens[:, 1:, :]                        # (B, N, C)
        B, _, C = patches.shape
        return patches.permute(0, 2, 1).view(B, C, self.grid, self.grid)

    def forward(self, x):
        """Returns 4 maps (B, 768, grid, grid): [shallow, early-mid, mid-deep, deep]."""
        self._cache = {}
        _ = self.clip_visual(x)   # fires hooks
        # Clamp activations — unclamped fp16 ViT outputs can blow up after many
        # epochs and cause NaN downstream (same guard clipsam.py uses).
        return [self._tokens_to_spatial(self._cache[l]).clamp(-100, 100)
                for l in range(len(self.HOOK_INDICES))]

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
    4-stage UNet decoder with CLIP skip connections + optional high-res raw-image skip.

    All CLIP taps are at the patch grid (g×g).  The decoder progressively
    upsamples ×2 per stage and fuses the corresponding (upsampled) CLIP tap:

        stage0  bottleneck : proj(deep)            → 256ch  @ g
        stage1             : up + proj(mid-deep)   → 128ch  @ 2g
        stage2             : up + proj(early-mid)  →  64ch  @ 4g
        stage3             : up + proj(shallow) [+ img_skip(32)] → 32ch @ 8g
        head               : up → input_size, Conv1×1 → 1 logit

    Stage sizes are derived from the feature grid and input size at runtime,
    so any --clip-size works (student hardcoded 14/28/56/112/224).
    """
    def __init__(self, clip_ch=768, dropout=0.2, use_raw_img_skip=False):
        super().__init__()
        self.use_raw_img_skip = use_raw_img_skip
        # Dropout schedule: heaviest at the wide bottleneck, tapering to 0 at
        # the head so fine boundary detail is not randomly zeroed near output.
        d0, d1, d2, d3 = dropout, dropout * 0.75, dropout * 0.5, 0.0
        self.proj_deep = nn.Sequential(   # deepest tap → bottleneck
            nn.Conv2d(clip_ch, 512, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.proj_md   = nn.Sequential(   # mid-deep tap
            nn.Conv2d(clip_ch, 256, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.proj_em   = nn.Sequential(   # early-mid tap
            nn.Conv2d(clip_ch, 128, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.proj_sh   = nn.Sequential(   # shallow tap
            nn.Conv2d(clip_ch, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # Optional high-res raw-image skip → recovers edge detail ViT's grid lacks.
        # Disable it for ablations that should use CLIP features only.
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
        maps    : [shallow, early-mid, mid-deep, deep], each (B, 768, g, g)
        raw_img : optional normalised input (B, 3, S, S). Required only when
                  ``use_raw_img_skip=True``; otherwise it is used only to infer
                  the final output size.
        """
        if self.use_raw_img_skip and raw_img is None:
            raise ValueError("raw_img must be provided when use_raw_img_skip=True")

        g  = maps[0].shape[-1]            # patch grid (e.g. 21 @ 336)
        S  = raw_img.shape[-1] if raw_img is not None else g * 16
        s1, s2, s3 = g * 2, g * 4, g * 8  # per-stage spatial sizes

        f_deep = self.proj_deep(maps[3])  # bottleneck
        f_md   = self.proj_md(maps[2])
        f_em   = self.proj_em(maps[1])
        f_sh   = self.proj_sh(maps[0])

        img_feat = None
        if self.use_raw_img_skip:
            img_feat = self.img_skip(self._up(raw_img, (s3, s3)))  # (B,32,8g,8g)

        x = self.stage0(f_deep)                                 # @ g
        x = self.stage1(torch.cat([self._up(x, (s1, s1)),
                                   self._up(f_md, (s1, s1))], 1))   # @ 2g
        x = self.stage2(torch.cat([self._up(x, (s2, s2)),
                                   self._up(f_em, (s2, s2))], 1))   # @ 4g
        stage3_inputs = [self._up(x, (s3, s3)), self._up(f_sh, (s3, s3))]
        if img_feat is not None:
            stage3_inputs.append(img_feat)
        x = self.stage3(torch.cat(stage3_inputs, 1))                # @ 8g
        x = self._up(x, (S, S))
        return self.head(x)                                      # (B, 1, S, S)

# ─────────────────────────────────────────────────────────────────────────────
#  FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────
class CLIPUNet(nn.Module):
    def __init__(self, clip_visual, clip_size, dropout=0.2,
                 use_raw_img_skip=False):
        super().__init__()
        self.encoder = CLIPEncoder(clip_visual, clip_size)
        self.decoder = CLIPUNetDecoder(clip_ch=CLIPEncoder.FEAT_DIM,
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
    fig.suptitle('CLIPUNet Training Curves (ISIC-2017)',
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
        'model': 'CLIPUNet',
        'checkpoint': {
            'best_epoch': best_epoch,
            'best_val_dice': round(float(best_dice), 6),
        },
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'clip_size': args.clip_size,
            'lr_clip': args.lr_clip,
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
            'lr_clip': [round(float(v), 10) for v in history.get('lr_clip', [])],
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
    p = argparse.ArgumentParser(description="CLIP-UNet ISIC-2017 segmentation")
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
    p.add_argument('--clip-size',   type=int,   default=CLIP_SIZE,
                   help="Input resolution for CLIP ViT-B/16. Default 336 gives a 21×21 patch grid for higher-resolution features; use 224 for lower memory.")
    p.add_argument('--lr-clip',     type=float, default=1e-5,
                   help="LR for the fine-tuned CLIP blocks")
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
                   help="Disable the high-resolution raw-image skip in the UNet decoder for CLIP-feature-only ablations.")
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

    # ── CLIP ViT-B/16 (open_clip — same loader as clipsam.py) ─────────────────
    print("[INFO] Loading CLIP ViT-B/16 …")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name='ViT-B-16',
        pretrained='openai',
        device=device,
        force_image_size=args.clip_size,
        force_quick_gelu=True)

    model = CLIPUNet(clip_model.visual, args.clip_size,
                     dropout=args.dropout,
                     use_raw_img_skip=not args.no_raw_img_skip).to(device)

    # Shape sanity check
    with torch.no_grad():
        dummy = torch.randn(2, 3, args.clip_size, args.clip_size, device=device)
        out   = model(dummy)
        assert out.shape == (2, 1, args.clip_size, args.clip_size), \
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
                                     args.clip_size, args.threshold, "Validation")
        val_results = {'mean_dice': round(float(val_d.mean()), 6),
                       'std_dice': round(float(val_d.std()), 6),
                       'mean_iou': round(float(val_j.mean()), 6),
                       'std_iou': round(float(val_j.std()), 6)}
        test_results = None
        if os.path.isdir(args.test_img):
            td, tj = evaluate_loop(model, args.test_img, args.test_mask, device,
                                   args.clip_size, args.threshold, "Test")
            test_results = {'mean_dice': round(float(td.mean()), 6),
                            'std_dice': round(float(td.std()), 6),
                            'mean_iou': round(float(tj.mean()), 6),
                            'std_iou': round(float(tj.std()), 6)}
            save_visualisations(model, args.test_img, args.test_mask, device,
                                args.clip_size,
                                os.path.join(args.output_dir, 'test_outputs'),
                                args.threshold)
        save_results_yaml(state.get('history', {}), state.get('best_dice', 0.0),
                          val_results, test_results, args, args.output_dir,
                          best_epoch=state.get('epoch'))
        return

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = SkinDataset(args.img_path, args.mask_path, args.clip_size, augment=True)
    if args.train_with_val_data:
        val_train_ds = SkinDataset(args.val_img, args.val_mask, args.clip_size,
                                   augment=True)
        train_ds = ConcatDataset([train_ds, val_train_ds])
        print("[WARN] Research mode enabled: validation split appended to "
              "training data. Validation metrics are no longer held-out.")
        print(f"[INFO] Combined train samples: {len(train_ds)}")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=True)

    # ── Optimiser: two LR groups (CLIP vs decoder) × decay/no-decay ───────────
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

    clip_d, clip_nd = split_decay(model.encoder)
    dec_d,  dec_nd  = split_decay(model.decoder)
    optimizer = torch.optim.AdamW([
        {'params': clip_d,  'lr': args.lr_clip, 'weight_decay': args.weight_decay},
        {'params': clip_nd, 'lr': args.lr_clip, 'weight_decay': 0.0},
        {'params': dec_d,   'lr': args.lr_dec,  'weight_decay': args.weight_decay},
        {'params': dec_nd,  'lr': args.lr_dec,  'weight_decay': 0.0},
    ])
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
    print("  Encoder : CLIP ViT-B/16  blocks 2,5,8,11  (lr=%.0e)" % args.lr_clip)
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
        'val_dice': [], 'val_iou': [], 'lr_clip': [], 'lr_decoder': [],
    }
    best_epoch = None
    diag_path = os.path.join(args.output_dir, 'training_diagnostics.csv')
    diag_fields = [
        'epoch', 'loss', 'train_dice', 'train_iou', 'val_dice', 'val_iou',
        'lr_clip', 'lr_decoder', 'grad_norm_mean', 'grad_norm_max',
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
                                     args.clip_size, args.threshold, "Validation")
        mean_val = float(val_d.mean())
        mean_val_iou = float(val_j.mean())
        lr_clip = optimizer.param_groups[0]['lr']
        lr_dec = optimizer.param_groups[2]['lr']

        history['loss'].append(tr_loss)
        history['train_dice'].append(tr_d)
        history['train_iou'].append(tr_j)
        history['val_dice'].append(mean_val)
        history['val_iou'].append(mean_val_iou)
        history['lr_clip'].append(lr_clip)
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
            'lr_clip': lr_clip,
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
                                 args.clip_size, args.threshold, "Validation")
    val_results = {'mean_dice': round(float(val_d.mean()), 6),
                   'std_dice': round(float(val_d.std()), 6),
                   'mean_iou': round(float(val_j.mean()), 6),
                   'std_iou': round(float(val_j.std()), 6)}

    test_results = None
    if os.path.isdir(args.test_img):
        td, tj = evaluate_loop(model, args.test_img, args.test_mask, device,
                               args.clip_size, args.threshold, "Test")
        test_results = {'mean_dice': round(float(td.mean()), 6),
                        'std_dice': round(float(td.std()), 6),
                        'mean_iou': round(float(tj.mean()), 6),
                        'std_iou': round(float(tj.std()), 6)}
        save_visualisations(model, args.test_img, args.test_mask, device,
                            args.clip_size,
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
