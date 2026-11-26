# -*- coding: utf-8 -*-
"""
ESC-Net adapted for SIP Salient Object Detection
Paper : arXiv:2411.14723v2  "ESC-Net: Effective SAM Combination for
         Open-Vocabulary Semantic Segmentation"
Domain: Binary salient object segmentation on the SIP dataset

Changes from original Colab notebook
─────────────────────────────────────
Bug fixes
  • Removed Google Colab / Google Drive dependencies
  • Removed !pip shell-magic (use requirements.txt instead)
  • Fixed syntax error  `model.train()S` → `model.train()`
  • Supports SIP-style mask naming where image and mask share a basename

Architectural alignment with the paper
  • CLIP backbone: RN50 → ViT-B/16 @ 336 × 336  (paper §3.1)
  • Multi-scale ViT feature extraction via forward hooks on
    transformer blocks [2, 5, 8, 11]
  • Full bidirectional cross-attention in SAM block: adds the missing
    cross_attn_token_to_image step  (paper §3.3)
  • VLF upgraded from a plain TransformerEncoderLayer to a Swin-style
    window attention block (window_size=7) for vision guidance  (§3.4)
  • Number of ESC blocks: 2 → 4  (paper uses N=4)

Training improvements
  • Dual learning rate:  2e-5 for pre-trained CLIP/SAM params,
                         2e-4 for new layers  (paper §4.1)
  • A100-tuned defaults: batch=16, num_workers=8
  • CLI argparse for every key hyperparameter
  • Salient-object caption/prompt bank for SIP experiments
"""

import os
import csv
import json
import shutil
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import open_clip
from segment_anything import sam_model_registry
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')          # headless — no display needed on server
import matplotlib.pyplot as plt
import yaml

from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset root: set SOD_DATA_ROOT, or place datasets under ./data (see README).
DATASET_ROOT = os.environ.get('SOD_DATA_ROOT', os.path.join(BASE_DIR, 'data'))
IMG_PATH               = os.path.join(DATASET_ROOT, 'DUTS_Dataset/DUTS-TR/DUTS-TR-Image')
MASK_PATH              = os.path.join(DATASET_ROOT, 'DUTS_Dataset/DUTS-TR/DUTS-TR-Mask')
VALIDATION_IMAGES_PATH = os.path.join(DATASET_ROOT, 'DUTS_Dataset/DUTS-TE/DUTS-TE-Image')
VALIDATION_MASK_PATH   = os.path.join(DATASET_ROOT, 'DUTS_Dataset/DUTS-TE/DUTS-TE-Mask')
# No default test path — use --eval-datasets for multi-dataset eval
TEST_IMAGES_PATH       = ''
TEST_MASK_PATH         = ''
OUTPUT_DIR             = os.path.join(BASE_DIR, 'outputs_clipSAM')
PRETRAINED_DIR         = os.path.join(BASE_DIR, 'pretrained_models')

SOD_DATASET_ROOT = DATASET_ROOT
BENCHMARK_REGISTRY = {
    'DUTS-TE': {
        'images': os.path.join(SOD_DATASET_ROOT, 'DUTS_Dataset/DUTS-TE/DUTS-TE-Image'),
        'masks': os.path.join(SOD_DATASET_ROOT, 'DUTS_Dataset/DUTS-TE/DUTS-TE-Mask'),
        'type': 'directory',
    },
    'DUT-O': {
        'images': os.path.join(SOD_DATASET_ROOT, 'DUT-O/DUT-OMRON-image'),
        'masks': os.path.join(SOD_DATASET_ROOT, 'DUT-O/pixelwiseGT-new-PNG'),
        'type': 'directory',
    },
    'ECSSD': {
        'images': os.path.join(SOD_DATASET_ROOT, 'ECSSD/images'),
        'masks': os.path.join(SOD_DATASET_ROOT, 'ECSSD/ground_truth_mask'),
        'type': 'directory',
    },
    'HKU-IS': {
        'images_npy': os.path.join(SOD_DATASET_ROOT, 'HKU-IS/images.npy'),
        'masks_npy': os.path.join(SOD_DATASET_ROOT, 'HKU-IS/gt.npy'),
        'type': 'npy',
    },
    # DEPRECATED (2026-11): the 256-image NPY subset gave broken results
    # (S-measure ~0.45, likely image/mask misalignment). Kept for records only.
    # Use 'PASCAL-S-full' (850 images) for all reported numbers.
    'PASCAL-S': {
        'images_npy': os.path.join(SOD_DATASET_ROOT, 'pascals/pascals_test_images.npy'),
        'masks_npy': os.path.join(SOD_DATASET_ROOT, 'pascals/pascals_test_masks.npy'),
        'type': 'npy',
    },
    # Full PASCAL-S (850 images). Images (.jpg) and masks (.png) share one
    # folder with matching stems, so only .jpg files are listed as images.
    'PASCAL-S-full': {
        'images': os.path.join(SOD_DATASET_ROOT, 'pascals-full/PASCALS'),
        'masks': os.path.join(SOD_DATASET_ROOT, 'pascals-full/PASCALS'),
        'type': 'directory',
        'image_exts': ('.jpg', '.jpeg'),
    },
    'SIP': {
        'images': os.path.join(SOD_DATASET_ROOT, 'SIP_Dataset/Test/Images'),
        'masks': os.path.join(SOD_DATASET_ROOT, 'SIP_Dataset/Test/Masks'),
        'type': 'directory',
    },
    # (2026-11) Full SIP (929 images = Train 1-800 + Test 801-929, symlinks).
    # Models are trained on DUTS-TR only, so all SIP images are unseen; RGB-D
    # papers report on all 929 images.
    'SIP-full': {
        'images': os.path.join(SOD_DATASET_ROOT, 'SIP-full/Images'),
        'masks': os.path.join(SOD_DATASET_ROOT, 'SIP-full/Masks'),
        'type': 'directory',
    },
}

# Salient-object caption (better CLIP alignment for SIP)
CAPTION = "a photo of a salient object"


def _safe_split_dirname(split_name: str) -> str:
    """Return a filesystem-safe split/dataset name for output folders."""
    safe = (split_name or 'split').strip().lower().replace(' ', '_').replace('-', '_')
    return safe or 'split'


def _save_predicted_mask(mask_bin: np.ndarray, output_dir: str, filename: str) -> str:
    """Save a binary predicted mask as an 8-bit PNG and return its path."""
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(filename))[0]
    out_path = os.path.join(output_dir, stem + '.png')
    mask_u8 = (np.asarray(mask_bin).astype(np.uint8) * 255)
    Image.fromarray(mask_u8, mode='L').save(out_path)
    return out_path

# ── Multi-prompt bank ─────────────────────────────────────────────────────────
# Diverse textual descriptions of salient objects.  Each prompt is encoded
# independently by the CLIP text encoder; their cosine-similarity maps with
# the visual features are max-pooled so that ANY prompt that activates a
# region (including small salient objects) contributes to the final correlation map.
SALIENT_PROMPT_BANK = [
    "a photo of a salient object",
    "an image containing the main foreground object",
    "a prominent object separated from the background",
    "the most visually distinctive object in the scene",
    "a foreground object with clear boundaries",
    "a salient region in a natural image",
    "the central object of interest",
    "a high contrast object against its background",
    "a complete object mask for the main subject",
    "the object that attracts visual attention",
]

RANDOM_PROMPT_BANK = [
    "a photograph taken outdoors",
    "an image with various colors",
    "a scene with natural lighting",
    "a picture containing different textures",
    "a visual with multiple elements",
    "an everyday scene captured on camera",
    "a composition with foreground and background",
    "a photograph of an everyday setting",
    "an image showing depth and perspective",
    "a captured moment in time",
]

# Improved prompt bank: semantically diverse prompts that cover different
# object categories and visual properties CLIP actually understands.
# Previous bank had 10 paraphrases of "salient object" — these are diverse.
DIVERSE_PROMPT_BANK = [
    # Category-specific (CLIP trained on real captions with these)
    "a person standing in the scene",
    "an animal in a natural environment",
    "a vehicle on the road",
    "a close-up of an everyday object",
    # Spatial/structural (foreground-background separation)
    "the main subject in the foreground",
    "a large object in the center of the image",
    "something that stands out from the background",
    # Visual properties CLIP understands from training data
    "a brightly colored object in focus",
    "a sharp subject with blurred background",
    "a clearly visible object against a plain background",
]


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
CLIP_SIZE = 336    # ViT-B/16 native resolution — matches paper
CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

# ─────────────────────────────────────────────────────────────────────────────
#  AUGMENTATION  — salient-object tuned (updated to 336 × 336)
# ─────────────────────────────────────────────────────────────────────────────
train_aug = A.Compose([
    A.Resize(CLIP_SIZE, CLIP_SIZE),
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.3),
    # ShiftScaleRotate is a special case of Affine; use Affine directly (v2+)
    A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
             scale=(0.85, 1.15), rotate=(-45, 45), p=0.4,
             border_mode=cv2.BORDER_REFLECT_101),
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
    # var_limit was removed in albumentations v2; std_range=(sqrt(10),sqrt(50))/255
    A.GaussNoise(std_range=(0.012, 0.028), p=0.2),
    # max_holes/fill_value API removed in albumentations v2
    A.CoarseDropout(num_holes_range=(1, 8),
                    hole_height_range=(1, 28), hole_width_range=(1, 28),
                    fill=0, p=0.1),
    A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})

val_aug = A.Compose([
    A.Resize(CLIP_SIZE, CLIP_SIZE),
    A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def generate_captions(images_dir, caption_text, save_path):
    captions = {}
    for fn in sorted(os.listdir(images_dir)):
        if fn.lower().endswith(IMAGE_EXTENSIONS):
            captions[fn] = caption_text
    with open(save_path, 'w') as f:
        json.dump(captions, f, indent=2)
    print(f"[INFO] captions.json saved → {save_path}  ({len(captions)} entries)")
    return captions


def inspect_folder(folder, label=""):
    if not os.path.exists(folder):
        print(f"[ERROR] Folder does not exist: {folder}")
        return []
    all_files = os.listdir(folder)
    from collections import Counter
    exts = Counter(os.path.splitext(f)[1].lower() for f in all_files)
    print(f"[DEBUG] {label} '{folder}' — {len(all_files)} items | {dict(exts)}")
    print(f"[DEBUG] Sample: {sorted(all_files)[:5]}")
    return all_files


def find_mask_path(masks_dir, image_fn):
    """
    SIP mask naming convention:
        image: <base>.<ext>
        mask : <base>.png or <base>.<ext>
    Falls back across common mask extensions.
    """
    base = os.path.splitext(image_fn)[0]
    for ext in ('.png', '.jpg', '.jpeg'):
        # p = os.path.join(masks_dir, base + '_segmentation' + ext)
        p = os.path.join(masks_dir, base + ext)
        if os.path.exists(p):
            return p
    return None

# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────
class SalientObjectDataset(Dataset):
    def __init__(self, images_dir, masks_dir, captions,
                 augment=False):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.captions   = captions
        self.transform  = train_aug if augment else val_aug

        all_files  = inspect_folder(images_dir, label="Images")
        if masks_dir:
            inspect_folder(masks_dir, label="Masks ")

        all_images = sorted([f for f in all_files if f.lower().endswith(IMAGE_EXTENSIONS)])
        print(f"[DEBUG] Image files detected: {len(all_images)}")

        if masks_dir is not None:
            valid, no_mask = [], []
            for fn in all_images:
                (valid if find_mask_path(masks_dir, fn) else no_mask).append(fn)
            if no_mask:
                print(f"[Warning] {len(no_mask)} images skipped (no mask). "
                      f"Example: {no_mask[:3]}")
            self.filenames = valid
        else:
            self.filenames = all_images

        print(f"[INFO] Dataset loaded: {len(self.filenames)} samples "
              f"(augment={augment})")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn      = self.filenames[idx]
        image   = np.array(Image.open(os.path.join(self.images_dir, fn)).convert('RGB'))
        caption = self.captions.get(fn, CAPTION)

        if self.masks_dir is None:
            out = self.transform(image=image)
            return out['image'].float(), caption, fn

        mask_np = np.array(
            Image.open(find_mask_path(self.masks_dir, fn)).convert('L'),
            dtype=np.uint8)

        out    = self.transform(image=image, mask=mask_np)
        img_t  = out['image'].float()
        mask_t = (out['mask'].float() / 255.0).unsqueeze(0)
        return img_t, caption, mask_t, fn

# ─────────────────────────────────────────────────────────────────────────────
#  SAM DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
def download_sam_checkpoint(path):
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        print(f"[INFO] Reusing SAM checkpoint → {path}")
        return path

    print("[INFO] Downloading SAM ViT-B checkpoint once …")
    import requests
    tmp_path = path + '.tmp'
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(tmp_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    os.replace(tmp_path, path)
    print(f"[INFO] SAM checkpoint saved → {path}")
    return path

# ─────────────────────────────────────────────────────────────────────────────
#  LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
class TverskyLoss(nn.Module):
    """FN-heavy Tversky — suited for imbalanced medical segmentation."""
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, pred_logits, target):
        p   = torch.sigmoid(pred_logits)
        p, t = p.view(-1), target.view(-1)
        tp   = (p * t).sum()
        fp   = (p * (1 - t)).sum()
        fn   = ((1 - p) * t).sum()
        return 1.0 - (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)


class FocalBCE(nn.Module):
    """Focal BCE — down-weights easy negatives."""
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred_logits, target, weight=None):
        bce  = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        pt   = (target * torch.sigmoid(pred_logits) +
                (1 - target) * (1 - torch.sigmoid(pred_logits))).clamp(1e-8, 1 - 1e-8)
        focal = ((1 - pt) ** self.gamma) * bce
        if weight is not None:
            focal = focal * weight
        return focal.mean()


def sobel_edge_weight(mask_gt, device):
    """Edge-amplified weight map from GT mask (amplifies boundary loss)."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    ky = kx.transpose(-2, -1)
    gx   = F.conv2d(mask_gt, kx, padding=1)
    gy   = F.conv2d(mask_gt, ky, padding=1)
    edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6).clamp(0, 1)
    edge = F.max_pool2d(edge, kernel_size=3, stride=1, padding=1)
    return 1.0 + 3.0 * edge          # range [1, 4]


class CompoundLoss(nn.Module):
    """Tversky + boundary-weighted FocalBCE + λ·MSE objective."""
    def __init__(self, w_tversky=0.6, w_focal=0.4, mse_lambda=0.1,
                 gamma=2.0, tversky_alpha=0.7, tversky_beta=0.3):
        super().__init__()
        self.w_tversky = w_tversky
        self.w_focal   = w_focal
        self.mse_lambda = mse_lambda
        self.tversky   = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.focal     = FocalBCE(gamma=gamma)

    def forward(self, pred_logits, target):
        edge_w = sobel_edge_weight(target, pred_logits.device)
        probs = torch.sigmoid(pred_logits)
        return (self.w_tversky * self.tversky(pred_logits, target) +
                self.w_focal   * self.focal(pred_logits, target, weight=edge_w) +
                self.mse_lambda * F.mse_loss(probs, target))

    def forward_with_stats(self, pred_logits, target):
        """Return (total_loss, dict) where dict has per-component scalar values.
        Avoids recomputing Tversky/Focal/MSE twice compared to calling forward + sub-losses."""
        edge_w  = sobel_edge_weight(target, pred_logits.device)
        tv_loss = self.tversky(pred_logits, target)
        fc_loss = self.focal(pred_logits, target, weight=edge_w)
        mse_loss = F.mse_loss(torch.sigmoid(pred_logits), target)
        total = (self.w_tversky * tv_loss +
                 self.w_focal * fc_loss +
                 self.mse_lambda * mse_loss)
        return total, {
            'tversky': tv_loss.item(),
            'focal': fc_loss.item(),
            'mse': mse_loss.item(),
        }

# ─────────────────────────────────────────────────────────────────────────────
#  WINDOW ATTENTION BLOCK  (Swin-like, no timm dependency)
#
#  Implements local self-attention over non-overlapping spatial windows.
#  For 21 × 21 feature maps with window_size=7: exactly 9 windows of 7×7.
# ─────────────────────────────────────────────────────────────────────────────
class WindowAttentionBlock(nn.Module):
    """
    Pre-norm Swin-style window attention block.
    Input / output shape: (B, C, H, W).
    """
    def __init__(self, dim, window_size=7, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.ws = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(p=0.0),
            nn.Linear(dim * 4, dim))

    # ── window helpers ────────────────────────────────────────────────────────
    def _partition(self, x):
        """(B, H, W, C) → (n_wins, ws*ws, C); also returns (B, Hp, Wp)"""
        B, H, W, C = x.shape
        ws = self.ws
        ph = (ws - H % ws) % ws
        pw = (ws - W % ws) % ws
        if ph or pw:
            x = F.pad(x.permute(0, 3, 1, 2), (0, pw, 0, ph)).permute(0, 2, 3, 1)
        Hp, Wp = H + ph, W + pw
        x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)
        return windows, B, H, W, Hp, Wp

    def _reverse(self, windows, B, H, W, Hp, Wp):
        """(n_wins, ws*ws, C) → (B, H, W, C)"""
        ws = self.ws
        C  = windows.shape[-1]
        x  = windows.view(B, Hp // ws, Wp // ws, ws, ws, C)
        x  = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)
        return x[:, :H, :W, :].contiguous()

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, H, W)"""
        # BCHW → BHWC
        x_hwc = x.permute(0, 2, 3, 1)

        # Window attention with residual
        # Force float32: the concat input is 1536-d (768+768) and QK^T in
        # fp16 with d_head=192 regularly overflows after ~20 epochs of
        # weight drift, producing all-NaN outputs.
        shortcut = x_hwc
        wins, B_, H_, W_, Hp, Wp = self._partition(self.norm1(x_hwc))
        orig_dtype = wins.dtype
        attn_out, _ = self.attn(
            wins.float(), wins.float(), wins.float(),
            need_weights=False)
        attn_out = attn_out.to(orig_dtype)
        x_hwc = self._reverse(attn_out, B_, H_, W_, Hp, Wp) + shortcut

        # MLP with residual
        x_hwc = x_hwc + self.mlp(self.norm2(x_hwc))

        return x_hwc.permute(0, 3, 1, 2)   # BHWC → BCHW

# ─────────────────────────────────────────────────────────────────────────────
#  PSEUDO PROMPT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
class PseudoPromptGenerator(nn.Module):
    """
    From a 2-D correlation map, generate N_o pseudo point prompts and
    pseudo mask prompts via softmax → binarise → K-means clustering.
    (Paper §3.2  "Pseudo Prompt Generator")
    """
    def __init__(self, threshold=0.1, num_regions=5,
                 top_fraction=0.15, min_pixels=None, adaptive_fallback=True):
        super().__init__()
        self.threshold    = threshold
        self.num_regions  = num_regions
        self.top_fraction = top_fraction  # fraction of map used as fallback floor
        self.min_pixels   = min_pixels    # explicit pixel floor (None → computed)
        self.adaptive_fallback = adaptive_fallback

    def forward(self, corr_maps):
        # corr_maps: (B, H, W)
        B, H, W    = corr_maps.shape
        P          = self.num_regions
        all_points = torch.zeros(B, P, 2,    dtype=torch.float32)
        all_labels = torch.zeros(B, P,       dtype=torch.int64)
        all_masks  = torch.zeros(B, P, H, W, dtype=torch.uint8)

        for b in range(B):
            prob    = F.softmax(corr_maps[b].detach().cpu().reshape(-1), dim=0).view(H, W)
            prob_np = prob.numpy()

            # ── Step 1: fixed-threshold binarisation ─────────────────────────
            bin_mask = (prob_np > self.threshold).astype(np.uint8)

            # ── Step 2: adaptive fallback — guarantee at least min_pix pixels ─
            # After softmax over H×W pixels the average probability is ~1/(H*W).
            # threshold=0.1 requires 44× the average to activate, so flat/diffuse
            # correlation maps (early training, small salient objects) yield zero pixels.
            # When that happens we fall back to the top top_fraction of the map
            # so SAMBlock always receives substantive spatial guidance.
            min_pix = (self.min_pixels
                       if self.min_pixels is not None
                       else max(self.num_regions * 4, int(self.top_fraction * H * W)))

            if self.adaptive_fallback and int(bin_mask.sum()) < min_pix:
                n_select = max(min_pix, int(self.top_fraction * H * W))
                flat     = prob_np.ravel()
                # argpartition is O(N) — faster than a full sort
                topk_idx = np.argpartition(flat, -n_select)[-n_select:]
                bin_mask = np.zeros(H * W, dtype=np.uint8)
                bin_mask[topk_idx] = 1
                bin_mask = bin_mask.reshape(H, W)

            coords     = np.column_stack(np.nonzero(bin_mask))
            if len(coords) == 0:
                # Naive threshold-only PPG: empty support emits zero prompts/masks.
                continue
            n_clusters = min(P, len(coords))
            # (2026-11) Single-threaded KMeans: on <600 points, multi-threading
            # gives no speedup and under CPU contention cost ~880 ms/call vs
            # ~12 ms single-threaded. Same algorithm and random_state.
            with threadpool_limits(limits=1):
                km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(coords)

            for slot in range(n_clusters):
                region = coords[km.labels_ == slot]
                vals   = [prob[int(y), int(x)].item() for y, x in region]
                y, x   = region[int(np.argmax(vals))]
                all_points[b, slot] = torch.tensor([float(x), float(y)], dtype=torch.float32)
                all_labels[b, slot] = 1
                m = np.zeros((H, W), dtype=np.uint8)
                m[region[:, 0], region[:, 1]] = 1
                all_masks[b, slot] = torch.from_numpy(m)

        return all_points, all_labels, all_masks

# ─────────────────────────────────────────────────────────────────────────────
#  SAM BLOCK  (bidirectional cross-attention)
#
#  Paper §3.3: SA(F_l) → BCA(F_v)
#  "bidirectional cross-attention (BCA) for image-to-prompt and
#   prompt-to-image, updating the image embedding feature F_v"
#
#  Implementation uses SAM's own Attention modules to match the pre-trained
#  weight layout exactly.
# ─────────────────────────────────────────────────────────────────────────────
class SAMBlock(nn.Module):
    def __init__(self, sam_decoder_block, prompt_encoder, feature_dim, num_prompts,
                 bidirectional=True):
        super().__init__()
        self.decoder_block  = sam_decoder_block
        self.prompt_encoder = prompt_encoder
        sam_dim             = sam_decoder_block.self_attn.q_proj.in_features
        self.sam_dim        = sam_dim
        self.num_prompts    = num_prompts
        self.bidirectional = bidirectional
        # Project CLIP features → SAM embedding dim
        self.proj  = nn.Conv2d(feature_dim, sam_dim, kernel_size=1)
        # Fuse per-prompt outputs back to feature_dim
        self.conv  = nn.Conv2d(sam_dim * num_prompts, feature_dim, kernel_size=1)

    def forward(self, feat_map, input_points, input_labels, input_masks):
        device     = feat_map.device
        B, _, H, W = feat_map.shape
        P          = input_points.shape[1]

        # Project features to SAM space
        feat_sam   = self.proj(feat_map)                               # (B, sam_dim, H, W)

        # Flatten prompt dims for SAM prompt encoder (treats each prompt separately)
        pts_f  = input_points.to(device).reshape(B * P, 1, 2).float()
        lbl_f  = input_labels.to(device).reshape(B * P).unsqueeze(1).float()
        masks_f = input_masks.to(device).reshape(B * P, 1, H, W).float()

        sparse_emb, dense_emb = self.prompt_encoder(
            points=(pts_f, lbl_f), boxes=None, masks=masks_f)

        # Reshape prompt embeddings
        T          = sparse_emb.shape[1]
        sparse_emb = sparse_emb.view(B, P, T, self.sam_dim)           # (B, P, T, sam_dim)
        dense_emb  = F.interpolate(dense_emb, size=(H, W),
                                   mode='bilinear', align_corners=False)
        dense_emb  = dense_emb.view(B, P, self.sam_dim, H, W)         # (B, P, sam_dim, H, W)

        # Image tokens for attention
        img_tokens = feat_sam.flatten(2).permute(0, 2, 1)             # (B, H*W, sam_dim)

        out_list = []
        for n in range(P):
            pt  = sparse_emb[:, n]                                     # (B, T, sam_dim)
            de  = dense_emb[:, n]                                      # (B, sam_dim, H, W)

            # 1) Prompt self-attention
            pt_sa = self.decoder_block.self_attn(pt, pt, pt)          # (B, T, sam_dim)

            # 2) Token-to-image cross-attention  (prompt queries, image K/V)
            #    Prompts gather spatial context from image features
            if self.bidirectional:
                pt_t2i = self.decoder_block.cross_attn_token_to_image(
                    pt_sa, img_tokens, img_tokens)                     # (B, T, sam_dim)
            else:
                pt_t2i = pt_sa

            # 3) Image-to-token cross-attention  (image queries, prompt K/V)
            #    Image features are refined by prompt information
            ir = self.decoder_block.cross_attn_image_to_token(
                img_tokens, pt_t2i, pt_t2i)                           # (B, H*W, sam_dim)

            # Reshape to spatial map and blend with dense prompt.
            # Clamp prevents fp16 overflow from the de*feat_sam product
            # accumulating over many training steps.
            xmap = ir.permute(0, 2, 1).view(B, self.sam_dim, H, W)
            out_list.append((xmap + de * feat_sam).clamp(-100, 100))

        # Pad if fewer prompts were generated than num_prompts
        while len(out_list) < self.num_prompts:
            out_list.append(torch.zeros_like(out_list[0]))

        return self.conv(torch.cat(out_list[:self.num_prompts], dim=1))  # (B, feat_dim, H, W)

# ─────────────────────────────────────────────────────────────────────────────
#  VISION-LANGUAGE FUSION BLOCK
#
#  Paper §3.4: two-stage refinement
#    Stage 1 (vision):  Swin-style window attention on [C_e; F'_v]
#    Stage 2 (text):    linear MHA with F_l as key and value
#                       (no positional embedding → class-agnostic)
# ─────────────────────────────────────────────────────────────────────────────
class VLFBlock(nn.Module):
    def __init__(self, feature_dim, text_dim, window_size=7):
        super().__init__()
        self.feature_dim = feature_dim
        in_ch = feature_dim * 2      # corr_embed (C) + feat_refined (C)

        # Stage 1: embed correlation → concat with feat → Swin window attention
        self.embed_corr  = nn.Conv2d(1, feature_dim, kernel_size=1)
        self.swin        = WindowAttentionBlock(in_ch, window_size=window_size, num_heads=8)
        self.proj_stage1 = nn.Conv2d(in_ch, feature_dim, 1, bias=False)

        # Stage 2: text-guided cross-attention (no positional encoding → order-agnostic)
        self.txt_proj = nn.Linear(text_dim, feature_dim)
        self.txt_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=8, batch_first=False)
        self.out_proj = nn.Linear(feature_dim, 1)          # project to scalar (corr map)

    def forward(self, corr_maps, feat_refined, txt_embeds):
        """
        corr_maps   : (B, N_c, H, W)
        feat_refined: (B, C, H, W)   refined by SAM block
        txt_embeds  : (B, text_dim)  CLIP text embeddings
        Returns     : (B, N_c, H, W)  refined correlation maps
        """
        B, N_c, H, W = corr_maps.shape

        # ── Stage 1: vision guidance ──────────────────────────────────────────
        corr_e   = self.embed_corr(corr_maps.view(B * N_c, 1, H, W))     # (B·Nc, C, H, W)
        feat_rep = feat_refined.unsqueeze(1).expand(-1, N_c, -1, -1, -1) \
                               .reshape(B * N_c, self.feature_dim, H, W) # (B·Nc, C, H, W)
        img_cat  = torch.cat([corr_e, feat_rep], dim=1)                   # (B·Nc, 2C, H, W)
        img_cat  = self.swin(img_cat)                                      # (B·Nc, 2C, H, W)
        img_cat  = self.proj_stage1(img_cat)                              # (B·Nc, C, H, W)

        # ── Stage 2: text guidance ────────────────────────────────────────────
        # Image tokens: (S, B·Nc, C) where S = H*W
        img_v   = img_cat.flatten(2).permute(2, 0, 1)
        # Text key/value: one text vector per batch item, expanded for N_c
        # Shape: (1, B·Nc, C)
        # Handle txt_embeds with batch=1 (shared caption) vs batch=B (per-image)
        # (2026-11, camera-ready) txt_embeds may also be (B, T, D): T text tokens
        # (prompt bank + caption). With a single token the softmax over keys is
        # always 1, so the output ignores corr/features (see GRADIENT_DIAGNOSTIC.md).
        if txt_embeds.dim() == 3:
            te = txt_embeds if txt_embeds.shape[0] == B else txt_embeds.expand(B, -1, -1)
            T = te.shape[1]
            txt_k = (self.txt_proj(te)                                    # (B, T, C)
                         .unsqueeze(1).expand(-1, N_c, -1, -1)            # (B, Nc, T, C)
                         .reshape(B * N_c, T, self.feature_dim)           # (B·Nc, T, C)
                         .permute(1, 0, 2))                               # (T, B·Nc, C)
            img_m, _ = self.txt_attn(query=img_v, key=txt_k, value=txt_k)
            out = self.out_proj(img_m)
            out = out.permute(1, 2, 0).view(B, N_c, H, W)
            if getattr(self, 'residual', False):
                out = out + corr_maps
            return out
        te = txt_embeds if txt_embeds.shape[0] == B else txt_embeds.expand(B, -1)
        txt_k   = (self.txt_proj(te)                                      # (B, C)
                       .unsqueeze(1).expand(-1, N_c, -1)                  # (B, Nc, C)
                       .reshape(B * N_c, self.feature_dim)                # (B·Nc, C)
                       .unsqueeze(0))                                      # (1, B·Nc, C)
        img_m, _ = self.txt_attn(query=img_v, key=txt_k, value=txt_k)    # (S, B·Nc, C)

        # Project to scalar correlation and reshape
        out = self.out_proj(img_m)                                        # (S, B·Nc, 1)
        out = out.permute(1, 2, 0).view(B, N_c, H, W)                   # (B, Nc, H, W)
        # (2026-11) --vlf-residual: refine instead of replace the correlation.
        if getattr(self, 'residual', False):
            out = out + corr_maps
        return out

# ─────────────────────────────────────────────────────────────────────────────
#  ESC BLOCK  (one ESC = PPG + SAMBlock + VLFBlock)
# ─────────────────────────────────────────────────────────────────────────────
class ESCBlock(nn.Module):
    def __init__(self, sam_dec_block, prompt_encoder,
                 feature_dim, num_prompts, text_dim,
                 adaptive_fallback=True, bidirectional_sam=True):
        super().__init__()
        self.ppg       = PseudoPromptGenerator(num_regions=num_prompts,
                                               adaptive_fallback=adaptive_fallback)
        self.sam_block = SAMBlock(sam_dec_block, prompt_encoder,
                                  feature_dim, num_prompts,
                                  bidirectional=bidirectional_sam)
        self.vlf       = VLFBlock(feature_dim, text_dim)

    def forward(self, feat_maps, text_embeds, corr_maps):
        # Use the first correlation channel (high-level) to guide PPG
        pts, labels, masks = self.ppg(corr_maps[:, 0])
        feat_ref           = self.sam_block(feat_maps['high'], pts, labels, masks)
        corr_ref           = self.vlf(corr_maps, feat_ref, text_embeds)
        new_feats = {**feat_maps, 'high': feat_ref}
        return new_feats, corr_ref, (pts, labels)

# ─────────────────────────────────────────────────────────────────────────────
#  MASK DECODER  (FPN + ASPP + correlation injection + deep supervision)
# ─────────────────────────────────────────────────────────────────────────────
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
            for r in rates])
        self.global_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        H, W  = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        ga    = F.interpolate(self.global_avg(x), size=(H, W),
                              mode='bilinear', align_corners=False)
        return self.proj(torch.cat(feats + [ga], dim=1))


class MaskDecoder(nn.Module):
    """
    FPN decoder with ASPP bottleneck, correlation injection, deep supervision.

    For ViT-B/16 features all skips have the same dim (768) and the same
    spatial size (21 × 21).  The decoder upsamples:
      21 → 42 → 84 → 168 (× 2 from the resolution the mask is compared at)
    The final bilinear resize to 336 × 336 is done in the train/eval loops.
    """
    def __init__(self, feature_dim, skips_dims, corr_ch=2, dropout=0.2):
        super().__init__()
        c1, c2, c3 = skips_dims

        self.aspp   = ASPP(feature_dim, feature_dim)

        # FPN lateral convolutions
        self.lat3   = nn.Conv2d(c3, feature_dim,       1)
        self.lat2   = nn.Conv2d(c2, feature_dim // 2,  1)
        self.lat1   = nn.Conv2d(c1, feature_dim // 4,  1)

        # Up-sample blocks
        self.up1    = self._up_block(feature_dim,       feature_dim // 2)
        self.up2    = self._up_block(feature_dim // 2,  feature_dim // 4)
        self.up3    = self._up_block(feature_dim // 4,  feature_dim // 8)

        # Correlation map projection (injected before final conv)
        self.corr_proj = nn.Sequential(
            nn.Conv2d(corr_ch, feature_dim // 4, 1, bias=False),
            nn.BatchNorm2d(feature_dim // 4), nn.ReLU(inplace=True))

        # Deep supervision heads (return logits)
        self.ds2    = nn.Conv2d(feature_dim // 2, 1, 1)
        self.ds3    = nn.Conv2d(feature_dim // 4, 1, 1)

        self.drop   = nn.Dropout2d(p=dropout)
        self.final  = nn.Sequential(
            nn.Conv2d(feature_dim // 8 + feature_dim // 4, feature_dim // 8, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 8), nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 8, 1, 1))

    @staticmethod
    def _up_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x, skips, corr_maps):
        x  = self.aspp(x)

        # Stage 1: 21 → 42
        x  = self.up1(x + F.interpolate(self.lat3(skips[2]), size=x.shape[-2:],
                                         mode='bilinear', align_corners=False))
        ds2_logits = self.ds2(x)

        # Stage 2: 42 → 84
        x  = self.up2(x + F.interpolate(self.lat2(skips[1]), size=x.shape[-2:],
                                         mode='bilinear', align_corners=False))
        ds3_logits = self.ds3(x)

        # Stage 3: 84 → 168
        x  = self.up3(x + F.interpolate(self.lat1(skips[0]), size=x.shape[-2:],
                                         mode='bilinear', align_corners=False))

        # Inject correlation cues
        corr_up   = F.interpolate(corr_maps, size=x.shape[-2:],
                                   mode='bilinear', align_corners=False)
        corr_feat = self.corr_proj(corr_up)
        x         = self.drop(torch.cat([x, corr_feat], dim=1))

        return self.final(x), ds2_logits, ds3_logits


class SimpleMaskDecoder(nn.Module):
    """
    Plain U-Net-style decoder for ablations.

    Uses bilinear upsampling and CLIP skip connections only.  It intentionally
    omits ASPP, correlation injection, and deep-supervision heads.
    """
    def __init__(self, feature_dim, skips_dims, corr_ch=2, dropout=0.2):
        super().__init__()
        c1, c2, c3 = skips_dims
        self.lat3 = nn.Conv2d(c3, feature_dim, 1)
        self.lat2 = nn.Conv2d(c2, feature_dim // 2, 1)
        self.lat1 = nn.Conv2d(c1, feature_dim // 4, 1)
        self.up1 = MaskDecoder._up_block(feature_dim, feature_dim // 2)
        self.up2 = MaskDecoder._up_block(feature_dim // 2, feature_dim // 4)
        self.up3 = MaskDecoder._up_block(feature_dim // 4, feature_dim // 8)
        self.drop = nn.Dropout2d(p=dropout)
        self.final = nn.Sequential(
            nn.Conv2d(feature_dim // 8, feature_dim // 8, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 8), nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 8, 1, 1))

    def forward(self, x, skips, corr_maps):
        x = self.up1(x + F.interpolate(self.lat3(skips[2]), size=x.shape[-2:],
                                        mode='bilinear', align_corners=False))
        x = self.up2(x + F.interpolate(self.lat2(skips[1]), size=x.shape[-2:],
                                        mode='bilinear', align_corners=False))
        x = self.up3(x + F.interpolate(self.lat1(skips[0]), size=x.shape[-2:],
                                        mode='bilinear', align_corners=False))
        return self.final(self.drop(x)), None, None

# ─────────────────────────────────────────────────────────────────────────────
#  ESCNet  (main model)
# ─────────────────────────────────────────────────────────────────────────────
class ESCNet(nn.Module):
    """
    ESC-Net adapted for binary salient object segmentation.

    Vision backbone : CLIP ViT-B/16 or ViT-L/14  (configurable via clip_arch)
    Multi-scale features extracted via forward hooks on transformer blocks,
    reshaped from (seq, B, C) → (B, C, H, W) spatial maps.
    """
    # Architecture configurations for supported CLIP models
    ARCH_CONFIG = {
        'ViT-B-16': {'feat_dim': 768, 'proj_dim': 512, 'hooks': [2, 5, 8, 11], 'patch_size': 16, 'num_blocks': 12},
        'ViT-L-14': {'feat_dim': 1024, 'proj_dim': 768, 'hooks': [5, 11, 17, 23], 'patch_size': 14, 'num_blocks': 24},
    }

    def __init__(self, clip_model, sam_model,
                 num_blocks=4, num_prompts=5,
                 adaptive_fallback=True, bidirectional_sam=True,
                 simple_decoder=False, clip_arch='ViT-L-14',
                 prompt_agg='max'):
        super().__init__()

        # ── Architecture config lookup ────────────────────────────────────────
        if clip_arch not in self.ARCH_CONFIG:
            raise ValueError(f"Unsupported clip_arch '{clip_arch}'. "
                             f"Choose from: {list(self.ARCH_CONFIG.keys())}")
        cfg = self.ARCH_CONFIG[clip_arch]
        self.feat_dim     = cfg['feat_dim']
        self.proj_dim     = cfg['proj_dim']
        self.hook_indices = cfg['hooks']
        self.patch_size   = cfg['patch_size']
        self.clip_arch    = clip_arch
        self.prompt_agg   = prompt_agg

        # ── CLIP visual encoder ───────────────────────────────────────────────
        self.clip_visual = clip_model.visual

        # Text encoding components (same for all CLIP architectures)
        self.token_embedding = clip_model.token_embedding
        self.pos_embed       = clip_model.positional_embedding
        self.ln_final        = clip_model.ln_final
        self.text_proj       = clip_model.text_projection   # Parameter (proj_dim, proj_dim)
        # (2026-11) Full CLIP model for the real text encoder. Stored in a list so it
        # is NOT registered as a submodule: state_dict keys (and old checkpoints) are
        # unchanged. Its weights are already on the target device.
        self._clip_full = [clip_model]
        # Legacy path skipped the text transformer (embedding depended only on token
        # length). Kept for reproducing the accepted-paper numbers.
        self.legacy_text_encoder = False

        # Visual → CLIP projection (feat_dim → proj_dim) for correlation computation.
        # Works whether clip_model.visual.proj is nn.Linear or nn.Parameter.
        vp = clip_model.visual.proj
        if isinstance(vp, nn.Parameter):
            proj_data = vp.data.clone()           # shape (feat_dim, proj_dim)
        elif isinstance(vp, nn.Linear):
            proj_data = vp.weight.data.T.clone()  # Linear stores (out, in); we need (feat_dim, proj_dim)
        else:
            raise TypeError(f"Unexpected visual.proj type: {type(vp)}")
        # Learnable parameter — fine-tuned together with new layers
        self.vis_proj_w = nn.Parameter(proj_data)

        # ── Hook-based multi-scale feature extraction ─────────────────────────
        # Hooks populate self._hook_cache during each forward pass.
        self._hook_cache: dict = {}
        self._hook_handles  = []
        for level, idx in enumerate(self.hook_indices):
            h = self.clip_visual.transformer.resblocks[idx].register_forward_hook(
                self._make_hook(level))
            self._hook_handles.append(h)

        # ── Derive patch grid size from CLIP model ────────────────────────────
        if hasattr(self.clip_visual, 'image_size'):
            img_sz = self.clip_visual.image_size
        else:
            img_sz = CLIP_SIZE
        # image_size can be an int or a (H, W) tuple depending on open_clip version
        if isinstance(img_sz, (tuple, list)):
            img_sz = img_sz[0]
        patch_sz = self.patch_size
        self.grid_h = self.grid_w = img_sz // patch_sz

        # ── SAM esc blocks ────────────────────────────────────────────────────
        try:
            dec_blocks = sam_model.mask_decoder.transformer.layers
        except AttributeError:
            dec_blocks = list(sam_model.mask_decoder.transformer.children())

        num_blocks = min(num_blocks, len(dec_blocks))
        text_dim   = self.proj_dim

        self.esc_blocks = nn.ModuleList([
            ESCBlock(dec_blocks[i], sam_model.prompt_encoder,
                     self.feat_dim, num_prompts, text_dim,
                     adaptive_fallback=adaptive_fallback,
                     bidirectional_sam=bidirectional_sam)
            for i in range(num_blocks)
        ])

        # ── Decoder  (all ViT skip dims = feat_dim) ─────────────────────────────
        decoder_cls = SimpleMaskDecoder if simple_decoder else MaskDecoder
        self.mask_decoder = decoder_cls(
            feature_dim = self.feat_dim,
            skips_dims  = [self.feat_dim,
                           self.feat_dim,
                           self.feat_dim],
            corr_ch = 2,
            dropout = 0.2)

    def _make_hook(self, level: int):
        """Create a forward hook that caches the block output at `level`."""
        def hook(_module, _input, output):
            # output shape: (seq_len, B, C)  — LND format in CLIP transformer
            self._hook_cache[level] = output
        return hook

    def _tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert ViT block output tokens to a spatial feature map.
        Supports both token layouts used by different open_clip versions:
          LND (legacy): (seq_len, B, C)  — CLS token at position [0]
          NLD (newer):  (B, seq_len, C)  — CLS token at position [:, 0, :]
        Returns: (B, C, H, W)
        """
        num_tokens = self.grid_h * self.grid_w + 1   # patches + CLS

        if tokens.shape[0] == num_tokens:
            # ── LND format: (seq_len, B, C) ──────────────────────────────────
            patches = tokens[1:]                         # (N_patches, B, C)
            B, C    = patches.shape[1], patches.shape[2]
            return patches.permute(1, 2, 0).view(B, C, self.grid_h, self.grid_w)
        else:
            # ── NLD format: (B, seq_len, C) ──────────────────────────────────
            patches = tokens[:, 1:, :]                   # (B, N_patches, C)
            B, _, C = patches.shape
            return patches.permute(0, 2, 1).view(B, C, self.grid_h, self.grid_w)

    def _extract_clip_feats(self, x: torch.Tensor) -> dict:
        """
        Run CLIP visual encoder; collect intermediate features from hooks.
        Returns dict with keys 'high', 'mid', 'skips'.
        """
        self._hook_cache = {}
        _ = self.clip_visual(x)   # populates self._hook_cache via hooks

        # Level 0 → block 2  (early)
        # Level 1 → block 5  (mid-low)
        # Level 2 → block 8  (mid-high)
        # Level 3 → block 11 (deep)
        # Clamp ViT activations to a safe range before they enter the rest of
        # the pipeline.  Unclamped fp16 ViT outputs can reach ±thousands after
        # 15+ epochs and cause NaN in subsequent attention/product ops.
        l0   = self._tokens_to_spatial(self._hook_cache[0]).clamp(-100, 100)
        l1   = self._tokens_to_spatial(self._hook_cache[1]).clamp(-100, 100)
        l2   = self._tokens_to_spatial(self._hook_cache[2]).clamp(-100, 100)
        high = self._tokens_to_spatial(self._hook_cache[3]).clamp(-100, 100)
        mid  = l1  # block-5 features used for mid-level correlation

        return {'high': high, 'mid': mid, 'skips': [l0, l1, l2]}

    def _encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """CLIP text encoder → L2-normalised text embeddings (B, proj_dim).

        (2026-11, camera-ready) Default runs the full, frozen CLIP text transformer.
        The legacy path below skipped the transformer, so every prompt embedding
        depended only on its token length (prompts of equal length were identical).
        """
        if not getattr(self, 'legacy_text_encoder', False):
            with torch.no_grad():
                emb = self._clip_full[0].encode_text(tokens)
            return F.normalize(emb.float(), dim=-1)
        # ── Legacy (accepted-paper) path: no text transformer ────────────────
        t   = self.token_embedding(tokens) + self.pos_embed
        t   = self.ln_final(t)
        eot = tokens.argmax(dim=-1)
        t   = t[torch.arange(t.shape[0]), eot]
        return F.normalize(t @ self.text_proj, dim=-1)

    def _project_visual(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Project spatial visual features from ViT space (feat_dim) to CLIP space (proj_dim).
        feat: (B, feat_dim, H, W) → (B, proj_dim, H, W)
        """
        B, C, H, W = feat.shape
        flat = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)     # (B*H*W, feat_dim)
        proj = flat @ self.vis_proj_w                               # (B*H*W, proj_dim)
        return F.normalize(proj.view(B, H, W, -1).permute(0, 3, 1, 2), dim=1)

    # ─────────────────────────────────────────────────────────────────────────
    #  Multi-prompt text bank
    # ─────────────────────────────────────────────────────────────────────────
    def setup_prompt_bank(self, bank_tokenizer, device,
                          prompts=None, bank_size=None):
        """
        Pre-encode a list of text prompts and register them as a non-trainable
        buffer.  Must be called once after the model is moved to `device`.

        Parameters
        ----------
        bank_tokenizer : callable
            open_clip tokenizer (same one used for captions).
        device : str or torch.device
        prompts : list[str] | None
            Defaults to SALIENT_PROMPT_BANK.
        bank_size : int | None
            If provided, slice prompts to the first K entries.
        """
        if prompts is None:
            prompts = SALIENT_PROMPT_BANK
        if bank_size is not None:
            prompts = prompts[:bank_size]
        tokens = bank_tokenizer(prompts).to(device)          # (K, L)
        with torch.no_grad():
            embeds = self._encode_text(tokens)               # (K, proj_dim)
        # register_buffer keeps the tensor on the right device when model is
        # moved, and excludes it from optimizer updates.
        self.register_buffer('_bank_embeds', embeds)
        print(f"[INFO] Prompt bank: {len(prompts)} descriptions encoded "
              f"({embeds.shape[1]}-d CLIP embeddings)")

    def _has_bank(self) -> bool:
        return hasattr(self, '_bank_embeds') and self._bank_embeds is not None

    # ─────────────────────────────────────────────────────────────────────────
    #  Improved Multi-Prompt Mechanisms
    # ─────────────────────────────────────────────────────────────────────────

    def setup_learnable_prompts(self, K: int, device, init_from_bank: bool = True):
        """
        Mode 'learnable': Replace frozen CLIP text embeddings with K trainable
        vectors in CLIP's projection space. Optionally initialized from the
        frozen bank embeddings so training starts from a reasonable point.

        These are nn.Parameters that receive gradients through the correlation →
        PPG → decoder → loss path, allowing the prompts to specialize for what
        ViT-L actually needs rather than what generic English describes.
        """
        if init_from_bank and self._has_bank():
            # Initialize from the first K frozen embeddings
            init_data = self._bank_embeds[:K].clone()
            if init_data.shape[0] < K:
                # Pad with small random vectors if bank is too small
                extra = torch.randn(K - init_data.shape[0], self.proj_dim,
                                    device=device) * 0.02
                extra = F.normalize(extra, dim=-1)
                init_data = torch.cat([init_data, extra], dim=0)
        else:
            # Random initialization on the unit sphere
            init_data = torch.randn(K, self.proj_dim, device=device) * 0.02
            init_data = F.normalize(init_data, dim=-1)

        self.learnable_prompts = nn.Parameter(init_data)  # (K, proj_dim)
        print(f"[INFO] Learnable prompts: {K} vectors ({self.proj_dim}-d), "
              f"init_from_bank={init_from_bank}")

    def setup_attention_aggregation(self, K: int, device):
        """
        Mode 'attention': Learn per-pixel attention weights over K prompts.

        Given K correlation maps (B, K, H, W), a small convolutional network
        predicts attention weights α(i,j,k) at each spatial location, then
        aggregates: corr(i,j) = Σ_k α(i,j,k) · corr_k(i,j).

        This is strictly more expressive than max: the model can learn to
        replicate max (put all weight on the strongest prompt) or learn soft
        combinations that suppress noisy prompts at specific locations.
        """
        # Lightweight attention network: takes K-channel correlation input,
        # outputs K-channel softmax weights
        self.prompt_attn = nn.Sequential(
            nn.Conv2d(K, K * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(K * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(K * 4, K, kernel_size=1, bias=True),
            # No softmax here — applied in compute_corr for numerical stability
        ).to(device)
        # Initialize the final conv bias to 0 → uniform attention at start
        nn.init.zeros_(self.prompt_attn[-1].bias)
        print(f"[INFO] Attention aggregation: {K} prompts, "
              f"conv attention network initialized")

    def setup_prompt_gating(self, K: int, device):
        """
        Mode 'gated': A lightweight gating network that takes the global image
        representation (GAP of visual features) and outputs K sigmoid gates.

        gate_k ∈ [0, 1] per image — determines how much each prompt contributes.
        Sparse: most gates can learn to be near 0 for irrelevant prompts.

        Aggregation: corr(i,j) = max_k [ gate_k · corr_k(i,j) ]
        (gated max — prompts with gate≈0 are effectively disabled)
        """
        self.prompt_gate = nn.Sequential(
            nn.Linear(self.proj_dim, K * 2),
            nn.ReLU(inplace=True),
            nn.Linear(K * 2, K),
            # Sigmoid applied in compute_corr
        ).to(device)
        # Initialize to slight positive bias → all prompts active initially
        nn.init.zeros_(self.prompt_gate[-1].weight)
        nn.init.constant_(self.prompt_gate[-1].bias, 1.0)  # sigmoid(1) ≈ 0.73
        print(f"[INFO] Prompt gating: {K} prompts, "
              f"gate network initialized (bias=1.0 → all active initially)")

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _minmax(t: torch.Tensor) -> torch.Tensor:
        """Per-sample min-max normalisation over (H, W)."""
        mn = t.flatten(1).min(1)[0].view(-1, 1, 1)
        mx = t.flatten(1).max(1)[0].view(-1, 1, 1)
        return (t - mn) / (mx - mn + 1e-6)

    def compute_corr(self, feats: dict, txt_embeds: torch.Tensor) -> torch.Tensor:
        """
        Compute normalised cosine-similarity correlation maps.

        Supports four prompt modes (controlled by self.prompt_agg):
          'max'       — original: fixed bank + element-wise max (WACV submission)
          'mean'      — fixed bank + element-wise mean
          'learnable' — trainable prompt embeddings + attention aggregation
          'attention' — fixed bank + learned per-pixel attention weights
          'gated'     — fixed bank + image-conditioned gate per prompt

        Without a bank and not in learnable mode, falls back to single-prompt.

        Returns (B, 2, H, W): [high-level corr, mid-level corr].
        """
        high_proj = self._project_visual(feats['high'])   # (B, proj_dim, H, W)
        mid_proj  = self._project_visual(feats['mid'])    # (B, proj_dim, H, W)

        # ── Determine prompt embeddings to use ────────────────────────────────
        if self.prompt_agg == 'learnable' and hasattr(self, 'learnable_prompts'):
            # Learnable mode: use trainable prompt vectors (normalized)
            bank = F.normalize(self.learnable_prompts, dim=-1)  # (K, proj_dim)
            high_bank = torch.einsum('bchw,kc->bkhw', high_proj, bank)
            mid_bank  = torch.einsum('bchw,kc->bkhw', mid_proj, bank)

            # Aggregate with learned soft attention (dot-product based)
            # Each prompt's contribution is weighted by how strongly it correlates
            # globally — strong prompts get more weight
            attn_weights = high_bank.mean(dim=(-2, -1))  # (B, K) global strength
            attn_weights = F.softmax(attn_weights / 0.1, dim=-1)  # temperature-scaled
            # Weighted sum: (B, K, H, W) × (B, K, 1, 1) → sum → (B, H, W)
            high_corr = (high_bank * attn_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
            mid_corr  = (mid_bank * attn_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

        elif self.prompt_agg == 'attention' and hasattr(self, 'prompt_attn'):
            # Attention mode: fixed bank + learned spatial attention
            bank = self._bank_embeds  # (K, proj_dim) frozen
            high_bank = torch.einsum('bchw,kc->bkhw', high_proj, bank)  # (B,K,H,W)
            mid_bank  = torch.einsum('bchw,kc->bkhw', mid_proj, bank)

            # Predict per-pixel attention weights from correlation patterns
            attn_logits = self.prompt_attn(high_bank)  # (B, K, H, W)
            attn_weights = F.softmax(attn_logits, dim=1)  # normalize over K prompts
            # Weighted sum aggregation
            high_corr = (high_bank * attn_weights).sum(dim=1)  # (B, H, W)
            # Use same attention weights for mid (they share spatial structure)
            mid_attn_logits = self.prompt_attn(mid_bank)
            mid_attn_weights = F.softmax(mid_attn_logits, dim=1)
            mid_corr = (mid_bank * mid_attn_weights).sum(dim=1)

        elif self.prompt_agg == 'gated' and hasattr(self, 'prompt_gate'):
            # Gated mode: fixed bank + image-conditioned gates
            bank = self._bank_embeds  # (K, proj_dim) frozen
            high_bank = torch.einsum('bchw,kc->bkhw', high_proj, bank)  # (B,K,H,W)
            mid_bank  = torch.einsum('bchw,kc->bkhw', mid_proj, bank)

            # Compute gates from global average of projected visual features
            # Use high_proj GAP as the image representation
            img_global = high_proj.mean(dim=(-2, -1))  # (B, proj_dim)
            gates = torch.sigmoid(self.prompt_gate(img_global))  # (B, K)

            # Apply gates: suppress irrelevant prompts, then max over active ones
            gates_hw = gates.unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
            high_gated = high_bank * gates_hw  # (B, K, H, W) — gated correlations
            mid_gated  = mid_bank * gates_hw

            # Max over gated prompts — disabled prompts (gate≈0) don't contribute
            high_corr = high_gated.max(dim=1).values  # (B, H, W)
            mid_corr  = mid_gated.max(dim=1).values

        elif self._has_bank():
            # ── Original multi-prompt path (max or mean) ──────────────────────
            bank = self._bank_embeds
            high_bank = torch.einsum('bchw,kc->bkhw', high_proj, bank)
            mid_bank  = torch.einsum('bchw,kc->bkhw', mid_proj, bank)

            # Correlation with per-sample caption:  (B, 1, H, W)
            high_cap = torch.einsum('bchw,bc->bhw', high_proj, txt_embeds).unsqueeze(1)
            mid_cap  = torch.einsum('bchw,bc->bhw', mid_proj, txt_embeds).unsqueeze(1)

            # Concatenate all K+1 maps and aggregate over the prompt axis
            if self.prompt_agg == 'mean':
                high_corr = torch.cat([high_bank, high_cap], dim=1).mean(dim=1)
                mid_corr  = torch.cat([mid_bank, mid_cap], dim=1).mean(dim=1)
            else:  # 'max' (default)
                high_corr = torch.cat([high_bank, high_cap], dim=1).max(dim=1).values
                mid_corr  = torch.cat([mid_bank, mid_cap], dim=1).max(dim=1).values

        else:
            # ── Single-prompt fallback ────────────────────────────────────────
            high_corr = torch.einsum('bchw,bc->bhw', high_proj, txt_embeds)
            mid_corr  = torch.einsum('bchw,bc->bhw', mid_proj, txt_embeds)

        high_corr = self._minmax(high_corr)
        mid_corr  = self._minmax(mid_corr)

        H, W = feats['high'].shape[-2:]
        return torch.cat([
            F.interpolate(high_corr.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False),
            F.interpolate(mid_corr .unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False),
        ], dim=1)   # (B, 2, H, W)

    def forward(self, image: torch.Tensor, text_tokens: torch.Tensor):
        img_feats  = self._extract_clip_feats(image)
        txt_embeds = self._encode_text(text_tokens)
        # Expand text embeddings to match image batch if needed (shared caption)
        B = image.shape[0]
        if txt_embeds.shape[0] == 1 and B > 1:
            txt_embeds = txt_embeds.expand(B, -1)
        corr       = self.compute_corr(img_feats, txt_embeds)

        # ── Correlation Residual Connection (Architecture B) ──────────────────
        # CHANGE LOG (2026-08-25):
        #   Architecture A (WACV submission, runs before 2026-08-25):
        #     - Initial correlation was used only for PPG point selection
        #     - VLF blocks completely overwrote correlation over 4 iterations
        #     - Prompt embeddings had NO gradient path to final output
        #     - Multi-prompt bank helped only via discrete PPG point locations
        #
        #   Architecture B (current, with residual):
        #     - Initial correlation is preserved and added back after ESC blocks
        #     - Prompt embeddings have a DIRECT gradient path: loss → decoder → corr → prompts
        #     - Enables learnable/attention/gated prompt modes to actually learn
        #     - Also improves fixed-max mode by giving prompt correlation direct influence
        #
        # The residual ensures the decoder always sees the prompt-derived spatial prior
        # in addition to the VLF-refined features. Without it, the prompt bank's
        # influence is limited to the initial PPG point sampling (non-differentiable).
        corr_init = corr

        # (2026-11) --vlf-multi-token: VLF attends over [bank prompts; caption]
        # instead of the caption alone. Default off = original behaviour.
        vlf_text = txt_embeds
        if getattr(self, 'vlf_multi_token', False) and self._has_bank():
            bank = self._bank_embeds.unsqueeze(0).expand(B, -1, -1)       # (B, K, D)
            vlf_text = torch.cat([bank, txt_embeds.unsqueeze(1)], dim=1)  # (B, K+1, D)

        pp_points = pp_labels = pp_size = None
        for i, block in enumerate(self.esc_blocks):
            img_feats, corr, (pts, labels) = block(img_feats, vlf_text, corr)
            if i == 0:
                pp_points, pp_labels, pp_size = pts, labels, corr.shape[-2:]

        # Add initial prompt correlation as residual to ESC-refined correlation.
        # (2026-11, camera-ready) Can be disabled with --no-prompt-residual for
        # the matched ablation; default (use_prompt_residual=True) is unchanged.
        if getattr(self, 'use_prompt_residual', True):
            corr = corr + corr_init

        mask, ds2, ds3 = self.mask_decoder(
            img_feats['high'], skips=img_feats['skips'], corr_maps=corr)

        aux = {'pp_points': pp_points, 'pp_labels': pp_labels,
               'pp_size'  : pp_size,
               'ds2': ds2, 'ds3': ds3}
        return mask, aux

# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
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
    p  = (preds_probs.detach() > threshold).bool().cpu()
    g  = (masks.detach() > 0.5).bool().cpu()
    ds, js = [], []
    for b in range(p.shape[0]):
        ds.append(dice_score(p[b, 0].numpy(), g[b, 0].numpy()))
        js.append(iou_score (p[b, 0].numpy(), g[b, 0].numpy()))
    return float(np.mean(ds)), float(np.mean(js))


# ─────────────────────────────────────────────────────────────────────────────
#  SIP / salient-object EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
def _safe_div(num, den, eps=1e-12):
    return num / (den + eps)


def _prepare_saliency_arrays(pred_prob: np.ndarray, gt_mask: np.ndarray):
    pred = np.asarray(pred_prob, dtype=np.float64)
    gt   = np.asarray(gt_mask, dtype=np.float64)

    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = np.clip(pred, 0, 255) / 255.0
    pred = np.clip(pred, 0.0, 1.0)
    gt_bin = gt > 0.5
    return pred, gt_bin


def f_measure_score(pred_bin: np.ndarray, gt_bin: np.ndarray,
                    beta2: float = 0.3) -> float:
    """Binary F-measure used by salient-object datasets such as SIP."""
    pred = np.asarray(pred_bin, dtype=bool)
    gt   = np.asarray(gt_bin, dtype=bool)
    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    precision = _safe_div(tp, pred.sum(dtype=np.float64))
    recall    = _safe_div(tp, gt.sum(dtype=np.float64))
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return float(_safe_div((1.0 + beta2) * precision * recall,
                           beta2 * precision + recall))


def weighted_f_measure(pred_prob: np.ndarray, gt_bin: np.ndarray,
                       beta2: float = 0.3) -> float:
    """Weighted F-measure (WFβ) for binary salient-object evaluation."""
    from scipy import ndimage

    pred, gt = _prepare_saliency_arrays(pred_prob, gt_bin)
    if not gt.any():
        return 0.0

    error = np.abs(pred - gt.astype(np.float64))
    dist, idx = ndimage.distance_transform_edt(~gt, return_indices=True)

    error_ext = error.copy()
    bg = ~gt
    error_ext[bg] = error[idx[0][bg], idx[1][bg]]
    error_smooth = ndimage.gaussian_filter(error_ext, sigma=5, mode='constant', cval=0.0)

    min_error = error.copy()
    fg = gt
    min_error[fg] = np.minimum(error[fg], error_smooth[fg])

    # Pixel-importance weighting for background errors, matching the original
    # WFβ implementation's distance-based falloff.
    weights = np.ones_like(pred, dtype=np.float64)
    weights[bg] = 2.0 - np.exp(np.log(0.5) / 5.0 * dist[bg])
    weighted_error = min_error * weights

    gt_sum = gt.sum(dtype=np.float64)
    tp_w = gt_sum - weighted_error[fg].sum(dtype=np.float64)
    fp_w = weighted_error[bg].sum(dtype=np.float64)
    recall = 1.0 - _safe_div(weighted_error[fg].sum(dtype=np.float64), gt_sum)
    precision = _safe_div(tp_w, tp_w + fp_w)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return float(_safe_div((1.0 + beta2) * precision * recall,
                           beta2 * precision + recall))


def enhanced_measure(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """Enhanced-alignment measure (E-measure) for a binary prediction."""
    pred = np.asarray(pred_bin, dtype=np.float64)
    gt   = np.asarray(gt_bin, dtype=bool)

    if gt.all():
        return float(pred.mean())
    if not gt.any():
        return float(1.0 - pred.mean())

    gt_float = gt.astype(np.float64)
    pred_align = pred - pred.mean()
    gt_align   = gt_float - gt_float.mean()
    align = _safe_div(2.0 * pred_align * gt_align,
                      pred_align ** 2 + gt_align ** 2)
    enhanced = ((align + 1.0) ** 2) / 4.0
    h, w = gt.shape
    return float(_safe_div(enhanced.sum(dtype=np.float64), h * w - 1.0))


def _s_object_component(region: np.ndarray) -> float:
    if region.size == 0:
        return 0.0
    mean = region.mean(dtype=np.float64)
    std = region.std(dtype=np.float64)
    return float(_safe_div(2.0 * mean, mean ** 2 + 1.0 + std + 1e-12))


def _s_object(pred: np.ndarray, gt: np.ndarray) -> float:
    fg = pred[gt]
    bg = 1.0 - pred[~gt]
    u = gt.mean(dtype=np.float64)
    return float(u * _s_object_component(fg) + (1.0 - u) * _s_object_component(bg))


def _centroid(gt: np.ndarray):
    h, w = gt.shape
    if not gt.any():
        return w // 2, h // 2
    ys, xs = np.nonzero(gt)
    x = int(np.round(xs.mean())) + 1
    y = int(np.round(ys.mean())) + 1
    return min(max(x, 1), w), min(max(y, 1), h)


def _divide_with_weights(mat: np.ndarray, x: int, y: int):
    h, w = mat.shape
    lt = mat[:y, :x]
    rt = mat[:y, x:]
    lb = mat[y:, :x]
    rb = mat[y:, x:]
    area = float(h * w)
    weights = [
        (x * y) / area,
        ((w - x) * y) / area,
        (x * (h - y)) / area,
        ((w - x) * (h - y)) / area,
    ]
    return (lt, rt, lb, rb), weights


def _ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    x = pred.mean(dtype=np.float64)
    y = gt.mean(dtype=np.float64)
    n = pred.size
    if n == 1:
        return 1.0 - abs(x - y)
    sigma_x = ((pred - x) ** 2).sum(dtype=np.float64) / (n - 1)
    sigma_y = ((gt - y) ** 2).sum(dtype=np.float64) / (n - 1)
    sigma_xy = ((pred - x) * (gt - y)).sum(dtype=np.float64) / (n - 1)
    alpha = 4.0 * x * y * sigma_xy
    beta = (x ** 2 + y ** 2) * (sigma_x + sigma_y)
    if alpha != 0.0:
        return float(_safe_div(alpha, beta))
    if alpha == 0.0 and beta == 0.0:
        return 1.0
    return 0.0


def _s_region(pred: np.ndarray, gt: np.ndarray) -> float:
    x, y = _centroid(gt)
    pred_parts, weights = _divide_with_weights(pred, x, y)
    gt_parts, _ = _divide_with_weights(gt.astype(np.float64), x, y)
    return float(sum(w * _ssim(p, g) for p, g, w in zip(pred_parts, gt_parts, weights)))


def structure_measure(pred_prob: np.ndarray, gt_bin: np.ndarray,
                      alpha: float = 0.5) -> float:
    """S-measure: object-aware and region-aware structural similarity."""
    pred, gt = _prepare_saliency_arrays(pred_prob, gt_bin)
    y = gt.mean(dtype=np.float64)
    if y == 0.0:
        return float(1.0 - pred.mean(dtype=np.float64))
    if y == 1.0:
        return float(pred.mean(dtype=np.float64))
    score = alpha * _s_object(pred, gt) + (1.0 - alpha) * _s_region(pred, gt)
    return float(max(0.0, score))


def sip_metric_scores(pred_prob: np.ndarray, gt_mask: np.ndarray,
                      threshold: float = 0.5) -> dict:
    """Return SIP metrics: E↑, S↑, weighted-F↑, F↑, MAE↓."""
    pred, gt = _prepare_saliency_arrays(pred_prob, gt_mask)
    pred_bin = pred >= threshold
    return {
        'e_measure':  enhanced_measure(pred_bin, gt),
        's_measure':  structure_measure(pred, gt),
        'weighted_f': weighted_f_measure(pred, gt),
        'f_measure':  f_measure_score(pred_bin, gt),
        'mae':        float(np.mean(np.abs(pred - gt.astype(np.float64)))),
    }


def standard_sod_metrics(pred_prob: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Standard SOD evaluation: max-Fβ (255 thresholds), mean-Em, Sm, WFβ, MAE."""
    pred, gt = _prepare_saliency_arrays(pred_prob, gt_mask)

    # Max F-measure across 255 thresholds
    f_scores = []
    e_scores = []
    for i in range(1, 256):
        thresh = i / 255.0
        pred_bin = pred >= thresh
        f_scores.append(f_measure_score(pred_bin, gt))
        e_scores.append(enhanced_measure(pred_bin, gt))

    max_f = max(f_scores) if f_scores else 0.0
    mean_e = float(np.mean(e_scores)) if e_scores else 0.0

    # Adaptive F-measure (threshold = 2 * mean of prediction)
    adaptive_thresh = min(2.0 * pred.mean(), 1.0)
    adaptive_pred_bin = pred >= adaptive_thresh
    adp_f = f_measure_score(adaptive_pred_bin, gt)

    return {
        'max_f': max_f,
        'mean_e': mean_e,
        'adp_f': adp_f,
        's_measure': structure_measure(pred, gt),
        'weighted_f': weighted_f_measure(pred, gt),
        'mae': float(np.mean(np.abs(pred - gt.astype(np.float64)))),
    }

# ─────────────────────────────────────────────────────────────────────────────
#  POINT CLASSIFICATION AUXILIARY LOSS
# ─────────────────────────────────────────────────────────────────────────────
def _point_cls_loss(preds_logits, masks_gt, aux, device):
    if aux['pp_points'] is None:
        return torch.tensor(0.0, device=device)
    Hpp, Wpp = aux['pp_size']
    pp_logits = F.interpolate(preds_logits, size=(Hpp, Wpp),
                               mode='bilinear', align_corners=False)
    gt_pp     = F.interpolate(masks_gt, size=(Hpp, Wpp), mode='nearest')
    pts       = aux['pp_points'].to(device)
    labs      = aux['pp_labels'].to(device)
    Bp, P, _  = pts.shape
    xs = pts[..., 0].round().clamp(0, Wpp - 1).long()
    ys = pts[..., 1].round().clamp(0, Hpp - 1).long()
    bidx = torch.arange(Bp, device=device).unsqueeze(1).expand(Bp, P)
    pred_vals = pp_logits[bidx, 0, ys, xs]
    gt_vals   = gt_pp   [bidx, 0, ys, xs]
    valid     = labs > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)
    return F.binary_cross_entropy_with_logits(pred_vals[valid], gt_vals[valid])

# ─────────────────────────────────────────────────────────────────────────────
#  POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def postprocess_mask(prob_map: np.ndarray, threshold: float = 0.5,
                     min_object_area: int = 300) -> np.ndarray:
    from scipy import ndimage
    binary = (prob_map > threshold).astype(np.uint8)
    filled = ndimage.binary_fill_holes(binary).astype(np.uint8)
    labeled, n = ndimage.label(filled)
    if n > 1:
        sizes = ndimage.sum(filled, labeled, range(1, n + 1))
        clean = np.zeros_like(filled)
        for i, sz in enumerate(sizes):
            if sz >= min_object_area:
                clean[labeled == (i + 1)] = 1
    else:
        clean = filled
    return clean.astype(bool)

# ─────────────────────────────────────────────────────────────────────────────
#  TEST-TIME AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def tta_predict(model, img_t, tokens):
    """Average predictions over 4 flips (original + h-flip + v-flip + both)."""
    preds = []
    with torch.no_grad():
        for flip_h, flip_v in [(False, False), (True, False),
                                (False, True),  (True,  True)]:
            x = img_t.clone()
            if flip_h: x = torch.flip(x, dims=[-1])
            if flip_v: x = torch.flip(x, dims=[-2])
            logits, _ = model(x, tokens)
            probs     = torch.sigmoid(logits)
            if flip_h: probs = torch.flip(probs, dims=[-1])
            if flip_v: probs = torch.flip(probs, dims=[-2])
            preds.append(probs)
    return torch.stack(preds).mean(0)

# ─────────────────────────────────────────────────────────────────────────────
#  LR SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup then segment-wise cosine decay with milestone drops.

    After the warmup phase the LR follows a fresh cosine curve within each
    segment.  At each milestone the *starting* LR of the new segment is
    multiplied by ``milestone_decay``, producing a staircase-cosine shape.

    Example  (total=100, milestones=[0.5, 0.75], decay=0.1, warmup=5):
      segment 0  ep  5-49   cosine  base_lr      → base_lr * 0.1
      segment 1  ep 50-74   cosine  base_lr*0.1  → base_lr * 0.01
      segment 2  ep 75-99   cosine  base_lr*0.01 → eta_min

    Args:
        milestones:      list of *fractions* of total_epochs (e.g. [0.5, 0.75]).
                         Defaults to [0.5, 0.75].
        milestone_decay: multiplicative factor applied at each milestone.
                         Default 0.1 (one decade per milestone).
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 milestones=None, milestone_decay=0.1,
                 eta_min=1e-6, last_epoch=-1):
        self.warmup_epochs   = warmup_epochs
        self.total_epochs    = total_epochs
        self.milestone_decay = milestone_decay
        self.eta_min         = eta_min

        if milestones is None:
            milestones = [0.5, 0.75]

        # Convert fractional milestones → integer epoch breakpoints.
        # Always include warmup_end and total_epochs; deduplicate & sort.
        ms_ep = [int(f * total_epochs) for f in milestones]
        self._breaks = sorted(set(
            [warmup_epochs]
            + [m for m in ms_ep if m > warmup_epochs]
            + [total_epochs]
        ))

        super().__init__(optimizer, last_epoch)

    # ------------------------------------------------------------------
    def _segment_for(self, ep):
        """Return (seg_index, seg_start, seg_end) for a given epoch."""
        breaks = self._breaks
        if len(breaks) < 2:
            # Degenerate case (very short training) — single segment from 0 to total
            return 0, 0, max(self.total_epochs, 1)
        for i in range(len(breaks) - 1):
            if ep < breaks[i + 1]:
                return i, breaks[i], breaks[i + 1]
        # ep >= total_epochs: clamp to last segment
        return len(breaks) - 2, breaks[-2], breaks[-1]

    def get_lr(self):
        ep = self.last_epoch

        # ── Warmup ───────────────────────────────────────────────────────────
        if ep < self.warmup_epochs:
            factor = (ep + 1) / max(self.warmup_epochs, 1)
            return [base_lr * factor for base_lr in self.base_lrs]

        # ── Segment-wise cosine ───────────────────────────────────────────────
        num_segs         = len(self._breaks) - 1
        seg, seg_s, seg_e = self._segment_for(ep)
        seg_len          = max(seg_e - seg_s, 1)
        progress         = (ep - seg_s) / seg_len          # 0 → 1 within seg

        d      = self.milestone_decay
        hi_fac = d ** seg                                   # seg-start multiplier
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        lrs = []
        for base_lr in self.base_lrs:
            lr_hi = base_lr * hi_fac
            if seg == num_segs - 1:
                # Last segment → decay all the way to eta_min
                lr = self.eta_min + (lr_hi - self.eta_min) * cosine
            else:
                lr_lo = base_lr * (d ** (seg + 1))
                lr = lr_lo + (lr_hi - lr_lo) * cosine
            lrs.append(max(lr, self.eta_min))
        return lrs

# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING LOOP
#   Returns history dict with per-epoch lists: 'loss', 'dice', 'iou', 'lr'
#   Saves two checkpoints every epoch:
#     escnet_best.pth  — whenever training Dice improves
#     escnet_last.pth  — overwritten at the end of every epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_loop(model, dataloader, optimizer, scheduler, device, epochs,
               save_path, pp_loss_weight=0.3, ds_weight=0.3, threshold=0.5,
               mse_lambda=0.1, plain_bce_loss=False, save_best_only=False):
    """
    Training loop with:
      • Per-batch NaN/Inf loss guard (skips weight update, logs breakdown)
      • Per-epoch component logging: Tversky, Focal, MSE, PP losses
      • Gradient norm tracking (mean & max per epoch)
      • AMP scale-drop counter (each drop = GradScaler detected overflow)
      • CSV diagnostics file for post-hoc analysis
    """
    criterion  = (nn.BCEWithLogitsLoss() if plain_bce_loss else
                  CompoundLoss(w_tversky=0.6, w_focal=0.4, mse_lambda=mse_lambda,
                               gamma=2.0, tversky_alpha=0.7, tversky_beta=0.3))
    # init_scale=4096: the default (65536) dropped 4× in epoch 1 every run,
    # so pre-set it to the stable value.  growth_interval=2000 avoids
    # re-growing aggressively and triggering new overflows.
    use_amp = torch.device(device).type == 'cuda'
    scaler  = GradScaler('cuda', init_scale=4096.0, growth_interval=2000,
                         enabled=use_amp)
    best_dice  = 0.0
    last_path  = save_path.replace('escnet_best.pth', 'escnet_last.pth')
    # "clean" checkpoint: only saved when the epoch has ZERO NaN batches.
    # Used as the recovery target so we never reload a checkpoint that was
    # itself produced during a corrupted epoch.
    clean_path = save_path.replace('escnet_best.pth', 'escnet_clean.pth')
    diag_path  = save_path.replace('escnet_best.pth', 'training_diagnostics.csv')

    history = {'loss': [], 'dice': [], 'iou': [], 'lr': []}

    # ── Write CSV header ──────────────────────────────────────────────────────
    _fields = ['epoch', 'loss', 'loss_tversky', 'loss_focal', 'loss_mse',
               'loss_pp', 'dice', 'iou', 'lr',
               'grad_norm_mean', 'grad_norm_max',
               'nan_batches', 'amp_scale_drops', 'amp_scale']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(diag_path, 'w', newline='') as _f:
        csv.DictWriter(_f, fieldnames=_fields).writeheader()
    print(f"[INFO] Diagnostics  → {diag_path}")

    print(f"\n{'Epoch':>6}  {'Loss':>8}  {'Dice':>8}  {'IoU':>8}  {'LR':>10}"
          f"  {'GradNorm':>9}  {'NaN/Skip':>8}  {'Saved':>8}")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        all_dice, all_iou = [], []
        acc_tv, acc_fc, acc_mse, acc_pp = 0.0, 0.0, 0.0, 0.0
        grad_norms: list = []
        nan_batches    = 0
        amp_scale_drops = 0
        consecutive_nan = 0          # tracks run of bad batches

        for batch_idx, (imgs, caps, masks, _) in enumerate(dataloader):
            imgs, masks = imgs.to(device), masks.to(device)
            tokens      = tokenizer(caps).to(device)

            with autocast(torch.device(device).type, enabled=use_amp):
                preds_logits, aux = model(imgs, tokens)

                # ── Detect NaN in model output before touching the loss ────────
                # All-NaN tv/fc/pp means preds_r was already NaN (model issue,
                # not a loss-function edge case).
                if not torch.isfinite(preds_logits).all():
                    nan_batches    += 1
                    consecutive_nan += 1
                    n_nan = (~torch.isfinite(preds_logits)).sum().item()
                    print(f"  [WARN] E{epoch:03d} B{batch_idx:04d}: "
                          f"NaN/Inf in model output ({n_nan} values) — "
                          f"skipping batch (streak={consecutive_nan})")
                    optimizer.zero_grad()

                    # ── Checkpoint recovery ───────────────────────────────────
                    # 3+ consecutive NaN batches means weights are corrupted.
                    # Always prefer the "clean" checkpoint (last epoch with 0
                    # NaN batches) over the "best" checkpoint which may itself
                    # have been saved during a partially-corrupted epoch.
                    if consecutive_nan >= 3:
                        recovery = (clean_path if os.path.exists(clean_path)
                                    else save_path)
                        if os.path.exists(recovery):
                            ckpt_r = torch.load(recovery, map_location=device,
                                                weights_only=False)
                            model.load_state_dict(ckpt_r['model_state'])
                            optimizer.load_state_dict(ckpt_r['optimizer_state'])
                            tag = 'clean' if recovery == clean_path else 'best'
                            print(f"  [RECOVER] Reloaded {tag} ckpt "
                                  f"(epoch {ckpt_r['epoch']}, "
                                  f"dice={ckpt_r.get('best_dice', ckpt_r.get('avg_dice', 0)):.4f})")
                        consecutive_nan = 0   # reset streak after reload

                    continue            # ← no scaler.update(); backward never called

                preds_r = F.interpolate(preds_logits, size=masks.shape[-2:],
                                        mode='bilinear', align_corners=False)

                # Main loss — get component scalars at no extra compute cost
                if plain_bce_loss:
                    loss = criterion(preds_r, masks)
                    _stats = {'tversky': 0.0, 'focal': loss.detach().item(), 'mse': 0.0}
                else:
                    loss, _stats = criterion.forward_with_stats(preds_r, masks)

                for ds_key in ('ds2', 'ds3'):
                    if aux.get(ds_key) is not None:
                        ds_r = F.interpolate(aux[ds_key], size=masks.shape[-2:],
                                             mode='bilinear', align_corners=False)
                        loss = loss + ds_weight * criterion(ds_r, masks)

                loss_pp = _point_cls_loss(preds_r, masks, aux, device)
                loss    = loss + pp_loss_weight * loss_pp

            # ── Guard: non-finite scalar loss (loss-function edge case) ────────
            if not torch.isfinite(loss):
                nan_batches += 1
                print(f"  [WARN] E{epoch:03d} B{batch_idx:04d}: "
                      f"non-finite loss  "
                      f"tv={_stats['tversky']:.4f}  "
                      f"fc={_stats['focal']:.4f}  "
                      f"mse={_stats['mse']:.4f}  "
                      f"pp={loss_pp.item():.4f}  — skipping update")
                optimizer.zero_grad()
                # scaler.update() must NOT be called here — backward was never
                # called so the scaler has no inf-checks recorded and would
                # raise: "No inf checks were recorded prior to update."
                continue

            # ── Normal backward pass ──────────────────────────────────────────
            prev_scale = scaler.get_scale()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            if scaler.get_scale() < prev_scale:
                amp_scale_drops += 1   # GradScaler detected overflow → skipped step

            consecutive_nan = 0        # successful update → reset streak
            grad_norms.append(float(gn))
            total_loss += loss.item()
            acc_tv     += _stats['tversky']
            acc_fc     += _stats['focal']
            acc_mse    += _stats['mse']
            acc_pp     += loss_pp.item()

            preds_probs = torch.sigmoid(preds_r)
            d, i = batch_dice_iou(preds_probs, masks, threshold)
            all_dice.append(d)
            all_iou.append(i)

        scheduler.step()

        n_valid  = max(len(dataloader) - nan_batches, 1)
        avg_loss = float(total_loss / n_valid)
        avg_dice = float(np.mean(all_dice) if all_dice else 0.0)
        avg_iou  = float(np.mean(all_iou)  if all_iou  else 0.0)
        avg_gn   = float(np.mean(grad_norms) if grad_norms else 0.0)
        max_gn   = float(max(grad_norms)     if grad_norms else 0.0)
        avg_tv   = float(acc_tv / n_valid)
        avg_fc   = float(acc_fc / n_valid)
        avg_mse  = float(acc_mse / n_valid)
        avg_pp   = float(acc_pp / n_valid)
        cur_lr   = float(optimizer.param_groups[0]['lr'])
        amp_scale = float(scaler.get_scale())

        history['loss'].append(avg_loss)
        history['dice'].append(avg_dice)
        history['iou'] .append(avg_iou)
        history['lr']  .append(cur_lr)

        # ── Append diagnostics row ────────────────────────────────────────────
        row = {'epoch': epoch, 'loss': avg_loss,
               'loss_tversky': avg_tv, 'loss_focal': avg_fc,
               'loss_mse': avg_mse, 'loss_pp': avg_pp,
               'dice': avg_dice, 'iou': avg_iou, 'lr': cur_lr,
               'grad_norm_mean': avg_gn, 'grad_norm_max': max_gn,
               'nan_batches': nan_batches,
               'amp_scale_drops': amp_scale_drops, 'amp_scale': amp_scale}
        with open(diag_path, 'a', newline='') as _f:
            csv.DictWriter(_f, fieldnames=_fields).writerow(row)

        # ── Checkpoint: last epoch (always overwrite) ─────────────────────────
        last_payload = {
            'epoch': epoch, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'avg_dice': avg_dice, 'history': history,
        }
        torch.save(last_payload, last_path)

        saved_tag = "last"

        # ── Checkpoint: "clean" — only when the epoch has ZERO NaN batches ────
        # This is the recovery target.  Using the "best" checkpoint as recovery
        # target was the bug: best may have been saved during a 55-NaN epoch,
        # so reloading it kept the model in a corrupted state.
        if nan_batches == 0:
            if not save_best_only:
                torch.save(last_payload, clean_path)
            saved_tag = "clean+last"

        # ── Checkpoint: best training Dice (only when epoch is clean) ─────────
        # Prevents a high-Dice epoch that also had many NaN batches from
        # overwriting the best checkpoint with a semi-corrupted state.
        if avg_dice > best_dice and nan_batches == 0:
            best_dice = avg_dice
            best_payload = {**last_payload, 'best_dice': best_dice}
            torch.save(best_payload, save_path)
            saved_tag = "best+clean+last" if not save_best_only else "best"

        # Skip saving last/clean if --save-best-only
        if save_best_only:
            if os.path.exists(last_path):
                os.remove(last_path)
            saved_tag = saved_tag.replace("+clean+last", "").replace("clean+last", "—").replace("last", "—")

        nan_tag = f"{nan_batches}nan" if nan_batches else f"{amp_scale_drops}ovf"
        print(f"{epoch:>6}  {avg_loss:>8.4f}  {avg_dice:>8.4f}  "
              f"{avg_iou:>8.4f}  {cur_lr:>10.2e}  "
              f"{avg_gn:>9.3f}  {nan_tag:>8}  {saved_tag:>8}")

    print(f"\n[INFO] Diagnostics CSV  : {diag_path}")
    print(f"[INFO] Best Train Dice  : {best_dice:.4f}  → {save_path}")
    print(f"[INFO] Last checkpoint  : epoch {epochs}    → {last_path}")
    print(f"[INFO] Last clean ckpt  : → {clean_path}")
    return history

# ─────────────────────────────────────────────────────────────────────────────
#  THRESHOLD CALIBRATION  (run once on validation, reuse on test)
# ─────────────────────────────────────────────────────────────────────────────
def calibrate_threshold(model, images_dir, masks_dir, device,
                        candidates=None, use_tta=False):
    """Sweep candidate thresholds on a held-out split and return the one that
    maximises mean Dice.

    Inference runs **once** — sigmoid probability maps are cached in memory,
    then every candidate threshold is evaluated offline.  This is fast and
    avoids the bias of using the same fixed 0.5 threshold on the test set.

    Args:
        candidates: list of floats to try (default: 0.10 → 0.90 step 0.05)

    Returns:
        (best_threshold: float, best_mean_dice: float)
    """
    if candidates is None:
        candidates = [round(t, 2) for t in np.arange(0.10, 0.91, 0.05).tolist()]

    fallback = (0.50, 0.0)
    if not (os.path.isdir(images_dir) and os.path.isdir(masks_dir)):
        print("[WARN] calibrate_threshold: images or masks dir missing; "
              "defaulting threshold to 0.50")
        return fallback

    model.eval()
    preds_np, gts_bin = [], []
    image_files = sorted([f for f in os.listdir(images_dir)
                          if f.lower().endswith(IMAGE_EXTENSIONS)])

    print(f"\n[INFO] Calibrating threshold — running inference on "
          f"{len(image_files)} validation images …")

    for fn in image_files:
        mask_path = find_mask_path(masks_dir, fn)
        if mask_path is None:
            continue

        image  = np.array(Image.open(os.path.join(images_dir, fn)).convert('RGB'))
        img_t  = val_aug(image=image)['image'].float().unsqueeze(0).to(device)
        gt_np  = np.array(Image.open(mask_path).convert('L'),
                          dtype=np.float32) / 255.0
        tokens = tokenizer([CAPTION]).to(device)

        if use_tta:
            prob = tta_predict(model, img_t, tokens)
        else:
            with torch.no_grad():
                logits, _ = model(img_t, tokens)
                prob = torch.sigmoid(logits)

        pred_np = F.interpolate(prob, size=gt_np.shape,
                                mode='bilinear', align_corners=False
                                )[0, 0].cpu().numpy()
        preds_np.append(pred_np)
        gts_bin.append(gt_np > 0.5)

    if not preds_np:
        print("[WARN] calibrate_threshold: no paired images found; "
              "defaulting threshold to 0.50")
        return fallback

    best_thresh, best_dice = 0.50, -1.0
    print(f"  {'Thresh':>7}  {'Mean Dice':>10}  {'Mean IoU':>10}")
    for t in candidates:
        dices = [dice_score(p > t, g) for p, g in zip(preds_np, gts_bin)]
        ious  = [iou_score (p > t, g) for p, g in zip(preds_np, gts_bin)]
        md, mi = float(np.mean(dices)), float(np.mean(ious))
        marker  = "  ←" if md > best_dice else ""
        print(f"   {t:.2f}      {md:.4f}      {mi:.4f}{marker}")
        if md > best_dice:
            best_dice, best_thresh = md, t

    print(f"[INFO] Best threshold : {best_thresh:.2f}  "
          f"(val Dice {best_dice:.4f})")
    return float(best_thresh), float(best_dice)


def threshold_sweep(model, images_dir, masks_dir, device,
                    candidates=None, use_tta=False,
                    highlight_thresh=None, split_name="Test"):
    """Evaluate *split* at every candidate threshold and return a list of row dicts.

    Inference runs once; all candidates are evaluated offline from cached
    probability maps.  Pass ``highlight_thresh`` to mark the val-calibrated
    value in the printed table.

    Saves nothing — caller writes the CSV.
    """
    if candidates is None:
        candidates = [round(t, 2) for t in np.arange(0.10, 0.91, 0.05).tolist()]

    model.eval()
    preds_np, gts_bin = [], []
    image_files = sorted([f for f in os.listdir(images_dir)
                          if f.lower().endswith(IMAGE_EXTENSIONS)])

    print(f"\n[INFO] Threshold sweep — collecting {split_name} predictions "
          f"({len(image_files)} images) …")

    for fn in image_files:
        mask_path = find_mask_path(masks_dir, fn)
        if mask_path is None:
            continue
        image  = np.array(Image.open(os.path.join(images_dir, fn)).convert('RGB'))
        img_t  = val_aug(image=image)['image'].float().unsqueeze(0).to(device)
        gt_np  = np.array(Image.open(mask_path).convert('L'),
                          dtype=np.float32) / 255.0
        tokens = tokenizer([CAPTION]).to(device)

        if use_tta:
            prob = tta_predict(model, img_t, tokens)
        else:
            with torch.no_grad():
                logits, _ = model(img_t, tokens)
                prob = torch.sigmoid(logits)

        pred_np = F.interpolate(prob, size=gt_np.shape,
                                mode='bilinear', align_corners=False
                                )[0, 0].cpu().numpy()
        preds_np.append(pred_np)
        gts_bin.append(gt_np > 0.5)

    if not preds_np:
        print(f"[WARN] threshold_sweep: no paired images in {split_name}; skipping")
        return []

    rows = []
    print(f"\n  Threshold sweep — {split_name}  ({len(preds_np)} images)")
    print(f"  {'Thresh':>7}  {'MeanDice':>9}  {'StdDice':>8}  "
          f"{'BestDice':>9}  {'MeanIoU':>8}  note")
    print("  " + "─" * 62)

    for t in candidates:
        dices = np.array([dice_score(p > t, g) for p, g in zip(preds_np, gts_bin)])
        ious  = np.array([iou_score (p > t, g) for p, g in zip(preds_np, gts_bin)])
        md, sd, bd = dices.mean(), dices.std(), dices.max()
        mi         = ious.mean()
        note = " ← val-calibrated" if (highlight_thresh is not None
                                        and abs(t - highlight_thresh) < 1e-9) else ""
        print(f"   {t:.2f}     {md:.4f}   {sd:.4f}    {bd:.4f}   {mi:.4f}{note}")
        rows.append({
            'threshold': t,
            'mean_dice': round(float(md), 6),
            'std_dice':  round(float(sd), 6),
            'best_dice': round(float(bd), 6),
            'mean_iou':  round(float(mi), 6),
            'std_iou':   round(float(ious.std()), 6),
        })

    best_row = max(rows, key=lambda r: r['mean_dice'])
    print(f"\n  Best on {split_name}: threshold={best_row['threshold']:.2f}  "
          f"Dice={best_row['mean_dice']:.4f}")
    if highlight_thresh is not None:
        cal_row = next((r for r in rows
                        if abs(r['threshold'] - highlight_thresh) < 1e-9), None)
        if cal_row:
            print(f"  Val-calibrated  : threshold={cal_row['threshold']:.2f}  "
                  f"Dice={cal_row['mean_dice']:.4f}  "
                  f"(Δ = {cal_row['mean_dice'] - best_row['mean_dice']:+.4f} vs test-best)")
    return rows


def evaluate_loop(model, images_dir, masks_dir, device,
                  threshold=0.5, split_name="", use_postprocess=False,
                  use_tta=False, output_dir=None, save_pred_masks=True,
                  sal_map_dir=None):
    model.eval()
    dice_list, iou_list, missing = [], [], 0
    sip_metrics = {
        'e_measure': [], 's_measure': [], 'weighted_f': [],
        'f_measure': [], 'mae': []
    }
    std_metrics_acc = {
        'max_f': [], 'mean_e': [], 'adp_f': [],
        's_measure': [], 'weighted_f': [], 'mae': []
    }
    per_image_rows = []
    pred_mask_dir = None
    if output_dir is not None and save_pred_masks:
        pred_mask_dir = os.path.join(output_dir, 'predicted_masks',
                                     _safe_split_dirname(split_name))
    # Auto-create saliency map directory if output_dir is set
    if sal_map_dir is None and output_dir is not None:
        sal_map_dir = os.path.join(output_dir, 'saliency_maps',
                                   _safe_split_dirname(split_name))
    if sal_map_dir is not None:
        os.makedirs(sal_map_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(images_dir)
                           if f.lower().endswith(IMAGE_EXTENSIONS)])
    print(f"\n[INFO] Evaluating {split_name}"
          + (" (TTA)" if use_tta else "") +
          f" — {len(image_files)} images")

    for fn in image_files:
        mask_path = find_mask_path(masks_dir, fn)
        if mask_path is None:
            missing += 1
            continue

        image  = np.array(Image.open(os.path.join(images_dir, fn)).convert('RGB'))
        img_t  = val_aug(image=image)['image'].float().unsqueeze(0).to(device)
        gt_np  = np.array(Image.open(mask_path).convert('L'), dtype=np.float32) / 255.0
        gt_bin = gt_np > 0.5

        tokens = tokenizer([CAPTION]).to(device)

        if use_tta:
            pred_probs = tta_predict(model, img_t, tokens)
        else:
            with torch.no_grad():
                logits, _ = model(img_t, tokens)
                pred_probs = torch.sigmoid(logits)

        pred_np  = F.interpolate(pred_probs, size=gt_np.shape,
                                  mode='bilinear', align_corners=False
                                  )[0, 0].cpu().numpy()

        # Save raw saliency map for PySODEvalToolkit
        if sal_map_dir is not None:
            sal_u8 = (np.clip(pred_np, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(sal_u8, mode='L').save(
                os.path.join(sal_map_dir, os.path.splitext(fn)[0] + '.png'))

        pred_bin = (postprocess_mask(pred_np, threshold)
                    if use_postprocess else pred_np >= threshold)
        if pred_mask_dir is not None:
            _save_predicted_mask(pred_bin, pred_mask_dir, fn)

        dice = dice_score(pred_bin, gt_bin)
        iou  = iou_score(pred_bin, gt_bin)
        sip  = sip_metric_scores(pred_np, gt_bin, threshold=threshold)
        std_met = standard_sod_metrics(pred_np, gt_bin)

        dice_list.append(dice)
        iou_list .append(iou)
        for key, value in sip.items():
            sip_metrics[key].append(value)
        for key, value in std_met.items():
            std_metrics_acc[key].append(value)

        per_image_rows.append({
            'filename': fn,
            'dice': round(float(dice), 6),
            'iou': round(float(iou), 6),
            **{k: round(float(v), 6) for k, v in sip.items()},
            **{f'std_{k}': round(float(v), 6) for k, v in std_met.items()},
        })

    if not dice_list:
        print(f"[WARN] evaluate_loop: no paired images in {split_name}; skipping")
        empty = np.array([], dtype=np.float64)
        return {'dice': empty, 'iou': empty,
                **{k: empty for k in sip_metrics},
                **{f'std_{k}': empty for k in std_metrics_acc}}

    dice_arr = np.array(dice_list)
    iou_arr  = np.array(iou_list)
    result = {
        'dice': dice_arr,
        'iou': iou_arr,
        **{k: np.array(v, dtype=np.float64) for k, v in sip_metrics.items()},
        **{f'std_{k}': np.array(v, dtype=np.float64) for k, v in std_metrics_acc.items()},
    }

    pp       = "ON" if use_postprocess else "OFF"
    print("=" * 68)
    print(f"  Split      : {split_name}   (post-process {pp})")
    print(f"  Evaluated  : {len(dice_list)} images"
          + (f"  | Skipped: {missing}" if missing else ""))
    print("-" * 68)
    print(f"  Mean Dice       : {dice_arr.mean():.4f}  ±  {dice_arr.std():.4f}")
    print(f"  Mean IoU        : {iou_arr.mean():.4f}  ±  {iou_arr.std():.4f}")
    print(f"  E-Measure ↑     : {result['e_measure'].mean():.4f}  ±  {result['e_measure'].std():.4f}")
    print(f"  S-Measure ↑     : {result['s_measure'].mean():.4f}  ±  {result['s_measure'].std():.4f}")
    print(f"  Weighted-F ↑    : {result['weighted_f'].mean():.4f}  ±  {result['weighted_f'].std():.4f}")
    print(f"  F-Measure ↑     : {result['f_measure'].mean():.4f}  ±  {result['f_measure'].std():.4f}")
    print(f"  MAE ↓           : {result['mae'].mean():.4f}  ±  {result['mae'].std():.4f}")
    print(f"  --- Standard SOD Protocol ---")
    print(f"  Max-F ↑         : {result['std_max_f'].mean():.4f}")
    print(f"  Mean-E ↑        : {result['std_mean_e'].mean():.4f}")
    print(f"  Adp-F ↑         : {result['std_adp_f'].mean():.4f}")
    print(f"  Best Dice       : {dice_arr.max():.4f}    Worst Dice : {dice_arr.min():.4f}")
    print("=" * 68)

    if output_dir is not None and per_image_rows:
        os.makedirs(output_dir, exist_ok=True)
        safe_split = _safe_split_dirname(split_name)
        metrics_csv = os.path.join(output_dir, f'{safe_split}_sip_metrics.csv')
        fields = ['filename', 'dice', 'iou', 'e_measure', 's_measure',
                  'weighted_f', 'f_measure', 'mae',
                  'std_max_f', 'std_mean_e', 'std_adp_f',
                  'std_s_measure', 'std_weighted_f', 'std_mae']
        with open(metrics_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(per_image_rows)
        print(f"[INFO] Per-image SIP metrics saved → {metrics_csv}")
    if sal_map_dir is not None:
        print(f"[INFO] Saliency maps saved → {sal_map_dir}")
    if pred_mask_dir is not None:
        print(f"[INFO] Predicted masks saved → {pred_mask_dir}")

    return result

# ─────────────────────────────────────────────────────────────────────────────
#  BENCHMARK EVALUATION (directory and NPY datasets)
# ─────────────────────────────────────────────────────────────────────────────
def _reshape_flat_image(arr: np.ndarray) -> np.ndarray:
    """Handle flattened benchmark NPY images such as (196608,) = 256*256*3."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        side3 = int(round(math.sqrt(arr.size / 3)))
        if side3 * side3 * 3 == arr.size:
            return arr.reshape(side3, side3, 3)
        side = int(round(math.sqrt(arr.size)))
        if side * side == arr.size:
            return arr.reshape(side, side)
    return arr


def _reshape_flat_mask(arr: np.ndarray) -> np.ndarray:
    """Handle flattened benchmark NPY masks such as (65536,) = 256*256."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        side = int(round(math.sqrt(arr.size)))
        if side * side == arr.size:
            return arr.reshape(side, side)
    return arr


def _as_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    arr = _reshape_flat_image(np.asarray(arr))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Cannot convert image array with shape {arr.shape} to RGB")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr[..., :3]


def _as_mask_float(arr: np.ndarray) -> np.ndarray:
    arr = _reshape_flat_mask(np.asarray(arr))
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Cannot convert mask array with shape {arr.shape} to HxW mask")
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _samples_from_dirs(images_dir: str, masks_dir: str, image_exts=None):
    """Load (filename, image, mask) triples.

    image_exts: optional tuple restricting which files count as images. Needed
    when images and masks live in the same folder (PASCAL-S-full), where the
    default IMAGE_EXTENSIONS would also pick up the .png masks.
    """
    if not (os.path.isdir(images_dir) and os.path.isdir(masks_dir)):
        return []
    exts = tuple(image_exts) if image_exts else IMAGE_EXTENSIONS
    same_folder = os.path.abspath(images_dir) == os.path.abspath(masks_dir)
    samples = []
    for fn in sorted(f for f in os.listdir(images_dir) if f.lower().endswith(exts)):
        if same_folder:
            # Mask must be the .png with the same stem, never the image itself.
            mask_path = os.path.join(masks_dir, os.path.splitext(fn)[0] + '.png')
            if not os.path.exists(mask_path) or mask_path == os.path.join(images_dir, fn):
                continue
        else:
            mask_path = find_mask_path(masks_dir, fn)
        if mask_path is None:
            continue
        image = np.array(Image.open(os.path.join(images_dir, fn)).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32) / 255.0
        samples.append((fn, image, mask))
    return samples


def _samples_from_npy(images_path: str, masks_path: str):
    if not (os.path.exists(images_path) and os.path.exists(masks_path)):
        return []
    images = np.load(images_path, allow_pickle=True)
    masks = np.load(masks_path, allow_pickle=True)
    if len(images) != len(masks):
        raise ValueError(f"NPY image/mask count mismatch: {images_path} ({len(images)}) vs {masks_path} ({len(masks)})")
    return [(f"{i:05d}.png", _as_rgb_uint8(images[i]), _as_mask_float(masks[i]))
            for i in range(len(images))]


class NpySalientObjectDataset(Dataset):
    """Training dataset for flattened or image-shaped NPY salient-object arrays."""
    def __init__(self, images_path: str, masks_path: str, augment=False):
        self.samples = _samples_from_npy(images_path, masks_path)
        if not self.samples:
            raise ValueError(f"No NPY samples loaded from {images_path} and {masks_path}")
        self.transform = train_aug if augment else val_aug
        print(f"[INFO] NPY dataset loaded: {len(self.samples)} samples from {images_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fn, image_np, mask_np = self.samples[idx]
        mask_u8 = (mask_np * 255.0).astype(np.uint8)
        out = self.transform(image=image_np, mask=mask_u8)
        img_t = out['image'].float()
        mask_t = (out['mask'].float() / 255.0).unsqueeze(0)
        return img_t, CAPTION, mask_t, fn


def load_benchmark_samples(root: str, name: str):
    key = name.lower().replace('_', '-').replace(' ', '-')
    if key == 'ecssd':
        return _samples_from_dirs(os.path.join(root, 'ECSSD', 'images'),
                                  os.path.join(root, 'ECSSD', 'ground_truth_mask'))
    if key in ('dut-o', 'dut-omron'):
        return _samples_from_dirs(os.path.join(root, 'DUT-O', 'DUT-OMRON-image'),
                                  os.path.join(root, 'DUT-O', 'pixelwiseGT-new-PNG'))
    if key in ('pascal-s', 'pascals'):
        return (_samples_from_npy(os.path.join(root, 'pascals', 'pascals_test_images.npy'),
                                  os.path.join(root, 'pascals', 'pascals_test_masks.npy')) or
                _samples_from_npy(os.path.join(root, 'pascals_test_images.npy'),
                                  os.path.join(root, 'pascals_test_masks.npy')))
    if key in ('hku-is', 'hkuis'):
        return _samples_from_npy(os.path.join(root, 'HKU-IS', 'images.npy'),
                                 os.path.join(root, 'HKU-IS', 'gt.npy'))
    if key in ('duts-te', 'duts'):
        candidates = [
            (os.path.join(root, 'DUTS_Dataset', 'DUTS-TE', 'DUTS-TE-Image'),
             os.path.join(root, 'DUTS_Dataset', 'DUTS-TE', 'DUTS-TE-Mask')),
            (os.path.join(root, 'DUTS_Dataset', 'DUTS-TE-Image'),
             os.path.join(root, 'DUTS_Dataset', 'DUTS-TE-Mask')),
            (os.path.join(root, 'DUTS_Dataset', 'DUTS-TE', 'Images'),
             os.path.join(root, 'DUTS_Dataset', 'DUTS-TE', 'Masks')),
        ]
        for img_dir, mask_dir in candidates:
            samples = _samples_from_dirs(img_dir, mask_dir)
            if samples:
                return samples
    return []


def evaluate_samples_loop(model, samples, device, threshold=0.5, split_name='',
                          use_postprocess=False, use_tta=False, output_dir=None,
                          save_pred_masks=True):
    model.eval()
    metrics = {'dice': [], 'iou': [], 'e_measure': [], 's_measure': [],
               'weighted_f': [], 'f_measure': [], 'mae': []}
    rows = []
    pred_mask_dir = None
    if output_dir is not None and save_pred_masks:
        pred_mask_dir = os.path.join(output_dir, 'predicted_masks',
                                     _safe_split_dirname(split_name))
    sal_map_dir = None
    if output_dir is not None:
        sal_map_dir = os.path.join(output_dir, 'saliency_maps',
                                   _safe_split_dirname(split_name))
    print(f"\n[INFO] Evaluating benchmark {split_name}"
          + (" (TTA)" if use_tta else "") + f" — {len(samples)} images")
    for fn, image_np, gt_np in samples:
        image_np = _as_rgb_uint8(image_np)
        gt_np = _as_mask_float(gt_np)
        gt_bin = gt_np > 0.5
        img_t = val_aug(image=image_np)['image'].float().unsqueeze(0).to(device)
        tokens = tokenizer([CAPTION]).to(device)
        if use_tta:
            pred_probs = tta_predict(model, img_t, tokens)
        else:
            with torch.no_grad():
                logits, _ = model(img_t, tokens)
                pred_probs = torch.sigmoid(logits)
        pred_np = F.interpolate(pred_probs, size=gt_np.shape,
                                mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
        pred_bin = postprocess_mask(pred_np, threshold) if use_postprocess else pred_np >= threshold
        if pred_mask_dir is not None:
            _save_predicted_mask(pred_bin, pred_mask_dir, fn)
        # Continuous uint8 saliency map for threshold-sweep metrics (PySODMetrics).
        if sal_map_dir is not None:
            os.makedirs(sal_map_dir, exist_ok=True)
            sal_u8 = (np.clip(pred_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            Image.fromarray(sal_u8, mode='L').save(
                os.path.join(sal_map_dir, os.path.splitext(fn)[0] + '.png'))
        sip = sip_metric_scores(pred_np, gt_bin, threshold=threshold)
        dice = dice_score(pred_bin, gt_bin)
        iou = iou_score(pred_bin, gt_bin)
        values = {'dice': dice, 'iou': iou, **sip}
        for k, v in values.items():
            metrics[k].append(v)
        rows.append({'filename': fn, **{k: round(float(v), 6) for k, v in values.items()}})

    if not rows:
        print(f"[WARN] No samples found for benchmark {split_name}")
        empty = np.array([], dtype=np.float64)
        return {k: empty for k in metrics}

    result = {k: np.array(v, dtype=np.float64) for k, v in metrics.items()}
    print("=" * 68)
    print(f"  Benchmark      : {split_name}")
    print(f"  Evaluated      : {len(rows)} images")
    print("-" * 68)
    print(f"  E-Measure ↑    : {result['e_measure'].mean():.4f}")
    print(f"  S-Measure ↑    : {result['s_measure'].mean():.4f}")
    print(f"  Weighted-F ↑   : {result['weighted_f'].mean():.4f}")
    print(f"  F-Measure ↑    : {result['f_measure'].mean():.4f}")
    print(f"  MAE ↓          : {result['mae'].mean():.4f}")
    print("=" * 68)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        safe = _safe_split_dirname(split_name)
        out_csv = os.path.join(output_dir, f'{safe}_sip_metrics.csv')
        fields = ['filename', 'dice', 'iou', 'e_measure', 's_measure',
                  'weighted_f', 'f_measure', 'mae']
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Benchmark per-image metrics saved → {out_csv}")
    if pred_mask_dir is not None:
        print(f"[INFO] Benchmark predicted masks saved → {pred_mask_dir}")
    return result


def evaluate_benchmark_suite(model, args, device, threshold, use_tta=False):
    requested = args.eval_benchmarks or []
    if any(x.lower() == 'all' for x in requested):
        requested = [ 'DUTS-TE', 'DUT-O','HKU-IS', 'ECSSD', 'PASCAL-S']
    summary = []
    for name in requested:
        samples = load_benchmark_samples(args.benchmark_root, name)
        result = evaluate_samples_loop(model, samples, device, threshold=threshold,
                                       split_name=name, use_tta=use_tta,
                                       output_dir=args.output_dir)
        if len(result['mae']) == 0:
            continue
        row = {
            'dataset': name,
            'threshold': round(float(threshold), 4),
            'e_measure': round(float(result['e_measure'].mean()), 6),
            's_measure': round(float(result['s_measure'].mean()), 6),
            'weighted_f': round(float(result['weighted_f'].mean()), 6),
            'f_measure': round(float(result['f_measure'].mean()), 6),
            'mae': round(float(result['mae'].mean()), 6),
            'dice': round(float(result['dice'].mean()), 6),
            'iou': round(float(result['iou'].mean()), 6),
            'n_samples': int(len(result['mae'])),
        }
        summary.append(row)

        safe = name.lower().replace(' ', '_').replace('-', '_')
        yaml_path = os.path.join(args.output_dir, f'{safe}_results.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump({'benchmark': row}, f, default_flow_style=False, sort_keys=False)
        print(f"[INFO] Benchmark YAML saved → {yaml_path}")
    if summary:
        out_csv = os.path.join(args.output_dir, 'benchmark_sip_results.csv')
        fields = ['dataset', 'threshold', 'e_measure', 's_measure', 'weighted_f',
                  'f_measure', 'mae', 'dice', 'iou', 'n_samples']
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)
        print(f"[INFO] Benchmark summary saved → {out_csv}")
    return summary

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#  TEST LOOP  (save visualisations)
# ─────────────────────────────────────────────────────────────────────────────
def _make_overlay(image_np: np.ndarray, mask_bin: np.ndarray,
                  colour: tuple, alpha: float = 0.45) -> np.ndarray:
    """
    Blend a boolean mask over an RGB image and draw a 1-px contour border.
    colour : (R, G, B) in [0, 255]
    Returns a uint8 RGB array.
    """
    from scipy.ndimage import binary_dilation
    out = image_np.copy().astype(np.float32)
    c   = np.array(colour, dtype=np.float32)
    # Semi-transparent fill
    out[mask_bin] = out[mask_bin] * (1 - alpha) + c * alpha
    # Solid 1-px border  (dilated XOR original)
    border = binary_dilation(mask_bin, iterations=2) & ~mask_bin
    out[border] = c
    return out.clip(0, 255).astype(np.uint8)


def test_loop(model, images_dir, output_dir, device,
              masks_dir=None, threshold=0.5, pred_mask_dir=None):
    """
    Save per-image visualisation panels for all images in `images_dir`.

    Layout when GT masks are available (masks_dir is set):
      ┌──────────────┬──────────────────┬──────────────────────────────┐
      │ Input image  │  GT overlay (⬜) │  Prediction overlay (🟥)     │
      │              │  (green)         │                              │
      └──────────────┴──────────────────┴──────────────────────────────┘

    Layout without GT (masks_dir=None or mask not found):
      ┌──────────────┬──────────────────────┐
      │ Input image  │  Prediction overlay  │
      └──────────────┴──────────────────────┘

    Overlays: green fill+border = GT, red fill+border = Prediction.
    Files are saved as <output_dir>/<stem>.png at 150 dpi.
    If pred_mask_dir is set, binary predicted masks are saved there as PNGs.
    """
    os.makedirs(output_dir, exist_ok=True)
    if pred_mask_dir is not None:
        os.makedirs(pred_mask_dir, exist_ok=True)
    model.eval()
    image_files = sorted([f for f in os.listdir(images_dir)
                          if f.lower().endswith(IMAGE_EXTENSIONS)])
    print(f"\n[INFO] Generating test visualisations → {output_dir}")

    saved = 0
    for fn in image_files:
        image    = Image.open(os.path.join(images_dir, fn)).convert('RGB')
        image_np = np.array(image)                                # H×W×3
        img_t    = val_aug(image=image_np)['image'].float().unsqueeze(0).to(device)
        tokens   = tokenizer([CAPTION]).to(device)

        # ── Prediction ────────────────────────────────────────────────────────
        pred_probs = tta_predict(model, img_t, tokens)
        pred_np    = F.interpolate(pred_probs,
                                   size=(image.height, image.width),
                                   mode='bilinear', align_corners=False
                                   )[0, 0].cpu().numpy()
        pred_bin   = postprocess_mask(pred_np, threshold)         # bool H×W
        if pred_mask_dir is not None:
            _save_predicted_mask(pred_bin, pred_mask_dir, fn)

        # ── GT (optional) ─────────────────────────────────────────────────────
        gt_bin = None
        if masks_dir is not None:
            gt_path = find_mask_path(masks_dir, fn)
            if gt_path is not None:
                gt_np  = np.array(Image.open(gt_path).convert('L'),
                                  dtype=np.float32) / 255.0
                gt_bin = gt_np > 0.5

        # ── Figure layout ─────────────────────────────────────────────────────
        n_cols = 3 if gt_bin is not None else 2
        fig, axes = plt.subplots(1, n_cols,
                                 figsize=(5.5 * n_cols, 5.5),
                                 gridspec_kw={'wspace': 0.04})
        fig.patch.set_facecolor('#1a1a1a')

        # Panel 1 — original image
        axes[0].imshow(image_np)
        axes[0].set_title('Input', color='white', fontsize=11, pad=5)
        axes[0].axis('off')

        col = 1

        # Panel 2 — GT overlay (green)
        if gt_bin is not None:
            axes[col].imshow(_make_overlay(image_np, gt_bin, (50, 220, 50)))
            axes[col].set_title('Ground Truth', color='#32de84',
                                fontsize=11, pad=5)
            axes[col].axis('off')
            col += 1

        # Panel 3 — Prediction overlay (red)
        pred_title = 'Prediction'

        axes[col].imshow(_make_overlay(image_np, pred_bin, (220, 60, 60)))
        axes[col].set_title(pred_title, color='#ff6b6b', fontsize=11, pad=5)
        axes[col].axis('off')

        # ── Save ──────────────────────────────────────────────────────────────
        stem     = os.path.splitext(fn)[0]
        out_path = os.path.join(output_dir, stem + '.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        saved += 1

    print(f"[INFO] Saved {saved} visualisation panels → {output_dir}")
    if pred_mask_dir is not None:
        print(f"[INFO] Saved {saved} predicted masks → {pred_mask_dir}")

# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING CURVES PLOT
# ─────────────────────────────────────────────────────────────────────────────
def save_training_plots(history: dict, output_dir: str):
    """
    Save a 3-panel figure:
      (1) Training loss vs epoch
      (2) Training Dice vs epoch
      (3) Training IoU  vs epoch
    File: <output_dir>/training_curves.png
    """
    epochs = list(range(1, len(history['loss']) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('ESC-Net Training Curves (SIP)', fontsize=13, fontweight='bold')

    panels = [
        ('loss', 'Compound Loss',   'tab:red'),
        ('dice', 'Dice Coefficient','tab:blue'),
        ('iou',  'IoU (Jaccard)',   'tab:green'),
    ]
    for ax, (key, title, color) in zip(axes, panels):
        ax.plot(epochs, history[key], color=color, linewidth=1.8)
        ax.fill_between(epochs, history[key], alpha=0.12, color=color)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.grid(True, linestyle='--', alpha=0.5)
        # Annotate best value
        best_ep  = int(np.argmax(history[key]) if key != 'loss'
                       else np.argmin(history[key])) + 1
        best_val = (max(history[key]) if key != 'loss' else min(history[key]))
        ax.axvline(best_ep, color=color, linestyle=':', alpha=0.6)
        ax.text(best_ep, best_val,
                f' ep {best_ep}\n {best_val:.4f}',
                fontsize=7, color=color, va='bottom')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Training curves saved → {out_path}")




# ─────────────────────────────────────────────────────────────────────────────
#  RESULTS  YAML
# ─────────────────────────────────────────────────────────────────────────────
def save_results_yaml(history: dict,
                      val_results: Optional[dict],
                      test_results: Optional[dict],
                      best_ckpt_info: dict,
                      args,
                      output_dir: str,
                      calibrated_threshold: Optional[float] = None):
    """
    Write a structured YAML file at <output_dir>/results.yaml that captures:
      • training config & per-epoch history
      • calibrated threshold (val-optimal) and fixed threshold
      • validation metrics (mean, std, best, worst Dice / IoU)
      • test metrics (same)
    """
    def arr_stats(arr: dict) -> dict:
        def stat_block(values: np.ndarray, higher_is_better: bool = True) -> dict:
            if values.size == 0:
                return {'mean': None, 'std': None, 'best': None, 'worst': None}
            best = values.max() if higher_is_better else values.min()
            worst = values.min() if higher_is_better else values.max()
            return {
                'mean':  float(f"{values.mean():.6f}"),
                'std':   float(f"{values.std():.6f}"),
                'best':  float(f"{best:.6f}"),
                'worst': float(f"{worst:.6f}"),
            }

        return {
            'dice':       stat_block(arr['dice']),
            'iou':        stat_block(arr['iou']),
            'e_measure':  stat_block(arr['e_measure']),
            's_measure':  stat_block(arr['s_measure']),
            'weighted_f': stat_block(arr['weighted_f']),
            'f_measure':  stat_block(arr['f_measure']),
            'mae':        stat_block(arr['mae'], higher_is_better=False),
            'n_samples':  int(len(arr['dice'])),
        }

    doc = {
        'training': {
            'best_epoch':      int(best_ckpt_info.get('epoch', -1)),
            'best_train_dice': float(f"{best_ckpt_info.get('best_dice', 0):.6f}"),
            'total_epochs':    int(args.epochs) if not getattr(args, 'test_only', False) else 0,
            'batch_size':      int(args.batch_size),
            'lr_pretrained':   float(args.lr_pretrained),
            'lr_new':          float(args.lr_new),
            'weight_decay':    float(args.weight_decay),
            'num_esc_blocks':  int(args.num_blocks),
            'num_prompts':     int(args.num_prompts),
            'clip_size':       int(args.clip_size),
            'warmup_epochs':   int(args.warmup_ep),
            'mse_lambda':      float(args.mse_lambda),
            # Full per-epoch history for later plotting or analysis
            'epoch_loss': [round(v, 6) for v in history['loss']],
            'epoch_dice': [round(v, 6) for v in history['dice']],
            'epoch_iou':  [round(v, 6) for v in history['iou']],
        },
        'threshold': {
            'calibrated': round(calibrated_threshold, 4) if calibrated_threshold is not None
                          else None,
            'fixed_arg':  float(args.threshold),
            'note': ('calibrated on validation set; applied to both val and test'
                     if calibrated_threshold is not None else 'fixed CLI value'),
        },
    }

    if val_results is not None:
        doc['validation'] = arr_stats(val_results)
    if test_results is not None:
        doc['test'] = arr_stats(test_results)

    out_path = os.path.join(output_dir, 'results.yaml')
    with open(out_path, 'w') as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] Results YAML saved  → {out_path}")

    summary_rows = []
    for split, results in [('validation', val_results), ('test', test_results)]:
        if results is None or len(results['dice']) == 0:
            continue
        summary_rows.append({
            'split': split,
            'threshold': round(calibrated_threshold, 4) if calibrated_threshold is not None else args.threshold,
            'e_measure': round(float(results['e_measure'].mean()), 6),
            's_measure': round(float(results['s_measure'].mean()), 6),
            'weighted_f': round(float(results['weighted_f'].mean()), 6),
            'f_measure': round(float(results['f_measure'].mean()), 6),
            'mae': round(float(results['mae'].mean()), 6),
            'dice': round(float(results['dice'].mean()), 6),
            'iou': round(float(results['iou'].mean()), 6),
            'n_samples': int(len(results['dice'])),
        })
    if summary_rows:
        summary_path = os.path.join(output_dir, 'sip_results.csv')
        fields = ['split', 'threshold', 'e_measure', 's_measure',
                  'weighted_f', 'f_measure', 'mae', 'dice', 'iou', 'n_samples']
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[INFO] SIP result summary saved → {summary_path}")

# ─────────────────────────────────────────────────────────────────────────────
#  OPTIMISER  — dual learning rate
# ─────────────────────────────────────────────────────────────────────────────
def build_optimizer(model: ESCNet, clip_model, _sam_model,
                    lr_pretrained: float, lr_new: float,
                    weight_decay: float, freeze_pretrained: bool = False):
    """
    Two parameter groups:
      pre-trained (CLIP attn + SAM decoder) → lr_pretrained (2e-5 default)
      new layers  (PPG / VLF / decoder)     → lr_new        (2e-4 default)

    Paper uses 2e-6 / 2e-4 for COCO-Stuff scale; we use 10× higher
    for the CLIP/SAM group because the SIP training set is smaller than large-scale pretraining datasets.
    """
    pretrained_ids: set = set()
    pretrained_params, new_params = [], []

    # Collect CLIP attention parameters (paper: fine-tune attn layers only)
    for name, param in clip_model.named_parameters():
        if 'attn' in name or 'self_attention' in name:
            pretrained_params.append(param)
            pretrained_ids.add(id(param))

    # Collect SAM decoder block parameters used inside ESC blocks
    for esc_block in model.esc_blocks:
        for param in esc_block.sam_block.decoder_block.parameters():
            if id(param) not in pretrained_ids:
                pretrained_params.append(param)
                pretrained_ids.add(id(param))
        for param in esc_block.sam_block.prompt_encoder.parameters():
            if id(param) not in pretrained_ids:
                pretrained_params.append(param)
                pretrained_ids.add(id(param))

    # Everything else is "new"
    for param in model.parameters():
        if id(param) not in pretrained_ids:
            new_params.append(param)

    # (2026-11) --freeze-pretrained: freeze the whole CLIP visual encoder and the
    # SAM decoder/prompt-encoder blocks; only new modules are trained.
    # Note: without this flag, CLIP visual non-attention weights fall in the
    # "new" group (lr_new) because only 'attn' names are collected above.
    if freeze_pretrained:
        frozen = set(id(p) for p in model.clip_visual.parameters()) | pretrained_ids
        for p in model.parameters():
            if id(p) in frozen:
                p.requires_grad_(False)
        pretrained_params = []
        new_params = [p for p in model.parameters() if p.requires_grad]

    print(f"[INFO] Optimiser: {len(pretrained_params)} pre-trained params "
          f"(lr={lr_pretrained:.1e}), "
          f"{len(new_params)} new params (lr={lr_new:.1e})"
          + (" [pretrained encoders FROZEN]" if freeze_pretrained else ""))

    groups = [{'params': new_params, 'lr': lr_new, 'weight_decay': weight_decay}]
    if pretrained_params:
        groups.insert(0, {'params': pretrained_params, 'lr': lr_pretrained, 'weight_decay': 0.0})
    return torch.optim.AdamW(groups)
    # OLD:
    # return torch.optim.AdamW([
    #     {'params': pretrained_params, 'lr': lr_pretrained, 'weight_decay': 0.0},
    #     {'params': new_params,        'lr': lr_new,        'weight_decay': weight_decay},
    # ])

# ─────────────────────────────────────────────────────────────────────────────
#  DEVICE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def resolve_device(args) -> str:
    """Select a training device and fail fast when CUDA is requested but unusable."""
    requested = args.device.lower()
    if requested == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if args.allow_cpu:
            print('[WARN] CUDA is not available; falling back to CPU because --allow-cpu was set.')
            return 'cpu'
        raise RuntimeError(
            'CUDA is not available to PyTorch, so training would run on CPU.\n'
            'Your nvidia-smi output can still show GPUs while PyTorch cannot use them. '
            'This commonly happens when the installed PyTorch CUDA build requires a newer '
            'driver than the system provides. In your log, PyTorch reported driver CUDA '
            'version 12020 while the installed build expects a newer driver.\n\n'
            'Fix one of these, then rerun:\n'
            '  1) Install a PyTorch build compatible with the cluster driver/CUDA 12.2, or\n'
            '  2) Ask the admin to update the NVIDIA driver, or\n'
            '  3) Use --allow-cpu only for a tiny smoke test.\n\n'
            'For A100 selection, you can also set CUDA_VISIBLE_DEVICES=0,1,2 before running.'
        )

    if requested == 'cpu':
        if not args.allow_cpu:
            raise RuntimeError('CPU was requested. Pass --allow-cpu to confirm this is intentional.')
        return 'cpu'

    if requested.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{args.device} was requested, but PyTorch reports CUDA is unavailable. "
                'Check your PyTorch CUDA build versus the installed NVIDIA driver, or run '
                '`python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`.'
            )
        return args.device

    raise ValueError(f"Unsupported --device value: {args.device!r}. Use auto, cpu, cuda, or cuda:N.")


def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch for repeatable ablation runs."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="ESC-Net SIP salient object segmentation")
    # Paths
    p.add_argument('--img-path',   default=IMG_PATH)
    p.add_argument('--mask-path',  default=MASK_PATH)
    p.add_argument('--train-images-npy', default=None,
                   help="Optional NPY training images; overrides --img-path/--mask-path when paired with --train-masks-npy")
    p.add_argument('--train-masks-npy', default=None,
                   help="Optional NPY training masks paired with --train-images-npy")
    p.add_argument('--val-img',    default=VALIDATION_IMAGES_PATH)
    p.add_argument('--val-mask',   default=VALIDATION_MASK_PATH)
    p.add_argument('--test-img',   default=TEST_IMAGES_PATH)
    p.add_argument('--test-mask',  default=TEST_MASK_PATH,
                   help="GT masks for SIP test set")
    p.add_argument('--output-dir', default=OUTPUT_DIR)
    p.add_argument('--pretrained-dir', default=PRETRAINED_DIR,
                   help="Directory for reusable pretrained checkpoints")
    p.add_argument('--sam-checkpoint', default=None,
                   help="Optional explicit SAM checkpoint path; otherwise uses --pretrained-dir/sam_vit_b.pth")
    p.add_argument('--benchmark-root', default=os.path.join(BASE_DIR, 'Datasets'),
                   help="Root containing ECSSD, DUT-O, DUTS_Dataset, and NPY benchmark files")
    p.add_argument('--eval-benchmarks', nargs='*', default=None,
                   help="Evaluate benchmark datasets by name after loading a checkpoint. Use ALL for ECSSD,DUT-O,DUTS-TE,HKU-IS,PASCAL-S.")
    p.add_argument('--no-calibrate', action='store_true',
                   help="Use --threshold directly instead of validation threshold calibration")
    p.add_argument('--skip-default-eval', action='store_true',
                   help="Skip the configured validation/test dirs and only run requested benchmark evaluation")
    # Training
    p.add_argument('--epochs',       type=int,   default=60)
    p.add_argument('--warmup-ep',    type=int,   default=5)
    p.add_argument('--batch-size',   type=int,   default=8,
                   help="Use 16 on A100 80 GB; 4-8 on smaller GPU")
    p.add_argument('--num-workers',  type=int,   default=4,
                   help="Use 8 on server; 2-4 locally")
    p.add_argument('--lr-pretrained',type=float, default=2e-6,
                   help="LR for CLIP attention + SAM decoder blocks")
    p.add_argument('--lr-new',       type=float, default=2e-5,
                   help="LR for new layers (PPG, VLF, decoder)")
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--clip-size',    type=int,   default=CLIP_SIZE)
    # Model
    p.add_argument('--clip-model',   default='ViT-L-14', choices=['ViT-L-14', 'ViT-B-16'],
                   help="CLIP vision backbone architecture")
    p.add_argument('--num-blocks',   type=int,   default=4,
                   help="Number of ESC blocks (paper uses N=4)")
    p.add_argument('--num-prompts',  type=int,   default=5)
    p.add_argument('--prompt-bank-size', type=int, default=10,
                   help='Number of prompts from the bank to use (1-20)')
    p.add_argument('--prompt-agg', default='max',
                   choices=['max', 'mean', 'learnable', 'attention', 'gated'],
                   help='Prompt aggregation mode: max (fixed bank + max-pool), '
                        'mean (fixed bank + mean), learnable (trainable prompt vectors), '
                        'attention (fixed bank + learned spatial attention), '
                        'gated (fixed bank + image-conditioned gates)')
    p.add_argument('--random-prompts', action='store_true',
                   help='Replace curated prompt bank with random generic sentences')
    p.add_argument('--diverse-prompts', action='store_true',
                   help='Use diverse category-specific prompt bank instead of default paraphrases')
    p.add_argument('--threshold',    type=float, default=0.5)
    p.add_argument('--mse-lambda',  type=float, default=0.1,
                   help="Lambda weight for the MSE term in the segmentation objective")
    p.add_argument('--eval-datasets', type=str, default='',
                   help='Comma-separated list of benchmarks to evaluate after training: DUTS-TE,DUT-O,HKU-IS,ECSSD,PASCAL-S,SIP')
    p.add_argument('--train-with-val-data', action='store_true',
                   help="Research-only: append the dedicated validation split to the training loader. Validation is still evaluated/calibrated after training, so reported validation metrics are no longer held-out.")
    # Misc
    p.add_argument('--resume',       default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument('--save-best-only', action='store_true',
                   help="Only save escnet_best.pth; skip last/clean checkpoints to save disk space")
    p.add_argument('--test-only',    action='store_true',
                   help="Skip training: load checkpoint (--resume, or escnet_best.pth "
                        "in --output-dir) and run threshold calibration + val/test "
                        "evaluation only.  No dataset or optimiser is constructed.")
    p.add_argument('--no-tta',         action='store_true',
                   help="Disable test-time augmentation during evaluation (this is now the default)")
    p.add_argument('--use-tta',        action='store_true',
                   help="Enable test-time augmentation during evaluation")
    p.add_argument('--no-prompt-bank', action='store_true',
                   help="Disable multi-prompt bank (use single caption only)")
    p.add_argument('--no-adaptive-fallback', action='store_true',
                   help="Disable PPG top-rho fallback; use threshold-only prompts")
    p.add_argument('--no-prompt-residual', action='store_true',
                   help="Disable corr_final = corr_N + corr_0 (decoder sees corr_N only). "
                        "For the matched residual ablation; default keeps the residual.")
    p.add_argument('--freeze-pretrained', action='store_true',
                   help="Freeze the CLIP visual encoder and SAM blocks; train only new modules")
    p.add_argument('--legacy-text-encoder', action='store_true',
                   help="Reproduce accepted-paper numbers: skip the CLIP text transformer "
                        "(embeddings depend only on token length). Default uses the real encoder.")
    p.add_argument('--vlf-multi-token', action='store_true',
                   help="VLF text cross-attention over prompt bank + caption (K+1 keys) "
                        "instead of the caption alone. Requires the prompt bank.")
    p.add_argument('--vlf-residual', action='store_true',
                   help="Each VLF block outputs corr_in + refinement instead of replacing "
                        "the correlation. Keeps the prompt prior through all ESC blocks.")
    p.add_argument('--no-bidirectional-sam', action='store_true',
                   help="Disable token-to-image SAM attention; use image-to-token only")
    p.add_argument('--simple-decoder', action='store_true',
                   help="Use plain U-Net-style decoder without ASPP/correlation/deep supervision")
    p.add_argument('--plain-bce-loss', action='store_true',
                   help="Use BCEWithLogitsLoss instead of the compound SOD loss")
    p.add_argument('--pp-loss-weight', type=float, default=0.3,
                   help="Point-classification auxiliary loss weight (0 disables it)")
    p.add_argument('--ds-weight', type=float, default=0.3,
                   help="Deep-supervision loss weight (0 disables it)")
    p.add_argument('--seed', type=int, default=0,
                   help="Random seed for reproducible runs")
    p.add_argument('--threshold-sweep', action='store_true',
                   help="After main evaluation, sweep all candidate thresholds "
                        "on the test set and save per-threshold Dice/IoU to "
                        "threshold_sweep_test.csv.  Requires test GT masks.")
    p.add_argument('--device', default='auto',
                   help="Device to use: auto, cuda, cuda:N, or cpu (default: auto)")
    p.add_argument('--allow-cpu', action='store_true',
                   help="Allow CPU execution when CUDA is unavailable or --device cpu is used")
    # LR milestones
    p.add_argument('--milestones', nargs='+', type=float, default=[0.5, 0.75],
                   metavar='F',
                   help="Fractional-epoch milestones for LR reduction "
                        "(e.g. --milestones 0.5 0.75).  "
                        "Each value is a fraction of --epochs.")
    p.add_argument('--milestone-decay', type=float, default=0.1,
                   help="Multiplicative LR factor applied at each milestone "
                        "(default 0.1 = one decade per drop)")
    return p.parse_args()


def main():
    args   = parse_args()
    set_seed(args.seed)
    device = resolve_device(args)
    print(f"[INFO] Device : {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── CLIP ViT @ 336 × 336  ─────────────────────────────────────────────────
    print(f"[INFO] Loading CLIP {args.clip_model} …")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name=args.clip_model,
        pretrained='openai',
        device=device,
        force_image_size=args.clip_size,   # 336 × 336
        force_quick_gelu=True)             # match OpenAI pretrained activation

    global tokenizer
    tokenizer = open_clip.get_tokenizer(args.clip_model)

    # ── SAM ViT-B  ────────────────────────────────────────────────────────────
    sam_ckpt = args.sam_checkpoint or os.path.join(args.pretrained_dir, 'sam_vit_b.pth')
    download_sam_checkpoint(sam_ckpt)
    print("[INFO] Loading SAM ViT-B …")
    sam = sam_model_registry['vit_b'](checkpoint=sam_ckpt).to(device)

    # ── Model  ────────────────────────────────────────────────────────────────
    model = ESCNet(clip_model, sam,
                   num_blocks=args.num_blocks,
                   num_prompts=args.num_prompts,
                   adaptive_fallback=not args.no_adaptive_fallback,
                   bidirectional_sam=not args.no_bidirectional_sam,
                   simple_decoder=args.simple_decoder,
                   clip_arch=args.clip_model,
                   prompt_agg=args.prompt_agg).to(device)
    model.legacy_text_encoder = args.legacy_text_encoder
    print(f"[INFO] Text encoder: {'LEGACY (no transformer)' if model.legacy_text_encoder else 'CLIP text transformer (frozen)'}")

    if not args.no_prompt_bank:
        if args.random_prompts:
            prompt_bank = RANDOM_PROMPT_BANK
        elif args.diverse_prompts:
            prompt_bank = DIVERSE_PROMPT_BANK
        else:
            prompt_bank = SALIENT_PROMPT_BANK
        model.setup_prompt_bank(tokenizer, device, prompts=prompt_bank,
                                bank_size=args.prompt_bank_size)

    # Prompt-residual switch (default True = original behaviour)
    model.use_prompt_residual = not args.no_prompt_residual
    print(f"[INFO] Prompt residual: {'ON' if model.use_prompt_residual else 'OFF'}")
    model.vlf_multi_token = args.vlf_multi_token
    print(f"[INFO] VLF text tokens: {'bank+caption' if model.vlf_multi_token else 'caption only'}")
    for _blk in model.esc_blocks:
        _blk.vlf.residual = args.vlf_residual
    print(f"[INFO] VLF in-block residual: {'ON' if args.vlf_residual else 'OFF'}")

    # ── Setup improved prompt mechanisms (if selected) ────────────────────────
    K = args.prompt_bank_size  # number of prompts
    if args.prompt_agg == 'learnable':
        model.setup_learnable_prompts(K, device, init_from_bank=True)
    elif args.prompt_agg == 'attention':
        model.setup_attention_aggregation(K, device)
    elif args.prompt_agg == 'gated':
        model.setup_prompt_gating(K, device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model parameters: {total / 1e6:.1f} M ({trainable / 1e6:.1f} M trainable)")

    best_path = os.path.join(args.output_dir, 'escnet_best.pth')
    use_tta   = args.use_tta  # Default: no TTA; enable with --use-tta
    history   = {'loss': [], 'dice': [], 'iou': [], 'lr': []}

    # ═════════════════════════════════════════════════════════════════════════
    #  TEST-ONLY PATH  — skip dataset/optimiser/training entirely
    # ═════════════════════════════════════════════════════════════════════════
    if args.test_only:
        ckpt_path = args.resume if args.resume else best_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"[ERROR] No checkpoint found at '{ckpt_path}'.\n"
                "  Pass --resume <path> to specify one, or ensure "
                "escnet_best.pth exists in --output-dir.")
        eval_ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(eval_ckpt['model_state'])
        print(f"[INFO] Loaded checkpoint: {ckpt_path}")
        print(f"       epoch {eval_ckpt.get('epoch', '?')}, "
              f"Dice {eval_ckpt.get('best_dice', 0):.4f}")

    # ═════════════════════════════════════════════════════════════════════════
    #  FULL TRAINING PATH
    # ═════════════════════════════════════════════════════════════════════════
    else:
        # ── Dataset & loader ─────────────────────────────────────────────────
        if args.train_images_npy or args.train_masks_npy:
            if not (args.train_images_npy and args.train_masks_npy):
                raise ValueError("Both --train-images-npy and --train-masks-npy are required for NPY training")
            train_ds = NpySalientObjectDataset(args.train_images_npy,
                                               args.train_masks_npy,
                                               augment=True)
            if args.train_with_val_data:
                print("[WARN] --train-with-val-data is ignored for NPY training.")
        else:
            captions_path = os.path.join(args.img_path, 'captions.json')
            if not os.path.exists(captions_path):
                captions = generate_captions(args.img_path, CAPTION, captions_path)
            else:
                with open(captions_path) as f:
                    captions = json.load(f)
                print(f"[INFO] Loaded captions.json ({len(captions)} entries)")

            train_ds = SalientObjectDataset(args.img_path, args.mask_path, captions,
                                            augment=True)
            if args.train_with_val_data:
                val_train_ds = SalientObjectDataset(args.val_img, args.val_mask, {},
                                                    augment=True)
                train_ds = ConcatDataset([train_ds, val_train_ds])
                print("[WARN] Research mode enabled: validation split appended to "
                      "training data. Validation metrics/threshold calibration "
                      "are no longer held-out.")
                print(f"[INFO] Combined train samples: {len(train_ds)}")
        pin_memory = torch.device(device).type == 'cuda'
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory,
                              drop_last=True)

        # ── Optimiser (dual LR) + scheduler ──────────────────────────────────
        optimizer = build_optimizer(
            model, clip_model, sam,
            lr_pretrained=args.lr_pretrained,
            lr_new=args.lr_new,
            weight_decay=args.weight_decay,
            freeze_pretrained=args.freeze_pretrained)

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=args.warmup_ep,
            total_epochs=args.epochs,
            milestones=args.milestones,
            milestone_decay=args.milestone_decay,
            eta_min=1e-6)

        # ── Resume from checkpoint ────────────────────────────────────────────
        if args.resume and os.path.exists(args.resume):
            ckpt_data = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt_data['model_state'])
            optimizer.load_state_dict(ckpt_data['optimizer_state'])
            resumed_epoch = ckpt_data.get('epoch', 1)
            print(f"[INFO] Resumed from epoch {resumed_epoch} "
                  f"(best Dice: {ckpt_data.get('best_dice', 0):.4f})")

        # ── Train ─────────────────────────────────────────────────────────────
        history = train_loop(
            model, train_dl, optimizer, scheduler, device,
            epochs=args.epochs,
            save_path=best_path,
            pp_loss_weight=args.pp_loss_weight, ds_weight=args.ds_weight,
            threshold=args.threshold,
            mse_lambda=args.mse_lambda,
            plain_bce_loss=args.plain_bce_loss,
            save_best_only=args.save_best_only)

        save_training_plots(history, args.output_dir)

        # ── Save final checkpoint (copy of last epoch) ────────────────────────
        last_path = os.path.join(args.output_dir, 'escnet_last.pth')
        final_path = os.path.join(args.output_dir, 'escnet_final.pth')
        if os.path.exists(last_path):
            shutil.copy2(last_path, final_path)
            print(f"[INFO] Final checkpoint saved → {final_path}")

        # ── Reload best checkpoint ────────────────────────────────────────────
        eval_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(eval_ckpt['model_state'])
        print(f"\n[INFO] Best checkpoint loaded — epoch {eval_ckpt['epoch']}, "
              f"train Dice {eval_ckpt['best_dice']:.4f}")

    # ═════════════════════════════════════════════════════════════════════════
    #  SHARED EVALUATION  (both training and test-only paths)
    # ═════════════════════════════════════════════════════════════════════════

    val_results = None
    test_results = None
    has_test_imgs = False

    # ── Calibrate threshold on validation set ────────────────────────────────
    # Run inference once on val, sweep candidate thresholds, pick the best one.
    # This threshold is then reused on the test set — never tuned on test data.
    if args.no_calibrate:
        cal_thresh = float(args.threshold)
        print(f"[INFO] Threshold calibration disabled; using --threshold={cal_thresh:.2f}")
    else:
        cal_thresh, _cal_dice = calibrate_threshold(
            model, args.val_img, args.val_mask, device, use_tta=use_tta)

    if not args.skip_default_eval:
        # ── Validation evaluation (with calibrated threshold) ────────────────────
        val_results = evaluate_loop(
            model, args.val_img, args.val_mask, device,
            threshold=cal_thresh, split_name="Validation",
            use_postprocess=False, use_tta=use_tta, output_dir=args.output_dir)

        # ── Test evaluation (GT available for SIP) ──────────────────────────
        test_results   = None
        has_test_imgs  = os.path.isdir(args.test_img)  and bool(os.listdir(args.test_img))
        has_test_masks = os.path.isdir(args.test_mask) and bool(os.listdir(args.test_mask))

        if has_test_imgs and has_test_masks:
            test_results = evaluate_loop(
                model, args.test_img, args.test_mask, device,
                threshold=cal_thresh, split_name="Test",
                use_postprocess=False, use_tta=use_tta, output_dir=args.output_dir)

            # ── Optional threshold sweep on test set ──────────────────────────
            if args.threshold_sweep:
                sweep_rows = threshold_sweep(
                    model, args.test_img, args.test_mask, device,
                    use_tta=use_tta, highlight_thresh=cal_thresh,
                    split_name="Test")
                if sweep_rows:
                    sweep_csv = os.path.join(args.output_dir,
                                             'threshold_sweep_test.csv')
                    _fields = ['threshold', 'mean_dice', 'std_dice',
                               'best_dice', 'mean_iou', 'std_iou']
                    with open(sweep_csv, 'w', newline='') as _f:
                        w = csv.DictWriter(_f, fieldnames=_fields)
                        w.writeheader()
                        w.writerows(sweep_rows)
                    print(f"[INFO] Threshold sweep saved → {sweep_csv}")

            test_loop(model, args.test_img,
                      os.path.join(args.output_dir, 'test_outputs'),
                      device, masks_dir=args.test_mask, threshold=cal_thresh,
                      pred_mask_dir=os.path.join(args.output_dir,
                                                 'predicted_masks', 'test'))
        elif has_test_imgs:
            print(f"\n[INFO] Test GT not found at {args.test_mask} — "
                  "generating visual outputs only (no metrics)")
            test_loop(model, args.test_img,
                      os.path.join(args.output_dir, 'test_outputs'),
                      device, threshold=cal_thresh,
                      pred_mask_dir=os.path.join(args.output_dir,
                                                 'predicted_masks', 'test'))
        else:
            print(f"\n[INFO] Test images not found at {args.test_img} — skipping test")

    else:
        print("[INFO] Skipping configured validation/test evaluation (--skip-default-eval)")

    if args.eval_benchmarks:
        evaluate_benchmark_suite(model, args, device, cal_thresh, use_tta=use_tta)

    # ── Multi-dataset evaluation (--eval-datasets) ────────────────────────────
    eval_dataset_results = {}
    if args.eval_datasets:
        eval_ds_names = [s.strip() for s in args.eval_datasets.split(',') if s.strip()]
        print(f"\n[INFO] Running multi-dataset evaluation: {eval_ds_names}")
        for ds_name in eval_ds_names:
            if ds_name in BENCHMARK_REGISTRY:
                reg = BENCHMARK_REGISTRY[ds_name]
                if reg['type'] == 'npy':
                    samples = _samples_from_npy(reg['images_npy'], reg['masks_npy'])
                else:
                    samples = _samples_from_dirs(reg['images'], reg['masks'],
                                                 image_exts=reg.get('image_exts'))
                if not samples:
                    print(f"[WARN] No samples found for {ds_name}; skipping")
                    continue
                result = evaluate_samples_loop(
                    model, samples, device, threshold=cal_thresh,
                    split_name=ds_name, use_postprocess=False, use_tta=use_tta,
                    output_dir=args.output_dir)
                if len(result.get('mae', [])) > 0:
                    eval_dataset_results[ds_name] = {
                        'e_measure': float(result['e_measure'].mean()),
                        's_measure': float(result['s_measure'].mean()),
                        'weighted_f': float(result['weighted_f'].mean()),
                        'f_measure': float(result['f_measure'].mean()),
                        'mae': float(result['mae'].mean()),
                        'dice': float(result['dice'].mean()),
                        'iou': float(result['iou'].mean()),
                        'n_samples': int(len(result['mae'])),
                    }
            else:
                print(f"[WARN] Dataset '{ds_name}' not in BENCHMARK_REGISTRY; skipping")

    # ── Save results_summary.md ───────────────────────────────────────────────
    summary_lines = ["# Results Summary\n"]
    summary_lines.append(f"**Model**: {args.clip_model}  \n")
    summary_lines.append(f"**Threshold**: {cal_thresh:.4f}  \n")
    summary_lines.append(f"**TTA**: {'Yes' if use_tta else 'No'}  \n")
    summary_lines.append(f"**Prompt Aggregation**: {args.prompt_agg}  \n")
    summary_lines.append(f"**Prompt Bank Size**: {args.prompt_bank_size}  \n\n")

    if val_results is not None and len(val_results.get('mae', [])) > 0:
        summary_lines.append("## Validation Results\n\n")
        summary_lines.append("| Metric | Value |\n|--------|-------|\n")
        summary_lines.append(f"| E-Measure ↑ | {val_results['e_measure'].mean():.4f} |\n")
        summary_lines.append(f"| S-Measure ↑ | {val_results['s_measure'].mean():.4f} |\n")
        summary_lines.append(f"| Weighted-F ↑ | {val_results['weighted_f'].mean():.4f} |\n")
        summary_lines.append(f"| F-Measure ↑ | {val_results['f_measure'].mean():.4f} |\n")
        summary_lines.append(f"| MAE ↓ | {val_results['mae'].mean():.4f} |\n")
        summary_lines.append(f"| Dice | {val_results['dice'].mean():.4f} |\n")
        summary_lines.append(f"| IoU | {val_results['iou'].mean():.4f} |\n\n")

    if test_results is not None and len(test_results.get('mae', [])) > 0:
        summary_lines.append("## Test Results\n\n")
        summary_lines.append("| Metric | Value |\n|--------|-------|\n")
        summary_lines.append(f"| E-Measure ↑ | {test_results['e_measure'].mean():.4f} |\n")
        summary_lines.append(f"| S-Measure ↑ | {test_results['s_measure'].mean():.4f} |\n")
        summary_lines.append(f"| Weighted-F ↑ | {test_results['weighted_f'].mean():.4f} |\n")
        summary_lines.append(f"| F-Measure ↑ | {test_results['f_measure'].mean():.4f} |\n")
        summary_lines.append(f"| MAE ↓ | {test_results['mae'].mean():.4f} |\n")
        summary_lines.append(f"| Dice | {test_results['dice'].mean():.4f} |\n")
        summary_lines.append(f"| IoU | {test_results['iou'].mean():.4f} |\n\n")

    if eval_dataset_results:
        summary_lines.append("## Benchmark Results\n\n")
        summary_lines.append("| Dataset | E-Measure ↑ | S-Measure ↑ | Weighted-F ↑ | F-Measure ↑ | MAE ↓ | Dice | IoU | N |\n")
        summary_lines.append("|---------|-------------|-------------|--------------|-------------|-------|------|-----|---|\n")
        for ds_name, metrics in eval_dataset_results.items():
            summary_lines.append(
                f"| {ds_name} | {metrics['e_measure']:.4f} | {metrics['s_measure']:.4f} | "
                f"{metrics['weighted_f']:.4f} | {metrics['f_measure']:.4f} | "
                f"{metrics['mae']:.4f} | {metrics['dice']:.4f} | {metrics['iou']:.4f} | "
                f"{metrics['n_samples']} |\n")
        summary_lines.append("\n")

    summary_md_path = os.path.join(args.output_dir, 'results_summary.md')
    with open(summary_md_path, 'w') as f:
        f.writelines(summary_lines)
    print(f"[INFO] Results summary saved → {summary_md_path}")

    # ── Save results YAML ─────────────────────────────────────────────────────
    save_results_yaml(history, val_results, test_results,
                      best_ckpt_info=eval_ckpt, args=args,
                      output_dir=args.output_dir,
                      calibrated_threshold=cal_thresh)

    print("\n[DONE] All outputs written to:", args.output_dir)
    if not args.test_only:
        print("  escnet_best.pth        — best training-Dice checkpoint")
        print("  escnet_last.pth        — last epoch checkpoint")
        print("  escnet_clean.pth       — last zero-NaN checkpoint")
        print("  training_curves.png    — loss / Dice / IoU plots")
        print("  training_diagnostics.csv")
    threshold_note = 'fixed --threshold' if args.no_calibrate else 'calibrated on val set'
    print(f"  threshold used         — {cal_thresh:.2f} ({threshold_note})")
    print("  results.yaml           — val & test metrics + threshold info")
    if has_test_imgs:
        print("  test_outputs/          — side-by-side visualisations")
        print("  predicted_masks/test/  — binary predicted mask PNGs")


if __name__ == '__main__':
    main()
