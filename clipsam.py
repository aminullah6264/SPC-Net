# -*- coding: utf-8 -*-
"""
ESC-Net adapted for ISIC-2017 Skin Lesion Segmentation
Paper : arXiv:2411.14723v2  "ESC-Net: Effective SAM Combination for
         Open-Vocabulary Semantic Segmentation"
Domain: Binary lesion segmentation on ISIC-2017 dermoscopy dataset

Changes from original Colab notebook
─────────────────────────────────────
Bug fixes
  • Removed Google Colab / Google Drive dependencies
  • Removed !pip shell-magic (use requirements.txt instead)
  • Fixed syntax error  `model.train()S` → `model.train()`
  • Fixed ISIC-2017 mask naming: `<base>_segmentation.png`
    (original code assumed an "imgx → imgy" convention that is wrong)

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
  • Better caption: "a dermoscopic image of a skin lesion …"
"""

import os
import json
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import open_clip
from segment_anything import sam_model_registry
from sklearn.cluster import KMeans
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
#  PATHS  (local server — extracted by setup_dataset.py)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET  = '/mnt/data/users/av543f/datasets'

IMG_PATH               = os.path.join(DATASET, 'ISIC-2017_Training_Data')
MASK_PATH              = os.path.join(DATASET, 'ISIC-2017_Training_Part1_GroundTruth')
VALIDATION_IMAGES_PATH = os.path.join(DATASET, 'ISIC-2017_Validation_Data')
VALIDATION_MASK_PATH   = os.path.join(DATASET, 'ISIC-2017_Validation_Part1_GroundTruth')
TEST_IMAGES_PATH       = os.path.join(DATASET, 'ISIC-2017_Test_v2_Data')
TEST_MASK_PATH         = os.path.join(DATASET, 'ISIC-2017_Test_v2_Part1_GroundTruth')    # GT released by ISIC-2017
OUTPUT_DIR             = os.path.join(BASE_DIR, 'outputs')

# More specific dermoscopy caption (better CLIP alignment)
CAPTION = "a dermoscopic image of a skin lesion with irregular border"

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
CLIP_SIZE = 336    # ViT-B/16 native resolution — matches paper
CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

# ─────────────────────────────────────────────────────────────────────────────
#  AUGMENTATION  — dermoscopy-tuned (updated to 336 × 336)
# ─────────────────────────────────────────────────────────────────────────────
train_aug = A.Compose([
    A.Resize(CLIP_SIZE, CLIP_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15,
                       rotate_limit=45, p=0.7,
                       border_mode=cv2.BORDER_REFLECT_101),
    A.ElasticTransform(alpha=80, sigma=8, p=0.3),
    A.OneOf([
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
        A.OpticalDistortion(distort_limit=0.3, p=1.0),
    ], p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30,
                         val_shift_limit=20, p=0.5),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
    A.CLAHE(clip_limit=4.0, p=0.3),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.MedianBlur(blur_limit=5, p=1.0),
        A.MotionBlur(p=1.0),
    ], p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.CoarseDropout(max_holes=8, max_height=28, max_width=28,
                    min_holes=1, fill_value=0, mask_fill_value=0, p=0.3),
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
    ISIC-2017 mask naming convention:
        image: ISIC_XXXXXXX.jpg
        mask : ISIC_XXXXXXX_segmentation.png
    Falls back to .jpg extension if .png not found.
    """
    base = os.path.splitext(image_fn)[0]
    for ext in ('.png', '.jpg', '.jpeg'):
        p = os.path.join(masks_dir, base + '_segmentation' + ext)
        if os.path.exists(p):
            return p
    return None

# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────
class ISICDataset(Dataset):
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
    if not os.path.exists(path):
        print("[INFO] Downloading SAM ViT-B checkpoint …")
        import requests
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[INFO] SAM checkpoint saved → {path}")

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
    """0.6 × Tversky  +  0.4 × boundary-weighted FocalBCE."""
    def __init__(self, w_tversky=0.6, w_focal=0.4, gamma=2.0,
                 tversky_alpha=0.7, tversky_beta=0.3):
        super().__init__()
        self.w_tversky = w_tversky
        self.w_focal   = w_focal
        self.tversky   = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.focal     = FocalBCE(gamma=gamma)

    def forward(self, pred_logits, target):
        edge_w = sobel_edge_weight(target, pred_logits.device)
        return (self.w_tversky * self.tversky(pred_logits, target) +
                self.w_focal   * self.focal(pred_logits, target, weight=edge_w))

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
        shortcut = x_hwc
        wins, B_, H_, W_, Hp, Wp = self._partition(self.norm1(x_hwc))
        attn_out, _ = self.attn(wins, wins, wins, need_weights=False)
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
    def __init__(self, threshold=0.1, num_regions=5):
        super().__init__()
        self.threshold   = threshold
        self.num_regions = num_regions

    def forward(self, corr_maps):
        # corr_maps: (B, H, W)
        B, H, W    = corr_maps.shape
        P          = self.num_regions
        all_points = torch.zeros(B, P, 2,    dtype=torch.float32)
        all_labels = torch.zeros(B, P,       dtype=torch.int64)
        all_masks  = torch.zeros(B, P, H, W, dtype=torch.uint8)

        for b in range(B):
            prob     = F.softmax(corr_maps[b].detach().cpu().reshape(-1), dim=0).view(H, W)
            bin_mask = (prob > self.threshold).numpy().astype(np.uint8)
            coords   = np.column_stack(np.nonzero(bin_mask))

            if len(coords) == 0:
                # Fallback: single centre point
                all_points[b, 0] = torch.tensor([W // 2, H // 2], dtype=torch.float32)
                all_labels[b, 0] = 1
                all_masks [b, 0] = torch.from_numpy(bin_mask)
                continue

            n_clusters = min(P, len(coords))
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
    def __init__(self, sam_decoder_block, prompt_encoder, feature_dim, num_prompts):
        super().__init__()
        self.decoder_block  = sam_decoder_block
        self.prompt_encoder = prompt_encoder
        sam_dim             = sam_decoder_block.self_attn.q_proj.in_features
        self.sam_dim        = sam_dim
        self.num_prompts    = num_prompts
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
            pt_t2i = self.decoder_block.cross_attn_token_to_image(
                pt_sa, img_tokens, img_tokens)                         # (B, T, sam_dim)

            # 3) Image-to-token cross-attention  (image queries, prompt K/V)
            #    Image features are refined by prompt information
            ir = self.decoder_block.cross_attn_image_to_token(
                img_tokens, pt_t2i, pt_t2i)                           # (B, H*W, sam_dim)

            # Reshape to spatial map and blend with dense prompt
            xmap = ir.permute(0, 2, 1).view(B, self.sam_dim, H, W)
            out_list.append(xmap + de * feat_sam)

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
        txt_k   = (self.txt_proj(txt_embeds)                              # (B, C)
                       .unsqueeze(1).expand(-1, N_c, -1)                  # (B, Nc, C)
                       .reshape(B * N_c, self.feature_dim)                # (B·Nc, C)
                       .unsqueeze(0))                                      # (1, B·Nc, C)
        img_m, _ = self.txt_attn(query=img_v, key=txt_k, value=txt_k)    # (S, B·Nc, C)

        # Project to scalar correlation and reshape
        out = self.out_proj(img_m)                                        # (S, B·Nc, 1)
        out = out.permute(1, 2, 0).view(B, N_c, H, W)                   # (B, Nc, H, W)
        return out

# ─────────────────────────────────────────────────────────────────────────────
#  ESC BLOCK  (one ESC = PPG + SAMBlock + VLFBlock)
# ─────────────────────────────────────────────────────────────────────────────
class ESCBlock(nn.Module):
    def __init__(self, sam_dec_block, prompt_encoder,
                 feature_dim, num_prompts, text_dim):
        super().__init__()
        self.ppg       = PseudoPromptGenerator(num_regions=num_prompts)
        self.sam_block = SAMBlock(sam_dec_block, prompt_encoder,
                                  feature_dim, num_prompts)
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

# ─────────────────────────────────────────────────────────────────────────────
#  ESCNet  (main model)
# ─────────────────────────────────────────────────────────────────────────────
class ESCNet(nn.Module):
    """
    ESC-Net adapted for binary lesion segmentation.

    Vision backbone : CLIP ViT-B/16  (replaces RN50 in original Colab code)
    Multi-scale features extracted via forward hooks on transformer blocks
    [2, 5, 8, 11], reshaped from (seq, B, C) → (B, C, H, W) spatial maps.
    """
    # ViT-B/16 constants
    VIT_FEAT_DIM  = 768    # ViT hidden dimension
    CLIP_PROJ_DIM = 512    # CLIP shared embedding space
    HOOK_INDICES  = [2, 5, 8, 11]   # transformer block indices (0-based)

    def __init__(self, clip_model, sam_model,
                 num_blocks=4, num_prompts=5):
        super().__init__()

        # ── CLIP visual encoder ───────────────────────────────────────────────
        self.clip_visual = clip_model.visual

        # Text encoding components (same for all CLIP architectures)
        self.token_embedding = clip_model.token_embedding
        self.pos_embed       = clip_model.positional_embedding
        self.ln_final        = clip_model.ln_final
        self.text_proj       = clip_model.text_projection   # Parameter (512, 512)

        # Visual → CLIP projection (768 → 512) for correlation computation.
        # Works whether clip_model.visual.proj is nn.Linear or nn.Parameter.
        vp = clip_model.visual.proj
        if isinstance(vp, nn.Parameter):
            proj_data = vp.data.clone()           # shape (768, 512)
        elif isinstance(vp, nn.Linear):
            proj_data = vp.weight.data.T.clone()  # Linear stores (out, in); we need (768, 512)
        else:
            raise TypeError(f"Unexpected visual.proj type: {type(vp)}")
        # Learnable parameter — fine-tuned together with new layers
        self.vis_proj_w = nn.Parameter(proj_data)

        # ── Hook-based multi-scale feature extraction ─────────────────────────
        # Hooks populate self._hook_cache during each forward pass.
        self._hook_cache: dict = {}
        self._hook_handles  = []
        for level, idx in enumerate(self.HOOK_INDICES):
            h = self.clip_visual.transformer.resblocks[idx].register_forward_hook(
                self._make_hook(level))
            self._hook_handles.append(h)

        # ── Derive patch grid size from CLIP model ────────────────────────────
        # For ViT-B/16 @ 336px: (336 / 16)^2 = 21^2 = 441 patches
        if hasattr(self.clip_visual, 'image_size'):
            img_sz = self.clip_visual.image_size
        else:
            img_sz = CLIP_SIZE
        # image_size can be an int or a (H, W) tuple depending on open_clip version
        if isinstance(img_sz, (tuple, list)):
            img_sz = img_sz[0]
        patch_sz = 16   # ViT-B/16
        self.grid_h = self.grid_w = img_sz // patch_sz   # = 21

        # ── SAM esc blocks ────────────────────────────────────────────────────
        try:
            dec_blocks = sam_model.mask_decoder.transformer.layers
        except AttributeError:
            dec_blocks = list(sam_model.mask_decoder.transformer.children())

        num_blocks = min(num_blocks, len(dec_blocks))
        text_dim   = self.CLIP_PROJ_DIM   # 512

        self.esc_blocks = nn.ModuleList([
            ESCBlock(dec_blocks[i], sam_model.prompt_encoder,
                     self.VIT_FEAT_DIM, num_prompts, text_dim)
            for i in range(num_blocks)
        ])

        # ── FPN decoder  (all ViT skip dims = 768) ────────────────────────────
        self.mask_decoder = MaskDecoder(
            feature_dim = self.VIT_FEAT_DIM,
            skips_dims  = [self.VIT_FEAT_DIM,
                           self.VIT_FEAT_DIM,
                           self.VIT_FEAT_DIM],
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
        l0   = self._tokens_to_spatial(self._hook_cache[0])
        l1   = self._tokens_to_spatial(self._hook_cache[1])
        l2   = self._tokens_to_spatial(self._hook_cache[2])
        high = self._tokens_to_spatial(self._hook_cache[3])
        mid  = l1  # block-5 features used for mid-level correlation

        return {'high': high, 'mid': mid, 'skips': [l0, l1, l2]}

    def _encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """CLIP text encoder → L2-normalised text embeddings (B, 512)."""
        t   = self.token_embedding(tokens) + self.pos_embed
        t   = self.ln_final(t)
        eot = tokens.argmax(dim=-1)
        t   = t[torch.arange(t.shape[0]), eot]
        return F.normalize(t @ self.text_proj, dim=-1)

    def _project_visual(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Project spatial visual features from ViT space (768) to CLIP space (512).
        feat: (B, 768, H, W) → (B, 512, H, W)
        """
        B, C, H, W = feat.shape
        flat = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)     # (B*H*W, 768)
        proj = flat @ self.vis_proj_w                               # (B*H*W, 512)
        return F.normalize(proj.view(B, H, W, -1).permute(0, 3, 1, 2), dim=1)

    def compute_corr(self, feats: dict, txt_embeds: torch.Tensor) -> torch.Tensor:
        """
        Compute normalised cosine-similarity correlation maps.
        Returns (B, 2, H, W): [high-level corr, mid-level corr].
        """
        def minmax(t):
            mn = t.flatten(1).min(1)[0].view(-1, 1, 1)
            mx = t.flatten(1).max(1)[0].view(-1, 1, 1)
            return (t - mn) / (mx - mn + 1e-6)

        high_proj = self._project_visual(feats['high'])             # (B, 512, 21, 21)
        mid_proj  = self._project_visual(feats['mid'])              # (B, 512, 21, 21)

        high_corr = minmax(torch.einsum('bchw,bc->bhw', high_proj, txt_embeds))
        mid_corr  = minmax(torch.einsum('bchw,bc->bhw', mid_proj,  txt_embeds))

        H, W = feats['high'].shape[-2:]
        return torch.cat([
            F.interpolate(high_corr.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False),
            F.interpolate(mid_corr .unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False),
        ], dim=1)   # (B, 2, H, W)

    def forward(self, image: torch.Tensor, text_tokens: torch.Tensor):
        img_feats  = self._extract_clip_feats(image)
        txt_embeds = self._encode_text(text_tokens)
        corr       = self.compute_corr(img_feats, txt_embeds)

        pp_points = pp_labels = pp_size = None
        for i, block in enumerate(self.esc_blocks):
            img_feats, corr, (pts, labels) = block(img_feats, txt_embeds, corr)
            if i == 0:
                pp_points, pp_labels, pp_size = pts, labels, corr.shape[-2:]

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
    """Linear warmup for `warmup_epochs`, then cosine decay to `eta_min`."""
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 eta_min=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.eta_min       = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ep = self.last_epoch
        if ep < self.warmup_epochs:
            factor = (ep + 1) / max(self.warmup_epochs, 1)
        else:
            progress = ((ep - self.warmup_epochs) /
                        max(self.total_epochs - self.warmup_epochs, 1))
            factor   = (self.eta_min / self.base_lrs[0] +
                        0.5 * (1 - self.eta_min / self.base_lrs[0]) *
                        (1 + math.cos(math.pi * progress)))
        return [base_lr * factor for base_lr in self.base_lrs]

# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING LOOP
#   Returns history dict with per-epoch lists: 'loss', 'dice', 'iou', 'lr'
#   Saves two checkpoints every epoch:
#     escnet_best.pth  — whenever training Dice improves
#     escnet_last.pth  — overwritten at the end of every epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_loop(model, dataloader, optimizer, scheduler, device, epochs,
               save_path, pp_loss_weight=0.3, ds_weight=0.3, threshold=0.5):
    criterion  = CompoundLoss(w_tversky=0.6, w_focal=0.4, gamma=2.0,
                              tversky_alpha=0.7, tversky_beta=0.3)
    scaler     = GradScaler('cuda')
    best_dice  = 0.0
    last_path  = save_path.replace('escnet_best.pth', 'escnet_last.pth')

    history = {'loss': [], 'dice': [], 'iou': [], 'lr': []}

    print(f"\n{'Epoch':>6}  {'Loss':>8}  {'Dice':>8}  {'IoU':>8}  {'LR':>10}"
          f"  {'Saved':>8}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        model.train()   # ← fixed: was `model.train()S` in original
        total_loss, all_dice, all_iou = 0.0, [], []

        for imgs, caps, masks, _ in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            tokens      = tokenizer(caps).to(device)

            with autocast('cuda'):
                preds_logits, aux = model(imgs, tokens)
                preds_r = F.interpolate(preds_logits, size=masks.shape[-2:],
                                        mode='bilinear', align_corners=False)
                loss = criterion(preds_r, masks)

                for ds_key in ('ds2', 'ds3'):
                    if aux.get(ds_key) is not None:
                        ds_r = F.interpolate(aux[ds_key], size=masks.shape[-2:],
                                             mode='bilinear', align_corners=False)
                        loss += ds_weight * criterion(ds_r, masks)

                loss += pp_loss_weight * _point_cls_loss(preds_r, masks, aux, device)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds_probs = torch.sigmoid(preds_r)
            d, i = batch_dice_iou(preds_probs, masks, threshold)
            all_dice.append(d)
            all_iou.append(i)

        scheduler.step()

        n        = len(dataloader)
        avg_loss = float(total_loss / n)
        avg_dice = float(np.mean(all_dice))
        avg_iou  = float(np.mean(all_iou))
        cur_lr   = float(optimizer.param_groups[0]['lr'])

        history['loss'].append(avg_loss)
        history['dice'].append(avg_dice)
        history['iou'] .append(avg_iou)
        history['lr']  .append(cur_lr)

        # ── Checkpoint: last epoch (always overwrite) ─────────────────────────
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        last_payload = {
            'epoch': epoch, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'history': history,
        }
        torch.save(last_payload, last_path)

        # ── Checkpoint: best training Dice ────────────────────────────────────
        saved_tag = "last"
        if avg_dice > best_dice:
            best_dice = avg_dice
            best_payload = {**last_payload, 'best_dice': best_dice}
            torch.save(best_payload, save_path)
            saved_tag = "best+last"

        print(f"{epoch:>6}  {avg_loss:>8.4f}  {avg_dice:>8.4f}  "
              f"{avg_iou:>8.4f}  {cur_lr:>10.2e}  {saved_tag:>8}")

    print(f"\n[INFO] Best Train Dice : {best_dice:.4f}  → {save_path}")
    print(f"[INFO] Last checkpoint : epoch {epochs}    → {last_path}")
    return history

# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_loop(model, images_dir, masks_dir, device,
                  threshold=0.5, split_name="", use_postprocess=True,
                  use_tta=True):
    model.eval()
    dice_list, iou_list, missing = [], [], 0

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
        pred_bin = (postprocess_mask(pred_np, threshold)
                    if use_postprocess else pred_np > threshold)

        dice_list.append(dice_score(pred_bin, gt_bin))
        iou_list .append(iou_score (pred_bin, gt_bin))

    dice_arr = np.array(dice_list)
    iou_arr  = np.array(iou_list)
    pp       = "ON" if use_postprocess else "OFF"
    print("=" * 56)
    print(f"  Split      : {split_name}   (post-process {pp})")
    print(f"  Evaluated  : {len(dice_list)} images"
          + (f"  | Skipped: {missing}" if missing else ""))
    print("-" * 56)
    print(f"  Mean Dice  : {dice_arr.mean():.4f}  ±  {dice_arr.std():.4f}")
    print(f"  Mean IoU   : {iou_arr.mean():.4f}  ±  {iou_arr.std():.4f}")
    print(f"  Best Dice  : {dice_arr.max():.4f}    Worst Dice : {dice_arr.min():.4f}")
    print("=" * 56)
    return dice_arr, iou_arr

# ─────────────────────────────────────────────────────────────────────────────
#  TEST LOOP  (save side-by-side visualisations)
# ─────────────────────────────────────────────────────────────────────────────
def test_loop(model, images_dir, output_dir, device, threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    image_files = sorted([f for f in os.listdir(images_dir)
                           if f.lower().endswith(IMAGE_EXTENSIONS)])
    print(f"\n[INFO] Generating test outputs → {output_dir}")

    for fn in image_files:
        image    = Image.open(os.path.join(images_dir, fn)).convert('RGB')
        image_np = np.array(image)
        img_t    = val_aug(image=image_np)['image'].float().unsqueeze(0).to(device)
        tokens   = tokenizer([CAPTION]).to(device)

        pred_probs = tta_predict(model, img_t, tokens)
        pred_np    = F.interpolate(pred_probs, size=(image.height, image.width),
                                    mode='bilinear', align_corners=False
                                    )[0, 0].cpu().numpy()
        mask_u8    = (postprocess_mask(pred_np, threshold).astype(np.uint8) * 255)
        mask_rgb   = Image.fromarray(mask_u8).convert('RGB')

        canvas = Image.new('RGB', (image.width * 2, image.height))
        canvas.paste(image,    (0, 0))
        canvas.paste(mask_rgb, (image.width, 0))
        canvas.save(os.path.join(output_dir, fn))

    print(f"[INFO] Saved {len(image_files)} output images")

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
    fig.suptitle('ESC-Net Training Curves (ISIC-2017)', fontsize=13, fontweight='bold')

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
                      output_dir: str):
    """
    Write a structured YAML file at <output_dir>/results.yaml that captures:
      • training config & per-epoch history
      • validation metrics (mean, std, best, worst Dice / IoU)
      • test metrics (same)
    """
    def arr_stats(arr: np.ndarray) -> dict:
        return {
            'mean_dice':  float(f"{arr['dice'].mean():.6f}"),
            'std_dice':   float(f"{arr['dice'].std():.6f}"),
            'best_dice':  float(f"{arr['dice'].max():.6f}"),
            'worst_dice': float(f"{arr['dice'].min():.6f}"),
            'mean_iou':   float(f"{arr['iou'].mean():.6f}"),
            'std_iou':    float(f"{arr['iou'].std():.6f}"),
            'best_iou':   float(f"{arr['iou'].max():.6f}"),
            'worst_iou':  float(f"{arr['iou'].min():.6f}"),
            'n_samples':  int(len(arr['dice'])),
        }

    doc = {
        'training': {
            'best_epoch':      int(best_ckpt_info.get('epoch', -1)),
            'best_train_dice': float(f"{best_ckpt_info.get('best_dice', 0):.6f}"),
            'total_epochs':    int(args.epochs),
            'batch_size':      int(args.batch_size),
            'lr_pretrained':   float(args.lr_pretrained),
            'lr_new':          float(args.lr_new),
            'weight_decay':    float(args.weight_decay),
            'num_esc_blocks':  int(args.num_blocks),
            'num_prompts':     int(args.num_prompts),
            'clip_size':       int(args.clip_size),
            'warmup_epochs':   int(args.warmup_ep),
            # Full per-epoch history for later plotting or analysis
            'epoch_loss': [round(v, 6) for v in history['loss']],
            'epoch_dice': [round(v, 6) for v in history['dice']],
            'epoch_iou':  [round(v, 6) for v in history['iou']],
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

# ─────────────────────────────────────────────────────────────────────────────
#  OPTIMISER  — dual learning rate
# ─────────────────────────────────────────────────────────────────────────────
def build_optimizer(model: ESCNet, clip_model, _sam_model,
                    lr_pretrained: float, lr_new: float,
                    weight_decay: float):
    """
    Two parameter groups:
      pre-trained (CLIP attn + SAM decoder) → lr_pretrained (2e-5 default)
      new layers  (PPG / VLF / decoder)     → lr_new        (2e-4 default)

    Paper uses 2e-6 / 2e-4 for COCO-Stuff scale; we use 10× higher
    for the CLIP/SAM group because the ISIC-2017 training set is much smaller.
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

    print(f"[INFO] Optimiser: {len(pretrained_params)} pre-trained params "
          f"(lr={lr_pretrained:.1e}), "
          f"{len(new_params)} new params (lr={lr_new:.1e})")

    return torch.optim.AdamW([
        {'params': pretrained_params, 'lr': lr_pretrained, 'weight_decay': 0.0},
        {'params': new_params,        'lr': lr_new,        'weight_decay': weight_decay},
    ])

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="ESC-Net ISIC-2017 segmentation")
    # Paths
    p.add_argument('--img-path',   default=IMG_PATH)
    p.add_argument('--mask-path',  default=MASK_PATH)
    p.add_argument('--val-img',    default=VALIDATION_IMAGES_PATH)
    p.add_argument('--val-mask',   default=VALIDATION_MASK_PATH)
    p.add_argument('--test-img',   default=TEST_IMAGES_PATH)
    p.add_argument('--test-mask',  default=TEST_MASK_PATH,
                   help="GT masks for test set — available for ISIC-2017")
    p.add_argument('--output-dir', default=OUTPUT_DIR)
    # Training
    p.add_argument('--epochs',       type=int,   default=60)
    p.add_argument('--warmup-ep',    type=int,   default=5)
    p.add_argument('--batch-size',   type=int,   default=8,
                   help="Use 16 on A100 80 GB; 4-8 on smaller GPU")
    p.add_argument('--num-workers',  type=int,   default=4,
                   help="Use 8 on server; 2-4 locally")
    p.add_argument('--lr-pretrained',type=float, default=2e-5,
                   help="LR for CLIP attention + SAM decoder blocks")
    p.add_argument('--lr-new',       type=float, default=2e-4,
                   help="LR for new layers (PPG, VLF, decoder)")
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--clip-size',    type=int,   default=CLIP_SIZE)
    # Model
    p.add_argument('--num-blocks',   type=int,   default=4,
                   help="Number of ESC blocks (paper uses N=4)")
    p.add_argument('--num-prompts',  type=int,   default=5)
    p.add_argument('--threshold',    type=float, default=0.5)
    # Misc
    p.add_argument('--resume',       default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument('--no-tta',       action='store_true',
                   help="Disable test-time augmentation during evaluation")
    return p.parse_args()


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Device : {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── CLIP ViT-B/16 @ 336 × 336  ───────────────────────────────────────────
    print("[INFO] Loading CLIP ViT-B/16 …")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name='ViT-B-16',
        pretrained='openai',
        device=device,
        force_image_size=args.clip_size)   # 336 × 336

    global tokenizer
    tokenizer = open_clip.get_tokenizer('ViT-B-16')

    # ── SAM ViT-B  ────────────────────────────────────────────────────────────
    ckpt = os.path.join(args.output_dir, 'sam_vit_b.pth')
    download_sam_checkpoint(ckpt)
    print("[INFO] Loading SAM ViT-B …")
    sam = sam_model_registry['vit_b'](checkpoint=ckpt).to(device)

    # ── Captions  ─────────────────────────────────────────────────────────────
    captions_path = os.path.join(args.img_path, 'captions.json')
    if not os.path.exists(captions_path):
        captions = generate_captions(args.img_path, CAPTION, captions_path)
    else:
        with open(captions_path) as f:
            captions = json.load(f)
        print(f"[INFO] Loaded captions.json ({len(captions)} entries)")

    # ── Dataset & loader  ─────────────────────────────────────────────────────
    train_ds = ISICDataset(args.img_path, args.mask_path, captions, augment=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=True)

    # ── Model  ────────────────────────────────────────────────────────────────
    model = ESCNet(clip_model, sam,
                   num_blocks=args.num_blocks,
                   num_prompts=args.num_prompts).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {total / 1e6:.1f} M")

    # ── Optimiser (dual LR) + scheduler  ─────────────────────────────────────
    optimizer = build_optimizer(
        model, clip_model, sam,
        lr_pretrained=args.lr_pretrained,
        lr_new=args.lr_new,
        weight_decay=args.weight_decay)

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_ep,
        total_epochs=args.epochs,
        eta_min=1e-6)

    # ── Resume from checkpoint  ───────────────────────────────────────────────
    best_path = os.path.join(args.output_dir, 'escnet_best.pth')
    if args.resume and os.path.exists(args.resume):
        ckpt_data = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt_data['model_state'])
        optimizer.load_state_dict(ckpt_data['optimizer_state'])
        resumed_epoch = ckpt_data.get('epoch', 1)
        print(f"[INFO] Resumed from epoch {resumed_epoch} "
              f"(best Dice: {ckpt_data.get('best_dice', 0):.4f})")

    # ── Train  ────────────────────────────────────────────────────────────────
    history = train_loop(
        model, train_dl, optimizer, scheduler, device,
        epochs=args.epochs,
        save_path=best_path,
        pp_loss_weight=0.3, ds_weight=0.3,
        threshold=args.threshold)

    # ── Save training curves  ─────────────────────────────────────────────────
    save_training_plots(history, args.output_dir)

    # ── Reload best checkpoint for evaluation  ────────────────────────────────
    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt['model_state'])
    print(f"\n[INFO] Best checkpoint loaded — epoch {best_ckpt['epoch']}, "
          f"train Dice {best_ckpt['best_dice']:.4f}")

    use_tta = not args.no_tta

    # ── Validation evaluation  ────────────────────────────────────────────────
    val_dice, val_iou = evaluate_loop(
        model, args.val_img, args.val_mask, device,
        threshold=args.threshold, split_name="Validation",
        use_postprocess=True, use_tta=use_tta)
    val_results = {'dice': val_dice, 'iou': val_iou}

    # ── Test evaluation (GT is available for ISIC-2017)  ──────────────────────
    test_results = None
    has_test_imgs  = os.path.isdir(args.test_img)  and bool(os.listdir(args.test_img))
    has_test_masks = os.path.isdir(args.test_mask) and bool(os.listdir(args.test_mask))

    if has_test_imgs and has_test_masks:
        test_dice, test_iou = evaluate_loop(
            model, args.test_img, args.test_mask, device,
            threshold=args.threshold, split_name="Test",
            use_postprocess=True, use_tta=use_tta)
        test_results = {'dice': test_dice, 'iou': test_iou}

        # Also save side-by-side visualisation images
        test_loop(model, args.test_img,
                  os.path.join(args.output_dir, 'test_outputs'),
                  device, threshold=args.threshold)
    elif has_test_imgs:
        print(f"\n[INFO] Test GT not found at {args.test_mask} — "
              "generating visual outputs only (no metrics)")
        test_loop(model, args.test_img,
                  os.path.join(args.output_dir, 'test_outputs'),
                  device, threshold=args.threshold)
    else:
        print(f"\n[INFO] Test images not found at {args.test_img} — skipping test")

    # ── Save results YAML  ────────────────────────────────────────────────────
    save_results_yaml(history, val_results, test_results,
                      best_ckpt_info=best_ckpt, args=args,
                      output_dir=args.output_dir)

    print("\n[DONE] All outputs written to:", args.output_dir)
    print("  escnet_best.pth        — best training-Dice checkpoint")
    print("  escnet_last.pth        — last epoch checkpoint")
    print("  training_curves.png    — loss / Dice / IoU plots")
    print("  results.yaml           — validation & test metrics")
    if has_test_imgs:
        print("  test_outputs/          — side-by-side visualisations")


if __name__ == '__main__':
    main()
