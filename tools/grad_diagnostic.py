#!/usr/bin/env python3
"""Gradient-flow diagnostic for the prompt residual and the VLF block (paper Sec. 3.5, 4.5).

Answers three questions with measurements instead of assumptions:
  1. Does the VLF output depend on its correlation input at all?
     (Its text cross-attention has a single key/value token.)
  2. Does SimpleMaskDecoder use corr_maps?
  3. On real DUTS-TR batches, what gradient reaches the correlation branch
     (vis_proj_w, prompt_attn, embed_corr) with the residual on vs off,
     for the simple and the full (ASPP) decoder?

Usage:
  CUDA_VISIBLE_DEVICES=0 python tools/grad_diagnostic.py \
      --batches 50 --out outputs/grad_diagnostic
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import argparse
import csv
import os
import random

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry
from torch.amp import autocast
from torch.utils.data import DataLoader

import clipsam
from clipsam import ESCNet, SALIENT_PROMPT_BANK, SalientObjectDataset


def build(clip_model, sam, tokenizer, device, simple_decoder, residual, multi_token=False,
          vlf_residual=False):
    m = ESCNet(clip_model, sam, clip_arch='ViT-L-14', prompt_agg='attention',
               simple_decoder=simple_decoder).to(device)
    m.setup_prompt_bank(tokenizer, device, prompts=SALIENT_PROMPT_BANK, bank_size=10)
    m.setup_attention_aggregation(10, device)
    m.use_prompt_residual = residual
    m.vlf_multi_token = multi_token
    for blk in m.esc_blocks:
        blk.vlf.residual = vlf_residual
    m.train()
    return m


def vlf_input_sensitivity(model, device, H=24, W=24, n_tokens=1):
    """Perturb corr_maps into one VLF block and measure the output change.

    n_tokens=1 reproduces the original single-caption VLF; n_tokens=11 mimics
    the --vlf-multi-token setting (10 bank prompts + caption).
    """
    vlf = model.esc_blocks[0].vlf
    vlf.eval()
    B, C = 2, model.feat_dim
    feat = torch.randn(B, C, H, W, device=device)
    if n_tokens == 1:
        txt = torch.randn(B, model.proj_dim, device=device)
    else:
        txt = torch.randn(B, n_tokens, model.proj_dim, device=device)
    c1 = torch.rand(B, 2, H, W, device=device)
    c2 = torch.rand(B, 2, H, W, device=device)
    with torch.no_grad():
        o1 = vlf(c1, feat, txt)
        o2 = vlf(c2, feat, txt)
        f2 = torch.randn_like(feat)
        o3 = vlf(c1, f2, txt)
    # Gradient of VLF output w.r.t. its corr input
    c = c1.clone().requires_grad_(True)
    vlf(c, feat, txt).sum().backward()
    return {
        'max_abs_change_when_corr_changes': float((o1 - o2).abs().max()),
        'max_abs_change_when_feat_changes': float((o1 - o3).abs().max()),
        'grad_norm_wrt_corr_input': float(c.grad.norm()),
        'output_std_across_pixels': float(o1.std()),
    }


def grad_norm(params):
    total = 0.0
    found = False
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().norm() ** 2)
            found = True
    return (total ** 0.5) if found else 0.0


def run_batches(model, loader, tokenizer, device, n_batches, use_amp=True):
    groups = {
        'vis_proj_w': [model.vis_proj_w],
        'prompt_attn': list(model.prompt_attn.parameters()),
        'vlf0_embed_corr': list(model.esc_blocks[0].vlf.embed_corr.parameters()),
        'vlf3_out_proj': list(model.esc_blocks[-1].vlf.out_proj.parameters()),
        'decoder': list(model.mask_decoder.parameters()),
    }
    crit = torch.nn.BCEWithLogitsLoss()
    rows = []
    for i, (imgs, caps, masks, _) in enumerate(loader):
        if i >= n_batches:
            break
        imgs, masks = imgs.to(device), masks.to(device)
        tokens = tokenizer(list(caps)).to(device)
        model.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=use_amp):
            logits, _ = model(imgs, tokens)
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear',
                                   align_corners=False)
            loss = crit(logits.float(), masks.float())
        loss.backward()  # unscaled: we only compare magnitudes
        row = {'batch': i, 'loss': float(loss)}
        for k, ps in groups.items():
            row[k] = grad_norm(ps)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--out', required=True)
    ap.add_argument('--only-multi', action='store_true',
                    help='Only run the --vlf-multi-token gradient configs')
    ap.add_argument('--fp32', action='store_true',
                    help='Disable autocast (avoids fp16 underflow of tiny gradients)')
    ap.add_argument('--vlf-residual', action='store_true',
                    help='Enable the in-block VLF residual in the multi-token configs')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    device = 'cuda'
    clip_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai', device=device,
        force_image_size=336, force_quick_gelu=True)
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    clipsam.tokenizer = tokenizer
    sam = sam_model_registry['vit_b'](
        checkpoint=os.path.join(clipsam.PRETRAINED_DIR, 'sam_vit_b.pth')).to(device)

    import json
    with open(os.path.join(clipsam.IMG_PATH, 'captions.json')) as f:
        caps = json.load(f)
    ds = SalientObjectDataset(clipsam.IMG_PATH, clipsam.MASK_PATH, caps, augment=False)
    g = torch.Generator(); g.manual_seed(0)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                        generator=g, drop_last=True)

    # 1+2: structural checks (simple decoder model is enough for VLF)
    m = build(clip_model, sam, tokenizer, device, simple_decoder=True, residual=True)
    sens = vlf_input_sensitivity(m, device, n_tokens=1)
    sens_multi = vlf_input_sensitivity(m, device, n_tokens=11)
    simple_uses_corr = 'corr_maps' in clipsam.SimpleMaskDecoder.forward.__code__.co_names
    del m; torch.cuda.empty_cache()

    # 3: gradient norms on identical batches
    configs = []
    if not args.only_multi:
        for simple in (True, False):
            for residual in (True, False):
                configs.append((simple, residual, False))
    for simple in (True, False):
        for residual in (True, False):
            configs.append((simple, residual, True))
    summary = {}
    for simple, residual, multi in configs:
        tag = (f"{'simple' if simple else 'aspp'}_dec__residual_{'on' if residual else 'off'}"
               f"{'__vlf_multi' if multi else ''}")
        torch.manual_seed(0)
        g.manual_seed(0)
        m = build(clip_model, sam, tokenizer, device, simple_decoder=simple,
                  residual=residual, multi_token=multi,
                  vlf_residual=(multi and args.vlf_residual))
        rows = run_batches(m, loader, tokenizer, device, args.batches, use_amp=not args.fp32)
        with open(os.path.join(args.out, f'{tag}.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        summary[tag] = {k: float(np.mean([r[k] for r in rows]))
                        for k in rows[0] if k != 'batch'}
        print(tag, {k: f"{v:.3e}" for k, v in summary[tag].items()}, flush=True)
        del m; torch.cuda.empty_cache()

    # Write report data
    with open(os.path.join(args.out, 'summary.csv'), 'w', newline='') as f:
        keys = list(next(iter(summary.values())).keys())
        w = csv.writer(f); w.writerow(['config', *keys])
        for tag, d in summary.items():
            w.writerow([tag, *[f"{d[k]:.6e}" for k in keys]])
    with open(os.path.join(args.out, 'structural.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['check', 'value'])
        for k, v in sens.items():
            w.writerow([f'vlf_1token_{k}', f"{v:.6e}"])
        for k, v in sens_multi.items():
            w.writerow([f'vlf_11token_{k}', f"{v:.6e}"])
        w.writerow(['simple_decoder_reads_corr_maps', simple_uses_corr])
    print('VLF sensitivity (1 token):', sens)
    print('VLF sensitivity (11 tokens):', sens_multi)
    print('SimpleMaskDecoder reads corr_maps:', simple_uses_corr)


if __name__ == '__main__':
    main()
