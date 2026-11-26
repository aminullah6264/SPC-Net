#!/bin/bash
# Configurations used in the paper. Sourced by the other scripts.
# Each configuration is a set of flags for clipsam.py; seeds are added separately.

# Shared training and evaluation settings (Sec. 4.1-4.2).
EVAL_DATASETS="DUTS-TE,DUT-O,ECSSD,HKU-IS,PASCAL-S-full,SIP-full"
COMMON="--clip-model ViT-L-14 --epochs 30 --batch-size 8 --num-workers 4 \
--mse-lambda 0.1 --save-best-only --eval-datasets ${EVAL_DATASETS} \
--no-calibrate --threshold 0.5"

# Simple decoder + BCE (no deep supervision, no point loss), image-to-token SAM attention.
SIMPLE="--no-bidirectional-sam --simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0"

declare -A CFG=(
  # ── Table 3: prompt-side adaptations ──────────────────────────────────────
  [baseline]="--no-prompt-bank --no-adaptive-fallback ${SIMPLE}"
  [base_af]="--no-prompt-bank ${SIMPLE}"
  [plus_mp]="--no-adaptive-fallback ${SIMPLE}"
  [ours]="${SIMPLE}"                                            # SPC-Net = MP + AF
  [ours_K1]="${SIMPLE} --prompt-bank-size 1"
  [ours_K5]="${SIMPLE} --prompt-bank-size 5"
  [ours_random]="${SIMPLE} --random-prompts"
  [ours_diverse]="${SIMPLE} --diverse-prompts"
  [ours_vlfk]="${SIMPLE} --vlf-multi-token --vlf-residual"
  [ours_vlfk_nores]="${SIMPLE} --vlf-multi-token"
  # ── Table 4: architecture and training adaptations ────────────────────────
  [ours_bisam]="--simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0"
  [corr_max]="--pp-loss-weight 0 --ds-weight 0.3"
  [corr_max_nores]="--pp-loss-weight 0 --ds-weight 0.3 --no-prompt-residual"
  [corr_attn]="--pp-loss-weight 0 --ds-weight 0.3 --prompt-agg attention"
  [corr_attn_nores]="--pp-loss-weight 0 --ds-weight 0.3 --prompt-agg attention --no-prompt-residual"
  [corr_max_lpt]="--pp-loss-weight 0.3 --ds-weight 0.3"
  [ours_frozen]="${SIMPLE} --freeze-pretrained"
)

# Seeds per configuration (paper: three seeds for the main settings).
declare -A SEEDS=(
  [baseline]="42 1 2" [base_af]="42" [plus_mp]="42" [ours]="42 1 2"
  [ours_K1]="42" [ours_K5]="42" [ours_random]="42" [ours_diverse]="42"
  [ours_vlfk]="42 1 2" [ours_vlfk_nores]="42"
  [ours_bisam]="42" [corr_max]="42 1 2" [corr_max_nores]="42 1 2"
  [corr_attn]="42 1 2" [corr_attn_nores]="42 1 2" [corr_max_lpt]="42 1 2"
  [ours_frozen]="42 1 2"
)

TABLE3="baseline base_af plus_mp ours ours_K1 ours_K5 ours_random ours_diverse ours_vlfk ours_vlfk_nores"
TABLE4="ours_bisam corr_max corr_max_nores corr_attn corr_attn_nores corr_max_lpt ours_frozen"
