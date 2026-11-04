# Experiment History & Architecture Change Log

## Architecture A (WACV submission + runs before 2026-08-25)

**Stored in:** `outputs_archA_no_residual/`

**Description:** ESC-Net adapted for SOD with ViT-L/14. The initial prompt-derived
correlation map feeds only into the PPG for point selection, then VLF blocks
completely overwrite it over 4 iterations. The prompt bank has NO gradient path
to the final output — its influence is limited to discrete PPG point locations.

**Key finding:** With ViT-L/14's strong backbone, the fixed multi-prompt bank
(max aggregation) provides negligible or negative improvement because:
1. ViT-L features already produce strong single-prompt correlations
2. The max operation introduces false positives (non-salient high-contrast regions)
3. The prompts cannot adapt because there's no gradient feedback

**Completed runs (Architecture A):**
- Full model (60 epochs): Sm=0.9256, Em=0.9651, Fm=0.9143, MAE=0.0271 (DUTS-TE)
- Ablation baseline (60 ep): Sm=0.9264
- Ablation +MP (60 ep): Sm=0.9253 (worse than baseline!)
- Ablation +BiSAM (60 ep): Sm=0.9270
- Ablation +AF (35 ep, interrupted): Sm=0.9241
- Ablation +Dec (31 ep, interrupted): Sm=0.9229
- Cross-dataset eval on DUT-O and SIP confirmed: prompts hurt everywhere

---

## Architecture B (current, with correlation residual — 2026-08-25 onwards)

**Stored in:** `outputs/`

**Change:** Added `corr = corr + corr_init` after the ESC block loop. This gives
the prompt-derived correlation a direct additive path to the decoder input.

**Why this matters:**
1. Gradient now flows: loss → decoder → corr_residual → compute_corr → prompt embeddings
2. Learnable prompt embeddings can actually be optimized end-to-end
3. The prompt bank's spatial prior directly influences the final mask prediction
4. Attention and gated aggregation modes become meaningful (can learn to suppress noise)

**Preliminary results (5-epoch smoke test, Architecture B):**
- Learnable prompts: Epoch 4 Dice = 0.9450
- Attention aggregation: Epoch 4 Dice = 0.9441
- Gated prompts: Epoch 4 Dice = 0.9431
- (For comparison, Arch A full model at epoch 4: Dice = 0.9457)

**Experiment plan:** Full re-run of all experiments with Architecture B.

---

## Code Changes Summary

1. `ESCNet.forward()`: Added `corr_init = corr` before ESC blocks and `corr = corr + corr_init` after
2. `ESCNet.compute_corr()`: Added 'learnable', 'attention', 'gated' aggregation modes
3. `ESCNet.setup_learnable_prompts()`: New — creates K trainable prompt vectors
4. `ESCNet.setup_attention_aggregation()`: New — Conv2d attention network over prompts
5. `ESCNet.setup_prompt_gating()`: New — MLP gate per prompt conditioned on image
6. `WarmupCosineScheduler._segment_for()`: Fixed crash with epochs <= warmup_epochs
7. `--prompt-agg` choices extended: {max, mean, learnable, attention, gated}
