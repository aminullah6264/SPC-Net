# Gradient Diagnostic: prompt-residual (camera-ready, 2026-11)

Script: `grad_diagnostic.py`. Raw data: `/mnt/data/projects/CAD_Adaptation/CLIPSAM_new_results/grad_diagnostic/` (`summary.csv`, `structural.csv`, one CSV per config).

Setup: ViT-L/14 + SAM ViT-B, attention prompt aggregation, K=10. 50 real DUTS-TR batches (batch 8, AMP, BCE loss), identical batches and seeds for every config. Gradient norms are from freshly initialised models. They measure which parameters receive gradient at all, not training dynamics.

## Result 1: the VLF block ignores its inputs (our implementation)

| Check | Value |
|---|---|
| Max output change when the correlation input changes | 0.0 |
| Max output change when the visual features change | 0.0 |
| Gradient norm of VLF output w.r.t. correlation input | 0.0 |

Cause: in `VLFBlock.forward`, stage 2 is a cross-attention whose key/value set has one token (the caption embedding). Softmax over a single key is always 1, so every query returns the same value vector, `W_o W_v txt`. The output depends only on the text embedding. It does not depend on the correlation or the SAM-refined features, and it is constant across pixels within an image. Stage 1 (`embed_corr`, Swin window attention, `proj_stage1`) therefore receives zero gradient (`vlf0_embed_corr` = 0 in every config).

This is a property of our port, not of ESC-Net. In multi-class ESC-Net there is one text token per class, so attention over several keys is not degenerate. Collapsing to a single caption token in the binary setting removes that.

Consequences:
- After ESC block 1, the correlation passed to the next block is spatially constant. PPG in blocks 2–4 runs on a constant map, so its point choice no longer reflects the image.
- Without the residual, the decoder's correlation input carries no image information, and nothing upstream of it gets gradient through that path.

## Result 2: `SimpleMaskDecoder` does not read the correlation

`SimpleMaskDecoder.forward(x, skips, corr_maps)` accepts `corr_maps` but never uses it (verified from the code object and by output equality below).

## Result 3: gradient norms (mean over 50 batches)

| Config | loss | vis_proj_w | prompt_attn | VLF embed_corr | VLF(last) out_proj | decoder |
|---|---|---|---|---|---|---|
| Simple decoder, residual ON | 0.7122 | 0 | 0 | 0 | 0 | 3.634 |
| Simple decoder, residual OFF | 0.7122 | 0 | 0 | 0 | 0 | 3.634 |
| ASPP decoder, residual ON | 0.6489 | 0.714 | 2.13e-3 | 0 | 1.01e-2 | 3.488 |
| ASPP decoder, residual OFF | 0.6466 | 0 | 0 | 0 | 0 | 5.462 |

## What this means for the paper's claims

1. **The residual matters only with the correlation-aware (ASPP) decoder.** There, it is the only route by which the prompt correlation reaches the output and by which `vis_proj_w` and the prompt-aggregation weights get gradient. Its necessity comes from the degenerate VLF above. It is not a general property of the SAM–CLIP pipeline.
2. **In all simple-decoder configurations the residual has no effect.** That covers Baseline, +MP, +AF, +BiSAM, and the two "attention" runs in Table 5. Losses are identical with it on or off. This includes the recommended +AF configuration.
3. **In simple-decoder configurations the prompt bank affects the output only through discrete PPG point selection in ESC block 1.** The prompt-aggregation network gets zero gradient, so in "Base+Attention" / "MP+Attention" the attention weights stayed at their random initialisation.
4. The original claim, "VLF iteratively overwrites the correlation", should be replaced by the measured mechanism: single-token cross-attention makes the VLF output independent of its spatial inputs.

## Related inconsistencies found while checking configurations

- **"Base+Attention" and "MP+Attention" are the same configuration.** `prompt_modes/baseline_attention` (run_gpu0.sh) and `prompt_modes/mp_attention` (run_all_remaining.sh) use identical flags. The two rows in Table 5 are two runs of one config. Their gap (DUTS-TE Sm 0.9244 vs 0.9257; SIP 0.9288 vs 0.9362) measures run-to-run variance. On SIP (129 images) that variance is about 0.007.
- **"K=20" is K=10.** `SALIENT_PROMPT_BANK` has 10 prompts and `--prompt-bank-size 20` slices `prompts[:20]`. The K20 log reports "Prompt bank: 10 descriptions". The K=20 row (0.9163) is a repeat of K=10 (0.9172).
- **"Residual γ" in Table 6 is the point-loss weight.** The code has no residual coefficient (`corr = corr + corr_init`).
