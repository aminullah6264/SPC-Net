# SPC-Net Camera-Ready Report (ACCV 2026, post-acceptance)

Living document for the camera-ready revision. Updated as runs finish.
Branch: `post-accept-camera-ready`. The accepted `main.tex` is not modified
(md5 `edbf649250c0211a2b49e929337cca60`, recorded in
`CLIPSAM_new_results/main_tex_md5_before.txt`). Paper edits will go in
`main_camera_ready.tex` with changes in red.

Results root: `/mnt/data/projects/CAD_Adaptation/CLIPSAM_new_results/`

## Decisions (user, 2026-11-05)

- Residual ablation: run the revised matched set (full model, residual off; seeds for baseline and +AF).
- VLF fix: implement and retrain. If it improves results, add it to the paper; if not, report the finding.
- Prompt bank stays at K <= 10. Drop the K=20 row (it was a duplicate of K=10).

## 1. PASCAL-S

- The 256-image NPY subset used before gave broken numbers (Sm 0.45). It is now marked deprecated in the code.
- New entry `PASCAL-S-full`: 850 images, `.jpg` images and same-stem `.png` masks in one folder.
- +AF (accepted recommended config) on PASCAL-S-full, PySODMetrics:

| Sm | maxF | meanF | adpF | wF | meanE | maxE | adpE | MAE |
|---|---|---|---|---|---|---|---|---|
| 0.8987 | 0.8986 | 0.8686 | 0.8558 | 0.8544 | 0.9242 | 0.9394 | 0.9154 | 0.0477 |

## 2. Gradient diagnostic (details: `GRADIENT_DIAGNOSTIC.md`)

Measured on 50 real DUTS-TR batches:

1. **The original VLF ignores its inputs.** Its text cross-attention has one key (the caption), so softmax = 1 and the output is a constant vector per image. Changing the correlation or the image features changes the output by exactly 0.
2. **`SimpleMaskDecoder` never reads the correlation.**
3. Hence the prompt-residual has **no effect** in every simple-decoder configuration (Baseline, +MP, +AF, +BiSAM, Base/MP+Attention): identical loss on/off. With the ASPP decoder, it is the only path from the prompt correlation to the output.
4. In simple-decoder configs the prompt-attention weights get zero gradient (they stay at random init).

Related corrections:
- "Base+Attention" and "MP+Attention" (Table 5) were run with identical flags, so they are one config run twice. Their SIP gap (0.9288 vs 0.9362) is run-to-run variance.
- "K=20" was K=10 (the bank has 10 prompts).
- Table 6 "Residual γ" is actually the point-loss weight. The code has no residual coefficient.

## 3. VLF fix (new flags, default off)

- `--vlf-multi-token`: VLF attends over the 10 bank prompts + caption (11 keys) instead of the caption alone.
- `--vlf-residual`: each VLF block outputs `corr_in + refinement` instead of replacing the correlation.

Sanity checks on one VLF block (random inputs, eval mode):

| Text keys | Output change when corr changes | Output change when features change | ‖∂out/∂corr‖ |
|---|---|---|---|
| 1 (original) | 0 | 0 | 0 |
| 11 (fix) | 3.0e-3 | 2.4e-2 | 7.4e-2 |

Gradient norms reaching the prompt branch, ASPP decoder, fp32, 20 batches:

| Config | vis_proj_w | prompt_attn |
|---|---|---|
| multi-token, top-level residual ON | 0.997 | 2.8e-3 |
| multi-token, top-level residual OFF | 1.9e-15 (vanishes) | 5.7e-18 |
| multi-token + VLF residual, top-level residual OFF | 0.998 | 2.8e-3 |

Multi-token alone makes VLF input-dependent, but the signal vanishes through four stacked blocks. With the in-block VLF residual, gradient reaches the prompt branch without the top-level residual. Both flags are used together in the VLF-fix runs.

Note: an accepted checkpoint evaluated with these flags turned on gives Sm 0.37 on SIP. That is expected (the flags change the network the weights were trained for), and it confirms the fix needs retraining.

## 4. Speed fix (no change in results)

The adaptive-fallback PPG runs scikit-learn KMeans 32 times per training step. Under CPU contention, multi-threaded KMeans took 880 ms per call (about 10 h/epoch) vs 12 ms single-threaded. KMeans is now run with `threadpool_limits(1)`. PPG outputs (points, labels, masks) are identical with and without the limit (5 trials × 8 maps). This also explains why +AF/full-model runs were slow in August.

## 5. Running experiments (`run_camera_ready.sh`, screens `cr_gpu0..3`)

All: ViT-L/14, DUTS-TR, 30 epochs, batch 8, `--save-best-only`, evaluated on DUTS-TE, DUT-O, ECSSD, HKU-IS, PASCAL-S-full, SIP with continuous maps saved. Outputs: `outputs_camera_ready/<name>/`.

| GPU | Queue |
|---|---|
| 0 | vlffix_plus_af_s42 → full_attention_noresid_s42 → plus_af_s2 |
| 1 | vlffix_full_max_s42 → full_max_noresid_s42 → baseline_s2 |
| 2 | vlffix_full_attention_s42 → baseline_s1 |
| 3 | vlffix_plus_mp_s42 → plus_af_s1 |

Matched comparisons these runs give (same seed 42, same 30 epochs, same data):
- Residual on vs off: full model (max) and full model + attention. The "on" runs already exist (`outputs/full_model`, `outputs/prompt_modes/attention`).
- VLF fix vs original: +MP, +AF, full (max), full + attention.
- Seed variance: baseline and +AF at seeds 42, 1, 2.

Unified re-evaluation of all 20 existing checkpoints (`run_unified_eval.sh`, screens `ue_A`/`ue_B`, outputs `unified_eval/<name>/`). Scored with `score_unified.py`.

## 6. Competitor numbers (`competitor_sources/`)

Extracted from the PDFs in `Openvocab-SD/Related Papers/`: 2048 values with paper, page and table (`COMPETITOR_SOURCES.csv`, `SOURCING_REPORT.md`).

- The earlier WACV table matches sourced values in all cells (mostly copied from the SODDCNet/CVPRW-2025 and STNet tables).
- The accepted paper's table does **not** match for DC-Net, ADMNet, BDFGNet and STNet on DUTS-TE, DUT-O and ECSSD. Those values appear in no source and must be replaced. PSGLoss, CSNet and MENet also have partly unsupported cells; RCSB was dropped.
- F-measure variants differ by paper (max vs mean vs unstated). The same method's DUTS-TE F ranges from 0.818 to 0.895 across papers.
- MDSAM (SAM ViT-B, ACM MM 2024, own paper; two rows, likely two input sizes): DUTS-TE Sm 0.919/0.920, maxF 0.934/0.937, MAE 0.025/0.024; DUT-O Sm 0.878/0.881; ECSSD Sm 0.946/0.948; HKU-IS Sm 0.941; PASCAL-S maxF 0.907/0.912.
- Not extractable: Zhou ICCV 2025 (no Sm/F/E/MAE), Positive-Incentive (tables are images).
- "CSNet" is not named in any source; mapped to CSF-R2 (assumption, to confirm).

## 7. Results so far (2026-11-05 23:45 UTC)

PySODMetrics on continuous maps (`score_unified.py`). The in-house S-measure agrees with PySODMetrics (+AF DUTS-TE 0.9284 vs 0.9292), so the old numbers were not miscomputed.

### VLF fix vs original, +MP (seed 42, 30 epochs)

| Sm | DUTS-TE | DUT-O | ECSSD | HKU-IS | PASCAL-S | SIP |
|---|---|---|---|---|---|---|
| +MP original | 0.9269 | 0.8806 | 0.9493 | 0.9397 | 0.8954 | 0.9379 |
| +MP VLF fix | 0.9252 | 0.8812 | 0.9496 | 0.9402 | 0.8957 | 0.9363 |

Differences are within ±0.002 and go both ways. That is expected: +MP uses the simple decoder, which ignores the correlation, so the fix can only change PPG point selection in ESC blocks 2–4. Seed runs (pending) will show whether these gaps are above noise. The full-model VLF-fix runs are the real test.

### Unified re-evaluation (9 of 20 checkpoints scored)

Sm on all six benchmarks, including PASCAL-S (850 images) for the first time:

| Config | DUTS-TE | DUT-O | ECSSD | HKU-IS | PASCAL-S | SIP |
|---|---|---|---|---|---|---|
| Baseline | 0.9270 | 0.8797 | 0.9475 | 0.9402 | 0.8965 | 0.9374 |
| +MP | 0.9269 | 0.8806 | 0.9493 | 0.9397 | 0.8954 | 0.9379 |
| +AF | **0.9292** | 0.8825 | 0.9498 | 0.9392 | **0.8987** | 0.9376 |
| +BiSAM | 0.9255 | 0.8768 | **0.9499** | **0.9403** | 0.8951 | 0.9369 |
| +Dec | 0.9262 | **0.8872** | 0.9492 | 0.9386 | 0.8975 | 0.9354 |
| Full + attention | 0.9243 | 0.8828 | 0.9473 | 0.9376 | 0.8933 | 0.9364 |
| Learnable | 0.9185 | 0.8759 | 0.9424 | 0.9327 | 0.8906 | 0.9334 |
| Gated | 0.9162 | 0.8774 | 0.9434 | 0.9298 | 0.8902 | 0.9253 |
| K=1 | 0.9058 | 0.8687 | 0.9326 | 0.9219 | 0.8810 | 0.9252 |

Max-F, mean-E and MAE tables: `unified_eval/<variant>/unified_metrics.csv`.

## 8. Final results (2026-11-06 17:55 UTC): all 10 runs and 20 re-evals done

S-measure, PySODMetrics. "orig" runs are the August seed-42 runs; new runs use seed 42 except the seed rows.

### Residual on vs off (matched seed and epochs)

| Config | DUTS-TE | DUT-O | ECSSD | HKU-IS | PASCAL-S | SIP |
|---|---|---|---|---|---|---|
| Full (max), residual ON | 0.9178 | 0.8739 | 0.9420 | 0.9328 | 0.8888 | 0.9305 |
| Full (max), residual OFF | 0.9209 | 0.8801 | 0.9471 | 0.9368 | 0.8916 | 0.9309 |
| Full + attn, residual ON | 0.9217 | 0.8767 | 0.9465 | 0.9359 | 0.8931 | 0.9367 |
| Full + attn, residual ON (rerun) | 0.9243 | 0.8828 | 0.9473 | 0.9376 | 0.8933 | 0.9364 |
| Full + attn, residual OFF | 0.9223 | 0.8776 | 0.9467 | 0.9362 | 0.8939 | 0.9349 |

Removing the residual does not hurt. With max aggregation it is better on all six datasets; with attention the off run lies between the two on runs.

### VLF fix vs original

| Config | DUTS-TE | DUT-O | ECSSD | HKU-IS | PASCAL-S | SIP |
|---|---|---|---|---|---|---|
| +MP orig / fix | 0.9269 / 0.9252 | 0.8806 / 0.8812 | 0.9493 / 0.9496 | 0.9397 / 0.9402 | 0.8954 / 0.8957 | 0.9379 / 0.9363 |
| +AF orig / fix | 0.9292 / 0.9278 | 0.8825 / 0.8837 | 0.9498 / 0.9482 | 0.9392 / 0.9406 | 0.8987 / 0.8973 | 0.9376 / 0.9351 |
| Full (max) orig / fix | 0.9178 / 0.9181 | 0.8739 / 0.8765 | 0.9420 / 0.9445 | 0.9328 / 0.9352 | 0.8888 / 0.8947 | 0.9305 / 0.9323 |
| Full + attn orig / fix | 0.9217 / 0.9248 | 0.8767 / 0.8826 | 0.9465 / 0.9481 | 0.9359 / 0.9369 | 0.8931 / 0.8940 | 0.9367 / 0.9331 |

Simple-decoder configs: mixed, within noise. Full model (max): better on all six (+0.0003 to +0.0059). Full + attention: better on five, worse on SIP (-0.0036). The fixed full models are still below the simple configs.

### Seed variance (seeds 42, 1, 2; mean ± std)

| Config | DUTS-TE | DUT-O | ECSSD | HKU-IS | PASCAL-S | SIP |
|---|---|---|---|---|---|---|
| Baseline | 0.9281 ± 0.0010 | 0.8803 ± 0.0009 | 0.9494 ± 0.0018 | 0.9406 ± 0.0006 | 0.8965 ± 0.0009 | 0.9357 ± 0.0015 |
| +AF | 0.9264 ± 0.0028 | 0.8811 ± 0.0013 | 0.9482 ± 0.0018 | 0.9396 ± 0.0005 | 0.8968 ± 0.0016 | 0.9340 ± 0.0031 |

Over three seeds, +AF does not beat the baseline: the single-seed gain reported in the paper (DUTS-TE 0.9261 → 0.9284) was within run-to-run noise.

## 9. Pending

- [ ] Finish the 10 training runs and score them
- [ ] Finish unified re-evaluation of the 20 existing checkpoints
- [ ] Residual on/off table, VLF-fix table, seed mean ± std
- [ ] Decide VLF-fix inclusion (user) and residual framing (user)
- [ ] `main_camera_ready.tex` with red changes + `REVIEW_RESPONSE_MAP.md`
