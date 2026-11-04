#!/usr/bin/env python3
"""Cumulative SIP ablation harness for SPC-Net/CLIPSAM.

Enhanced with --gpu, --clip-model, --variants selection, CUDA_VISIBLE_DEVICES
control, and markdown output alongside CSV/LaTeX.

Usage:
    python run_ablation.py --gpu 2 --epochs 60 --seed 42 --clip-model ViT-L-14
    python run_ablation.py --gpu 2 --variants baseline,+mp,+af
    python run_ablation.py --smoke  # verify output format
"""
import argparse
import csv
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path


# Editable cumulative ablation sequence. Values are extra flags relative to the
# full model defaults in clipsam.py.
ABLATION_VARIANTS = OrderedDict([
    ("baseline", [
        "--no-prompt-bank", "--no-adaptive-fallback", "--no-bidirectional-sam",
        "--simple-decoder", "--plain-bce-loss", "--pp-loss-weight", "0", "--ds-weight", "0",
    ]),
    ("+mp", [
        "--no-adaptive-fallback", "--no-bidirectional-sam", "--simple-decoder",
        "--plain-bce-loss", "--pp-loss-weight", "0", "--ds-weight", "0",
    ]),
    ("+af", [
        "--no-bidirectional-sam", "--simple-decoder", "--plain-bce-loss",
        "--pp-loss-weight", "0", "--ds-weight", "0",
    ]),
    ("+bisam", [
        "--simple-decoder", "--plain-bce-loss", "--pp-loss-weight", "0", "--ds-weight", "0",
    ]),
    ("+dec", ["--pp-loss-weight", "0", "--ds-weight", "0.3"]),
    ("+lpt", ["--pp-loss-weight", "0.3", "--ds-weight", "0.3"]),
])

METRIC_COLUMNS = ["s_measure", "f_measure", "e_measure", "weighted_f", "mae"]
TEX_COLUMNS = [("s_measure", "S_m", True), ("f_measure", "F_\\beta", True),
               ("e_measure", "E_m", True), ("mae", "MAE", False)]


def parse_args():
    p = argparse.ArgumentParser(description="Run cumulative SPC-Net SIP ablations")
    p.add_argument("--gpu", type=int, default=None, help="GPU device ID (sets CUDA_VISIBLE_DEVICES)")
    p.add_argument("--clip-model", default="ViT-L-14", help="CLIP model variant (default: ViT-L-14)")
    p.add_argument("--variants", type=str, default=None,
                   help="Comma-separated subset of variants to run (e.g. 'baseline,+mp,+af')")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="outputs_clipSAM")
    p.add_argument("--smoke", action="store_true", help="Write tiny deterministic example outputs without training")
    p.add_argument("clipsam_args", nargs=argparse.REMAINDER,
                   help="Additional arguments forwarded to clipsam.py after --")
    return p.parse_args()


def fmt_flags(flags):
    return " ".join(flags) if flags else "(full defaults)"


def read_metrics(run_dir):
    import yaml
    import json
    yaml_path = Path(run_dir) / "results.yaml"
    
    # Try reading from results.yaml
    if yaml_path.exists():
        with open(yaml_path) as f:
            doc = yaml.safe_load(f)
        # Priority: test > validation > benchmarks
        for section in ["test", "validation"]:
            if section in doc and doc[section] is not None:
                try:
                    return {key: float(doc[section][key]["mean"]) for key in METRIC_COLUMNS}
                except (KeyError, TypeError):
                    continue
    
    # Fallback: try eval_results.json from benchmark evaluation
    json_path = Path(run_dir) / "saliency_maps" / "eval_results.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        if "DUTS-TE" in data:
            m = data["DUTS-TE"]
            return {
                's_measure': m.get('Sm', 0),
                'f_measure': m.get('maxFm', 0),
                'e_measure': m.get('meanEm', 0),
                'weighted_f': m.get('wFm', 0),
                'mae': m.get('MAE', 0),
            }
    
    # Fallback: try results_summary.md or benchmark_sip_results.csv
    csv_path = Path(run_dir) / "sip_results.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                return {key: float(row[key]) for key in METRIC_COLUMNS if key in row}
    
    raise RuntimeError(f"No metrics found in {run_dir}")


def write_csv(rows, out_path):
    fields = ["variant", *METRIC_COLUMNS, "flags"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_results_md(rows, out_path):
    """Write ablation results as a markdown table."""
    lines = [
        "# Ablation Study Results\n",
        "| Variant | Sm ↑ | Fβ ↑ | Em ↑ | wFβ ↑ | MAE ↓ |",
        "|---------|------|------|------|-------|-------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {float(r['s_measure']):.4f} | {float(r['f_measure']):.4f} | "
            f"{float(r['e_measure']):.4f} | {float(r['weighted_f']):.4f} | {float(r['mae']):.4f} |"
        )
    lines.append("")
    Path(out_path).write_text("\n".join(lines))


def format_tex(rows):
    best = {}
    for key, _label, higher in TEX_COLUMNS:
        vals = [float(r[key]) for r in rows]
        best[key] = max(vals) if higher else min(vals)
    lines = []
    for r in rows:
        cells = [r["variant"]]
        for key, _label, _higher in TEX_COLUMNS:
            value = float(r[key])
            text = f"{value:.4f}"
            if abs(value - best[key]) <= 1e-12:
                text = f"\\textbf{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def print_summary(rows):
    widths = {"variant": max(7, *(len(r["variant"]) for r in rows))}
    for col in METRIC_COLUMNS:
        widths[col] = max(len(col), 8)
    header = "  ".join(["variant".ljust(widths["variant"]), *[c.rjust(widths[c]) for c in METRIC_COLUMNS]])
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print("  ".join([r["variant"].ljust(widths["variant"]), *[f"{float(r[c]):.4f}".rjust(widths[c]) for c in METRIC_COLUMNS]]))


def smoke_rows():
    rows = []
    for i, (name, flags) in enumerate(ABLATION_VARIANTS.items()):
        rows.append({
            "variant": name,
            "s_measure": 0.7000 + i * 0.01,
            "f_measure": 0.6500 + i * 0.01,
            "e_measure": 0.7600 + i * 0.01,
            "weighted_f": 0.6000 + i * 0.01,
            "mae": 0.1000 - i * 0.005,
            "flags": fmt_flags(flags),
        })
    return rows


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ablation_sip.csv"
    tex_path = out_dir / "ablation_sip.tex"
    md_path = out_dir / "ablation_results.md"

    if args.smoke:
        rows = smoke_rows()
    else:
        rows = []
        extra = args.clipsam_args[1:] if args.clipsam_args[:1] == ["--"] else args.clipsam_args

        # Determine which variants to run
        if args.variants:
            selected = [v.strip() for v in args.variants.split(",")]
        else:
            selected = list(ABLATION_VARIANTS.keys())

        for variant in selected:
            if variant not in ABLATION_VARIANTS:
                print(f"[WARN] Unknown variant: {variant}, skipping")
                continue
            flags = ABLATION_VARIANTS[variant]
            run_dir = out_dir / "ablation_sip" / variant.replace("+", "plus_")
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, "clipsam.py",
                "--clip-model", args.clip_model,
                "--epochs", str(args.epochs),
                "--seed", str(args.seed),
                "--batch-size", "8",
                "--num-workers", "8",
                "--mse-lambda", "0.1",
                "--output-dir", str(run_dir),
                "--eval-datasets", "DUTS-TE",
                *flags, *extra,
            ]

            # Set CUDA_VISIBLE_DEVICES if --gpu specified
            env = os.environ.copy()
            if args.gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

            print(f"\n[ABLATION] {variant}: {' '.join(cmd)}")
            proc = subprocess.run(cmd, env=env)
            if proc.returncode != 0:
                print(f"[ERROR] {variant} failed (exit {proc.returncode})")
                continue

            try:
                metrics = read_metrics(run_dir)
                rows.append({"variant": variant, **metrics, "flags": fmt_flags(flags)})
            except Exception as e:
                print(f"[ERROR] Could not read metrics for {variant}: {e}")

    write_csv(rows, csv_path)
    tex_path.write_text(format_tex(rows))
    write_results_md(rows, md_path)
    print_summary(rows)
    print(f"\n[INFO] Wrote {csv_path}")
    print(f"[INFO] Wrote {tex_path}")
    print(f"[INFO] Wrote {md_path}")
    print("\nLaTeX body:\n" + tex_path.read_text())


if __name__ == "__main__":
    main()
