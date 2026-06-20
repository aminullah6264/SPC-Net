#!/usr/bin/env python3
"""Cumulative SIP ablation harness for SPC-Net/CLIPSAM."""
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
    with open(Path(run_dir) / "results.yaml") as f:
        doc = yaml.safe_load(f)
    if "test" not in doc:
        raise RuntimeError(f"No test metrics found in {run_dir}/results.yaml")
    return {key: float(doc["test"][key]["mean"]) for key in METRIC_COLUMNS}


def write_csv(rows, out_path):
    fields = ["variant", *METRIC_COLUMNS, "flags"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


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

    if args.smoke:
        rows = smoke_rows()
    else:
        rows = []
        extra = args.clipsam_args[1:] if args.clipsam_args[:1] == ["--"] else args.clipsam_args
        for variant, flags in ABLATION_VARIANTS.items():
            run_dir = out_dir / "ablation_sip" / variant.replace("+", "plus_")
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "clipsam.py", "--epochs", str(args.epochs),
                   "--seed", str(args.seed), "--output-dir", str(run_dir), *flags, *extra]
            print(f"\n[ABLATION] {variant}: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            metrics = read_metrics(run_dir)
            rows.append({"variant": variant, **metrics, "flags": fmt_flags(flags)})

    write_csv(rows, csv_path)
    tex_path.write_text(format_tex(rows))
    print_summary(rows)
    print(f"\n[INFO] Wrote {csv_path}")
    print(f"[INFO] Wrote {tex_path}")
    print("\nLaTeX body:\n" + tex_path.read_text())


if __name__ == "__main__":
    main()
