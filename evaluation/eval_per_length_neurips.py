#!/usr/bin/env python3
"""
Compute per-length designability and diversity for the 8 selected NeurIPS-selfd runs
and write csvs/metrics.csv in the schema plot_per_length.py expects:
    run_name, ckpt_name, length, designability, diversity

run_name format must match RUN_RE in plot_per_length.py: Broteina_<nstep>_SC_<sc>.

Run from /homes/kasram/broteina/SiD_Protein/evaluation/ inside an env that has
foldseek + pandas (e.g. sid_protein_env).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# Resolve foldseek absolutely so subprocesses don't depend on PATH activation.
FOLDSEEK_BIN = (
    shutil.which("foldseek")
    or str(Path(sys.executable).resolve().parent / "foldseek")
)
if not Path(FOLDSEEK_BIN).is_file():
    raise FileNotFoundError(
        f"foldseek not found via PATH or {FOLDSEEK_BIN}. "
        "Install it or activate an env that ships foldseek."
    )

DESIGN_EVAL_BASE = Path(
    "/homes/kasram/broteina/SiD_Protein/design_eval_neurips/neurips_selfd"
)

# (run_name, relative-path-under-DESIGN_EVAL_BASE)
RUNS: List[Tuple[str, str]] = [
    ("Broteina_1_SC_1",    "1/network-snapshot-1.000000-005561.pkl"),
    ("Broteina_1_SC_0.7",  "1/network-snapshot-1.000000-005561.pkl_sc_0.7"),
    ("Broteina_5_SC_1",    "5/5step_network-snapshot-1.000000-000626.pkl"),
    ("Broteina_5_SC_0.8",  "5/5step_network-snapshot-1.000000-000626.pkl_sc_0.8"),
    ("Broteina_8_SC_1",    "8/network-snapshot-1.000000-000462.pkl"),
    ("Broteina_8_SC_0.8",  "8/network-snapshot-1.000000-000462.pkl_sc_0.8"),
    ("Broteina_16_SC_1",   "16/network-snapshot-1.000000-000069.pkl"),
    ("Broteina_16_SC_0.8", "16/network-snapshot-1.000000-000069.pkl_sc_0.8"),
]

LENGTHS = [0, 50, 100, 150, 200, 250]  # 0 = aggregate over all lengths
EVAL_TMP = Path("eval_tmp_neurips")
FOLDSEEK_TMP = Path("foldseek_tmp_neurips")


def list_with_prefix(directory: Path, prefix: str) -> List[Path]:
    if not directory.exists():
        return []
    if prefix == "":
        return [p for p in directory.iterdir() if p.is_file()]
    return [
        p for p in directory.iterdir()
        if p.is_file() and p.name.startswith(prefix)
    ]


def evaluate_one(run_name: str, rel_path: str, length: int) -> dict:
    ckpt_dir = DESIGN_EVAL_BASE / rel_path
    src_des = ckpt_dir / "pdbs" / "designable"
    src_und = ckpt_dir / "pdbs" / "undesignable"

    prefix = "" if length == 0 else f"{length}_"

    des_files = list_with_prefix(src_des, prefix)
    und_files = list_with_prefix(src_und, prefix)
    n_des, n_und = len(des_files), len(und_files)
    denom = n_des + n_und
    designability = (n_des / denom) if denom > 0 else 0.0

    if n_des == 0:
        diversity: float = 0.0
    elif n_des == 1:
        diversity = 1.0
    else:
        ckpt_tag = Path(rel_path).name
        # Stage symlinks (cheaper than copies) into a per-(run,length) directory.
        dst = EVAL_TMP / ckpt_tag / str(length)
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        for src in des_files:
            (dst / src.name).symlink_to(src)

        fs_workdir = FOLDSEEK_TMP / ckpt_tag / str(length)
        if fs_workdir.exists():
            shutil.rmtree(fs_workdir)
        fs_workdir.mkdir(parents=True, exist_ok=True)
        res_prefix = fs_workdir / "res"

        cmd = [
            FOLDSEEK_BIN, "easy-cluster",
            str(dst), str(res_prefix), str(fs_workdir),
            "--alignment-type", "1",
            "--cov-mode", "0",
            "--min-seq-id", "0",
            "--tmscore-threshold", "0.5",
        ]
        proc = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        cluster_tsv = fs_workdir / "res_cluster.tsv"
        if proc.returncode != 0 or not cluster_tsv.exists():
            diversity = float("nan")
        else:
            df = pd.read_csv(
                cluster_tsv, sep="\t", header=None, names=["cluster", "protein"]
            )
            diversity = (
                float(df["cluster"].nunique()) / float(len(df))
                if len(df) > 0 else float("nan")
            )

    return {
        "run_name": run_name,
        "ckpt_name": Path(rel_path).name,
        "length": int(length),
        "designability": designability,
        "diversity": diversity,
        "n_des": n_des,
        "n_und": n_und,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workers", type=int, default=8,
        help="Parallel (run, length) jobs (default: 8).",
    )
    ap.add_argument(
        "--out", default="csvs/metrics.csv",
        help="Output CSV (default: csvs/metrics.csv, the path plot_per_length.py reads).",
    )
    args = ap.parse_args()

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    EVAL_TMP.mkdir(parents=True, exist_ok=True)
    FOLDSEEK_TMP.mkdir(parents=True, exist_ok=True)

    jobs = [(rn, rp, L) for rn, rp in RUNS for L in LENGTHS]
    print(f"[eval] {len(jobs)} (run, length) jobs, workers={args.workers}")

    records: List[dict] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(evaluate_one, *j) for j in jobs]
            for f in as_completed(futs):
                rec = f.result()
                records.append(rec)
                print(
                    f"[eval] {rec['run_name']:24s} L={rec['length']:3d}  "
                    f"des={rec['designability']:.3f} div={rec['diversity']:.3f} "
                    f"(n_des={rec['n_des']}, n_und={rec['n_und']})"
                )
    else:
        for j in jobs:
            rec = evaluate_one(*j)
            records.append(rec)
            print(
                f"[eval] {rec['run_name']:24s} L={rec['length']:3d}  "
                f"des={rec['designability']:.3f} div={rec['diversity']:.3f} "
                f"(n_des={rec['n_des']}, n_und={rec['n_und']})"
            )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["run_name", "length"]).reset_index(drop=True)
    df_out = df[["run_name", "ckpt_name", "length", "designability", "diversity"]]
    df_out.to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv.resolve()} ({len(df_out)} rows)")


if __name__ == "__main__":
    main()
