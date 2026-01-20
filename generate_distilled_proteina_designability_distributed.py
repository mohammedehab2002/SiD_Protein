#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = "~/broteina/SiD_Protein/"
CONDA_ENV = "sid_protein_env"
SCRIPT = "generate_distilled_proteina_designability.py"

MODEL_PATH = (
    "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/"
    "proteina_multistep/00028-uncond-proteina-glr5e-05-lr0.0001-"
    "initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/"
    # "network-snapshot-1.000000-001052.pkl"
    # "network-snapshot-1.000000-000397.pkl"
    # "network-snapshot-1.000000-000954.pkl"
    "network-snapshot-1.000000-001265.pkl"

    # "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/"
    # "proteina_multistep/00031-uncond-proteina-glr5e-05-lr0.0001-"
    # "initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/"
    # "network-snapshot-1.000000-002006.pkl"

    # "/homes/kasram/broteina/SiD_Protein_scratch/protein_experiment/sid-train-runs/"
    # "proteina_multistep/00002-uncond-proteina-glr5e-05-lr0.0001-"
    # "initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep10/"
    # "network-snapshot-1.000000-000233.pkl"

    # "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/"
    # "proteina_multistep/00035-uncond-proteina-glr5e-05-lr0.0001-"
    # "initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/"
    # "network-snapshot-1.000000-002800.pkl"
)

OUT_DIR = "design_eval/"
NUM_BATCH = 4
NOISE_SCALE = 1


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def tmux_session_exists(name: str) -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _count_entries(dir_path: Path, length: int) -> int:
    """
    Count entries exactly like:
      ls -1A <dir> | wc -l
    Returns 0 if the directory doesn't exist.
    """
    # Expand ~ just in case and normalize
    p = Path(dir_path).expanduser()

    if not p.exists() or not p.is_dir():
        return 0

    if length == 0:
        cmd = f"ls -1A {str(p)!s} 2>/dev/null | wc -l"
    else:
        cmd = f'ls -1A "{p}" 2>/dev/null | grep -E "^{str(length)}_" | wc -l'
    out = subprocess.check_output(["bash", "-lc", cmd], text=True).strip()
    try:
        return int(out)
    except ValueError:
        return 0


def count(model_path: str = MODEL_PATH) -> None:
    mp = Path(model_path)
    model_name = mp.name
    noise_scale_identifier = f"_sc_{str(NOISE_SCALE)}" if NOISE_SCALE != 1.0 else ""
    base = Path(PROJECT_DIR) / OUT_DIR / (model_name + noise_scale_identifier) / "pdbs"

    for i in range(0, 6):
        length = i * 50

        designable = _count_entries(base / "designable", length)
        undesignable = _count_entries(base / "undesignable", length)
        denom = designable + undesignable
        designability = (designable / denom) if denom > 0 else 0.0

        if i != 0:
            print()
            print("\t", end="")
        print(f"Sampled a total of {denom} proteins.")
        if i != 0:
            print("\t", end="")
        print(f"Designability is: {designability}")


def main() -> int:
    # Quick sanity check that tmux is available
    try:
        subprocess.run(["tmux", "-V"], check=True, stdout=subprocess.DEVNULL)
    except Exception as e:
        print("Error: tmux not found or not working:", e, file=sys.stderr)
        return 1

    for i in range(1, 7):
        session = f"eval{i}"

        # If the session already exists, kill it so we can recreate cleanly
        if tmux_session_exists(session):
            run(["tmux", "kill-session", "-t", session])

        # Create a new detached tmux session
        run(["tmux", "new-session", "-d", "-s", session])

        # # In tmux, conda often requires sourcing conda.sh to make `conda activate` work.
        # # Adjust CONDA_SH if your conda install lives elsewhere.
        # conda_sh = "$HOME/miniconda3/etc/profile.d/conda.sh"

        cmd = (
            f"cd {PROJECT_DIR} && "
            # f"source {conda_sh} && "
            f"conda activate {CONDA_ENV} && "
            f"CUDA_VISIBLE_DEVICES={i} python {SCRIPT} "
            f"--model_path {MODEL_PATH} "
            f"--out_dir {OUT_DIR} "
            f"--num_batch {NUM_BATCH} "
            f"--noise_scale {NOISE_SCALE} "
            f"--seed {i}"
            f"; echo '[{session}] finished'; exec bash"
        )

        # Send the command to the tmux session and press Enter
        run(["tmux", "send-keys", "-t", session, cmd, "C-m"])

        print(f"Started tmux session: {session}")

    print("\nTo attach to one session, run: tmux attach -t eval0 (or eval1..eval6)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count",
        action="store_true",
        help="Compute and print designability from design_eval/<model_name>/pdbs/",
    )
    parser.add_argument(
        "--model_path",
        default=MODEL_PATH,
        help="Model path used to extract model_name for --count (default: MODEL_PATH in script).",
    )
    args = parser.parse_args()

    if args.count:
        count(args.model_path)
        raise SystemExit(0)
    else:
        raise SystemExit(main())
