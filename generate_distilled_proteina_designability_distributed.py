#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import pandas as pd
from pathlib import Path

PROJECT_DIR = "~/broteina/SiD_Protein/"
CONDA_ENV = "sid_protein_env"
SCRIPT = "generate_distilled_proteina_designability.py"
CONDA_SH = str(
    (
        Path(
            os.environ.get(
                "CONDA_EXE",
                str(Path.home() / "miniconda3" / "bin" / "conda"),
            )
        ).resolve().parent.parent
    )
    / "etc"
    / "profile.d"
    / "conda.sh"
)

MODEL_PATH = (
    # "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/"
    # "proteina_multistep/00028-uncond-proteina-glr5e-05-lr0.0001-"
    # "initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/"
    # # "network-snapshot-1.000000-001052.pkl"
    # # "network-snapshot-1.000000-000397.pkl"
    # # "network-snapshot-1.000000-000954.pkl"
    # "network-snapshot-1.000000-001265.pkl"

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

    # "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/"
    # "proteina_multistep/00037-uncond-proteina-glr5e-05-lr0.0001-"
    # "initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep2/"
    # "network-snapshot-1.000000-000987.pkl"

    # "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/"
    # "proteina_multistep/00038-uncond-proteina-glr5e-05-lr0.0001-"
    # "initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep2/"
    # # "network-snapshot-1.000000-002219.pkl"
    # "network-snapshot-1.000000-002268.pkl"

    # "/homes/kasram/broteina/SiD_Protein_scratch/protein_experiment/sid-train-runs/"
    # "proteina_multistep/00008-uncond-proteina-glr5e-05-lr0.0001-"
    # "initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/"
    # "network-snapshot-1.000000-002350.pkl"

    # "/homes/kasram/broteina/SiD_Protein_scratch/sid_checkpoint/network-snapshot-1.000000-000823.pkl"
    # "/homes/kasram/broteina/SiD_Protein_scratch/protein_experiment/sid-train-runs/proteina_multistep/00008-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-000827.pkl"

    # "Best one-step yet"
    # "/homes/kasram/broteina/SiD_Protein_scratch/protein_experiment/sid-train-runs/proteina_multistep/00008-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-000958.pkl"

    # "Best one-step yet, 34%"
    # "/homes/kasram/broteina/SiD_Protein_scratch_2/protein_experiment/sid-train-runs/proteina_multistep/00006-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-002862.pkl"

    # Designability: 0.278
    # Diversity: 0.35251798561151076
    # "/homes/kasram/broteina/SiD_Protein_scratch_2/protein_experiment/sid-train-runs/proteina_multistep/00006-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-001092.pkl"
    # "/homes/kasram/broteina/SiD_Protein_scratch_2/protein_experiment/sid-train-runs/proteina_multistep/00006-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-002010.pkl"

    # "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/proteina_multistep/00041-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-001134.pkl"
    # "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/proteina_multistep/00041-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-001232.pkl"

    # 5 step generator
    # "/homes/kasram/broteina/SiD_Protein_4/protein_experiment/sid-train-runs/proteina_multistep/00001-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep5/network-snapshot-1.000000-000315.pkl"
    # "/homes/kasram/broteina/SiD_Protein_4/protein_experiment/sid-train-runs/proteina_multistep/00004-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep5/network-snapshot-1.000000-000368.pkl"
    # "/homes/kasram/broteina/SiD_Protein_4/protein_experiment/sid-train-runs/proteina_multistep/00004-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep5/network-snapshot-1.000000-000434.pkl"

    # 8 step generator
    # "/homes/kasram/broteina/SiD_Protein_3/protein_experiment/sid-train-runs/proteina_multistep/00005-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-000135.pkl"
    # "/homes/kasram/broteina/SiD_Protein_3/protein_experiment/sid-train-runs/proteina_multistep/00006-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-000188.pkl"
    # "/homes/kasram/broteina/SiD_Protein_3/protein_experiment/sid-train-runs/proteina_multistep/00006-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-000335.pkl"

    # 2 step generator
    # "/homes/kasram/broteina/SiD_Protein_4/protein_experiment/sid-train-runs/proteina_multistep/00006-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep2/network-snapshot-1.000000-000905.pkl"
    # "/homes/kasram/broteina/SiD_Protein_4/protein_experiment/sid-train-runs/proteina_multistep/00006-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep2/network-snapshot-1.000000-001429.pkl"
    # "/homes/kasram/broteina/SiD_Protein_4/protein_experiment/sid-train-runs/proteina_multistep/00007-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep2/network-snapshot-1.000000-001564.pkl"
    # "/homes/kasram/broteina/SiD_Protein_4/protein_experiment/sid-train-runs/proteina_multistep/00007-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep2/network-snapshot-1.000000-001892.pkl"

    # 1 step generator: 50-150
    # "/homes/kasram/broteina/SiD_Protein_3/protein_experiment/sid-train-runs/proteina_multistep/00007-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-001413.pkl"
    # "/homes/kasram/broteina/SiD_Protein_3/protein_experiment/sid-train-runs/proteina_multistep/00007-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-001789.pkl"

    # 1 step generator: 200-250
    # "/homes/kasram/broteina/SiD_Protein_scratch_2/protein_experiment/sid-train-runs/proteina_multistep/00005-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-002134.pkl"

    # New 1-step generator from scratch
    # "/homes/kasram/broteina/SiD_Protein_4/protein_experiment/sid-train-runs/proteina_multistep/00011-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-000856.pkl"
    # "/homes/kasram/broteina/SiD_Protein_4/protein_experiment/sid-train-runs/proteina_multistep/00011-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-001511.pkl"
    # "/homes/kasram/broteina/SiD_Protein_4/protein_experiment/sid-train-runs/proteina_multistep/00011-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-001740.pkl"

    # + SFT
    # "/homes/kasram/broteina/SiD_Protein_3/protein_experiment/sid-train-runs/proteina_multistep/00011-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-002416.pkl"
    # "/homes/kasram/broteina/SiD_Protein_3/protein_experiment/sid-train-runs/proteina_multistep/00011-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-002875.pkl"

    # 1-step generator linear 50-250, 1-stage ablation
    # "/homes/kasram/broteina/SiD_Protein_3/protein_experiment/sid-train-runs/proteina_multistep/00010-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-001396.pkl"
    # "/homes/kasram/broteina/SiD_Protein_3/protein_experiment/sid-train-runs/proteina_multistep/00010-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-001724.pkl"
)

runs = {
    "filtered": {
        # Single stage:
        # 8: "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/proteina_multistep/00042-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-000053.pkl"
        # 8: "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/proteina_multistep/00042-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-000724.pkl"
        # 8: "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/proteina_multistep/00042-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-000774.pkl"
        # 8: "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/proteina_multistep/00042-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-001069.pkl"
        8: "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/proteina_multistep/00042-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-001642.pkl"
        # -> pretty good 10% beta

        # Second stage:
        # 8: "/homes/kasram/broteina/SiD_Protein_scratch_2/protein_experiment/sid-train-runs/proteina_multistep/00012-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-001007.pkl"
        # 8: "/homes/kasram/broteina/SiD_Protein_scratch_2/protein_experiment/sid-train-runs/proteina_multistep/00012-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-001056.pkl"
        # 8: "/homes/kasram/broteina/SiD_Protein_scratch_2/protein_experiment/sid-train-runs/proteina_multistep/00012-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-001269.pkl"
    },
    "uniform": {
        # Single stage:
        # 8: "/homes/kasram/broteina/SiD_Protein_3/protein_experiment/sid-train-runs/proteina_multistep/00014-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-000053.pkl"
        # 8: "/homes/kasram/broteina/SiD_Protein_3/protein_experiment/sid-train-runs/proteina_multistep/00014-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-000364.pkl"
        8: "/homes/kasram/broteina/SiD_Protein_3/protein_experiment/sid-train-runs/proteina_multistep/00014-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-000610.pkl",

        # Second stage:
        # 8: "/homes/kasram/broteina/SiD_Protein_3/protein_experiment/sid-train-runs/proteina_multistep/00015-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-000728.pkl",
        # 8: "/homes/kasram/broteina/SiD_Protein_3/protein_experiment/sid-train-runs/proteina_multistep/00015-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-000842.pkl",
        # 8: "/homes/kasram/broteina/SiD_Protein_3/protein_experiment/sid-train-runs/proteina_multistep/00015-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-001023.pkl",
        # 8: "/homes/kasram/broteina/SiD_Protein_3/protein_experiment/sid-train-runs/proteina_multistep/00015-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-001301.pkl",
        # 8: "/homes/kasram/broteina/SiD_Protein_3/protein_experiment/sid-train-runs/proteina_multistep/00015-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-002956.pkl",

        # 1: "/homes/kasram/broteina/SiD_Protein_4/protein_experiment/sid-train-runs/proteina_multistep/00012-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-000987.pkl"
        # 1: "/homes/kasram/broteina/SiD_Protein_4/protein_experiment/sid-train-runs/proteina_multistep/00012-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-001167.pkl"
        # 1: "/homes/kasram/broteina/SiD_Protein_4/protein_experiment/sid-train-runs/proteina_multistep/00012-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-001380.pkl"
        # 1: "/homes/kasram/broteina/SiD_Protein_4/protein_experiment/sid-train-runs/proteina_multistep/00012-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-002265.pkl"
        # 1: "/homes/kasram/broteina/SiD_Protein_4/protein_experiment/sid-train-runs/proteina_multistep/00012-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-003772.pkl"
        1: "/homes/kasram/broteina/SiD_Protein_4/protein_experiment/sid-train-runs/proteina_multistep/00012-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-005214.pkl"
    },
    "neurips_selfd": {
        # 16: "/homes/kasram/broteina/SiD_Protein_3/protein_experiment/sid-train-runs/proteina_multistep/00018-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep16/network-snapshot-1.000000-000004.pkl",
        # 16: "/homes/kasram/broteina/SiD_Protein_3/protein_experiment/sid-train-runs/proteina_multistep/00018-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep16/network-snapshot-1.000000-000102.pkl",
        16: "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/proteina_multistep/00047-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep16/network-snapshot-1.000000-000069.pkl",
        # 16: "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/proteina_multistep/00047-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep16/network-snapshot-1.000000-000184.pkl",
        # 16: "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/proteina_multistep/00047-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep16/network-snapshot-1.000000-000233.pkl",
        # 16: "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/proteina_multistep/00047-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep16/network-snapshot-1.000000-000249.pkl",

        8: "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/proteina_multistep/00044-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-000462.pkl",
        # 8: "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/proteina_multistep/00045-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-000925.pkl"
        # 8: "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/proteina_multistep/00045-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-001269.pkl"
        # 8: "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/proteina_multistep/00045-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-001367.pkl"

        5: "/homes/kasram/broteina/SiD_Protein/pretrained_checkpoints/5step_network-snapshot-1.000000-000626.pkl",

        # 1: "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/proteina_multistep/00046-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-002953.pkl",
        # 1: "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/proteina_multistep/00046-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-003952.pkl",
        1: "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/proteina_multistep/00050-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-005561.pkl",
        # 1: "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/proteina_multistep/00050-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-005152.pkl",
        # 1: "/homes/kasram/broteina/SiD_Protein/protein_experiment/sid-train-runs/proteina_multistep/00050-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep1/network-snapshot-1.000000-006348.pkl",
    }
}

STD_LENGHS = "50,100,150,200,250"
LONG_LENGTHS_DICT = {
    0: "300,400,500,600",
    1: "500",
    2: "400"
}
LONG_TYPE = 2
LONG_LENGTHS = LONG_LENGTHS_DICT[LONG_TYPE]
EVAL_LONG = False

RUN_TYPE = "neurips_selfd"
NSTEP = 16
MODEL_PATH = runs[RUN_TYPE][NSTEP]

LENGTHS = LONG_LENGTHS if EVAL_LONG else STD_LENGHS
BATCH_SIZE = 5

OUT_DIR = f"design_eval_neurips/{RUN_TYPE}{'_LONG' if EVAL_LONG else ''}/{NSTEP}"
NUM_BATCH = 5
NOISE_SCALE = 0.4
# GPUS = range(4, 8)
GPUS = range(0, 4)


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

    count = 0
    
    try:
        with os.scandir(p) as entries:
            if length == 0:
                for _ in entries:
                    count += 1
            else:
                prefix = f"{length}_"
                for entry in entries:
                    if entry.name.startswith(prefix):
                        count += 1
    except OSError:
        return 0
    return count


def count(model_path: str = MODEL_PATH) -> None:
    mp = Path(model_path)
    model_name = mp.name
    noise_scale_identifier = f"_sc_{str(NOISE_SCALE)}" if NOISE_SCALE != 1.0 else ""
    base = Path(PROJECT_DIR) / OUT_DIR / (model_name + noise_scale_identifier) / "pdbs"
    csv_path = base.parent / "scRMSD_scores.csv"
    df = pd.read_csv(csv_path)

    lengths = [0] + eval(f"[{LENGTHS}]")
    for length in lengths:
        designable = _count_entries(base / "designable", length)
        undesignable = _count_entries(base / "undesignable", length)
        denom = designable + undesignable
        designability = (designable / denom) if denom > 0 else 0.0

        if length == 0:
            avg_scrmsd = df['scRMSD'].mean()
        else:
            avg_scrmsd = df.loc[df['n'] == length, 'scRMSD'].mean()

        scrmsd_str = f"{avg_scrmsd:.3f}" if pd.notna(avg_scrmsd) else "N/A"

        if length != 0:
            print()
            print("\t", end="")
        
        label = "Total" if length == 0 else f"Length {length}"
        print(f"[{label}] Sampled: {denom} proteins.")
        
        if length != 0:
            print("\t", end="")
        print(f"Designability: {designability:.3f}")
        
        if length != 0:
            print("\t", end="")
        print(f"Avg scRMSD:    {scrmsd_str}")


def main() -> int:
    # Quick sanity check that tmux is available
    try:
        subprocess.run(["tmux", "-V"], check=True, stdout=subprocess.DEVNULL)
    except Exception as e:
        print("Error: tmux not found or not working:", e, file=sys.stderr)
        return 1

    for i in GPUS:
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
            f"source {CONDA_SH} && "
            f"conda deactivate >/dev/null 2>&1 || true && "
            f"conda activate {CONDA_ENV} && "
            f"export PYTHONPATH=$(pwd)/training/proteina:$(pwd):${{PYTHONPATH:-}} && "
            f"CUDA_VISIBLE_DEVICES={i} python {SCRIPT} "
            f"--model_path {MODEL_PATH} "
            f"--lengths {LENGTHS} "
            f"--batch_size {BATCH_SIZE} "
            f"--out_dir {OUT_DIR} "
            f"--num_batch {NUM_BATCH} "
            f"--noise_scale {NOISE_SCALE} "
            f"--nstep {NSTEP} "
            f"--seed {i}"
            f"; echo '[{session}] finished'; exec bash"
        )

        # Send the command to the tmux session and press Enter
        run(["tmux", "send-keys", "-t", session, cmd, "C-m"])

        print(f"Started tmux session: {session}")

    print("\nTo attach to one session, run: tmux attach -t eval{i}")
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
