from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import argparse
import biotite.structure.io.pdb as pdb
import biotite.structure.sse as annotate
import numpy as np
from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data
from tqdm import tqdm


def _list_files_with_prefix(directory: Path, prefix: str) -> List[Path]:
    if not directory.exists():
        return []
    return [p for p in directory.iterdir() if p.is_file() and (prefix == "0_" or p.name.startswith(prefix))]


def analyze_secondary_structure(pdb_path):
    # 1. Load the structure
    pdb_file = pdb.PDBFile.read(pdb_path)
    array = pdb_file.get_structure(model=1)
    
    # 2. Filter for CA atoms (P-SEA uses Carbon-alpha positions)
    ca_atoms = array[array.atom_name == "CA"]
    
    # 3. Run the P-SEA algorithm
    # Returns an array of characters: 'a' (alpha), 'b' (beta), 'c' (coil)
    sse = annotate.annotate_sse(ca_atoms)
    
    # 4. Count occurrences
    total = len(sse)
    alpha_count = np.count_nonzero(sse == 'a')
    beta_count = np.count_nonzero(sse == 'b')
    coil_count = np.count_nonzero(sse == 'c')
    
    return np.array([alpha_count, beta_count, coil_count]) / total


def evaluate_metrics(ckpt_name: str, length: int) -> List[float]:
    """
    Returns [designability, diversity] for a given checkpoint and length.

    Side effects:
      - Creates/overwrites eval_tmp/{ckpt_name}/{length}/ with filtered designable files
      - Creates/overwrites foldseek_tmp/{ckpt_name}/{length}/ for clustering output
    """
    length = int(length)
    prefix = f"{length}_"

    # Ensure eval_tmp exists
    eval_root = Path("eval_tmp")
    eval_root.mkdir(exist_ok=True)

    # Source directories
    design_eval_root = Path("design_eval") / ckpt_name / "pdbs"
    src_designable = design_eval_root / "designable"
    src_undesignable = design_eval_root / "undesignable"

    # Filtered counts for designability (prefix match in BOTH designable and undesignable)
    designable_files = _list_files_with_prefix(src_designable, prefix)
    undesignable_files = _list_files_with_prefix(src_undesignable, prefix)

    n_designable = len(designable_files)
    n_undesignable = len(undesignable_files)
    denom = n_designable + n_undesignable
    designability = (n_designable / denom) if denom > 0 else 0

    # Prepare destination directory: eval_tmp/{ckpt_name}/{length}/
    dst_dir = eval_root / ckpt_name / str(length)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy filtered designable files
    for src in designable_files:
        shutil.copy2(src, dst_dir / src.name)

    # Diversity: foldseek easy-cluster on filtered designable directory
    # If there are 0 or 1 proteins, diversity is trivial.
    if n_designable == 0:
        diversity = 0
        return [designability, diversity]
    if n_designable == 1:
        diversity = 1.0
        return [designability, diversity]

    foldseek_root = Path("foldseek_tmp")
    foldseek_root.mkdir(parents=True, exist_ok=True)

    fs_workdir = foldseek_root / ckpt_name / str(length)
    if fs_workdir.exists():
        shutil.rmtree(fs_workdir)
    fs_workdir.mkdir(parents=True, exist_ok=True)

    # foldseek outputs:
    #   - result prefix: fs_workdir / "res"
    #   - temp/work dir: fs_workdir
    res_prefix = fs_workdir / "res"

    cmd = (
        f"foldseek easy-cluster "
        f"{dst_dir} "
        f"{res_prefix} "
        f"{fs_workdir} "
        f"--alignment-type 1 --cov-mode 0 --min-seq-id 0 --tmscore-threshold 0.5"
    )

    try:
        subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError:
        # If foldseek fails, return NaN for diversity but still return designability
        diversity = float("nan")
        return [designability, diversity]

    cluster_tsv = fs_workdir / "res_cluster.tsv"
    if not cluster_tsv.exists():
        diversity = float("nan")
        return [designability, diversity]

    df = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["cluster", "protein"])
    if len(df) == 0:
        diversity = float("nan")
    else:
        diversity = float(df["cluster"].nunique()) / float(len(df))

    return [designability, diversity]


def evaluate_structure(ckpt_name):
    # Source directories
    design_eval_root = Path("design_eval") / ckpt_name / "pdbs"
    src_designable = design_eval_root / "designable"

    designable_files = _list_files_with_prefix(src_designable, "")
    sec = np.zeros((3,))

    for f in designable_files:
        sec += analyze_secondary_structure(f)
    return list(sec / len(designable_files))


def evaluate_novelty(ckpt_name, dataset):
    path = Path(f"foldseek_tmp/{ckpt_name}/novelty_{dataset}")
    if not os.path.exists(path):
        subprocess.run(f"foldseek easy-search design_eval/{ckpt_name}/pdbs/designable ./additional_files/foldseek_databases/{dataset} foldseek_tmp/{ckpt_name}/novelty_{dataset} foldseek_tmp/{ckpt_name}  --alignment-type 1 --exhaustive-search --tmscore-threshold 0.0 --max-seqs 10000000000 --format-output query,target,alntmscore,lddt", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    df = pd.read_csv(f"foldseek_tmp/{ckpt_name}/novelty_{dataset}", header=None, names=["protein","target","TM","lddt"], sep="\t")
    
    tot_tm = 0
    for protein in tqdm(df["protein"].unique()):
        tot_tm += df[df["protein"] == protein]["TM"].max()
    return tot_tm / len(df["protein"].unique())


def evaluate_tm_diversity(ckpt_name):
    tm_sum = dict(zip(range(50,251,50),[0 for _ in range(5)]))
    tm_count = dict(zip(range(50,251,50),[0 for _ in range(5)]))

    def nresidue(file):
        with open(file, "rb") as f:
            num_lines = sum(1 for _ in f)
            return num_lines - 4
        
    base_dir = Path(f"design_eval/{ckpt_name}/pdbs/designable")
    designable_list = os.listdir(base_dir)
    struc = {}
    nresidues = {}
    for f in designable_list:
        struc[f] = get_structure(base_dir / f)
        nresidues[f] = nresidue(base_dir / f)


    for idx,f1 in enumerate(tqdm(designable_list)):
        # nres = int(f1.split('_')[0])
        nres = nresidues[f1]
        coords1, seq1 = get_residue_data(next(struc[f1].get_chains()))
        for f2 in designable_list[idx+1:]:
            # if int(f2.split('_')[0]) == nres:
            if nresidues[f2] == nres:
                coords2, seq2 = get_residue_data(next(struc[f2].get_chains()))
                result = tm_align(coords1, coords2, seq1, seq2)
                tm_sum[nres] += result.tm_norm_chain1
                tm_count[nres] += 1

    tm_avg = []
    for nres in range(50,251,50):
        if tm_count[nres]:
            tm_avg.append(tm_sum[nres] / tm_count[nres])
    if len(tm_avg) == 0:
        return 2.0
    return sum(tm_avg) / len(tm_avg)


def main() -> None:
    ckpt_names = [
        # ("Broteina_1_SC_1", "network-snapshot-1.000000-002875.pkl"),
        # ("Broteina_1_SC_0.75", "network-snapshot-1.000000-002875.pkl_sc_0.75"),

        # ("Broteina_2_SC_1", "network-snapshot-1.000000-001892.pkl"),
        # ("Broteina_2_SC_0.45", "network-snapshot-1.000000-001892.pkl_sc_0.45"),

        # ("Broteina_5_SC_1", "network-snapshot-1.000000-000434.pkl"),
        # ("Broteina_5_SC_0.45", "network-snapshot-1.000000-000434.pkl_sc_0.45"),

        # ("Broteina_8_SC_1", "network-snapshot-1.000000-000335.pkl"),
        # ("Broteina_8_SC_0.35", "network-snapshot-1.000000-000188.pkl_sc_0.35"),

        ("M_SYN_SC_1", "finetuned_designables_epoch=00000000_step=000000002000"),
        ("M_FS_SC_1", "proteina_v1.2_DFS_200M_notri_sc_1"),
    ]

    # ckpt_names = [
    #     ("B__SC_1", "network-snapshot-1.000000-002875.pkl"),
    #     ("B__SC_0.75", "network-snapshot-1.000000-002875.pkl_sc_0.75"),

    #     ("B_UNI_SC_1", "network-snapshot-1.000000-001740.pkl"),
    #     ("B_UNI_SC_0.75", "network-snapshot-1.000000-001740.pkl_sc_0.75"),

    #     ("B_LIN_SC_1", "network-snapshot-1.000000-001724.pkl"),
    #     ("B_LIN_SC_0.75", "network-snapshot-1.000000-001724.pkl_sc_0.75"),
    # ]

    # ckpt_names = [
    #     ("B_50-150_SC_1", "network-snapshot-1.000000-001789.pkl"),
    #     ("B_50-150_SC_0.75", "network-snapshot-1.000000-001789.pkl_sc_0.75"),

    #     ("B_200-250_SC_1", "network-snapshot-1.000000-002134.pkl"),
    #     ("B_200-250_SC_0.75", "network-snapshot-1.000000-002134.pkl_sc_0.75"),
    # ]

    lengths = [0]#, 50, 100, 150, 200, 250]
    # out_csv = "evaluation/csvs/metrics_2.csv"
    # out_csv = "evaluation/csvs/structure.csv"
    out_csv = "evaluation/csvs/misc_metrics_2.csv"
    # out_csv = "evaluation/csvs/metrics_ablation_stage.csv"
    # out_csv = "evaluation/csvs/metrics_ablation_narrow.csv"

    records: List[Dict[str, Any]] = []
    for run_name, ckpt in ckpt_names:
        for L in lengths:
            # designability, diversity = evaluate_metrics(ckpt, L)

            a,b,c = evaluate_structure(ckpt)

            foldseek_root = Path(f"foldseek_tmp/{ckpt}")
            foldseek_root.mkdir(parents=True, exist_ok=True)

            novelty_pdb = evaluate_novelty(ckpt, "pdb")
            novelty_afdb = evaluate_novelty(ckpt, "afdb")

            tm_diversity = evaluate_tm_diversity(ckpt)

            records.append(
                {
                    "run_name": run_name,
                    "ckpt_name": ckpt,
                    "length": int(L),
                    # "designability": designability,
                    # "diversity": diversity,
                    "alpha": a,
                    "beta": b,
                    "coil": c,
                    "novelty_pdb": novelty_pdb,
                    "novelty_afdb": novelty_afdb,
                    "tm_diversity": tm_diversity,
                }
            )


    df = pd.DataFrame.from_records(records).sort_values(["ckpt_name", "length"]).reset_index(drop=True)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Wrote results to: {out_path.resolve()}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
