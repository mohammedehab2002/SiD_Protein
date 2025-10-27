# Copyright (c) 2025, Liyang Xie. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

import re
import os
import torch
import json
import shutil
import hydra
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import subprocess
import tempfile
import glob
import argparse

from training.proteina.evaluations.inverse_fold_models.proteinmpnn import ProteinMPNN
from training.proteina.evaluations.fold_models.esmfold import ESMFold
from training.proteina.evaluations.pipeline import Pipeline
from training.proteina.proteinfoundation.metrics.metric_factory import (
    GenerationMetricFactory,
    generation_metric_from_list,
    DatasetWrapper,
)

load_dotenv()

def run_designability_evaluation(eval_input_dir, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # inverse fold model
    inverse_fold_model = ProteinMPNN(device=device)

	# fold model
    fold_model = ESMFold(device=device)

	# pipeline
    pipeline = Pipeline(inverse_fold_model, fold_model)
    eval_output_dir = eval_input_dir + "/scores"
    if os.path.exists(eval_output_dir):
        shutil.rmtree(eval_output_dir)

    pipeline.evaluate(eval_input_dir, eval_output_dir, verbose=False)

def run_foldseek(eval_input_dir, base_db_path):
    info_file = os.path.join(eval_input_dir, "scores/info.csv")
    pdb_dir = os.path.join(eval_input_dir, "pdbs")

    if not os.path.exists(info_file) or not os.path.exists(pdb_dir):
        print(f"Skipping {eval_input_dir}: missing info.csv or pdbs/")
        return None

    # Load info.csv
    df = pd.read_csv(info_file)

    # Filter scRMSD < 2
    filtered = df[df["scRMSD"] < 2]
    domains = filtered["domain"].tolist()

    if len(domains) == 0:
        print(f"Skipping {eval_input_dir}: no domains with scRMSD < 2")
        return None

    # Database paths
    databases = {
        "pdb": os.path.join(base_db_path, "pdb", "pdb"),
        "afdb": os.path.join(base_db_path, "afdb", "afdb"),
    }

    tmpdir = tempfile.mkdtemp()
    try:
        # Prepare input dir for foldseek
        input_dir = os.path.join(tmpdir, "inputs")
        os.makedirs(input_dir, exist_ok=True)

        for domain in domains:
            pattern = os.path.join(pdb_dir, f"{domain}*.pdb")
            matches = glob.glob(pattern)

            if not matches:
                raise FileNotFoundError(f"No PDB file found for domain: {domain}")
            elif len(matches) > 1:
                raise FileExistsError(f"Multiple PDB files found for domain {domain}: {matches}")

            pdb_src = matches[0]
            pdb_dst = os.path.join(input_dir, f"{domain}.pdb")
            shutil.copy(pdb_src, pdb_dst)

        #
        # --- Part 1: easy-cluster ---
        #
        output_base = os.path.join(tmpdir, "res")
        cmd_cluster = [
            "foldseek", "easy-cluster",
            input_dir, output_base, tmpdir,
            "--alignment-type", "1",
            "--cov-mode", "0",
            "--min-seq-id", "0",
            "--tmscore-threshold", "0.5",
            "-v", "1"
        ]
        subprocess.run(cmd_cluster, check=True)

        cluster_file = output_base + "_cluster.tsv"
        if not os.path.exists(cluster_file):
            raise FileNotFoundError("Foldseek did not produce cluster file")

        cluster_df = pd.read_csv(cluster_file, sep="\t", header=None)
        num_clusters = cluster_df[0].nunique()
        ratio = num_clusters / len(domains)

        #
        # --- Part 2: easy-search for pdb + afdb ---
        #
        avg_scores = {}
        for db_name, db_path in databases.items():
            search_out = os.path.join(tmpdir, f"search_{db_name}.m8")
            cmd_search = [
                "foldseek", "easy-search",
                input_dir, db_path, search_out, tmpdir,
                "--alignment-type", "1",
                "--exhaustive-search",
                "--tmscore-threshold", "0.0",
                "--max-seqs", "10000000000",
                "--format-output", "query,target,alntmscore,lddt",
                "-v", "1"
            ]
            subprocess.run(cmd_search, check=True)

            search_df = pd.read_csv(search_out, sep="\t", header=None,
                                    names=["query", "target", "alntmscore", "lddt"])

            if search_df.empty:
                avg_scores[db_name] = 0.0
            else:
                max_scores = search_df.groupby("query")["alntmscore"].max()
                avg_scores[db_name] = max_scores.mean()

        #
        # --- Save results ---
        #
        step_num = extract_step(os.path.basename(eval_input_dir))
        results = {
            "folder": os.path.basename(eval_input_dir),
            "step": step_num,
            "num_domains": len(domains),
            "num_clusters": num_clusters,
            "cluster_ratio": ratio,
        }
        for db_name, avg_score in avg_scores.items():
            results[f"avg_max_tmscore_{db_name}"] = avg_score

        results_df = pd.DataFrame([results])
        out_file = os.path.join(eval_input_dir, "distilled_conditional_novelty_results.csv")
        results_df.to_csv(out_file, index=False)

        print(f"Processed {eval_input_dir} → saved {out_file}")
        return results

    finally:
        shutil.rmtree(tmpdir)

def run_fid_evaluation(eval_input_dir, device=None):
    config_path = 'training/proteina/configs/experiment_config/'
    config_name = 'inference_fid_ca'
    with hydra.initialize(config_path, version_base=hydra.__version__):
        cfg = hydra.compose(config_name=config_name)

    eval_pdb_dir = os.path.join(eval_input_dir, "pdbs")
    list_of_pdbs = [os.path.join(eval_pdb_dir, f) for f in os.listdir(eval_pdb_dir) if f.endswith('.pdb')]
    res = {}
    for cfg_mf in cfg.metric_factory:
        assert cfg_mf.ca_only == True, "Please turn on ca_only for CAFlow model"
        metric_factory = GenerationMetricFactory(**cfg_mf).cuda()
        metrics = generation_metric_from_list(list_of_pdbs, metric_factory)
        for k, v in metrics.items():
            res[k] = v.cpu().item()
    df = pd.DataFrame([res])
    df.to_csv(os.path.join(eval_input_dir, 'fid_results.csv'), index=False)
    return res

def gather_designability_results(eval_input_dir):
    eval_output_dir = os.path.join(eval_input_dir, "scores")
    output_info = pd.read_csv(os.path.join(eval_output_dir, 'info.csv'))
    scTM = np.mean(output_info.scTM)
    scRMSD = np.mean(output_info.scRMSD)
    pLDDT = np.mean(output_info.pLDDT)
    pAE = np.mean(output_info.pAE)
    pct_helix = np.mean(output_info.pct_helix)
    pct_strand = np.mean(output_info.pct_strand)
    designability_TM = np.sum(output_info.scTM > 0.5) / len(output_info.scTM)
    designability_RMSD = np.sum(output_info.scRMSD < 2.0) / len(output_info.scRMSD)
    pair_info = pd.read_csv(os.path.join(eval_output_dir, 'pair_info.csv'))
    designable_domains = output_info.loc[output_info['scRMSD'] < 2, 'domain']
    filtered_pairs = pair_info[pair_info['domain_1'].isin(designable_domains) & pair_info['domain_2'].isin(designable_domains)]
    average_tm = filtered_pairs['tm'].mean()
    
    res = dict(scTM=scTM, scRMSD=scRMSD, pLDDT=pLDDT, pAE=pAE, designability_TM=designability_TM, \
                designability_RMSD=designability_RMSD, diversity_TM=average_tm, pct_helix=pct_helix, pct_strand=pct_strand)
    df = pd.DataFrame([res])
    df.to_csv(os.path.join(eval_input_dir, 'designability_summary.csv'), index=False)
    return res

def run_all_evaluations(eval_input_dir, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Running designability evaluation...")
    run_designability_evaluation(eval_input_dir, device=device)
    print("Gathering designability results...")
    designability_results = gather_designability_results(eval_input_dir)
    print("Running foldseek evaluation...")
    foldseek_results = run_foldseek(eval_input_dir, os.path.join(os.getenv("DATA_PATH"), "foldseek_databases"))
    print("Running FID evaluation...")
    fid_results = run_fid_evaluation(eval_input_dir, device=device)
    print("All evaluations completed.")
    nstep = extract_step(os.path.basename(eval_input_dir))
    designability_results['nstep'] = nstep

    # Calculate effective sampling time
    time_results = pd.read_csv(os.path.join(eval_input_dir, "total_time.csv"))
    total_time = time_results['time'].values[0]
    nsample = time_results['nsample'].values[0]
    effective_time_per_sample = total_time / (nsample * designability_results['designability_RMSD']) \
        if designability_results['designability_RMSD'] > 0 else float('inf')
    designability_results['effective_time'] = effective_time_per_sample
    
    # Save combined results
    merged = combine_results(designability_results, foldseek_results, fid_results)
    merged.to_csv(os.path.join(eval_input_dir, "combined_results.csv"), index=False)
    print("Saved combined_results.csv")

    # Round all numeric values to 2 decimals
    rounded = merged.round(2)

    # Convert to LaTeX table
    latex_str = rounded.to_latex(index=False, escape=False)

    # Save to file
    with open(os.path.join(eval_input_dir, "results_table.tex"), "w") as f:
        f.write(latex_str)
    print("Saved results_table.tex")

    return merged

def combine_results(designability_results, foldseek_results, fid_results):
    merged = {**designability_results, **foldseek_results, **fid_results}
    merged = pd.DataFrame([merged])

    # Calculate average PDB_fjSD and AFDB_fjSD, fS, and secondary structure summary
    merged["PDB_fJSD"] = merged[["PDB_fJSD_C", "PDB_fJSD_A", "PDB_fJSD_T"]].mean(axis=1) * 10
    merged["AFDB_fJSD"] = merged[["AFDB_fJSD_C", "AFDB_fJSD_A", "AFDB_fJSD_T"]].mean(axis=1) * 10
    merged["fS"] = merged["IS_C"].map("{:.2f}".format).astype(str) + " / " + merged["IS_A"].map("{:.2f}".format).astype(str) + " / " + merged["IS_T"].map("{:.2f}".format).astype(str)
    merged["Sec. Struct."] = (merged["pct_helix"] * 100).map("{:.1f}".format).astype(str) + " / " + (merged["pct_strand"]*100).map("{:.1f}".format).astype(str)
    merged["diversity_cluster"] = merged["cluster_ratio"].map("{:.2f}".format).astype(str) + " (" + merged["num_clusters"].astype(int).astype(str) + ")"
    merged["designability_RMSD"] = merged["designability_RMSD"] * 100
    # Rearrange columns
    desired_order = [
        "nstep",
        "designability_RMSD",
        "diversity_cluster",
        "diversity_TM",
        "avg_max_tmscore_pdb",
        "avg_max_tmscore_afdb",
        "PDB_FID",
        "AFDB_FID",
        "fS",
        "PDB_fJSD",
        "AFDB_fJSD",
        "Sec. Struct.",
        "effective_time"
    ]
    # Add any remaining columns from merged that aren’t in desired_order
    # remaining = [col for col in merged.columns if col not in desired_order]
    # merged = merged[desired_order + remaining]
    merged = merged[desired_order]

    return merged

def extract_step(folder_name):
    """
    Extract the number of steps from folder name.
    Example: "proteina_uncond_distilled_1step_noise0.45_output" → 1
    """
    match = re.search(r"_(\d+)step_", folder_name)
    if match:
        return int(match.group(1))
    else:
        return None

def unpack_logfile(logfile):
    scTM_values = []
    scRMSD_values = []
    designability_values = []
    with open(logfile, "r") as f:
        for line in f:
            if len(scTM_values) >= 7:
                break
            # Case 1: dictionary style {'scTM': 0.97, ...}
            # match_dict = re.search(r"'scTM':\s*([\d\.]+)", line)
            # if match_dict:
            #     scTM_values.append(float(match_dict.group(1)))
            #     continue

            # Case 2: JSON style {"results": {"scTM": 0.97, ...}}
            match_json = re.search(r'"scTM":\s*([\d\.]+)', line)
            if match_json:
                scTM_values.append(float(match_json.group(1)))
                scRMSD_values.append(float(re.search(r'"scRMSD":\s*([\d\.]+)', line).group(1)))
                designability_values.append(float(re.search(r'"designability_RMSD":\s*([\d\.]+)', line).group(1)))
                continue

    return scTM_values, scRMSD_values, designability_values

# Define a function to extract scTM values from the output log file
def extract_sctm_values(filename):
    sctm_values = []
    timestamps = []  # List to store timestamp (or other x-axis values if needed)

    # Open the output file and read line by line
    with open(filename, 'r') as file:
        for line in file:
            # Try to find lines with dictionary-like format (e.g. {'scTM': 0.094})
            dict_match = re.search(r"'\s*scTM\s*':\s*([\d\.]+)", line)
            if dict_match:
                sctm_values.append(float(dict_match.group(1)))
                timestamps.append(len(sctm_values))  # Use the index as the x-axis
                continue

    return timestamps, sctm_values

def extract_max_tm(out_dir, foldseek_filename):
    # Read your CSV file
    df = pd.read_csv(os.path.join(out_dir, foldseek_filename), header=None, sep="\t", names=["query", "target", "alntmscore", "lddt"])

    # Group by 'query' and get the row with the maximum 'alntmscore' per group
    best_hits = df.loc[df.groupby('query')['alntmscore'].idxmax()]

    # Save to a new CSV
    best_hits.to_csv(os.path.join(out_dir, "novelty_" + foldseek_filename), index=False)

    # Print average
    print(best_hits['alntmscore'].mean())

def plot_scTM(filename):
    # Extract scTM values
    timestamps, sctm_values = extract_sctm_values(filename)

    # Generate a plot of scTM values
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, sctm_values, marker='o', linestyle='-', color='b', label='scTM')
    plt.xlabel('Evaluation Step (Tick)', fontsize=12)
    plt.ylabel('scTM Value', fontsize=12)
    plt.title('scTM Values Over Time', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the plot as an image file (optional)
    plt.savefig(filename + '_sctm_values_plot.png')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate protein generation outputs.")
    parser.add_argument('--eval_input_dir', type=str, required=True, help='Directory containing generated protein outputs.')
    args = parser.parse_args()
    res = run_all_evaluations(args.eval_input_dir)
