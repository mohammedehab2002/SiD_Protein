import pandas as pd
import argparse
import subprocess
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute stats for a given checkpoint")
    parser.add_argument('--ckpt_name', '-c', help='Name of the checkpoint to process', required=True)
    args = parser.parse_args()
    ckpt_name = args.ckpt_name

    num_designable = len([f for f in os.listdir(f"protein_out/{ckpt_name}/pdbs/designable")])
    num_undesignable = len([f for f in os.listdir(f"protein_out/{ckpt_name}/pdbs/undesignable")])

    os.makedirs("foldseek_tmp", exist_ok=True)

    subprocess.run(f"foldseek easy-cluster protein_out/{ckpt_name}/pdbs/designable foldseek_tmp/{ckpt_name}/res foldseek_tmp/{ckpt_name} --alignment-type 1 --cov-mode 0 --min-seq-id 0 --tmscore-threshold 0.5", shell=True)

    df = pd.read_csv(f"foldseek_tmp/{ckpt_name}/res_cluster.tsv", sep="\t", header=None, names=["cluster", "protein"])
    print("Designability:", num_designable / (num_designable + num_undesignable))
    print("Diversity:", len(df["cluster"].unique()) / len(df))