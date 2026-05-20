import pandas as pd
import argparse
import subprocess
import shutil
import os
import biotite.structure.io.pdb as pdb
import biotite.structure.sse as annotate
import numpy as np
from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data
from tqdm import tqdm

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute stats for a given checkpoint")
    parser.add_argument('--ckpt_name', '-c', help='Name of the checkpoint to process', default='8-step-checkpoint')
    parser.add_argument(
        '--foldseek_pdb_db',
        help='Path to the Foldseek PDB database. If omitted, novelty search is skipped.',
        default=os.environ.get("FOLDSEEK_PDB_DB"),
    )
    args = parser.parse_args()
    ckpt_name = args.ckpt_name

    designable_list = os.listdir(f"design_eval/{ckpt_name}/pdbs/designable")
    num_designable = len(designable_list)
    num_undesignable = len(os.listdir(f"design_eval/{ckpt_name}/pdbs/undesignable"))
    print("Designability:", num_designable / (num_designable + num_undesignable))

    sec = np.zeros((3,))
    struc = {}

    for f in designable_list:
        sec += analyze_secondary_structure(f"design_eval/{ckpt_name}/pdbs/designable/{f}")
        struc[f] = get_structure(f"design_eval/{ckpt_name}/pdbs/designable/{f}")

    print("Secondary Structure Content:", sec / num_designable)

    # tm_sum = dict(zip(range(50,251,50),[0 for _ in range(5)]))
    # tm_count = dict(zip(range(50,251,50),[0 for _ in range(5)]))

    # for idx,f1 in enumerate(tqdm(designable_list)):
    #     nres = int(f1.split('_')[0])
    #     coords1, seq1 = get_residue_data(next(struc[f1].get_chains()))
    #     for f2 in designable_list[idx+1:]:
    #         if int(f2.split('_')[0]) == nres:
    #             coords2, seq2 = get_residue_data(next(struc[f2].get_chains()))
    #             result = tm_align(coords1, coords2, seq1, seq2)
    #             tm_sum[nres] += result.tm_norm_chain1
    #             tm_count[nres] += 1

    # tm_avg = []
    # for nres in range(50,251,50):
    #     tm_avg.append(tm_sum[nres] / tm_count[nres])
    # print("TMScore by Length:", tm_avg)
    # print("Average TMScore:", sum(tm_avg) / len(tm_avg))

    os.makedirs("foldseek_tmp", exist_ok=True)
    # if os.path.exists(f"foldseek_tmp/{ckpt_name}"):
    #     shutil.rmtree(f"foldseek_tmp/{ckpt_name}")
    # subprocess.run(f"foldseek easy-cluster design_eval/{ckpt_name}/pdbs/designable foldseek_tmp/{ckpt_name}/res foldseek_tmp/{ckpt_name} --alignment-type 1 --cov-mode 0 --min-seq-id 0 --tmscore-threshold 0.5", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # df = pd.read_csv(f"foldseek_tmp/{ckpt_name}/res_cluster.tsv", sep="\t", header=None, names=["cluster", "protein"])
    # print("Diversity:", len(df["cluster"].unique()) / len(df))

    if args.foldseek_pdb_db:
        subprocess.run(
            f"foldseek easy-search design_eval/{ckpt_name}/pdbs/designable "
            f"{args.foldseek_pdb_db} foldseek_tmp/{ckpt_name}/novelty_pdb foldseek_tmp/{ckpt_name} "
            f"--alignment-type 1 --exhaustive-search --tmscore-threshold 0.0 "
            f"--max-seqs 10000000000 --format-output query,target,alntmscore,lddt",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    else:
        print("Skipping Foldseek novelty search because --foldseek_pdb_db was not provided.")
    # subprocess.run(f"foldseek easy-search design_eval/{ckpt_name}/pdbs/designable ./additional_files/foldseek_databases/afdb foldseek_tmp/{ckpt_name}/novelty_afdb foldseek_tmp/{ckpt_name}  --alignment-type 1 --exhaustive-search --tmscore-threshold 0.0 --max-seqs 10000000000 --format-output query,target,alntmscore,lddt", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
