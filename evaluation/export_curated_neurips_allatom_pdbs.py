#!/usr/bin/env python3
import csv
import random
from pathlib import Path

import biotite.structure.io.pdb as biotite_pdb
import biotite.structure.sse as annotate
import numpy as np
import torch
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
from transformers.models.esm.openfold_utils.protein import to_pdb


OUTPUT_DIR = Path(
    "/homes/kasram/broteina/SiD_Protein/design_eval_neurips/curated_allatom_pdbs"
)
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"
LENGTHS = [50, 100, 150, 200, 250]
RNG = random.Random(7)

MODEL_ROOTS = {
    "B1": Path(
        "/homes/kasram/broteina/SiD_Protein/design_eval_neurips/neurips_selfd/1/"
        "network-snapshot-1.000000-005561.pkl_sc_0.7"
    ),
    "B5": Path(
        "/homes/kasram/broteina/SiD_Protein/design_eval_neurips/neurips_selfd/5/"
        "5step_network-snapshot-1.000000-000626.pkl_sc_0.8"
    ),
    "B8": Path(
        "/homes/kasram/broteina/SiD_Protein/design_eval_neurips/neurips_selfd/8/"
        "network-snapshot-1.000000-000462.pkl_sc_0.8"
    ),
    "B16": Path(
        "/homes/kasram/broteina/SiD_Protein/design_eval_neurips/neurips_selfd/16/"
        "network-snapshot-1.000000-000069.pkl_sc_0.8"
    ),
}


def load_ca_coords_from_pdb(pdb_path: Path) -> np.ndarray:
    pdb_file = biotite_pdb.PDBFile.read(str(pdb_path))
    array = pdb_file.get_structure(model=1)
    ca_atoms = array[array.atom_name == "CA"]
    return ca_atoms.coord.astype(np.float64)


def analyze_secondary_structure(pdb_path: Path) -> tuple[float, float, float]:
    pdb_file = biotite_pdb.PDBFile.read(str(pdb_path))
    array = pdb_file.get_structure(model=1)
    ca_atoms = array[array.atom_name == "CA"]
    sse = annotate.annotate_sse(ca_atoms)
    total = len(sse)
    alpha = float(np.count_nonzero(sse == "a") / total)
    beta = float(np.count_nonzero(sse == "b") / total)
    coil = float(np.count_nonzero(sse == "c") / total)
    return alpha, beta, coil


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    P_cent = P - P.mean(axis=0)
    Q_cent = Q - Q.mean(axis=0)
    C = P_cent.T @ Q_cent
    V, _, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    U = V @ D @ Wt
    P_rot = P_cent @ U
    diff = P_rot - Q_cent
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def load_viz_atom37_and_ca(viz_pt: Path) -> tuple[dict, np.ndarray]:
    outputs = torch.load(viz_pt, map_location="cpu")
    atom37 = atom14_to_atom37(outputs["positions"][-1], outputs).cpu().numpy()
    ca = atom37[:, 1, :].astype(np.float64)
    return outputs, ca


def export_viz_outputs_to_pdb(outputs: dict, output_pdb: Path) -> None:
    atom37 = atom14_to_atom37(outputs["positions"][-1], outputs).cpu().numpy()
    pred = OFProtein(
        aatype=outputs["aatype"].cpu().numpy(),
        atom_positions=atom37,
        atom_mask=outputs["atom37_atom_exists"].cpu().numpy(),
        residue_index=(outputs["residue_index"] + 1).cpu().numpy(),
        b_factors=outputs["plddt"].cpu().numpy(),
        chain_index=outputs["chain_index"].cpu().numpy()
        if "chain_index" in outputs
        else None,
    )
    output_pdb.write_text(to_pdb(pred))


def rank_designable_samples(root: Path, length: int) -> list[dict]:
    ranked = []
    for pdb_path in sorted((root / "pdbs" / "designable").glob(f"{length}_*.pdb")):
        alpha, beta, coil = analyze_secondary_structure(pdb_path)
        ranked.append(
            {
                "pdb_path": pdb_path,
                "alpha": alpha,
                "beta": beta,
                "coil": coil,
            }
        )
    ranked.sort(key=lambda x: (x["beta"], x["coil"] * -1.0), reverse=True)
    return ranked


def pick_samples(ranked: list[dict]) -> list[tuple[str, dict]]:
    if not ranked:
        return []
    highbeta = ranked[0]
    if len(ranked) == 1:
        return [("highbeta", highbeta)]
    remaining = [x for x in ranked if x["pdb_path"] != highbeta["pdb_path"]]
    random_pick = RNG.choice(remaining)
    return [("highbeta", highbeta), ("random", random_pick)]


def build_viz_cache(root: Path, length: int) -> list[tuple[Path, dict, np.ndarray]]:
    cache = []
    for viz_pt in sorted((root / "viz").glob(f"{length}_*.pt")):
        try:
            outputs, ca = load_viz_atom37_and_ca(viz_pt)
        except Exception as exc:
            print(f"Skipping corrupt viz file: {viz_pt} ({exc})")
            continue
        cache.append((viz_pt, outputs, ca))
    return cache


def match_viz_pt_from_cache(
    viz_cache: list[tuple[Path, dict, np.ndarray]], target_ca: np.ndarray
) -> tuple[Path, dict, float]:
    best = None
    for viz_pt, outputs, ca in viz_cache:
        if ca.shape != target_ca.shape:
            continue
        rmsd = kabsch_rmsd(ca, target_ca)
        if best is None or rmsd < best[2]:
            best = (viz_pt, outputs, rmsd)
    return best


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for model_label, root in MODEL_ROOTS.items():
        for length in LENGTHS:
            viz_cache = build_viz_cache(root, length)
            ranked = rank_designable_samples(root, length)
            for selection_type, sample in pick_samples(ranked):
                source_pdb = sample["pdb_path"]
                target_ca = load_ca_coords_from_pdb(source_pdb)
                matched = match_viz_pt_from_cache(viz_cache, target_ca)
                if matched is None:
                    raise FileNotFoundError(
                        f"No matching viz pt for model {model_label}, length {length}"
                    )
                viz_pt, outputs, match_rmsd = matched
                out_name = (
                    f"{model_label}_L{length}_{selection_type}"
                    f"_beta_{sample['beta']:.3f}_matchRMSD_{match_rmsd:.3f}.pdb"
                )
                output_pdb = OUTPUT_DIR / out_name
                export_viz_outputs_to_pdb(outputs, output_pdb)
                manifest_rows.append(
                    {
                        "model": model_label,
                        "length": length,
                        "selection_type": selection_type,
                        "beta_fraction": f"{sample['beta']:.6f}",
                        "alpha_fraction": f"{sample['alpha']:.6f}",
                        "coil_fraction": f"{sample['coil']:.6f}",
                        "source_designable_pdb": str(source_pdb),
                        "matched_viz_pt": str(viz_pt),
                        "match_ca_rmsd": f"{match_rmsd:.6f}",
                        "output_pdb": str(output_pdb),
                    }
                )

    with MANIFEST_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "length",
                "selection_type",
                "beta_fraction",
                "alpha_fraction",
                "coil_fraction",
                "source_designable_pdb",
                "matched_viz_pt",
                "match_ca_rmsd",
                "output_pdb",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(OUTPUT_DIR)
    print(MANIFEST_PATH)


if __name__ == "__main__":
    main()
