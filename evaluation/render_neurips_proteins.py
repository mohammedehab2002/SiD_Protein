#!/usr/bin/env python3
import argparse
from pathlib import Path

import biotite.structure.io.pdb as biotite_pdb
import biotite.structure.sse as annotate
import numpy as np
import pymol
from pymol import cmd
from pymol.parsing import QuietException


DEFAULT_RENDER_DIR = Path("/homes/kasram/broteina/SiD_Protein/design_eval_neurips/renders")
MODEL_ROOTS = {
    "B1": Path(
        "/homes/kasram/broteina/SiD_Protein/design_eval_neurips/neurips_selfd/1/"
        "network-snapshot-1.000000-005561.pkl_sc_0.7/pdbs"
    ),
    "B5": Path(
        "/homes/kasram/broteina/SiD_Protein/design_eval_neurips/neurips_selfd/5/"
        "5step_network-snapshot-1.000000-000626.pkl_sc_0.8/pdbs"
    ),
    "B8": Path(
        "/homes/kasram/broteina/SiD_Protein/design_eval_neurips/neurips_selfd/8/"
        "network-snapshot-1.000000-000462.pkl_sc_0.8/pdbs"
    ),
    "B16": Path(
        "/homes/kasram/broteina/SiD_Protein/design_eval_neurips/neurips_selfd/16/"
        "network-snapshot-1.000000-000069.pkl_sc_0.8/pdbs"
    ),
}


def analyze_secondary_structure(pdb_path: Path) -> np.ndarray:
    pdb_file = biotite_pdb.PDBFile.read(str(pdb_path))
    array = pdb_file.get_structure(model=1)
    ca_atoms = array[array.atom_name == "CA"]
    sse = annotate.annotate_sse(ca_atoms)
    total = len(sse)
    alpha_count = np.count_nonzero(sse == "a")
    beta_count = np.count_nonzero(sse == "b")
    coil_count = np.count_nonzero(sse == "c")
    return np.array([alpha_count, beta_count, coil_count]) / total


def apply_textbook_style() -> None:
    def safe_set(name: str, value) -> None:
        try:
            cmd.set(name, value)
        except QuietException:
            pass

    cmd.dss()
    cmd.hide("everything")
    cmd.show("cartoon")
    safe_set("cartoon_ca_only", 1)
    safe_set("cartoon_oval_length", 1.50)
    safe_set("cartoon_oval_width", 0.25)
    safe_set("cartoon_rect_length", 1.50)
    safe_set("cartoon_rect_width", 0.25)
    safe_set("cartoon_loop_radius", 0.12)
    safe_set("cartoon_fancy_helices", 1)
    safe_set("cartoon_dumbbell_length", 1.5)
    safe_set("cartoon_highlight_color", "grey90")
    cmd.color("firebrick", "ss h")
    cmd.color("slate", "ss s")
    cmd.color("forest", "ss l+")
    safe_set("ray_trace_mode", 1)
    safe_set("ray_shadows", 1)
    safe_set("light_count", 2)
    safe_set("spec_reflect", 0.2)
    safe_set("ray_trace_gain", 0.4)
    cmd.bg_color("white")
    safe_set("ray_opaque_background", 1)


def select_best_candidate(model_label: str, length: int) -> tuple[Path, str, float]:
    root = MODEL_ROOTS[model_label]
    best: tuple[Path, str, float] | None = None
    for subset in ("designable", "undesignable"):
        for pdb_path in sorted((root / subset).glob(f"{length}_*.pdb")):
            sec = analyze_secondary_structure(pdb_path)
            beta = float(sec[1])
            if best is None or beta > best[2]:
                best = (pdb_path, subset, beta)
    if best is None:
        raise FileNotFoundError(f"No PDB found for {model_label} length {length}")
    return best


def render_pdb(input_pdb: Path, output_png: Path, width: int, height: int) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    cmd.reinitialize()
    cmd.load(str(input_pdb), "protein")
    apply_textbook_style()
    cmd.orient()
    cmd.zoom("protein", complete=1)
    cmd.clip("slab", 100)
    cmd.ray(width, height)
    cmd.png(str(output_png))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_ROOTS))
    parser.add_argument("--length", required=True, type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=1200)
    args = parser.parse_args()

    output = args.output or (DEFAULT_RENDER_DIR / f"{args.model}_{args.length}.png")
    input_pdb, subset, beta = select_best_candidate(args.model, args.length)
    pymol.finish_launching(["pymol", "-cq"])
    render_pdb(input_pdb, output, args.width, args.height)
    print(f"input_pdb={input_pdb}")
    print(f"subset={subset}")
    print(f"beta_fraction={beta:.4f}")
    print(f"output_png={output}")


if __name__ == "__main__":
    main()
