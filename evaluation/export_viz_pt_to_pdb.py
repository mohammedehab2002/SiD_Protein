#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
from transformers.models.esm.openfold_utils.protein import to_pdb


def export_viz_pt_to_pdb(input_pt: Path, output_pdb: Path) -> None:
    outputs = torch.load(input_pt, map_location="cpu")

    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    pred = OFProtein(
        aatype=outputs["aatype"].cpu().numpy(),
        atom_positions=final_atom_positions.cpu().numpy(),
        atom_mask=outputs["atom37_atom_exists"].cpu().numpy(),
        residue_index=(outputs["residue_index"] + 1).cpu().numpy(),
        b_factors=outputs["plddt"].cpu().numpy(),
        chain_index=outputs["chain_index"].cpu().numpy()
        if "chain_index" in outputs
        else None,
    )

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_pdb.write_text(to_pdb(pred))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pt", type=Path, required=True)
    parser.add_argument("--output_pdb", type=Path, required=True)
    args = parser.parse_args()
    export_viz_pt_to_pdb(args.input_pt, args.output_pdb)
    print(f"input_pt={args.input_pt}")
    print(f"output_pdb={args.output_pdb}")


if __name__ == "__main__":
    main()
