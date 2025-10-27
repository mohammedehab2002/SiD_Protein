# Copyright (c) 2025, Liyang Xie. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

from omegaconf import OmegaConf
import os
import shutil
import torch
import numpy as np
import pandas as pd

from training.proteina.proteinfoundation.utils.ff_utils.pdb_utils import mask_cath_code_by_level, write_prot_to_pdb
from training.proteina.proteinfoundation.metrics.designability import scRMSD
from training.proteina.proteina_utils import interpolate, sample_reference, samples_to_atom37

def compute_scRMSD(opts):
    G = opts.G
    batch_size = 8
    device = opts.device
    with torch.no_grad():
        batch = {'nres': torch.tensor([100]), 'dt': torch.tensor([0.0025], dtype=torch.float32), 'nsamples': torch.tensor([batch_size])}
        batch_shape = batch['nsamples']
        n = batch['nres'].item()
        mask = torch.ones((batch_shape, n), device=device, dtype=torch.bool)
        t = 0.286 * torch.ones(batch_shape, device=device)
        batch["t"] = t
        batch["mask"] = mask
        x_0 = sample_reference(
            n=n, shape=(batch_shape,), device=device, mask=mask
        )
        x_1 = torch.zeros_like(x_0)
        x_t = interpolate(x_0, x_1, t)
        x_t = (1 - t[..., None, None]) * x_0
        batch["x_t"] = x_t
        if "cath_code" in batch:
            batch.pop("cath_code")
        x_g = G(batch)
        coors_atom37 = samples_to_atom37(x_g)
    cfg = opts.dataset_kwargs.cfg
    flat_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    flat_dict = pd.json_normalize(flat_cfg, sep="_").to_dict(orient="records")[0]
    flat_dict = {k: str(v) for k, v in flat_dict.items()}
    columns = list(flat_dict.keys())
    # Add some columns to store per-sample results
    columns += ["id_gen", "pdb_path", "L"]
    if cfg.compute_designability:
        columns += ["_res_scRMSD", "_res_scRMSD_all"]

    root_path = "./proteina_eval/sid" + str(torch.cuda.current_device())
    results = []
    designability = []
    for i in range(coors_atom37.shape[0]):
        # Create directory where everything related to this sample will be stored
        dir_name = f"n_{n}_id_{i}"
        sample_root_path = os.path.join(
            root_path, dir_name
        )  # ./inference/conf_{}/n_{}_id_{}
        os.makedirs(sample_root_path, exist_ok=True)

        # Save generated structure as pdb
        fname = dir_name + ".pdb"
        pdb_path = os.path.join(sample_root_path, fname)
        write_prot_to_pdb(
            coors_atom37[i].cpu().numpy(),
            pdb_path,
            overwrite=True,
            no_indexing=True,
        )

        res_row = list(flat_dict.values()) + [i, pdb_path, n]

        # If needed run designability, storing all intermediate values generated in sample_root_path
        if cfg.compute_designability:
            res_designability = scRMSD(
                pdb_path, ret_min=False, tmp_path=sample_root_path
            )
            res_row += [min(res_designability), res_designability]
            designability.append(res_designability)

        results.append(res_row)

    df = pd.DataFrame(results, columns=columns)
    csv_file = os.path.join(root_path, "..", "results_test.csv")
    df.to_csv(csv_file, index=False)

    return dict(scRMSD=np.mean(designability))