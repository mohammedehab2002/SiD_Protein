# Copyright (c) 2025, Liyang Xie. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

import os
import shutil
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from training.proteina.evaluations.inverse_fold_models.proteinmpnn import ProteinMPNN
from training.proteina.evaluations.fold_models.esmfold import ESMFold
from training.proteina.evaluations.pipeline import Pipeline
from training.proteina.proteina_utils import interpolate, sample_reference, samples_to_atom37
from training.proteina.proteinfoundation.metrics import designability
from training.proteina.proteinfoundation.utils.coors_utils import nm_to_ang
from training.sid_utils import generator_step

def compute_scTM(opts):
    G = opts.G
    batch_size = opts.network_kwargs.eval_batch
    noise_scale = opts.network_kwargs.noise_scale
    device = opts.device
    nstep = opts.network_kwargs.nstep
    # inverse fold model
    inverse_fold_model = ProteinMPNN(device=opts.device)

	# fold model
    fold_model = ESMFold(device=opts.device)

	# pipeline
    pipeline = Pipeline(inverse_fold_model, fold_model)
    if not os.path.exists("evaluation_cache"):
        os.mkdir("evaluation_cache")
    eval_input_dir = "evaluation_cache/pipeline_output" + str(opts.rank)
    clean_cache_dir(eval_input_dir)
    eval_coords_dir = eval_input_dir + "/coords"
    clean_cache_dir(eval_coords_dir)
    eval_output_dir = eval_input_dir + "/scores"
    if os.path.exists(eval_output_dir):
        shutil.rmtree(eval_output_dir)

    mean_dist_sum = 0    # To calculate mean consecutive CA-CA distance

    # Define time steps
    if nstep == 1:
        t_steps = [opts.network_kwargs.t_init]
    else:
        t_steps = torch.round(torch.linspace(opts.network_kwargs.t_init, opts.network_kwargs.t, steps=nstep))

    # Randomly sample proteins with 5 different lengths
    for n in torch.randint(opts.network_kwargs.min_n_res, opts.network_kwargs.max_n_res+1, size=(5,)).tolist():
        # Generate batch_size samples of length n unconditionally
        batch = {'nres': torch.tensor([n]), 'nsamples': torch.tensor([batch_size])}
        batch_shape = (batch['nsamples'],)
        mask = torch.ones((batch_size, n), device=device, dtype=torch.bool)
        batch["mask"] = mask
        x_g = torch.zeros((batch_size, n, 3), device=device, dtype=torch.float32)    # Start with pure noise

        # Sampling process along time trajectory defined by t_steps
        for t_step in t_steps:
            x_g = generator_step(G, batch, batch_shape, n, mask, x_g, t_step, nstep, opts.network_kwargs.t, device, noise_scale=noise_scale)

        # Save generated coordinates for evaluation
        for sample_idx in range(x_g.shape[0]):
            coords = x_g[sample_idx].detach().cpu().numpy()
            if opts.network_kwargs.class_name == 'training.networks.ProteinaWrapper':
                coords = nm_to_ang(coords)
            coords = coords[:n]
            np.savetxt(os.path.join(eval_coords_dir, f'{n}_{sample_idx}.npy'), coords, fmt='%.3f', delimiter=',')
            mean_dist_sum += calculate_consecutive_distances(coords)

    # Evaluate generated structures
    pipeline.evaluate(eval_input_dir, eval_output_dir, verbose=False)
    output_info = pd.read_csv(os.path.join(eval_output_dir, 'info.csv'))
    scTM = np.mean(output_info.scTM)
    scRMSD = np.mean(output_info.scRMSD)
    pLDDT = np.mean(output_info.pLDDT)
    pAE = np.mean(output_info.pAE)
    mean_dist = mean_dist_sum / len(output_info.scTM)
    designability_TM = np.sum(output_info.scTM > 0.5) / len(output_info.scTM)
    designability_RMSD = np.sum(output_info.scRMSD < 2.0) / len(output_info.scTM)
    return dict(scTM=scTM, scRMSD=scRMSD, pLDDT=pLDDT, pAE=pAE, mean_dist=mean_dist, designability_TM=designability_TM, designability_RMSD=designability_RMSD)

def calculate_consecutive_distances(coords):
    # Calculate the differences between consecutive points
    differences = np.diff(coords, axis=0)
    
    # Calculate the distances using the Euclidean formula
    distances = np.sqrt(np.sum(differences**2, axis=1))
    
    return float(distances.mean())

def clean_cache_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)