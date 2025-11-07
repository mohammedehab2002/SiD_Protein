# Copyright (c) 2025, Liyang Xie. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

import os
import shutil
import random
import numpy as np
import torch
import click
import pandas as pd
import hydra
import csv
from dotenv import load_dotenv
from pytorch_lightning import seed_everything
from tqdm import tqdm
import time

from training.proteina.proteinfoundation.proteinflow.proteina import Proteina
from training.proteina.proteinfoundation.utils.coors_utils import nm_to_ang
from training.proteina.proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb, mask_cath_code_by_level
from training.proteina.proteinfoundation.inference import parse_len_cath_code
from training.proteina.proteina_utils import interpolate, sample_reference, samples_to_atom37
from training.networks import ProteinaWrapper


@click.command()
@click.option("--model_path", type=str, required=True, help="Path to the distilled model file.")
@click.option("--out_dir", type=str, required=True, help="Directory to save generated structures.")
@click.option("--lengths", type=str, default="50,100,150,200,250", help="Comma separated list of protein lengths to generate.")
@click.option("--conditional", is_flag=True, help="Whether to use fold-class conditional generation.")
@click.option("--num_batch", type=int, default=10, help="Number of batches to generate.")
@click.option("--batch_size", type=int, default=5, help="Batch size for generation.")
@click.option("--seed", type=int, default=5, help="Random seed for generation.")
@click.option("--nstep", type=int, default=10, help="Number of diffusion steps.")
@click.option("--noise_scale", type=float, default=0.4, help="Noise scale at each step.")
def main(model_path, out_dir, lengths, conditional, num_batch, batch_size, seed, nstep, noise_scale):
    # Initialize environment and settings
    load_dotenv()
    seed_everything(seed)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    lengths = parse_int_list(lengths)
    out_dir += f"/{seed}"
    os.makedirs(out_dir, exist_ok=True)

    # Load the distilled model
    print(f"Loading checkpoint {model_path}")
    G = ProteinaWrapper('proteina/configs/experiment_config/', 'inference_ucond_200m_notri')
    G.load_state_dict(torch.load(model_path))
    G = G.to(device)
    G.eval().requires_grad_(False)

    if conditional:
        # If the conditional flag is set, sample from the empirical length-CATH code distribution and generate structures conditionally
        config_name = 'inference_cond_sampling_specific_codes'
        eval_input_dir = generate_conditional(out_dir, config_name, G, num_batch, batch_size, device, nstep, noise_scale)
    else:
        # Unconditional generation
        eval_input_dir = generate_unconditional(out_dir, G, lengths, num_batch, batch_size, device, nstep, noise_scale)
    print(f"Generated structures are saved in {eval_input_dir}")

def parse_int_list(s):
    # Target protein lengths should be a comma separated list of integers
    int_list = [int(s.strip()) for s in s.split(',')]
    return int_list

def generate_unconditional(out_dir, G, lengths, num_batch, batch_size, device, nstep, noise_scale):
    # Create output directories
    eval_input_dir = out_dir # f"{out_dir}/proteina_uncond_distilled_{nstep}step_noise{noise_scale}_output"
    eval_coords_dir, eval_pdb_dir = create_directories(eval_input_dir)

    start_time = time.time()
    for niter in tqdm(range(num_batch), desc="Generating unconditional batches"):
        if niter % 10 == 0:
            os.makedirs(f"{eval_coords_dir}/{niter // 10}", exist_ok=False)
            os.makedirs(f"{eval_pdb_dir}/{niter // 10}", exist_ok=False)
        for n in lengths:
            # Prepare batch. Here all proteins in the batch have the same length n.
            batch = {'nres': torch.tensor([n]), 'dt': torch.tensor([0.0025], dtype=torch.float32), 
                     'nsamples': torch.tensor([batch_size])}
            batch_shape = batch['nsamples']
            mask = torch.ones((batch_shape, n), device=device, dtype=torch.bool)
            batch["mask"] = mask
            x_g = torch.zeros((batch_size, n, 3), device=device, dtype=torch.float32)

            # Generate structures based on the number of sampling steps
            if nstep is None or nstep == 1:
                x_g = generate_onestep(G, batch, batch_shape, n, mask, x_g, device)
            else:
                x_g = generate_multistep(G, batch, batch_shape, n, mask, x_g, nstep, noise_scale, device)
            save_structures(x_g, f"{eval_coords_dir}/{niter // 10}", f"{eval_pdb_dir}/{niter // 10}", n, niter)
    
    # end_time = time.time()
    # with open(os.path.join(eval_input_dir, "total_time.csv"), "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["time", "nsample"])  # optional header
    #     writer.writerow([end_time - start_time, num_batch * batch_size * len(lengths)])

    return eval_input_dir

def generate_conditional(out_dir, config_name, G, num_batch, batch_size, device, nstep, noise_scale):
    # Create output directories
    eval_input_dir = f"{out_dir}/proteina_cond_distilled_{nstep}step_noise{noise_scale}_output"
    eval_coords_dir, eval_pdb_dir = create_directories(eval_input_dir)

    # Load and sample from empirical length-CATH code distribution
    config_path = 'training/proteina/configs/experiment_config/'
    with hydra.initialize(config_path, version_base=hydra.__version__):
        cfg = hydra.compose(config_name=config_name)    
    len_cath_code = parse_len_cath_code(cfg)
    len_cath_code = [(k,v) for k,v in len_cath_code if len(v)==1]
    len_cath_code = random.sample(len_cath_code, num_batch) 
    
    start_time = time.time()
    sampled_cath_codes = {}
    for niter in tqdm(range(num_batch), desc="Generating conditional batches"):
        n, cath_code = len_cath_code[niter]
        masked_cath_codes = [cath_code]
        for level in ["T", "A"]:
            masked_cath_codes.append(mask_cath_code_by_level(masked_cath_codes[-1], level=level)) # Mask CATH code to each level progressively
        for masked_cath_code in masked_cath_codes:  
            # Prepare batch. Here all proteins in the batch have the same length n and CATH code.
            batch = {'nres': torch.tensor([n]), 'dt': torch.tensor([0.0025], dtype=torch.float32), 'nsamples': torch.tensor([batch_size]), 
                        'cath_code': [masked_cath_code] * batch_size}
            batch_shape = batch['nsamples']
            mask = torch.ones((batch_shape, n), device=device, dtype=torch.bool)
            batch["mask"] = mask
            x_g = torch.zeros((batch_shape, n, 3), device=device, dtype=torch.float32)
            # Generate structures based on the number of sampling steps
            if nstep is None or nstep == 1:
                x_g = generate_onestep(G, batch, batch_shape, n, mask, x_g, device)
            else:
                x_g = generate_multistep(G, batch, batch_shape, n, mask, x_g, nstep, noise_scale, device)
            save_structures(x_g, eval_coords_dir, eval_pdb_dir, n, niter, masked_cath_code, sampled_cath_codes)
    # Save the sampled CATH codes for reference
    cath_code_df = pd.DataFrame(list(sampled_cath_codes.items()), columns=['domain', 'cath_code'])
    cath_code_df.to_csv(os.path.join(eval_input_dir, "cath_code_list.csv"), index=False)

    end_time = time.time()
    with open(os.path.join(eval_input_dir, "total_time.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "nsample"])  # optional header
        writer.writerow([end_time - start_time, num_batch * batch_size * 3]) # 3 levels: C, A, T

    return eval_input_dir

def generate_onestep(G, batch, batch_shape, n, mask, x_g, device):
    t_init = 0.37
    t = t_init * torch.ones(batch_shape, device=device)
    batch["t"] = t
    x_0 = sample_reference(
        n=n, shape=(batch_shape,), device=device, mask=mask
    )
    x_t = interpolate(x_0, x_g, t)
    batch["x_t"] = x_t
    x_g, _ = G(batch)
    return x_g

def generate_multistep(G, batch, batch_shape, n, mask, x_g, nstep, noise_scale, device):
    t_steps = torch.round(torch.linspace(30, 400, steps=nstep))
    for t_step in t_steps:
        t = 1.0 - 10 ** (-(t_step / 400) * 2.0) * torch.ones(batch_shape, device=device)
        x_0 = sample_reference(
            n=n, shape=(batch_shape,), device=device, mask=mask
        )
        x_t = interpolate(noise_scale * x_0, x_g, t)
        batch["x_t"] = x_t
        batch["t"] = t
        if nstep == 20:
            x_g, _ = G.predict_clean(batch)
        else:
            x_g, _ = G(batch)
    return x_g

def save_structures(x_g, eval_coords_dir, eval_pdb_dir, n, niter, cath_code=None, sampled_cath_codes=None):
    for sample_idx in range(x_g.shape[0]):
        coords = x_g[sample_idx].detach()
        coords_atom37 = samples_to_atom37(coords)
        coords = nm_to_ang(coords)
        coords = coords.cpu().numpy()
        if cath_code is None:
            filename = f'{n}_{niter}_{sample_idx}'
        else:
            filename = f'{n}_{niter}_{sample_idx}' + cath_code[0].replace('.', '_')
        np.savetxt(os.path.join(eval_coords_dir, filename + '.npy'), coords, fmt='%.3f', delimiter=',')
        write_prot_to_pdb(coords_atom37.cpu().numpy(), 
                        os.path.join(eval_pdb_dir, filename + '.pdb'), 
                        overwrite=True,
                        no_indexing=True
        )
        if sampled_cath_codes is not None:
            sampled_cath_codes[f'{n}_{niter}_{sample_idx}'] = cath_code

def create_directories(eval_input_dir):
    #if os.path.exists(eval_input_dir):
    #    shutil.rmtree(eval_input_dir)
    os.makedirs(eval_input_dir, exist_ok=True)
    eval_coords_dir = eval_input_dir + "/coords"
    #if os.path.exists(eval_coords_dir):
    #    shutil.rmtree(eval_coords_dir)
    os.makedirs(eval_coords_dir, exist_ok=True)
    eval_output_dir = eval_input_dir + "/scores"
    #if os.path.exists(eval_output_dir):
    #    shutil.rmtree(eval_output_dir)
    eval_pdb_dir = eval_input_dir + "/pdbs"
    #if os.path.exists(eval_pdb_dir):
    #    shutil.rmtree(eval_pdb_dir)
    os.makedirs(eval_pdb_dir, exist_ok=True)
    return eval_coords_dir, eval_pdb_dir

if __name__ == "__main__":
    main()