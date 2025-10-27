# Copyright (c) 2025, Liyang Xie. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

import torch
from training.proteina.proteina_utils import interpolate, sample_reference, extract_clean_sample

def sample_t(t_step, nstep, max_t, batch_shape, device):
    if nstep == 1:
        t = 0.37 * torch.ones(batch_shape, device=device)
    else:
        t = 1.0 - 10 ** (-(t_step / max_t) * 2.0) * torch.ones(batch_shape, device=device)
    return t

def sample_training_parameters(network_kwargs, nstep, batch_gpu, device):
    # Randomly sample lengths for proteins in batch
    lengths = torch.randint(network_kwargs.min_n_res, network_kwargs.max_n_res + 1, (batch_gpu,))
    # Create mask based on the lengths. Dimension : (batch_size, max_length)
    n = lengths.max().item()
    range_vector = torch.arange(n).unsqueeze(0)
    mask = range_vector < lengths.unsqueeze(1)
    mask = mask.to(device)
    # Create batch dict as input to the network
    batch = {'nres': torch.tensor([n]), 'nsamples': torch.tensor([batch_gpu]), 'mask': mask}
    batch_shape = (batch['nsamples'],)
    # The first step starts from pure noise, i.e. signal = 0
    x_1 = torch.zeros((batch_gpu, n, 3), device=device)
    # Randomly select one time step from t_steps to train
    train_step = torch.randint(0, nstep, (1,)).item()
    return batch, batch_shape, n, mask, x_1, train_step

def generator_step(G_ddp, batch, batch_shape, n, mask, x_1, t_step, nstep, max_t, device, noise_scale=1.0):
    # Sample time t
    t = sample_t(t_step, nstep, max_t, batch_shape, device)
    # Sample Gaussian noise x_0
    x_0 = sample_reference(
        n=n, shape=batch_shape, device=device, mask=mask
    )
    # Interpolate x_0 and x_1 to get x_t
    x_t = interpolate(noise_scale * x_0, x_1, t)
    # Prepare batch for network input
    batch["x_t"] = x_t
    batch["t"] = t
    # Forward pass through the network
    x_g, _ = G_ddp(batch)
    return x_g