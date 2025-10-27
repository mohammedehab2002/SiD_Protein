import torch
import numpy as np

def calculate_consecutive_distances(opts):
    G = opts.G
    batch_size = G.batch_size
    
    z = torch.randn([batch_size, G.length, 3], device=opts.device)
    z = G.sqrt_one_minus_alphas_cumprod[G.t_init] * z
    mask = torch.ones((batch_size, G.length), device=opts.device)
    x_g = G(z, torch.tensor([G.t_init] * batch_size, device=z.device), mask)

    x_g = np.array(x_g.cpu())
    
    # Calculate the differences between consecutive points
    differences = np.diff(x_g, axis=0)
    
    # Calculate the distances using the Euclidean formula
    distances = np.sqrt(np.sum(differences**2, axis=1))
    
    return float(distances.mean())