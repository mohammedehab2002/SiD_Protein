#!/usr/bin/env python3

import csv
import os
import pickle as pkl
import random
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import load_dotenv
from pytorch_lightning import seed_everything
from tqdm import tqdm

from training.proteina.proteina_utils import interpolate, sample_reference, samples_to_atom37
from training.proteina.proteinfoundation.utils.coors_utils import nm_to_ang
from training.proteina.proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb


def set_generation_seed(base_seed: int, split_id: int, seed_stride: int) -> int:
    effective_seed = int(base_seed) + int(split_id) * int(seed_stride)
    seed_everything(effective_seed, workers=True)
    random.seed(effective_seed)
    np.random.seed(effective_seed % (2**32 - 1))
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(effective_seed)
        torch.cuda.manual_seed_all(effective_seed)
    return effective_seed


def build_work_items(min_len: int, max_len: int, step_len: int, samples_per_len: int, max_batch_size: int):
    items = []
    for n in range(min_len, max_len + 1, step_len):
        offset = 0
        while offset < samples_per_len:
            count = min(max_batch_size, samples_per_len - offset)
            items.append({"n": n, "count": count, "offset": offset})
            offset += count
    return items


def generate_onestep(G, batch, batch_shape, n, mask, x_g, noise_scale, device):
    t_init = 0.37
    t = t_init * torch.ones(batch_shape, device=device)
    batch["t"] = t
    x_0 = sample_reference(n=n, shape=(batch_shape,), device=device, mask=mask)
    x_t = interpolate(noise_scale * x_0, x_g, t)
    batch["x_t"] = x_t
    x_g, _ = G(batch)
    return x_g


def generate_multistep(G, batch, batch_shape, n, mask, x_g, nstep, noise_scale, device):
    t_steps = torch.round(torch.linspace(30, 400, steps=nstep))
    for t_step in t_steps:
        t = 1.0 - 10 ** (-(t_step / 400) * 2.0) * torch.ones(batch_shape, device=device)
        x_0 = sample_reference(n=n, shape=(batch_shape,), device=device, mask=mask)
        x_t = interpolate(noise_scale * x_0, x_g, t)
        batch["x_t"] = x_t
        batch["t"] = t
        if nstep == 20:
            x_g, _ = G.predict_clean(batch)
        else:
            x_g, _ = G(batch)
    return x_g


@click.command()
@click.option("--model_path", type=str, required=True)
@click.option("--out_dir", type=str, required=True)
@click.option("--nstep", type=int, required=True)
@click.option("--noise_scale", type=float, default=1.0)
@click.option("--seed", type=int, default=5)
@click.option("--split_id", type=int, default=0)
@click.option("--num_splits", type=int, default=1)
@click.option("--seed_stride", type=int, default=1000003)
@click.option("--min_len", type=int, default=60)
@click.option("--max_len", type=int, default=255)
@click.option("--step_len", type=int, default=5)
@click.option("--samples_per_len", type=int, default=125)
@click.option("--max_batch_size", type=int, default=8)
def main(
    model_path,
    out_dir,
    nstep,
    noise_scale,
    seed,
    split_id,
    num_splits,
    seed_stride,
    min_len,
    max_len,
    step_len,
    samples_per_len,
    max_batch_size,
):
    load_dotenv()
    assert num_splits >= 1
    assert 0 <= split_id < num_splits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_seed = set_generation_seed(seed, split_id, seed_stride)
    print(
        f"Using device={device}, split={split_id + 1}/{num_splits}, "
        f"base_seed={seed}, effective_seed={effective_seed}, noise_scale={noise_scale}"
    )

    with open(model_path, "rb") as f:
        G = pkl.load(f)["ema"]
    G = G.to(device)
    G.eval().requires_grad_(False)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples_fid_sharded"
    samples_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = out_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    for old_file in samples_dir.glob(f"split{split_id:02d}_*_fid.pdb"):
        old_file.unlink()
    metadata_path = metadata_dir / f"split{split_id:02d}.csv"
    if metadata_path.exists():
        metadata_path.unlink()

    work_items = build_work_items(min_len, max_len, step_len, samples_per_len, max_batch_size)
    shard_items = work_items[split_id::num_splits]
    print(f"Total work items={len(work_items)}, shard work items={len(shard_items)}")

    rows = []
    sample_count = 0
    for item_idx, item in enumerate(tqdm(shard_items, desc=f"split{split_id:02d}")):
        n = item["n"]
        count = item["count"]
        offset = item["offset"]

        batch_shape = torch.tensor([count])
        batch = {
            "nres": torch.tensor([n]),
            "dt": torch.tensor([0.0025], dtype=torch.float32),
            "nsamples": batch_shape,
        }
        mask = torch.ones((count, n), device=device, dtype=torch.bool)
        batch["mask"] = mask
        x_g = torch.zeros((count, n, 3), device=device, dtype=torch.float32)

        if nstep == 1:
            x_g = generate_onestep(G, batch, batch_shape, n, mask, x_g, noise_scale, device)
        else:
            x_g = generate_multistep(G, batch, batch_shape, n, mask, x_g, nstep, noise_scale, device)

        for sample_idx in range(count):
            global_idx_for_length = offset + sample_idx
            fname = f"split{split_id:02d}_{n}_{global_idx_for_length:03d}_fid.pdb"
            pdb_path = samples_dir / fname
            coords_atom37 = samples_to_atom37(x_g[sample_idx].detach())
            write_prot_to_pdb(
                coords_atom37.cpu().numpy(),
                str(pdb_path),
                overwrite=True,
                no_indexing=True,
            )
            rows.append(
                {
                    "split_id": split_id,
                    "effective_seed": effective_seed,
                    "n": n,
                    "offset": offset,
                    "sample_idx_in_batch": sample_idx,
                    "global_idx_for_length": global_idx_for_length,
                    "pdb_path": str(pdb_path),
                    "noise_scale": noise_scale,
                    "nstep": nstep,
                    "model_path": model_path,
                }
            )
            sample_count += 1

    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "split_id", "effective_seed", "n", "offset", "sample_idx_in_batch",
            "global_idx_for_length", "pdb_path", "noise_scale", "nstep", "model_path"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {sample_count} samples to {samples_dir}")


if __name__ == "__main__":
    main()
