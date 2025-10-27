# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

"""Distill pretraind diffusion-based generative model using the techniques described in the
paper "Score identity Distillation: Exponentially Fast Distillation of
Pretrained Diffusion Models for One-Step Generation"."""

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
import wandb
import hydra
from contextlib import nullcontext
from functools import partial
from torch.amp import autocast, GradScaler
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from dotenv import load_dotenv
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pandas as pd

from metrics import sid_metric_main as metric_main
from training.proteina.proteina_utils import interpolate, sample_reference, extract_clean_sample
from training.proteina.proteinfoundation.inference import parse_len_cath_code
from training.sid_utils import sample_training_parameters, generator_step

#----------------------------------------------------------------------------
# Helper methods
def save_data(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

def save_pt(pt, fname):
    torch.save(pt, fname)


def calculate_metric(metric,  G, init_sigma, network_kwargs, dataset_kwargs, num_gpus, rank, local_rank, device,data_stat):
    return metric_main.calc_metric(metric=metric,G=G, init_sigma=init_sigma, network_kwargs=network_kwargs,
        dataset_kwargs=dataset_kwargs, num_gpus=num_gpus, rank=rank, local_rank=local_rank, device=device,data_stat=data_stat)

def append_line(jsonl_line, fname):
    with open(fname, 'at') as f:
        f.write(jsonl_line + '\n')

def init_wandb(config, project="proteina", run_name="test_run"):
    run_name = run_name + "_" + time.strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project=project,
        name=run_name,
        config=config,
    )

def is_loss_nan_check(loss: torch.Tensor) -> bool:
    """check the validness of the current loss

    Args:
        loss: the loss from the model

    Returns:
        bool: if True, loss is not nan or inf
    """

    def is_nan(x):
        return torch.isnan(x).any() or torch.isinf(x).any()

    def all_reduce_tensor(tensor, op=torch.distributed.ReduceOp.SUM):
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=op)
        return tensor

    nan_flag = torch.tensor(
        1.0 if is_nan(loss) else 0.0,
        device=loss.device if torch.cuda.is_available() else None,
    )  # support cpu
    # avoid "Watchdog caught collective operation timeout" error
    all_reduce_tensor(nan_flag)
    if nan_flag.item() > 0.0:
        return True
    return False

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    fake_score_optimizer_kwargs   = {},       # Options for fake score network optimizer.
    g_optimizer_kwargs    = {},     # Options for generator optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    #
    loss_scaling_G      = 100,       # Loss scaling factor of G for reducing FP16 under/overflows.
    #
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot for initialization, None = random initialization.
    resume_training     = None,     # Resume training from the given network snapshot.
    resume_kimg         = 0,        # Start from the given training progress.
    alpha               = 1,         # loss = L2-alpha*L1
    tmax                = 800,        #We add noise at steps 0 to tmax, tmax <= 1000
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    metrics             = None,
    init_sigma          = None,
    data_stat           = None,
    use_sida            = False,
):
    # Initialize.
    load_dotenv()
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    
    # Load dataset for the empirical distribution of lengths and cath codes. Uncomment for conditional training.
    # if network_kwargs.class_name == 'training.networks.ProteinaWrapper':
    #     version_base = hydra.__version__
    #     config_path = "proteina/configs/experiment_config/inference_cond_sampling_specific_codes"
    #     with hydra.initialize(config_path, version_base=hydra.__version__):
    #         cfg = hydra.compose(
    #             config_name="inference_cond_sampling_specific_codes",
    #             return_hydra_config=True,
    #         )
    #     len_cath_code = parse_len_cath_code(cfg)

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dnnlib.EasyDict()

    #Construct the pretrained (true) score network f_phi
    true_score = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    true_score.eval().requires_grad_(False).to(device)

    fake_score = copy.deepcopy(true_score)
    fake_score.train().requires_grad_(True).to(device)
    G = copy.deepcopy(true_score)
    G.train().requires_grad_(True).to(device)

    network_dtype = torch.bfloat16 if network_kwargs.use_fp16 else torch.float32
    enable_amp = (
        torch.autocast(
            device_type="cuda", dtype=network_dtype, cache_enabled=False, enabled=True
        )
        if torch.cuda.is_available() and network_kwargs.use_fp16 
        else nullcontext()
    )
    fake_score_loss_scaler = GradScaler(enabled=False)
    generator_loss_scaler = GradScaler(enabled=False)

    if dist.get_rank() == 0:
        config = {
            "batch_size": batch_size,
            **fake_score_optimizer_kwargs,
            **g_optimizer_kwargs,
            **network_kwargs,
            **loss_kwargs,
        }
        init_wandb(config, project='proteina_multistep')

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss
    dist.print0('Loss function: ' + loss_kwargs.class_name)
    fake_score_optimizer = dnnlib.util.construct_class_by_name(params=fake_score.parameters(), **fake_score_optimizer_kwargs) # subclass of torch.optim.Optimizer
    g_optimizer = dnnlib.util.construct_class_by_name(params=G.parameters(), **g_optimizer_kwargs) # subclass of torch.optim.Optimizer
    g_scheduler = CosineAnnealingWarmRestarts(
        g_optimizer,
        T_0=1000,     # Number of iterations for the first cycle
        T_mult=2,     # Multiplier for cycle length (doubles each time)
        eta_min=1e-6  # Minimum LR to decay to
    )
    fake_score_scheduler = CosineAnnealingWarmRestarts(
        fake_score_optimizer,
        T_0=1000,     # Number of iterations for the first cycle
        T_mult=2,     # Multiplier for cycle length (doubles each time)
        eta_min=1e-6  # Minimum LR to decay to
    )
    
    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from URL "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        # with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
        #     data = pickle.load(f)
        with open(resume_pkl, "rb") as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        
        dist.print0('Loading network completed')
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=true_score, require_all=False)
        
        if resume_training is not None:
            dist.print0('checkpoint path:',resume_training)
            data = torch.load(resume_training, map_location=torch.device('cpu'))
            misc.copy_params_and_buffers(src_module=data['fake_score'], dst_module=fake_score, require_all=True)
            misc.copy_params_and_buffers(src_module=data['G'], dst_module=G, require_all=True)
            G_ema = copy.deepcopy(G).eval().requires_grad_(False)
            misc.copy_params_and_buffers(src_module=data['G_ema'], dst_module=G_ema, require_all=True)
            G_ema.eval().requires_grad_(False)
            fake_score_optimizer.load_state_dict(data['fake_score_optimizer_state'])
            g_optimizer.load_state_dict(data['g_optimizer_state'])
            del data # conserve memory
            dist.print0('Loading checkpoint completed')
            dist.print0('Setting up optimizer...')
            fake_score_ddp = torch.nn.parallel.DistributedDataParallel(fake_score, device_ids=[device], broadcast_buffers=False,find_unused_parameters=False)
            G_ddp = torch.nn.parallel.DistributedDataParallel(G, device_ids=[device], broadcast_buffers=False,find_unused_parameters=False)

        else:     
            # Setup optimizer.
            misc.copy_params_and_buffers(src_module=data['ema'], dst_module=fake_score, require_all=False)
            misc.copy_params_and_buffers(src_module=data['ema'], dst_module=G, require_all=False)
            dist.print0('Setting up optimizer...')
            fake_score_ddp = torch.nn.parallel.DistributedDataParallel(fake_score, device_ids=[device], broadcast_buffers=False,find_unused_parameters=False)
            G_ddp = torch.nn.parallel.DistributedDataParallel(G, device_ids=[device], broadcast_buffers=False,find_unused_parameters=False)
            G_ema = copy.deepcopy(G).eval().requires_grad_(False)
            misc.copy_params_and_buffers(src_module=data['ema'], dst_module=G_ema, require_all=False)
            del data # conserve memory
        fake_score_ddp.eval().requires_grad_(False)
        G_ddp.eval().requires_grad_(False)
    else:
        fake_score_ddp = torch.nn.parallel.DistributedDataParallel(fake_score, device_ids=[device], broadcast_buffers=False,find_unused_parameters=False)
        G_ddp = torch.nn.parallel.DistributedDataParallel(G, device_ids=[device], broadcast_buffers=False,find_unused_parameters=False)
        fake_score_ddp.eval().requires_grad_(False)
        G_ddp.eval().requires_grad_(False)
        G_ema = copy.deepcopy(G)
        G_ema.eval().requires_grad_(False)

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    stats_metrics = dict()

    nstep = network_kwargs.nstep
    if nstep == 1:
        t_steps = [network_kwargs.t_init]
    else:
        t_steps = torch.round(torch.linspace(network_kwargs.t_init, network_kwargs.t, steps=network_kwargs.nstep))
    cath_code = None
    reset_counter = 1
    torch.cuda.empty_cache()

    while True:  
        # Periodically reset fake score optimizer state and learning rate schedulers benefit training stability      
        if cur_nimg > 400000 * reset_counter:
            reset_counter += 1
            fake_score = copy.deepcopy(true_score)
            fake_score.train().requires_grad_(True).to(device)
            fake_score_ddp = torch.nn.parallel.DistributedDataParallel(fake_score, device_ids=[device], broadcast_buffers=False,find_unused_parameters=False)
            fake_score_optimizer = dnnlib.util.construct_class_by_name(params=fake_score.parameters(), **fake_score_optimizer_kwargs) # subclass of torch.optim.Optimizer
            g_scheduler = CosineAnnealingWarmRestarts(
                g_optimizer,
                T_0=1000,     # Number of iterations for the first cycle
                T_mult=2,     # Multiplier for cycle length (doubles each time)
                eta_min=1e-6  # Minimum LR to decay to
            )
            fake_score_scheduler = CosineAnnealingWarmRestarts(
                fake_score_optimizer,
                T_0=1000,     # Number of iterations for the first cycle
                T_mult=2,     # Multiplier for cycle length (doubles each time)
                eta_min=1e-6  # Minimum LR to decay to
            )

        #Update fake score network f_psi
        # Accumulate gradients.
        fake_score_ddp.train().requires_grad_(True)
        fake_score_optimizer.zero_grad(set_to_none=True)

        for round_idx in range(num_accumulation_rounds):
            batch, batch_shape, n, mask, x_1, train_step = sample_training_parameters(network_kwargs, nstep, batch_gpu, device)
            with misc.ddp_sync(G_ddp, False):
                for i, t_step in enumerate(t_steps):
                    # Only compute gradients for the selected time step
                    if i == train_step:
                        x_g = generator_step(G_ddp, batch, batch_shape, n, mask, x_1, t_step, nstep, network_kwargs.t, device)
                        break
                    # For other time steps, no gradient computation
                    else:
                        with torch.no_grad():
                            x_1 = generator_step(G_ddp, batch, batch_shape, n, mask, x_1, t_step, nstep, network_kwargs.t, device)
            # Accumulate gradients for fake score network
            with misc.ddp_sync(fake_score_ddp, (round_idx == num_accumulation_rounds - 1)):
                with enable_amp:
                    fake_score_loss = loss_fn(fake_score=fake_score_ddp, batch=batch, x_g=x_g, tmax=tmax)
                    fake_score_loss=fake_score_loss.sum().mul(loss_scaling / batch_gpu_total)
                if is_loss_nan_check(fake_score_loss):
                    dist.print0(f"Skip iteration with NaN loss: {cur_tick} ticks")
                    fake_score_loss = torch.tensor(0.0, device=fake_score_loss.device, requires_grad=True)
                if use_sida and is_loss_nan_check(fake_loss_D):
                    dist.print0(f"Skip iteration with NaN loss: {cur_tick} ticks")
                    fake_loss_D = torch.tensor(0.0, device=fake_loss_D.device, requires_grad=True)

            # Backpropagate fake score loss
            if not use_sida:
                fake_score_loss_scaler.scale(fake_score_loss).backward()
            else:
                fake_score_loss_scaler.scale(fake_score_loss + fake_loss_D).backward()
        
        # Check if gradients are valid
        if dist.get_rank() == 0:
            fake_grad_norm = 0
            fake_grad_max = 0
            for p in fake_score_ddp.parameters():
                if p.grad is not None:
                    param_norm = p.grad.norm(2)  # Compute L2 norm of gradients
                    fake_grad_norm += param_norm.item() ** 2
                    param_max = p.grad.abs().max().item()
                    fake_grad_max = max(fake_grad_max, param_max)
            wandb.log({'train/fake_grad_norm': fake_grad_norm ** 0.5, 'train/fake_grad_max': fake_grad_max}, step=cur_nimg)
        
        # Apply gradients to update fake score network
        torch.nn.utils.clip_grad_norm_(
            fake_score_ddp.parameters(), 1
        )
        fake_score_loss_scaler.unscale_(fake_score_optimizer)
        for param in fake_score_ddp.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        fake_score_loss_scaler.step(fake_score_optimizer)
        fake_score_loss_scaler.update()
        fake_score_optimizer.zero_grad(set_to_none=True)
        fake_score_scheduler.step()

        # Log training stats for fake score network
        if dist.get_rank() == 0:
            fake_score_lr = fake_score_optimizer.param_groups[0]['lr']
            wandb.log({'train/fake_score_lr': fake_score_lr}, step=cur_nimg)
        loss_fake_score_print = fake_score_loss.item()
        if use_sida:
            loss_fake_D_print = fake_loss_D.item()
        training_stats.report('fake_score_Loss/loss', loss_fake_score_print)
        if dist.get_rank() == 0:
            wandb.log({'train/fake_score_loss': loss_fake_score_print}, step=cur_nimg)
            if use_sida:
                wandb.log({'train/fake_score_loss_D': loss_fake_D_print}, step=cur_nimg)
        fake_score_ddp.eval().requires_grad_(False)
        if use_sida:
            del fake_loss_D
        del fake_score_loss
        torch.cuda.empty_cache()

        #Update generator G_theta
        G_ddp.train().requires_grad_(True)
        g_optimizer.zero_grad(set_to_none=True)

        for round_idx in range(num_accumulation_rounds):
            batch, batch_shape, n, mask, x_1, train_step = sample_training_parameters(network_kwargs, nstep, batch_gpu, device)
            with misc.ddp_sync(G_ddp, (round_idx == num_accumulation_rounds - 1)):
                for i, t_step in enumerate(t_steps):
                    # Only compute gradients for the selected time step
                    if i == train_step:
                        x_g = generator_step(G_ddp, batch, batch_shape, n, mask, x_1, t_step, nstep, network_kwargs.t, device)
                        break
                    # For other time steps, no gradient computation
                    else:
                        with torch.no_grad():
                            x_1 = generator_step(G_ddp, batch, batch_shape, n, mask, x_1, t_step, nstep, network_kwargs.t, device)
            
            # Accumulate gradients for generator     
            with misc.ddp_sync(fake_score_ddp, False):
                with enable_amp:
                    G_loss, real_fake_loss, real_G_loss = loss_fn.generator_loss(true_score=true_score, fake_score=fake_score_ddp, batch=batch, \
                                                                                    x_g=x_g,alpha=alpha,tmax=tmax, network_kwargs=network_kwargs)
                    G_loss=G_loss.sum().mul(loss_scaling_G / batch_gpu_total)
                if is_loss_nan_check(G_loss):
                    dist.print0(f"Skip iteration with NaN loss: {cur_tick} ticks")
                    G_loss = torch.tensor(0.0, device=G_loss.device, requires_grad=True)
                if use_sida and is_loss_nan_check(G_loss_D):
                    dist.print0(f"Skip iteration with NaN loss: {cur_tick} ticks")
                    G_loss_D = torch.tensor(0.0, device=G_loss_D.device, requires_grad=True)

            # Backpropagate generator loss
            if not use_sida:
                generator_loss_scaler.scale(G_loss).backward()
            else:
                generator_loss_scaler.scale(G_loss + G_loss_D).backward()

        # Check if gradients are valid
        if dist.get_rank() == 0:
            G_grad_norm = 0
            G_grad_max = 0
            for p in G_ddp.parameters():
                if p.grad is not None:
                    param_norm = p.grad.norm(2)  # Compute L2 norm of gradients
                    G_grad_norm += param_norm.item() ** 2
                    param_max = p.grad.abs().max().item()
                    G_grad_max = max(fake_grad_max, param_max)
            wandb.log({'train/generator_grad_norm': G_grad_norm ** 0.5, 'train/generator_grad_max': G_grad_max}, step=cur_nimg)
            wandb.log({'Loss/y_real-y_fake': real_fake_loss, 'Loss/y_real-y': real_G_loss}, step=cur_nimg)
        torch.nn.utils.clip_grad_norm_(
            G_ddp.parameters(), 1
        )

        # Apply gradients to update generator
        generator_loss_scaler.unscale_(g_optimizer)
        for param in G_ddp.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        generator_loss_scaler.step(g_optimizer)
        generator_loss_scaler.update()
        g_optimizer.zero_grad(set_to_none=True)
        g_scheduler.step()

        # Log training stats for generator
        if dist.get_rank() == 0:
            g_lr = g_optimizer.param_groups[0]['lr']
            wandb.log({'train/g_lr': g_lr}, step=cur_nimg)
        lossG_print = G_loss.item()
        if use_sida:
            lossG_D_print = G_loss_D.item()
        training_stats.report('G_Loss/loss', lossG_print)
        if dist.get_rank() == 0:
            wandb.log({'train/generator_loss': lossG_print}, step=cur_nimg)
            if use_sida:
                wandb.log({'train/generator_loss_D': lossG_D_print}, step=cur_nimg)
        G_ddp.eval().requires_grad_(False)
        if use_sida:
            del G_loss_D
        del G_loss
        torch.cuda.empty_cache()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
                
        for p_ema, p_true_score in zip(G_ema.parameters(), G.parameters()):
            p_ema.copy_(p_true_score.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        fields += [f"loss_fake_score {training_stats.report0('fake_score_Loss/loss', loss_fake_score_print):<6.2f}"]
        fields += [f"loss_G {training_stats.report0('G_Loss/loss', lossG_print):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')
                        
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0 or cur_tick in [10,20,30,40,50,60,70,80,90,100]):
            dist.print0('Evaluating metrics...')
            for metric in metrics:
                result_dict = calculate_metric(metric=metric, G=G, init_sigma=init_sigma, network_kwargs=network_kwargs,
                    dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), local_rank=dist.get_local_rank(), device=device,data_stat=data_stat)
                if dist.get_rank() == 0:
                    print(result_dict.results)
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=f'fakes_{alpha:03f}_{cur_nimg//1000:06d}.png', alpha=alpha)  
                    wandb.log(result_dict.results, step=cur_nimg)
                stats_metrics.update(result_dict.results)

                
            data = dict(ema=G_ema)
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    data[key] = value.cpu()
                del value # conserve memory
                
            if dist.get_rank() == 0:
                save_data(data=data, fname=os.path.join(run_dir, f'network-snapshot-{alpha:03f}-{cur_nimg//1000:06d}.pkl'))               
            del data # conserve memory

        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            dist.print0(f'saving checkpoint: training-state-{cur_nimg//1000:06d}.pt')
            save_pt(pt=dict(fake_score=fake_score, G=G, G_ema=G_ema, fake_score_optimizer_state=fake_score_optimizer.state_dict(), g_optimizer_state=g_optimizer.state_dict()), fname=os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))
            if (result_dict.results['scRMSD'] < 2):
                torch.save(G.model.state_dict(), 'vsd_proteina_generator.pth')    
        dist.print0("Evaluation Done")
        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                append_line(jsonl_line=json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n', fname=os.path.join(run_dir, f'stats_{alpha:03f}.jsonl'))

        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
