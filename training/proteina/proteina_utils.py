# Copyright (c) 2025, Liyang Xie. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

from proteinfoundation.utils.coors_utils import ang_to_nm, trans_nm_to_atom37
import torch
from jaxtyping import Bool, Float
from torch import Dict, Tensor
from tqdm import tqdm
from typing import Callable, List, Literal, Optional, Tuple
from scipy.spatial.transform import Rotation
from math import prod

from proteinfoundation.utils.align_utils.align_utils import mean_w_mask

def samples_to_atom37(samples):
        """
        Transforms samples to atom37 representation.

        Args:
            samples: Tensor of shape [b, n, 3]

        Returns:
            Samples in atom37 representation, shape [b, n, 37, 3].
        """
        return trans_nm_to_atom37(samples)  # [b, n, 37, 3]

def _force_zero_com(
        x: Float[Tensor, "* n 3"], mask: Optional[Bool[Tensor, "* n"]] = None
    ) -> Dict[str, Tensor]:
        """
        Centers tensor over n dimension.

        Args:
            x: Tensor of shape [*, n, 3]
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Centered x = x - mean(x, dim=-2), shape [*, n, 3].
        """
        if mask is None:
            x = x - torch.mean(x, dim=-2, keepdim=True)
        else:
            x = (x - mean_w_mask(x, mask, keepdim=True)) * mask[..., None]
        return x

def _apply_mask(
        x: Float[Tensor, "* n 3"], mask: Optional[Bool[Tensor, "* n"]] = None
    ) -> Dict[str, Tensor]:
        """
        Applies mask to x. Sets masked elements to zero.

        Args:
            x: Tensor of shape [*, n, 3]
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Masked x of shape [*, n, 3]
        """
        if mask is None:
            return x
        return x * mask[..., None]  # [*, n, 3]

def _extend_t(n: int, t: Float[Tensor, "*"]) -> Float[Tensor, "* n"]:
        """
        Extends t shape with n. Needed to use flow matching utils.

        Args:
            n (int): Number of elements per sample (e.g. number of residues)
            t: Float vector, shape [*]

        Returns:
            Extended t vector of shape [*, n] compatible with flow matching utils.
        """
        return t[..., None].expand(t.shape + (n,))

def _mask_and_zero_com(
        x, mask: Optional[Bool[Tensor, "* n"]] = None, zero_com: bool = True
    ) -> Dict[str, Tensor]:
        """
        Applies mask to and centers x if needed (if zero_com=True).

        Args:
            x: Batch of samples, batch shape *
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Masked (and possibly center) samples.
        """
        x = _apply_mask(x, mask)
        if zero_com:
            x = _force_zero_com(x, mask)
        return x

def interpolate(
        x_0: Float[Tensor, "* n 3"],
        x_1: Float[Tensor, "* n 3"],
        t: Float[Tensor, "*"],
        mask: Optional[Bool[Tensor, "* n"]] = None,
    ) -> Dict[str, Tensor]:
        """
        Interpolates between rigids x_0 (base) and x_1 (data) using t.

        Args:
            x_0: Tensor sampled from reference, shape [*, n, 3]
            x_1: Tensor sampled from target, shape [*, n, 3]
            t: Interpolation times, shape [*]
            mask (optional): Binary mask, shape [*, n]

        Returns:
            x_t: Interpolated tensor, shape [*, n, 3]
        """
        x_0, x_1 = map(
            lambda args: _mask_and_zero_com(*args), ((x_0, mask), (x_1, mask))
        )
        # x_0 should already be masked (reference), x_1 depends on dataloader
        # x_0 should be centered (reference), x_1 depends on dataloader

        n = x_0.shape[-2]
        t = _extend_t(n, t)  # [*, n]
        t = t[..., None]  # [*, n, 1]
        trans_t = (1.0 - t) * x_0 + t * x_1
        return trans_t  # Masking nor centering necessary since x_0 and x_1 are

def sample_reference(
        n: int,
        shape: Tuple = tuple(),
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        mask: Optional[Bool[Tensor, "* n"]] = None,
    ) -> Dict[str, Tensor]:
        """
        Samples reference distribution std Gaussian (possibly centered).

        Args:
            n: number of frames in a single sample, int
            shape: tuple (if empty then single sample)
            dtype (optional): torch.dtype used
            device (optional): torch device used
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Samples from refenrece [N(0, I_3)]^n shape [*shape, n, 3]
        """
        scale_ref = 1
        x = (
            torch.randn(
                shape
                + (
                    n,
                    3,
                ),
                device=device,
                dtype=dtype,
            )
            * scale_ref
        )
        return _mask_and_zero_com(x, mask)

def compute_fm_loss(
        x_1: Float[Tensor, "* n 3"],
        x_1_pred: Float[Tensor, "* n 3"],
        x_t: Float[Tensor, "* n 3"],
        t: Float[Tensor, "*"],
        mask: Bool[Tensor, "* nres"],
    ) -> Float[Tensor, "*"]:
        """
        Computes and logs flow matching loss.

        Args:
            x_1: True clean sample, shape [*, n, 3].
            x_1_pred: Predicted clean sample, shape [*, n, 3].
            x_t: Sample at interpolation time t (used as input to predict clean sample), shape [*, n, 3].
            t: Interpolation time, shape [*].
            mask: Boolean residue mask, shape [*, nres].

        Returns:
            Flow matching loss.
        """
        nres = torch.sum(mask, dim=-1) * 3  # [*]

        err = (x_1 - x_1_pred) * mask[..., None]  # [*, n, 3]
        loss = torch.sum(err**2, dim=(-1, -2)) / nres  # [*]

        total_loss_w = 1.0 / ((1.0 - t) ** 2 + 1e-5)

        loss = loss * total_loss_w  # [*]
        return loss

def compute_auxiliary_loss(
        x_1: Float[Tensor, "* n 3"],
        x_1_pred: Float[Tensor, "* n 3"],
        x_t: Float[Tensor, "* n 3"],
        t: Float[Tensor, "*"],
        mask: Bool[Tensor, "* n"],
        nn_out: Dict[str, Tensor],
        batch: Dict[str, Tensor] = None,
    ) -> Float[Tensor, ""]:
        """
        Computes and logs auxiliary losses.

        Args:
            x_1: True clean sample, shape [*, n, 3].
            x_1_pred: Predicted clean sample, shape [*, n, 3].
            x_t: Sample at interpolation time t (used as input to predict clean sample), shape [*, n, 3].
            t: Interpolation time, shape [*].
            mask: Boolean residue mask, shape [*, n].
            nn_out: Dictionary of output from neural network

        Returns:
            Auxiliary loss.
        """
        bs = x_1.shape[0]
        n = x_1.shape[1]
        nres = mask.sum(-1)  # [*]

        gt_ca_coors = x_1 * mask[..., None]  # [*, n, 3]
        pred_ca_coors = x_1_pred * mask[..., None]  # [*, n, 3]
        pair_mask = mask[..., None, :] * mask[..., None]  # [*, n, n]

        # Pairwise distances
        gt_pair_dists = torch.linalg.norm(
            gt_ca_coors[:, :, None, :] - gt_ca_coors[:, None, :, :], dim=-1
        )  # [*, n, n]
        pred_pair_dists = torch.linalg.norm(
            pred_ca_coors[:, :, None, :] - pred_ca_coors[:, None, :, :], dim=-1
        )  # [*, n, n]
        gt_pair_dists = gt_pair_dists * pair_mask  # [*, n, n]
        pred_pair_dists = pred_pair_dists * pair_mask  # [*, n, n]

        # Add mask to only account for pairs that are closer than thr in ground truth
        max_dist = 0.6 # training config.loss.thres_aux_2d_loss 
        if max_dist is None:
            max_dist = 1e10
        pair_mask_thr = gt_pair_dists < max_dist  # [*, n, n]
        total_pair_mask = pair_mask * pair_mask_thr  # [*, n, n]

        # Compute loss
        den = torch.sum(total_pair_mask, dim=(-1, -2)) - nres
        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * total_pair_mask, dim=(-1, -2)
        )  # [*]
        dist_mat_loss = dist_mat_loss / den  # [*]

        # Distogram loss
        num_dist_buckets = 64 # training config.loss.num_dist_buckets
        pair_pred = nn_out.get("pair_pred", None)
        if num_dist_buckets and pair_pred is not None:
            assert (
                num_dist_buckets == pair_pred.shape[-1]
            ), "The number of distance buckets should be equal with the output dim of pair pred head"
            assert num_dist_buckets > 1, "Need more than one bucket for distogram loss"

            # Bucketize pair distance
            max_dist_boundary = 1.0 # training config.loss.max_dist_boundary
            boundaries = torch.linspace(
                0.0, max_dist_boundary, num_dist_buckets - 1, device=pair_pred.device
            )
            gt_pair_dist_bucket = torch.bucketize(
                gt_pair_dists, boundaries
            )  # [*, n, n], each value in [0, num_dist_buckets)

            # Distogram loss
            pair_pred = pair_pred.view(bs * n * n, num_dist_buckets)
            gt_pair_dist_bucket = gt_pair_dist_bucket.view(bs * n * n)
            distogram_loss = torch.nn.functional.cross_entropy(
                pair_pred, gt_pair_dist_bucket, reduction="none"
            )  # [bs * n * n]
            distogram_loss = distogram_loss.view(bs, n, n)
            distogram_loss = torch.sum(distogram_loss * pair_mask, dim=(-1, -2))  # [*]
            distogram_loss = distogram_loss / (
                pair_mask.sum(dim=(-1, -2)) + 1e-10
            )  # [*]
        else:
            distogram_loss = dist_mat_loss * 0

        auxiliary_loss = (
            distogram_loss
            * (t > 0.3)
            * 1.0
        )
        auxiliary_loss_no_w = distogram_loss * (t > 0.3)
        motif_aux_loss_weight = 0
        scaffold_aux_loss_weight = 0
        if scaffold_aux_loss_weight > 0:
            scaffold_loss = scaffold_aux_loss_weight * compute_fm_loss(
                        x_1=x_1,
                        x_1_pred=x_1_pred,
                        x_t=x_t,
                        mask=~batch["fixed_sequence_mask"]*batch["mask"],
                        t=t
                    )
            auxiliary_loss += scaffold_loss
        elif motif_aux_loss_weight:
            mask_to_use = batch["fixed_sequence_mask"] * batch["mask"]
            check_weight = 1.0
            if not batch["fixed_sequence_mask"].any():
                check_weight = 0
                mask_to_use = batch["mask"]
            motif_loss = motif_aux_loss_weight * self.compute_fm_loss(
                x_1=x_1,
                x_1_pred=x_1_pred,
                x_t=x_t,
                mask=mask_to_use,
                t=t,
                log_prefix=None,
            )
            auxiliary_loss += check_weight * motif_loss
        return auxiliary_loss

def sample_uniform_rotation(
    shape=tuple(), dtype=None, device=None
) -> Float[Tensor, "*batch 3 3"]:
    """
    Samples rotations distributed uniformly.

    Args:
        shape: tuple (if empty then samples single rotation)
        dtype: used for samples
        device: torch.device

    Returns:
        Uniformly samples rotation matrices [*shape, 3, 3]
    """
    return torch.tensor(
        Rotation.random(prod(shape)).as_matrix(),
        device=device,
        dtype=dtype,
    ).reshape(*shape, 3, 3)

def xt_dot(
    x_1: Float[Tensor, "* n 3"],
    x_t: Float[Tensor, "* n 3"],
    t: Float[Tensor, "*"],
    mask: Optional[Bool[Tensor, "* n"]] = None,
) -> Dict[str, Tensor]:
    """
    Computes \dot{x_t} for the interpolation scheme defined
    above. This is the target used in flow matching loss.

    Args:
        x_1: Sample tensor from target, shape [*, n, 3]
        x_t: Interpolated tensor, shape [*, n, 3]
        t: Interpolation times, shape [*]
        mask (optional): Binary mask of shape [*, n]

    Returns:
        dx_t / dt, with shapes [*, n, 3].
    """
    x_1, x_t = map(
        lambda args: _mask_and_zero_com(*args), ((x_1, mask), (x_t, mask))
    )
    # x_t should be masked (interp or sampling), x_1 depdnds on dataloader and pred network
    # x_t should already be centered, x_1 not necessarily (data or pred)

    n = x_1.shape[-2]
    t = _extend_t(n, t)  # [*, n]
    t = t[..., None]  # [*, n, 1]
    x_t_dot = (x_1 - x_t) / (1.0 - t)
    return x_t_dot
    # Masking not necessary since both x_1 and x_t masked

def vf_to_score(
    x_t: Float[Tensor, "* n 3"],
    v: Float[Tensor, "* n 3"],
    t: Float[Tensor, "* n"],
):
    """
    Compute score of noisy density given the vector field learned by flow matching. With
    our interpolation scheme these are related by

    v(x_t, t) = (1 / t) (x_t + scale_ref ** 2 * (1 - t) * s(x_t, t)),

    or equivalently,

    s(x_t, t) = (t * v(x_t, t) - x_t) / (scale_ref ** 2 * (1 - t)).

    Args:
        x_t: Noisy sample, shape [*, dim]
        v: Vector field, shape [*, dim]
        t: Interpolation time, shape [*]

    Returns:
        Score of intermediate density, shape [*, dim].
    """
    assert torch.all(t < 1.0), "vf_to_score requires t < 1 (strict)"
    num = t[..., None] * v - x_t  # [*, n, 3]
    den = (1.0 - t)[..., None]  # [*, n, 1]
    score = num / den
    return score  # [*, dim]

def get_gt(
    t: Float[Tensor, "s"],
    mode: str,
    param: float,
    clamp_val: Optional[float] = None,
    eps: float = 1e-2,
) -> Float[Tensor, "s"]:
    """
    Computes gt for different modes.

    Args:
        t: times where we'll evaluate, covers [0, 1), shape [nsteps]
        mode: "us" or "tan"
        param: parameterized transformation
        clamp_val: value to clamp gt, no clamping if None
        eps: small value leave as it is

    Returns
    """

    # Function to get variants for some gt mode
    def transform_gt(gt, f_pow=1.0):
        # 1.0 means no transformation
        if f_pow == 1.0:
            return gt

        # First we somewhat normalize between 0 and 1
        log_gt = torch.log(gt)
        mean_log_gt = torch.mean(log_gt)
        log_gt_centered = log_gt - mean_log_gt
        normalized = torch.nn.functional.sigmoid(log_gt_centered)
        # Transformation here
        normalized = normalized**f_pow
        # Undo normalization with the transformed variable
        log_gt_centered_rec = torch.logit(normalized, eps=1e-6)
        log_gt_rec = log_gt_centered_rec + mean_log_gt
        gt_rec = torch.exp(log_gt_rec)
        return gt_rec

    # Numerical reasons for some schedule
    t = torch.clamp(t, 0, 1 - 1e-5)

    if mode == "us":
        num = 1.0 - t
        den = t
        gt = num / (den + eps)
    elif mode == "tan":
        num = torch.sin((1.0 - t) * torch.pi / 2.0)
        den = torch.cos((1.0 - t) * torch.pi / 2.0)
        gt = (torch.pi / 2.0) * num / (den + eps)
    elif mode == "1/t":
        num = 1.0
        den = t
        gt = num / (den + eps)
    else:
        raise NotImplementedError(f"gt not implemented {mode}")
    gt = transform_gt(gt, f_pow=param)
    gt = torch.clamp(gt, 0, clamp_val)  # If None no clamping
    return gt  # [s]

def extract_clean_sample(batch):
        """
        Extracts clean sample, mask, batch size, protein length n, and dtype from the batch.
        Applies augmentations if those are required.

        Args:
            batch: batch from dataloader.

        Returns:
            Tuple (x_1, mask, batch_shape, n, dtype)
        """
        x_1 = batch["coords"][:,:,1,:]  # [b, n, 3]
        mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n] boolean
        batch_shape = x_1.shape[:-2]
        n = x_1.shape[-2]
        return (
            ang_to_nm(x_1),
            mask,
            batch_shape,
            n,
            x_1.dtype,
        )
