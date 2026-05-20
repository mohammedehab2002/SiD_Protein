# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Any, List, Tuple

import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class ProteinFoldScore(Metric):

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    features: List

    def __init__(
        self,
        splits: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        Calculate Fold Score which is used to access the quality of protein structures.

        Args:
            splits (int): Integer determining how many splits the fold score calculation should be split among.
                Defaults to 10.

        """
        super().__init__(**kwargs)

        rank_zero_warn(
            "Metric `FoldScore` will save all extracted features in buffer."
            " For large datasets this may lead to large memory footprint.",
            UserWarning,
        )

        self.splits = splits
        self.add_state("features", [], dist_reduce_fx=None)

    def update(self, features: Tensor) -> None:
        """Update the state with extracted features."""
        self.features.append(features)

    def compute(self) -> Tuple[Tensor, Tensor]:
        """Compute metric."""
        features = dim_zero_cat(self.features)
        idx = torch.randperm(features.shape[0])
        features = features[idx]

        prob = features.softmax(dim=1)
        log_prob = features.log_softmax(dim=1)

        prob = prob.chunk(self.splits, dim=0)
        log_prob = log_prob.chunk(self.splits, dim=0)

        mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
        kl_ = [
            p * (log_p - (m_p + 1e-10).log())
            for p, log_p, m_p in zip(prob, log_prob, mean_prob)
        ]
        kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
        kl = torch.stack(kl_)

        return kl.mean()
