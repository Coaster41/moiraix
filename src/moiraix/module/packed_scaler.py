#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Optional

import torch
from einops import reduce
from jaxtyping import Bool, Float, Int
from torch import nn

from moiraix.common.torch_util import safe_div


class PackedScaler(nn.Module):
    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"] = None,
        sample_id: Int[torch.Tensor, "*batch seq_len"] = None,
        variate_id: Optional[Int[torch.Tensor, "*batch seq_len"]] = None,
    ):
        if observed_mask is None:
            observed_mask = torch.ones_like(target, dtype=torch.bool)
        if sample_id is None:
            sample_id = torch.zeros(
                target.shape[:-1], dtype=torch.long, device=target.device
            )
        if variate_id is None:
            variate_id = torch.zeros(
                target.shape[:-1], dtype=torch.long, device=target.device
            )

        loc, scale = self._get_loc_scale(
            target.double(), observed_mask, sample_id, variate_id
        )
        return loc.float(), scale.float()

    def _get_loc_scale(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> tuple[
        Float[torch.Tensor, "*batch seq_len #dim"],
        Float[torch.Tensor, "*batch seq_len #dim"],
    ]:
        raise NotImplementedError


class PackedNOPScaler(PackedScaler):
    def _get_loc_scale(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> tuple[
        Float[torch.Tensor, "*batch 1 #dim"], Float[torch.Tensor, "*batch 1 #dim"]
    ]:
        loc = torch.zeros_like(target, dtype=target.dtype)
        scale = torch.ones_like(target, dtype=target.dtype)
        return loc, scale


class PackedStdScaler(PackedScaler):
    def __init__(self, correction: int = 1, minimum_scale: float = 1e-5):
        super().__init__()
        self.correction = correction
        self.minimum_scale = minimum_scale

    def _get_loc_scale(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> tuple[
        Float[torch.Tensor, "*batch 1 #dim"], Float[torch.Tensor, "*batch 1 #dim"]
    ]:
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        tobs = reduce(
            id_mask * reduce(observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loc = reduce(
            id_mask * reduce(target * observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loc = safe_div(loc, tobs)
        var = reduce(
            id_mask
            * reduce(
                ((target - loc) ** 2) * observed_mask,
                "... seq dim -> ... 1 seq",
                "sum",
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        var = safe_div(var, (tobs - self.correction))
        scale = torch.sqrt(var + self.minimum_scale)
        # Out-of-place: an in-place write here mutates the sqrt/div outputs and
        # breaks backward through `target` (input-gradient work, e.g. adversarial
        # attacks). Values are identical.
        pad = (sample_id == 0).unsqueeze(-1)
        loc = torch.where(pad, torch.zeros_like(loc), loc)
        scale = torch.where(pad, torch.ones_like(scale), scale)
        return loc, scale


class PackedCausalStdScaler(PackedScaler):
    """RevIN-causal standardization (TimesFM ``revin_causal`` semantics, adapted
    to packed sequences).

    Unlike :class:`PackedStdScaler`, which produces a single ``(loc, scale)`` per
    ``(sample, variate)`` group, this scaler gives every *position* its own
    running statistics: ``loc[i]`` / ``scale[i]`` are the mean / std over all
    observed patch values of the same ``(sample_id, variate_id)`` group whose
    ``time_id`` is ``<=`` the query position's ``time_id`` (inclusive). This
    mirrors TimesFM's per-patch running RevIN stats, computed here in a single
    ``O(S^2)`` vectorized pass (same complexity as ``PackedStdScaler``) rather
    than a Python loop over positions.

    Variance uses the ``E[x^2] - E[x]^2`` identity (in double precision, matching
    the base scaler) so a single query-dependent ``loc`` need not be materialized
    per key.
    """

    def __init__(self, correction: int = 1, minimum_scale: float = 1e-5):
        super().__init__()
        self.correction = correction
        self.minimum_scale = minimum_scale

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"] = None,
        sample_id: Int[torch.Tensor, "*batch seq_len"] = None,
        time_id: Int[torch.Tensor, "*batch seq_len"] = None,
        variate_id: Optional[Int[torch.Tensor, "*batch seq_len"]] = None,
    ):
        if observed_mask is None:
            observed_mask = torch.ones_like(target, dtype=torch.bool)
        if sample_id is None:
            sample_id = torch.zeros(
                target.shape[:-1], dtype=torch.long, device=target.device
            )
        if variate_id is None:
            variate_id = torch.zeros(
                target.shape[:-1], dtype=torch.long, device=target.device
            )
        if time_id is None:
            # Fall back to positional order along the sequence axis.
            time_id = torch.arange(
                target.shape[-2], device=target.device
            ).expand(target.shape[:-1])

        loc, scale = self._get_loc_scale(
            target.double(), observed_mask, sample_id, variate_id, time_id
        )
        return loc.float(), scale.float()

    def _get_loc_scale(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> tuple[
        Float[torch.Tensor, "*batch seq_len 1"], Float[torch.Tensor, "*batch seq_len 1"]
    ]:
        # [*B, Sq, Sk]: key is in the same group AND causally precedes the query.
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        causal = time_id.unsqueeze(-1) >= time_id.unsqueeze(-2)
        cmask = (id_mask & causal).to(target.dtype)

        obs = observed_mask.to(target.dtype)
        cnt_k = reduce(obs, "... seq dim -> ... 1 seq", "sum")
        sum_k = reduce(target * obs, "... seq dim -> ... 1 seq", "sum")
        sumsq_k = reduce((target**2) * obs, "... seq dim -> ... 1 seq", "sum")

        tobs = reduce(cmask * cnt_k, "... seq1 seq2 -> ... seq1 1", "sum")
        sum_x = reduce(cmask * sum_k, "... seq1 seq2 -> ... seq1 1", "sum")
        sum_x2 = reduce(cmask * sumsq_k, "... seq1 seq2 -> ... seq1 1", "sum")

        loc = safe_div(sum_x, tobs)
        var = safe_div(sum_x2 - tobs * loc**2, tobs - self.correction)
        var = var.clamp(min=0.0)
        scale = torch.sqrt(var + self.minimum_scale)

        # Out-of-place — see PackedStdScaler._get_loc_scale.
        pad = (sample_id == 0).unsqueeze(-1)
        loc = torch.where(pad, torch.zeros_like(loc), loc)
        scale = torch.where(pad, torch.ones_like(scale), scale)
        return loc, scale


class PackedAbsMeanScaler(PackedScaler):
    def _get_loc_scale(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> tuple[
        Float[torch.Tensor, "*batch 1 #dim"], Float[torch.Tensor, "*batch 1 #dim"]
    ]:
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        tobs = reduce(
            id_mask * reduce(observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        scale = reduce(
            id_mask
            * reduce(target.abs() * observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        scale = safe_div(scale, tobs)
        loc = torch.zeros_like(scale)

        # Out-of-place — see PackedStdScaler._get_loc_scale.
        pad = (sample_id == 0).unsqueeze(-1)
        scale = torch.where(pad, torch.ones_like(scale), scale)
        return loc, scale
