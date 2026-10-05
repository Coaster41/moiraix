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

from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Bool, Float, Int
from torch import nn

from moiraix.common.torch_util import (
    mask_fill,
    packed_attention_mask,
    packed_causal_attention_mask,
)
from moiraix.module.norm import RMSNorm
from moiraix.module.packed_scaler import (
    PackedCausalStdScaler,
    PackedNOPScaler,
    PackedStdScaler,
)
from moiraix.module.position import (
    BinaryAttentionBias,
    QueryKeyProjection,
    RotaryProjection,
)
from moiraix.module.transformer import TransformerEncoder
from moiraix.module.ts_embed import ResidualBlock


class MoiraiXModule(
    nn.Module,
    PyTorchModelHubMixin,
):
    """
    Universal/extensible Moirai module.

    This single module unifies the formerly-separate ``moiraic`` (causal decoder)
    and ``moiraie`` (bidirectional encoder) variants. Every behavioral axis that
    used to distinguish them is now an orthogonal flag, so arbitrary combinations
    can be trained (e.g. a bidirectional model with multi-token next-token / AR
    training). The flags are serialized to ``config.json`` via
    ``PyTorchModelHubMixin`` so checkpoints are self-describing.

    Behavioral flags:
      - ``causal``: causal vs bidirectional attention.
      - ``mask_inputs``: if True, prediction-window tokens are replaced by a learned
        ``mask_encoding`` embedding before the transformer (BERT-style); if False the
        scaled raw values pass through and only the attention mask prevents leakage
        (GPT-style). Controls whether ``self.mask_encoding`` exists, keeping checkpoint
        weight keys aligned with the old variants.
      - ``predict_next``: target alignment / objective. True ⇒ next-token (the loss
        shifts targets +1, the forecaster reads the token before each slot); False ⇒
        current-token reconstruction. Stored so the loss, validation, and forecaster
        all read one source of truth.

    KV-cache (``past_cache`` / ``return_cache``) is only meaningful with causal
    attention and is driven by the forecaster.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_layers: int,
        patch_size: int,
        max_seq_len: int,
        attn_dropout_p: float,
        dropout_p: float,
        scaling: bool = True,
        num_heads: Optional[int] = None,
        num_predict_token: int = 1,
        quantile_levels: tuple[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_scale: float = 1e-5,
        causal: bool = False,
        mask_inputs: bool = True,
        predict_next: bool = False,
        train_scale_full_observed: bool = False,
        scale_prefix_ratio: Optional[float] = None,
        arcsinh_norm: bool = False,
        revin_causal: bool = False,
        num_spread_token: int = 0,
    ):
        """
        :param d_model: model hidden dimensions
        :param num_layers: number of transformer layers
        :param patch_size: patch size
        :param max_seq_len: maximum sequence length for inputs
        :param attn_dropout_p: dropout probability for attention layers
        :param dropout_p: dropout probability for all other layers
        :param scaling: whether to apply scaling (standardization)
        :param num_heads: number of attention heads. ``None`` defaults to
            ``d_model // 64`` (see ``TransformerEncoder``), i.e. a fixed head_dim
            of 64, with ``num_groups`` following ``num_heads`` (standard MHA).
        :param num_predict_token: number of patches predicted per position
        :param quantile_levels: quantile levels of the output head
        :param min_scale: minimum scale for std scaler
        :param causal: causal vs bidirectional attention
        :param mask_inputs: replace prediction-window tokens with a learned mask embedding
        :param predict_next: next-token (AR) objective vs current-token reconstruction
        :param train_scale_full_observed: if True, compute the scaler loc/scale from
            the *entire* observation mask (context **and** prediction tokens) during
            training only. This intentionally lets the normalization "peek" at the
            prediction window — a hyperparameter for studying its effect. It applies
            only while the module is in training mode (``self.training``); validation,
            inference and forecasting always use the honest context-only statistics.
            Default (False) preserves the original context-only behavior.
        :param scale_prefix_ratio: if set, restricts the scaler's visible context
            further to only the first ``scale_prefix_ratio`` fraction of each
            (sample, variate)'s timeline (by ``time_id``), instead of every
            unmasked context patch scattered across the whole window. Matches the
            Moirai2 paper's instance-norm statistics computed "solely from the
            first 30% of the time series" (see FIX D in
            HANDOFF_MOIRAI2_PARITY.md). ``None`` preserves the original behavior.
        :param arcsinh_norm: if True, apply ``arcsinh`` after z-scoring —
            ``arcsinh((x - loc) / scale)`` — so both the input tokens and the
            loss targets live in arcsinh space (~identity for |z| ≲ 1,
            logarithmic for large |z|; tames the huge scaled targets produced
            when a near-constant context meets a burst in the prediction
            window). Predictions are inverted with ``sinh`` before
            de-normalization at inference, so forecasts stay in data space;
            sinh is monotone, so quantile ordering is preserved. exp-014.
        :param revin_causal: if True, replace the single-``(loc, scale)``-per-row
            standardization with TimesFM-style RevIN-causal statistics: every
            position is normalized by the running mean/std over the observed
            context patches of its own ``(sample, variate)`` group whose
            ``time_id`` is ``<=`` its own (see
            :class:`~moiraix.module.packed_scaler.PackedCausalStdScaler`). Requires
            ``scaling=True``; only affects the prefill/forward path (not the
            causal KV-cache decode path, which asserts against it). Default
            (False) preserves the original ``PackedStdScaler`` behavior.
        :param num_spread_token: if > 0, add a second output head
            (``out_proj_spread``) that predicts ``num_spread_token`` patches of
            quantiles from a single position — typically far more than
            ``num_predict_token``. At inference the AR loop supplies the median
            trajectory and this head supplies the quantile *offsets* around it
            (TimesFM 2.5's ``use_continuous_quantile_head``). ``0`` (default)
            disables the head entirely, so the state dict is byte-identical to
            checkpoints trained without it.
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.num_predict_token = num_predict_token
        self.max_seq_len = max_seq_len
        self.scaling = scaling
        self.num_heads = num_heads
        self.causal = causal
        self.mask_inputs = mask_inputs
        self.predict_next = predict_next
        self.train_scale_full_observed = train_scale_full_observed
        self.scale_prefix_ratio = scale_prefix_ratio
        self.arcsinh_norm = arcsinh_norm
        self.revin_causal = revin_causal
        self.quantile_levels = quantile_levels
        self.num_quantiles = len(quantile_levels)

        # Only create the learned mask embedding when it is actually used, so the
        # state-dict keys match checkpoints trained without input masking.
        if self.mask_inputs:
            self.mask_encoding = nn.Embedding(num_embeddings=1, embedding_dim=d_model)

        if scaling and revin_causal:
            self.scaler = PackedCausalStdScaler(minimum_scale=min_scale)
        elif scaling:
            self.scaler = PackedStdScaler(minimum_scale=min_scale)
        else:
            self.scaler = PackedNOPScaler()
        self.in_proj = ResidualBlock(
            input_dims=patch_size * 2,
            hidden_dims=d_model,
            output_dims=d_model,
        )
        self.encoder = TransformerEncoder(
            d_model,
            num_layers,
            num_heads=num_heads,
            pre_norm=True,
            attn_dropout_p=attn_dropout_p,
            dropout_p=dropout_p,
            norm_layer=RMSNorm,
            activation=F.silu,
            use_glu=True,
            use_qk_norm=True,
            var_attn_bias_layer=partial(BinaryAttentionBias),
            time_qk_proj_layer=partial(
                QueryKeyProjection,
                proj_layer=RotaryProjection,
                kwargs=dict(max_len=max_seq_len),
                partial_factor=(0.0, 0.5),
            ),
            shared_var_attn_bias=False,
            shared_time_qk_proj=True,
            d_ff=d_ff,
        )
        self.out_proj = ResidualBlock(
            input_dims=d_model,
            hidden_dims=d_model,
            output_dims=num_predict_token * self.num_quantiles * patch_size,
        )
        self.num_spread_token = num_spread_token
        # Only create the head when used, so state-dict keys match checkpoints
        # trained without it (same pattern as ``mask_encoding`` above).
        if num_spread_token > 0:
            self.out_proj_spread = ResidualBlock(
                input_dims=d_model,
                hidden_dims=d_model,
                output_dims=num_spread_token * self.num_quantiles * patch_size,
            )
        self.get_reprs = False

    def _prefix_mask(
        self,
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Bool[torch.Tensor, "*batch seq_len"]:
        """True for positions within the first ``scale_prefix_ratio`` of their
        (sample, variate)'s timeline, per ``time_id``."""
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        time_id_bc = time_id.unsqueeze(-2).expand_as(id_mask)
        group_max_time = time_id_bc.masked_fill(~id_mask, -1).amax(dim=-1)
        cutoff = torch.ceil((group_max_time + 1) * self.scale_prefix_ratio).clamp(min=1)
        return time_id < cutoff

    def _embed(
        self,
        target: Float[torch.Tensor, "*batch seq_len patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len patch"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        loc: Float[torch.Tensor, "*batch seq_len 1"],
        scale: Float[torch.Tensor, "*batch seq_len 1"],
    ) -> Float[torch.Tensor, "*batch seq_len d_model"]:
        """Scale, project, and (optionally) mask prediction-window tokens."""
        scaled_target = (target - loc) / scale
        if self.arcsinh_norm:
            scaled_target = torch.asinh(scaled_target)
        input_tokens = torch.cat(
            [scaled_target, observed_mask.to(torch.float32)], dim=-1
        )
        reprs = self.in_proj(input_tokens)
        if self.mask_inputs:
            reprs = mask_fill(reprs, prediction_mask, self.mask_encoding.weight)
        return reprs, scaled_target

    def _spread_head(
        self,
        reprs: Float[torch.Tensor, "*batch seq_len d_model"],
        spread_positions: Optional[Int[torch.Tensor, "*batch num_pos"]],
        loc: Float[torch.Tensor, "*batch seq_len 1"],
        scale: Float[torch.Tensor, "*batch seq_len 1"],
        training_mode: bool,
    ) -> Float[torch.Tensor, "*batch num_pos num_spread_token*num_quantiles*patch"]:
        """Long-horizon quantile head, optionally evaluated at a sparse set of
        positions only.

        ``spread_positions`` gathers rows of ``reprs`` (and the matching
        ``loc``/``scale``) before the projection. This matters: the head's output
        dim is ``num_spread_token * num_quantiles * patch_size``, so materializing
        it at every packed position is an order of magnitude more activation
        memory than the point head. Inference needs one position per variate;
        training samples a handful (see ``PackedQuantileSpreadLoss``).
        """
        assert self.num_spread_token > 0, (
            "return_spread=True requires num_spread_token > 0"
        )
        if spread_positions is not None:
            idx = spread_positions.unsqueeze(-1)  # [*B, K, 1]
            reprs = reprs.gather(-2, idx.expand(*idx.shape[:-1], reprs.shape[-1]))
            loc = loc.gather(-2, idx.expand(*idx.shape[:-1], loc.shape[-1]))
            scale = scale.gather(-2, idx.expand(*idx.shape[:-1], scale.shape[-1]))
        spread = self.out_proj_spread(reprs)
        if not training_mode:
            if self.arcsinh_norm:
                spread = torch.sinh(spread)
            spread = spread * scale + loc
        return spread

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        training_mode: Bool = True,
        past_cache: Optional[dict] = None,
        return_cache: bool = False,
        return_attn_weights: bool = False,
        return_spread: bool = False,
        spread_positions: Optional[Int[torch.Tensor, "*batch num_pos"]] = None,
    ):
        """
        Forward pass of MoiraiXModule (expects processed inputs).

        1. Apply scaling to observations
        2. Project from observations to representations
        3. (optional) replace prediction window with learned mask embedding
        4. Apply transformer layers (causal or bidirectional)
        5. Project from representations to quantile parameters
        """
        # RevIN-causal stats are per-position and cannot be reduced to the
        # per-variate (loc, scale) lookup the KV-cache decode path relies on.
        assert not (self.revin_causal and (return_cache or past_cache is not None)), (
            "revin_causal is not supported with the KV-cache decode path"
        )
        if past_cache is None:
            # ------------------------------------------------------------------
            # Prefill path
            # ------------------------------------------------------------------
            # By default the scaler only sees context (non-prediction) tokens so the
            # normalization cannot leak information about the prediction window. When
            # ``train_scale_full_observed`` is set we instead feed the full observation
            # mask *during training only* (``self.training``), letting loc/scale be
            # computed from prediction tokens too. Validation/inference keep the honest
            # context-only statistics so metrics stay comparable.
            if self.train_scale_full_observed and self.training:
                scaler_observed_mask = observed_mask
            else:
                scaler_observed_mask = observed_mask * ~prediction_mask.unsqueeze(-1)
            if self.scale_prefix_ratio is not None:
                scaler_observed_mask = scaler_observed_mask & self._prefix_mask(
                    sample_id, variate_id, time_id
                ).unsqueeze(-1)
            if self.revin_causal:
                loc, scale = self.scaler(
                    target,
                    scaler_observed_mask,
                    sample_id,
                    time_id,
                    variate_id,
                )
            else:
                loc, scale = self.scaler(
                    target,
                    scaler_observed_mask,
                    sample_id,
                    variate_id,
                )
            reprs, scaled_target = self._embed(
                target, observed_mask, prediction_mask, loc, scale
            )

            attn_mask = (
                packed_causal_attention_mask(sample_id, time_id)
                if self.causal
                else packed_attention_mask(sample_id)
            )
            if return_cache:
                reprs, layer_kvs = self.encoder(
                    reprs,
                    attn_mask,
                    time_id=time_id,
                    var_id=variate_id,
                    return_kvs=True,
                )
            elif return_attn_weights:
                reprs, all_attn_weights = self.encoder(
                    reprs,
                    attn_mask,
                    time_id=time_id,
                    var_id=variate_id,
                    return_attn_weights=True,
                )
            else:
                reprs = self.encoder(
                    reprs,
                    attn_mask,
                    time_id=time_id,
                    var_id=variate_id,
                )

            if self.get_reprs:
                preds = self.out_proj(reprs)
                result = (reprs, torch.cat((loc, scale), dim=-1))
            else:
                preds = self.out_proj(reprs)
                if training_mode:
                    result = (preds, scaled_target)
                else:
                    if self.arcsinh_norm:
                        preds = torch.sinh(preds)
                    result = preds * scale + loc

            spread = (
                self._spread_head(reprs, spread_positions, loc, scale, training_mode)
                if return_spread
                else None
            )

            if return_cache:
                cache = self._build_context_cache(
                    layer_kvs=layer_kvs,
                    sample_id=sample_id,
                    time_id=time_id,
                    variate_id=variate_id,
                    prediction_mask=prediction_mask,
                    loc=loc,
                    scale=scale,
                )
            outs = [result]
            if return_spread:
                outs.append(spread)
            if return_cache:
                outs.append(cache)
            if return_attn_weights:
                outs.append(all_attn_weights)
            return outs[0] if len(outs) == 1 else tuple(outs)

        # ----------------------------------------------------------------------
        # Decode path (causal + KV cache) — inputs cover only the NEW tokens
        # (caller slices). Only reachable when ``causal`` is True.
        # ----------------------------------------------------------------------
        # 1) Look up loc/scale per new token using its variate_id.
        var_idx = variate_id.unsqueeze(-1)  # [..., new_len, 1]
        new_loc = past_cache["variate_loc"].gather(-2, var_idx)  # [..., new_len, 1]
        new_scale = past_cache["variate_scale"].gather(-2, var_idx)  # [..., new_len, 1]

        # 2) Scale + project (+ optional mask) new tokens.
        reprs, scaled_target = self._embed(
            target, observed_mask, prediction_mask, new_loc, new_scale
        )

        # 3) Build attention mask spanning [cached_ctx ⊕ new] keys.
        full_sample_id = torch.cat([past_cache["kv_sample_id"], sample_id], dim=-1)
        full_time_id = torch.cat([past_cache["kv_time_id"], time_id], dim=-1)
        new_len = target.shape[-2]
        decode_attn_mask = packed_causal_attention_mask(full_sample_id, full_time_id)[
            ..., -new_len:, :
        ]

        # 4) Encoder over new tokens with cached context K/V.
        reprs = self.encoder(
            reprs,
            decode_attn_mask,
            time_id=time_id,
            var_id=variate_id,
            past_kvs=past_cache["layer_kvs"],
            past_kv_var_id=past_cache["kv_var_id"],
            past_kv_time_id=past_cache["kv_time_id"],
            return_kvs=False,  # context cache is invariant; nothing to update
        )

        # 5) Output projection on new tokens only.
        if self.get_reprs:
            preds = self.out_proj(reprs)
            result = (reprs, torch.cat((new_loc, new_scale), dim=-1))
        else:
            preds = self.out_proj(reprs)
            if training_mode:
                result = (preds, scaled_target)
            else:
                if self.arcsinh_norm:
                    preds = torch.sinh(preds)
                result = preds * new_scale + new_loc

        # Note: ``spread_positions`` here would be relative to the new-token
        # slice, not the full sequence. The forecaster never requests spread on
        # a decode pass (spread comes from prefill only) — kept symmetric.
        spread = (
            self._spread_head(
                reprs, spread_positions, new_loc, new_scale, training_mode
            )
            if return_spread
            else None
        )

        outs = [result]
        if return_spread:
            outs.append(spread)
        if return_cache:
            # Pass through the same context cache for chained AR calls.
            outs.append(past_cache)
        return outs[0] if len(outs) == 1 else tuple(outs)

    def _build_context_cache(
        self,
        layer_kvs,
        sample_id,
        time_id,
        variate_id,
        prediction_mask,
        loc,
        scale,
    ) -> dict:
        """
        Filter the prefill cache down to context positions only and build
        a per-variate (loc, scale) lookup so that decode can rescale new
        tokens without re-running the scaler.

        Assumes the forecast layout: context positions form a contiguous
        prefix, prediction positions a contiguous suffix.
        """
        ctx_mask = ~prediction_mask  # [*B, S]; True where token is context
        ctx_lens = ctx_mask.sum(dim=-1)
        ctx_len = int(ctx_lens.flatten()[0].item())
        assert (ctx_lens == ctx_len).all(), (
            "MoiraiXModule cache assumes a uniform, contiguous context prefix; "
            f"got per-row context lengths {ctx_lens.tolist()}"
        )

        # Slice K/V along seq dim. layer_kvs[i] = (k, v) of shape
        # [..., group, hpg, S, head_dim]; we keep [..., :ctx_len, :].
        ctx_layer_kvs = [
            (k[..., :ctx_len, :].contiguous(), v[..., :ctx_len, :].contiguous())
            for (k, v) in layer_kvs
        ]
        ctx_var_id = variate_id[..., :ctx_len]
        ctx_time_id = time_id[..., :ctx_len]
        ctx_sample_id = sample_id[..., :ctx_len]

        # Build per-variate loc/scale lookup tables: shape [*B, num_variates, 1].
        num_variates = int(variate_id.max().item()) + 1
        batch_shape = variate_id.shape[:-1]
        variate_loc = torch.zeros(
            *batch_shape, num_variates, 1, dtype=loc.dtype, device=loc.device
        )
        variate_scale = torch.ones(
            *batch_shape, num_variates, 1, dtype=scale.dtype, device=scale.device
        )

        for v in range(num_variates):
            is_pred_v = prediction_mask & (variate_id == v)  # [*B, S]
            if not is_pred_v.any():
                continue
            first_v = is_pred_v & (
                is_pred_v.int().cumsum(-1) == 1
            )  # first True per row
            b_idx, s_idx = first_v.nonzero(as_tuple=True)
            variate_loc[b_idx, v] = loc[b_idx, s_idx]
            variate_scale[b_idx, v] = scale[b_idx, s_idx]

        return {
            "layer_kvs": ctx_layer_kvs,
            "kv_var_id": ctx_var_id,
            "kv_time_id": ctx_time_id,
            "kv_sample_id": ctx_sample_id,
            "variate_loc": variate_loc,
            "variate_scale": variate_scale,
            "ctx_len": ctx_len,
        }
