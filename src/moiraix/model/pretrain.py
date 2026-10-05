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

import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Optional

import lightning as L
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import nn

from moiraix.loss.packed import (
    PackedLoss,
    PackedQuantileLoss,
    PackedQuantileMTPLoss,
    PackedQuantileSpreadLoss,
)
from moiraix.module.norm import RMSNorm
from moiraix.module.position import (
    BinaryAttentionBias,
    LearnedEmbedding,
    LearnedProjection,
)
from moiraix.module.ts_embed import ResidualBlock
from moiraix.optim import SchedulerType, get_scheduler
from moiraix.transform import (
    AddObservedMask,
    AddTimeIndex,
    AddVariateIndex,
    CausalMeanImputation,
    ContiguousPatchMasking,
    ContiguousPatchPrediction,
    ContextPatchMasking,
    DefaultPatchSizeConstraints,
    ExtendMask,
    FlatPackCollection,
    FlatPackFields,
    GetPatchSize,
    Identity,
    ImputeTimeSeries,
    MaskedPrediction,
    PackFields,
    PatchCrop,
    Patchify,
    SampleDimension,
    SelectFields,
    Transformation,
    ZScoreSampleFilter,
)

from .module import MoiraiXModule


class MoiraiXPretrain(L.LightningModule):
    #: Module class instantiated from ``module_kwargs``. Preset shims override this.
    module_class: type = MoiraiXModule

    seq_fields: tuple[str, ...] = (
        "target",
        "observed_mask",
        "time_id",
        "variate_id",
        "prediction_mask",
    )
    pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]] = {
        "target": np.zeros,
        "observed_mask": np.zeros,
        "time_id": np.zeros,
        "variate_id": np.zeros,
        "prediction_mask": np.zeros,
    }

    def __init__(
        self,
        min_patches: int,
        min_mask_ratio: float,
        max_mask_ratio: float,
        max_dim: int,
        num_training_steps: int,
        num_warmup_steps: int,
        module_kwargs: Optional[dict[str, Any]] = None,
        module: Optional[MoiraiXModule] = None,
        patch_mask_ratio: float = 0.0,
        cpm_c_mask_max: int = 4,
        cpm_p_mask_max: float = 0.0,
        cpm_pred_c_mask_max: int = 4,
        cpm_pred_p_mask_max: float = 0.0,
        cpm_pred_p_fixed: Optional[float] = None,
        cpm_pred_protect_prefix_ratio: float = 0.0,
        zscore_filter_prefix_ratio: Optional[float] = None,
        zscore_filter_threshold: float = 6.0,
        zscore_filter_max_frac_exceed: float = 0.01,
        num_samples: int = 100,
        beta1: float = 0.9,
        beta2: float = 0.98,
        loss_func: PackedQuantileLoss = PackedQuantileMTPLoss(),
        val_metric: Optional[PackedLoss | list[PackedLoss]] = None,
        val_min_mask_ratio: Optional[float] = None,
        val_max_mask_ratio: Optional[float] = None,
        val_cpm_c_mask_max: Optional[int] = None,
        val_cpm_p_mask_max: Optional[float] = None,
        val_cpm_pred_c_mask_max: Optional[int] = None,
        val_cpm_pred_p_mask_max: Optional[float] = None,
        val_cpm_pred_p_fixed: Optional[float] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        log_on_step: bool = False,
        spread_loss_func: Optional[PackedQuantileSpreadLoss] = None,
        spread_loss_weight: float = 1.0,
        num_spread_positions: int = 8,
        freeze_pattern: Optional[str] = None,
        freeze_unlock_last_n_layers: int = 0,
        train_spread_only: bool = False,
        robust_epoch_trim_frac: float = 0.01,
    ):
        assert (module is not None) or (
            module_kwargs is not None
        ), "if module is not provided, module_kwargs is required"
        assert (
            num_warmup_steps <= num_training_steps
        ), f"num_warmup_steps ({num_warmup_steps}) should be <= num_training_steps ({num_training_steps})."
        super().__init__()
        self.save_hyperparameters(ignore=["module"])
        self.module = self.module_class(**module_kwargs) if module is None else module

        # The loss objective (shift) must agree with the model objective
        # (predict_next): the module is the single source of truth.
        loss_func = self.hparams.loss_func
        if hasattr(loss_func, "shift") and loss_func.shift != self.module.predict_next:
            warnings.warn(
                f"loss_func.shift={loss_func.shift} disagrees with "
                f"module.predict_next={self.module.predict_next}; coercing loss "
                f"shift to match the model objective."
            )
            loss_func.shift = self.module.predict_next

        if self.hparams.spread_loss_func is not None:
            assert self.module.num_spread_token > 0, (
                "spread_loss_func given but module.num_spread_token == 0"
            )
            slf = self.hparams.spread_loss_func
            if getattr(slf, "shift", None) != self.module.predict_next:
                warnings.warn(
                    f"spread_loss_func.shift={slf.shift} disagrees with "
                    f"module.predict_next={self.module.predict_next}; coercing."
                )
                slf.shift = self.module.predict_next
        self.use_spread = self.hparams.spread_loss_func is not None

        self._apply_freeze()

        # Per-epoch scalar buffers for the robust epoch aggregates. Values are
        # kept as detached device tensors and stacked once at epoch end -- do
        # NOT call .item() per step, which would force a host sync every batch.
        self._robust_train: list[torch.Tensor] = []
        self._robust_val: dict[str, list[torch.Tensor]] = {}

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        training_mode: Bool = True,
        spread_positions: Optional[Int[torch.Tensor, "*batch num_pos"]] = None,
    ):
        if spread_positions is None:
            # unchanged path -- identical 2-tuple return as today
            preds, scaled_target = self.module(
                target=target,
                observed_mask=observed_mask,
                sample_id=sample_id,
                time_id=time_id,
                variate_id=variate_id,
                prediction_mask=prediction_mask,
                training_mode=training_mode,
            )
            return preds, scaled_target
        (preds, scaled_target), spread = self.module(
            target=target,
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id,
            prediction_mask=prediction_mask,
            training_mode=training_mode,
            return_spread=True,
            spread_positions=spread_positions,
        )
        return preds, scaled_target, spread

    def _sample_spread_positions(
        self,
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        num_positions: int,
    ) -> Int[torch.Tensor, "*batch num_positions"]:
        """Positions whose first predicted patch lands on a prediction token --
        the same geometry the forecaster's anchor has."""
        nst = self.module.num_spread_token
        off = 1 if self.module.predict_next else 0
        S = prediction_mask.shape[-1]
        W = S - nst + 1  # number of valid windows
        assert W > 0, (
            f"num_spread_token={nst} exceeds max_seq_len={S}; the spread head "
            "can never see a full target window."
        )
        # eligible anchors t in [0, W): prediction_mask[t + off] is True
        elig = prediction_mask[..., off : off + W].float() + 1e-9
        flat = elig.reshape(-1, W)
        pos = torch.multinomial(flat, num_positions, replacement=True)
        return pos.reshape(*prediction_mask.shape[:-1], num_positions)

    def _apply_freeze(self) -> None:
        """Freeze everything except the spread head (and optionally the last N
        transformer layers). ``configure_optimizers`` already drops frozen
        params from both the decay and no-decay groups, so nothing else needs
        to change."""
        pattern = self.hparams.freeze_pattern
        if pattern is None:
            return
        if pattern != "spread_head":
            raise ValueError(f"unknown freeze_pattern: {pattern}")
        assert self.module.num_spread_token > 0, (
            "freeze_pattern='spread_head' requires num_spread_token > 0"
        )

        for p in self.module.parameters():
            p.requires_grad = False
        for p in self.module.out_proj_spread.parameters():
            p.requires_grad = True

        n = self.hparams.freeze_unlock_last_n_layers
        if n > 0:
            for layer in self.module.encoder.layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"[freeze] {pattern}: {trainable:,}/{total:,} params trainable "
            f"({100 * trainable / total:.1f}%)"
        )

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        if self.use_spread:
            spread_positions = self._sample_spread_positions(
                batch["prediction_mask"], self.hparams.num_spread_positions
            )
            preds, scaled_target, spread = self(
                **{f: batch[f] for f in list(self.seq_fields) + ["sample_id"]},
                spread_positions=spread_positions,
            )
        else:
            spread_positions, spread = None, None
            preds, scaled_target = self(
                **{f: batch[f] for f in list(self.seq_fields) + ["sample_id"]}
            )
        loss = self.hparams.loss_func(
            pred=preds,
            target=scaled_target,
            **{
                field: batch[field]
                for field in [
                    "prediction_mask",
                    "observed_mask",
                    "sample_id",
                    "variate_id",
                ]
            },
        )
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )

        if self.use_spread:
            spread_loss = self.hparams.spread_loss_func(
                pred=spread,
                target=scaled_target,
                positions=spread_positions,
                **{
                    field: batch[field]
                    for field in [
                        "prediction_mask",
                        "observed_mask",
                        "sample_id",
                        "variate_id",
                    ]
                },
            )
            self.log(
                "train/spread_loss",
                spread_loss,
                on_step=self.hparams.log_on_step,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
                rank_zero_only=True,
            )
            if self.hparams.train_spread_only:
                self.log(
                    "train/point_loss_frozen",
                    loss.detach(),
                    on_step=self.hparams.log_on_step,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                    batch_size=batch_size,
                    rank_zero_only=True,
                )
                loss = self.hparams.spread_loss_weight * spread_loss
            else:
                loss = loss + self.hparams.spread_loss_weight * spread_loss

        self.log(
            f"train/{self.hparams.loss_func.__class__.__name__}",
            loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self._robust_train.append(loss.detach())
        return loss

    def _extract_next_step_forecast(
        self,
        preds: Float[
            torch.Tensor, "*batch seq_len num_predict_token*num_quantiles*patch_size"
        ],
        scaled_target: Float[torch.Tensor, "*batch seq_len patch_size"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len patch_size"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> dict[str, torch.Tensor]:
        """Extract a one-step median forecast aligned with the target.

        For ``predict_next`` models position t's prediction targets position t+1, so
        the first predict-token is sliced and target/masks are shifted by one. For
        current-token (reconstruction) models position t targets position t, so no
        shift is applied.
        """
        num_predict_token = self.module.num_predict_token
        num_quantiles = self.module.num_quantiles
        patch_size = self.module.patch_size

        pred = rearrange(
            preds,
            "... seq_len (predict_token num_quantiles patch_size) "
            "-> ... seq_len predict_token num_quantiles patch_size",
            predict_token=num_predict_token,
            num_quantiles=num_quantiles,
            patch_size=patch_size,
        )
        median_idx = num_quantiles // 2

        if self.module.predict_next:
            # position t's prediction → target at position t+1
            pred = pred[..., :-1, 0, median_idx, :]
            return {
                "pred": pred,
                "target": scaled_target[..., 1:, :],
                "prediction_mask": prediction_mask[..., 1:],
                "observed_mask": observed_mask[..., 1:, :],
                "sample_id": sample_id[..., 1:],
                "variate_id": variate_id[..., 1:],
            }
        # position t's prediction → target at position t
        pred = pred[..., 0, median_idx, :]
        return {
            "pred": pred,
            "target": scaled_target,
            "prediction_mask": prediction_mask,
            "observed_mask": observed_mask,
            "sample_id": sample_id,
            "variate_id": variate_id,
        }

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        if self.use_spread:
            spread_positions = self._sample_spread_positions(
                batch["prediction_mask"], self.hparams.num_spread_positions
            )
            preds, scaled_target, spread = self(
                training_mode=True,
                **{
                    field: batch[field]
                    for field in list(self.seq_fields) + ["sample_id"]
                },
                spread_positions=spread_positions,
            )
        else:
            preds, scaled_target = self(
                training_mode=True,
                **{
                    field: batch[field]
                    for field in list(self.seq_fields) + ["sample_id"]
                },
            )

        val_loss = self.hparams.loss_func(
            pred=preds,
            target=scaled_target,
            **{
                field: batch[field]
                for field in [
                    "prediction_mask",
                    "observed_mask",
                    "sample_id",
                    "variate_id",
                ]
            },
        )

        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )

        if self.use_spread:
            spread_loss = self.hparams.spread_loss_func(
                pred=spread,
                target=scaled_target,
                positions=spread_positions,
                **{
                    field: batch[field]
                    for field in [
                        "prediction_mask",
                        "observed_mask",
                        "sample_id",
                        "variate_id",
                    ]
                },
            )
            self.log(
                "val/spread_loss",
                spread_loss,
                on_step=self.hparams.log_on_step,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
                rank_zero_only=True,
            )

        self._robust_val.setdefault(
            f"val/{self.hparams.loss_func.__class__.__name__}", []
        ).append(val_loss.detach())

        self.log(
            f"val/{self.hparams.loss_func.__class__.__name__}",
            val_loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )

        if self.hparams.val_metric is not None:
            aligned = self._extract_next_step_forecast(
                preds,
                scaled_target,
                batch["prediction_mask"],
                batch["observed_mask"],
                batch["sample_id"],
                batch["variate_id"],
            )

            val_metrics = (
                self.hparams.val_metric
                if isinstance(self.hparams.val_metric, list)
                else [self.hparams.val_metric]
            )
            for metric_func in val_metrics:
                metric = metric_func(**aligned)
                self._robust_val.setdefault(
                    f"val/{metric_func.__class__.__name__}", []
                ).append(metric.detach())
                self.log(
                    f"val/{metric_func.__class__.__name__}",
                    metric,
                    on_step=self.hparams.log_on_step,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                    batch_size=batch_size,
                    rank_zero_only=True,
                )

        return val_loss

    def _log_robust_epoch(self, buffers: dict[str, list[torch.Tensor]]) -> None:
        """Log a median and an upper-trimmed mean for each buffered metric.

        The batch mean that ``self.log(on_epoch=True)`` produces is a faithful
        estimate of the objective but is close to useless for monitoring,
        because the scaled targets are heavy-tailed: ``scale`` floors at
        ``sqrt(min_scale)``, so a near-constant context followed by a burst in
        the prediction window yields a normalized target of order 1e4-1e5 and a
        matching quantile loss. Measured on a 30M run, ~0.9% of batches carried
        ~90% of the epoch mean (median 0.36, mean 20.1, max 2.4e5), leaving the
        epoch curve trendless while the model trained normally. ``arcsinh_norm``
        fixes this at the source; these aggregates make the curve readable for
        runs that do not use it.

        Trimming is one-sided because the tail is: the loss is non-negative and
        only the upper end is pathological. The median is the reliable one at
        any buffer size; the trimmed mean needs at least
        ``1 / robust_epoch_trim_frac`` batches to drop anything at all, so on a
        short validation loader it degenerates to the plain mean and is
        skipped. Aggregation is per-rank, matching the convention the existing
        ``rank_zero_only=True`` logs already use.
        """
        frac = self.hparams.robust_epoch_trim_frac
        for name, values in buffers.items():
            if not values:
                continue
            stacked = torch.stack([v.float().reshape(()) for v in values])
            n = stacked.numel()
            stats = {"median": stacked.median()}
            n_drop = int(n * frac) if frac > 0 else 0
            if n_drop > 0:
                stats["trimmed"] = torch.sort(stacked).values[: n - n_drop].mean()
            for suffix, value in stats.items():
                self.log(
                    f"{name}_epoch_{suffix}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                    batch_size=1,
                    rank_zero_only=True,
                )

    def on_train_epoch_end(self) -> None:
        self._log_robust_epoch(
            {f"train/{self.hparams.loss_func.__class__.__name__}": self._robust_train}
        )
        self._robust_train.clear()

    def on_validation_epoch_end(self) -> None:
        # Fires once per validation pass, so with val_check_interval < 1 epoch
        # each pass gets its own aggregate rather than one blended over the
        # training epoch.
        self._log_robust_epoch(self._robust_val)
        self._robust_val.clear()

    def configure_optimizers(self) -> dict:
        decay = set()
        no_decay = set()

        whitelist_params = (
            LearnedProjection,
            nn.Linear,
            ResidualBlock,
        )
        blacklist_params = (
            BinaryAttentionBias,
            LearnedEmbedding,
            RMSNorm,
            nn.Embedding,
            nn.LayerNorm,
        )

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue

                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_params):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_params):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        optim_groups = [
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(decay))],
                ),
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(no_decay))],
                ),
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=1e-6,
        )
        scheduler = get_scheduler(
            SchedulerType.COSINE_WITH_RESTARTS,
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step",
            },
        }

    def _build_transform(
        self, min_mask_ratio: float, max_mask_ratio: float,
            c_pred_mask_max: float, p_pred_mask_max: float,
            c_mask_max: float, p_mask_max: float,
            p_pred_fixed: Optional[float] = None,
            p_pred_protect_prefix_ratio: float = 0.0,
    ) -> Transformation:
        """Build the patch / mask / pack pipeline used for both train and val."""
        return (
            SampleDimension(
                max_dim=self.hparams.max_dim,
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
            )
            + GetPatchSize(
                min_time_patches=self.hparams.min_patches,
                target_field="target",
                patch_sizes=self.module.patch_size,
                patch_size_constraints=DefaultPatchSizeConstraints(),
                offset=True,
            )
            + PatchCrop(
                min_time_patches=self.hparams.min_patches,
                max_patches=self.module.max_seq_len,
                will_flatten=True,
                offset=True,
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
            )
            + PackFields(
                output_field="target",
                fields=("target",),
                feat=False,
            )
            + (
                ZScoreSampleFilter(
                    prefix_ratio=self.hparams.zscore_filter_prefix_ratio,
                    z_threshold=self.hparams.zscore_filter_threshold,
                    max_frac_exceed=self.hparams.zscore_filter_max_frac_exceed,
                )
                if self.hparams.zscore_filter_prefix_ratio is not None
                else Identity()
            )
            + PackFields(
                output_field="past_feat_dynamic_real",
                fields=tuple(),
                optional_fields=("past_feat_dynamic_real",),
                feat=False,
            )
            + AddObservedMask(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                observed_mask_field="observed_mask",
                collection_type=dict,
            )
            + ImputeTimeSeries(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                imputation_method=CausalMeanImputation(),
            )
            + Patchify(
                max_patch_size=self.module.patch_size,
                fields=("target", "observed_mask"),
                optional_fields=("past_feat_dynamic_real",),
            )
            + AddVariateIndex(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                variate_id_field="variate_id",
                expected_ndim=3,
                max_dim=self.hparams.max_dim,
                randomize=True,
                collection_type=dict,
            )
            + AddTimeIndex(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                time_id_field="time_id",
                expected_ndim=3,
                collection_type=dict,
            )
            + MaskedPrediction(
                min_mask_ratio=min_mask_ratio,
                max_mask_ratio=max_mask_ratio,
                target_field="target",
                truncate_fields=("variate_id", "time_id", "observed_mask"),
                optional_truncate_fields=("past_feat_dynamic_real",),
                prediction_mask_field="prediction_mask",
                expected_ndim=3,
            )
            + ContiguousPatchPrediction(
                c_mask_max=c_pred_mask_max,
                p_mask_max=p_pred_mask_max,
                p_fixed=p_pred_fixed,
                protect_prefix_ratio=p_pred_protect_prefix_ratio,
            )
            + ContiguousPatchMasking(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                c_mask_max=c_mask_max,
                p_mask_max=p_mask_max,
            )
            + ExtendMask(
                fields=tuple(),
                optional_fields=("past_feat_dynamic_real",),
                mask_field="prediction_mask",
                expected_ndim=3,
            )
            + FlatPackCollection(field="variate_id", feat=False)
            + FlatPackCollection(field="time_id", feat=False)
            + FlatPackCollection(field="prediction_mask", feat=False)
            + FlatPackCollection(field="observed_mask", feat=True)
            + FlatPackFields(
                output_field="target",
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                feat=True,
            )
            + SelectFields(fields=list(self.seq_fields))
        )

    @property
    def train_transform_map(self) -> dict[str, Callable[..., Transformation]]:
        def default_train_transform():
            return self._build_transform(
                self.hparams.min_mask_ratio, self.hparams.max_mask_ratio,
                self.hparams.cpm_pred_c_mask_max, self.hparams.cpm_pred_p_mask_max,
                self.hparams.cpm_c_mask_max, self.hparams.cpm_p_mask_max,
                p_pred_fixed=self.hparams.cpm_pred_p_fixed,
                p_pred_protect_prefix_ratio=self.hparams.cpm_pred_protect_prefix_ratio,
            )

        return defaultdict(lambda: default_train_transform)

    @property
    def val_transform_map(self) -> dict[str, Callable[..., Transformation]]:
        def default_val_transform():
            return self._build_transform(
                min_mask_ratio=self.hparams.val_min_mask_ratio if self.hparams.val_min_mask_ratio is not None else self.hparams.min_mask_ratio,
                max_mask_ratio=self.hparams.val_max_mask_ratio if self.hparams.val_max_mask_ratio is not None else self.hparams.max_mask_ratio,
                c_pred_mask_max=self.hparams.val_cpm_pred_c_mask_max if self.hparams.val_cpm_pred_c_mask_max is not None else self.hparams.cpm_pred_c_mask_max,
                p_pred_mask_max=self.hparams.val_cpm_pred_p_mask_max if self.hparams.val_cpm_pred_p_mask_max is not None else self.hparams.cpm_pred_p_mask_max,
                c_mask_max=self.hparams.val_cpm_c_mask_max if self.hparams.val_cpm_c_mask_max is not None else self.hparams.cpm_c_mask_max,
                p_mask_max=self.hparams.val_cpm_p_mask_max if self.hparams.val_cpm_p_mask_max is not None else self.hparams.cpm_p_mask_max,
                p_pred_fixed=self.hparams.val_cpm_pred_p_fixed if self.hparams.val_cpm_pred_p_fixed is not None else self.hparams.cpm_pred_p_fixed,
            )

        return defaultdict(lambda: default_val_transform)
