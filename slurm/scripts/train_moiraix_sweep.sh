#!/bin/bash
#SBATCH --job-name=moiraix_sweep
#SBATCH --partition=datalab.p
#SBATCH --nodelist=markov-2
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=200gb
#SBATCH --time=24:00:00
# --output is set on the sbatch command line per run.
#
# Parametrized launcher for the 4-run "synth + cpm-mask split" sweep.
# Holds the enc_cpm_2 best recipe fixed (encoder / cpm / npred=1 / patch16 /
# HPO optimizer config / max_mask_ratio=0.2) and varies ONLY:
#   DATA  - which pretrain data config (synthetic-ratio axis)
#   EXTRA - extra model.* overrides (cpm-mask axis)
#
# Usage (4 runs of the split):
#   sbatch --output=slurm/logs/sweep_synth_p10_%j.txt \
#     --export=ALL,RUN_NAME=sweep_synth_p10,DATA=gift_eval_pretrain_moirai2_p10 \
#     slurm/scripts/train_moiraix_sweep.sh
#   sbatch --output=slurm/logs/sweep_synth_p45_%j.txt \
#     --export=ALL,RUN_NAME=sweep_synth_p45,DATA=gift_eval_pretrain_moirai2_p45 \
#     slurm/scripts/train_moiraix_sweep.sh
#   sbatch --output=slurm/logs/sweep_mask_ctxmask_%j.txt \
#     --export=ALL,RUN_NAME=sweep_mask_ctxmask,DATA=gift_eval_pretrain_moirai2,EXTRA='model.cpm_p_mask_max=0.15' \
#     slurm/scripts/train_moiraix_sweep.sh
#   sbatch --output=slurm/logs/sweep_mask_predp40_%j.txt \
#     --export=ALL,RUN_NAME=sweep_mask_predp40,DATA=gift_eval_pretrain_moirai2,EXTRA='model.cpm_pred_p_mask_max=0.4' \
#     slurm/scripts/train_moiraix_sweep.sh

set -euo pipefail
cd /srv/disk00/ctadler/moiraix
PY=/srv/disk00/ctadler/moiraix/venv/bin/python

: "${RUN_NAME:?set RUN_NAME}" "${DATA:?set DATA}"
EXTRA="${EXTRA:-}"

srun echo "Start of ${RUN_NAME} (data=${DATA} extra=${EXTRA})"
srun ${PY} -u -m cli.train \
  -cp conf/pretrain \
  run_name=${RUN_NAME} \
  model=moiraix \
  data=${DATA} \
  val_data=gift_eval_test \
  train_dataloader.batch_size=256 \
  val_dataloader.batch_size=256 \
  train_dataloader.num_workers=6 \
  trainer.gradient_clip_val=2.17524 \
  model.weight_decay=0.127608 \
  model.beta1=0.804699 \
  model.beta2=0.957911 \
  model.lr=0.00015 \
  seed=1 \
  model.module_kwargs.patch_size=16 \
  model.module_kwargs.causal=false \
  model.module_kwargs.mask_inputs=true \
  model.module_kwargs.predict_next=false \
  model.module_kwargs.num_predict_token=1 \
  model.loss_func.shift=false \
  model.min_mask_ratio=0 \
  model.max_mask_ratio=0.2 \
  model.cpm_pred_c_mask_max=5 \
  model.cpm_pred_p_mask_max=0.25 \
  model.val_cpm_pred_p_mask_max=0 \
  model.val_min_mask_ratio=0.3 \
  model.val_max_mask_ratio=0.3 \
  ${EXTRA}
