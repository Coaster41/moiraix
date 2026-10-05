#!/bin/bash
#SBATCH --job-name=moiraix_30m_scale_v1
#SBATCH --output=/srv/disk00/ctadler/moiraix/slurm/logs/moiraix_30m_scale_v1_%j.txt
#SBATCH --partition=datalab.p
#SBATCH --nodelist=markov-2
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=200gb
#SBATCH --time=2-00:00:00
set -euo pipefail
cd /srv/disk00/ctadler/moiraix
PY=/srv/disk00/ctadler/moiraix/venv/bin/python

# ---------------------------------------------------------------------------
# Track C arm 1 of 2: the 30M CONTROL, run FIRST and alone on all 8 GPUs.
#
# Recipe = `moiraix_enc_cpm_arcsinh` EXACTLY (the in-house best, WQL 0.1110,
# short-55), replayed from that run's .hydra/overrides.yaml. The ONLY change is
# `data=gift_eval_pretrain_scale_v1`. So:
#
#   this run  vs  moiraix_enc_cpm_arcsinh (0.1110)   = the data delta, clean
#   400M arm  vs  this run                           = the scale delta, clean
#
# That second comparison is the actual question. The previous 400M attempt
# (moiraix_enc_cpm_gift_300m_0) tied the 30M model on the OLD mix; if scale
# still buys nothing once the corpus is unthrottled, the plateau is not about
# data at all.
#
# Batch geometry: the arcsinh reference ran 256 x accum2 x 4 GPUs = 2048 global.
# Here 128 x accum2 x 8 GPUs = 2048 global, and 1000 opt-steps/epoch either way,
# so tokens/epoch is IDENTICAL to the reference and to the 400M arm. Do not
# "simplify" this to bs=256 on 8 ranks -- that would double the global batch and
# silently change the effective learning rate.
# ---------------------------------------------------------------------------
RUN_NAME="${RUN_NAME:-moiraix_enc_cpm_scale_v1_0}"
DATA="${DATA:-gift_eval_pretrain_scale_v1}"
EPOCHS="${EPOCHS:-100}"
export RUN_NAME DATA EPOCHS

CKPT_DIR="outputs/pretrain/moiraix/${DATA}/${RUN_NAME}/checkpoints"
CKPT_ARG=""
if [ -f "${CKPT_DIR}/last.ckpt" ]; then
  CKPT_ARG="ckpt_path=last"
  echo "Resuming from ${CKPT_DIR}/last.ckpt"
else
  echo "No checkpoint in ${CKPT_DIR} -- starting fresh"
fi
export CKPT_ARG

srun echo "Start of ${RUN_NAME} (data=${DATA}, ${EPOCHS} epochs)"
srun bash -c '
  exec ${PY} -u -m cli.train \
  -cp conf/pretrain \
  run_name=${RUN_NAME} \
  model=moiraix \
  data=${DATA} \
  val_data=gift_eval_test \
  ${CKPT_ARG} \
  trainer.max_epochs=${EPOCHS} \
  train_dataloader.batch_size=128 \
  val_dataloader.batch_size=128 \
  train_dataloader.num_workers=6 \
  trainer.accumulate_grad_batches=2 \
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
  +model.module_kwargs.arcsinh_norm=true \
  model.loss_func.shift=false \
  model.min_mask_ratio=0 \
  model.max_mask_ratio=0.2 \
  model.cpm_pred_c_mask_max=5 \
  model.cpm_pred_p_mask_max=0.25 \
  model.val_cpm_pred_p_mask_max=0 \
  model.val_min_mask_ratio=0.3 \
  model.val_max_mask_ratio=0.3
'
