#!/bin/bash
#SBATCH --job-name=moiraix_decoder
#SBATCH --partition=datalab.p
#SBATCH --nodelist=markov-2
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=200gb
#SBATCH --time=36:00:00
# --output is set on the sbatch command line per run.
#
# Causal-NTP (decoder / GPT-style) retrain launcher — the moiraic_base_11 twin
# protocol. `model=moiraix` defaults ARE the old moiraic recipe (causal=true,
# mask_inputs=false, predict_next=true, npt4, mask 0.6-0.75, lr 1.5e-4,
# wd 0.05, betas 0.9/0.95, clip 1.0, 100 ep). Only DATA varies vs base_11;
# 4 GPUs x bs256 x accum2 = effective 2048 (matches base_11's 8x256x1).
#
# Usage:
#   sbatch --output=slurm/logs/moiraix_dec_ntp_gift_%j.txt \
#     --export=ALL,RUN_NAME=moiraix_dec_ntp_gift_0,DATA=gift_eval_pretrain_trackb_p30_nosde_gift \
#     slurm/scripts/train_moiraix_decoder.sh

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
  trainer.accumulate_grad_batches=2 \
  seed=1 \
  ${EXTRA}
