#!/bin/bash
#SBATCH --job-name=moiraix_ctrl_seed
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
# Training-variance replicate of the CONTROL recipe (moiraix_dec_ntp_gift50_0):
# model=moiraix defaults (causal-NTP, npt4, mask 0.6-0.75, lr 1.5e-4, wd 0.05,
# betas 0.9/0.95, clip 1.0), 50ep, warmup 2500, 4xbs256xaccum2. ONLY the seed
# varies vs the seed=1 run — a true replicate for the seed-variance estimate.
#
# Usage:
#   sbatch --output=slurm/logs/moiraix_ctrl_%j.txt \
#     --export=ALL,RUN_NAME=moiraix_dec_ntp_gift50_s2,SEED=2 \
#     slurm/scripts/train_moiraix_control_seed.sh

set -euo pipefail
cd /srv/disk00/ctadler/moiraix

: "${RUN_NAME:?set RUN_NAME}" "${SEED:?set SEED}"
DATA="${DATA:-gift_eval_pretrain_trackb_p30_nosde_gift}"
PY=/srv/disk00/ctadler/moiraix/venv/bin/python

srun echo "Start of ${RUN_NAME} (control replicate, data=${DATA} seed=${SEED})"
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
  seed=${SEED} \
  trainer.max_epochs=50 \
  model.num_warmup_steps=2500
