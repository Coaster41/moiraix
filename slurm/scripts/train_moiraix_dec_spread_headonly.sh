#!/bin/bash
#SBATCH --job-name=moiraix_dec_spread_headonly
#SBATCH --partition=datalab.p
#SBATCH --nodelist=markov-2
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=200gb
#SBATCH --time=06:00:00
# --output is set on the sbatch command line per run.
#
# Mode A (HANDOFF_SPREAD_HEAD.md §7.A): graft a long-horizon quantile-spread
# head (num_spread_token=64) onto the moiraix_dec_ntp_gift causal-NTP
# checkpoint and train ONLY the head (trunk frozen). Step budget is
# MAX_EPOCHS x BATCHES_PER_EPOCH (defaults below reproduce the original
# 1024-step smoke/de-risking run; override both for a real run, e.g.
# MAX_EPOCHS=15 BATCHES_PER_EPOCH=1000 = 15,000 steps, the handoff's
# recommended 10-20k range for a real Mode-A run).
#
# Usage (smoke test, defaults):
#   sbatch --output=slurm/logs/moiraix_dec_spread_headonly_%j.txt \
#     --export=ALL,RUN_NAME=moiraix_dec_spread_headonly_0 \
#     slurm/scripts/train_moiraix_dec_spread_headonly.sh
# Usage (real run):
#   sbatch --output=slurm/logs/moiraix_dec_spread_headonly_%j.txt \
#     --export=ALL,RUN_NAME=moiraix_dec_spread_headonly64_15ep,MAX_EPOCHS=15,BATCHES_PER_EPOCH=1000,WARMUP_STEPS=500 \
#     slurm/scripts/train_moiraix_dec_spread_headonly.sh

set -euo pipefail
cd /srv/disk00/ctadler/moiraix
PY=/srv/disk00/ctadler/moiraix/venv/bin/python

: "${RUN_NAME:?set RUN_NAME}"
EXTRA="${EXTRA:-}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
BATCHES_PER_EPOCH="${BATCHES_PER_EPOCH:-1024}"
ACCUM_GRAD_BATCHES="${ACCUM_GRAD_BATCHES:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"

srun echo "Start of ${RUN_NAME} (epochs=${MAX_EPOCHS} batches/epoch=${BATCHES_PER_EPOCH} accum=${ACCUM_GRAD_BATCHES} warmup=${WARMUP_STEPS} extra=${EXTRA})"
srun ${PY} -u -m cli.train \
  -cp conf/pretrain \
  run_name=${RUN_NAME} \
  model=moiraix_dec_spread_headonly \
  data=gift_eval_pretrain_trackb_p30_nosde_gift \
  val_data=gift_eval_test \
  train_dataloader.batch_size=256 \
  val_dataloader.batch_size=256 \
  train_dataloader.num_workers=6 \
  train_dataloader.num_batches_per_epoch=${BATCHES_PER_EPOCH} \
  trainer.max_epochs=${MAX_EPOCHS} \
  trainer.accumulate_grad_batches=${ACCUM_GRAD_BATCHES} \
  model.num_warmup_steps=${WARMUP_STEPS} \
  seed=1 \
  ${EXTRA}
