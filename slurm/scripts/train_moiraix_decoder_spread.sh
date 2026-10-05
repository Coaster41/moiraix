#!/bin/bash
#SBATCH --job-name=moiraix_decoder_spread
#SBATCH --partition=datalab.p
#SBATCH --nodelist=markov-2
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=200gb
#SBATCH --time=12:00:00
# --output is set on the sbatch command line per run.
#
# Mode B (HANDOFF_SPREAD_HEAD.md §7.B): causal-NTP decoder + long-horizon
# quantile-spread head, trained JOINTLY FROM RANDOM INIT (not a graft/
# finetune of an existing checkpoint like train_moiraix_dec_spread_headonly.sh
# / Mode A). Twins train_moiraix_decoder.sh's moiraix_dec_ntp_gift recipe
# exactly (model=moiraix defaults, same data/batch/lr/warmup ratio) plus the
# spread head trained end-to-end (full backprop through the trunk, no
# freeze_pattern). num_spread_token=64 / num_spread_positions=32 match the
# Mode A config already validated in tsfm-mi-experiments/ar_calibration
# exp-000 (partial positive: spread+naive beats naive alone by -2.1% gWQL).
#
# Compute overhead measured empirically (bf16-mixed A/B, isolated processes):
# +28% params but only +2.0% step time (head only touches 32/512 sampled
# positions per sample, dwarfed by full-sequence trunk compute). Anchored to
# the real moiraix_dec_ntp_gift_0 run (100 epochs/100k steps in 16h33m ->
# 9.93 min/epoch), a 50-epoch run costs ~8.28h without the head, ~8.44h with
# it -- about 10 extra minutes total.
#
# Usage:
#   sbatch --output=slurm/logs/moiraix_decoder_spread_%j.txt \
#     --export=ALL,RUN_NAME=moiraix_dec_spread_scratch50_0 \
#     slurm/scripts/train_moiraix_decoder_spread.sh

set -euo pipefail
cd /srv/disk00/ctadler/moiraix
PY=/srv/disk00/ctadler/moiraix/venv/bin/python

: "${RUN_NAME:?set RUN_NAME}"
DATA="${DATA:-gift_eval_pretrain_trackb_p30_nosde_gift}"
EXTRA="${EXTRA:-}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCHES_PER_EPOCH="${BATCHES_PER_EPOCH:-1000}"
ACCUM_GRAD_BATCHES="${ACCUM_GRAD_BATCHES:-2}"
WARMUP_STEPS="${WARMUP_STEPS:-2500}"          # 5% of 50k steps, matches moiraix_dec_ntp_gift's 5%/100k ratio
NUM_SPREAD_TOKEN="${NUM_SPREAD_TOKEN:-64}"
NUM_SPREAD_POSITIONS="${NUM_SPREAD_POSITIONS:-32}"
SPREAD_LOSS_WEIGHT="${SPREAD_LOSS_WEIGHT:-1.0}"

srun echo "Start of ${RUN_NAME} (Mode B joint from-scratch: epochs=${MAX_EPOCHS} batches/epoch=${BATCHES_PER_EPOCH} accum=${ACCUM_GRAD_BATCHES} warmup=${WARMUP_STEPS} nst=${NUM_SPREAD_TOKEN} nsp=${NUM_SPREAD_POSITIONS} spread_wt=${SPREAD_LOSS_WEIGHT} data=${DATA} extra=${EXTRA})"
srun ${PY} -u -m cli.train \
  -cp conf/pretrain \
  run_name=${RUN_NAME} \
  model=moiraix_decoder_spread \
  data=${DATA} \
  val_data=gift_eval_test \
  model.module_kwargs.num_spread_token=${NUM_SPREAD_TOKEN} \
  model.num_spread_positions=${NUM_SPREAD_POSITIONS} \
  model.spread_loss_weight=${SPREAD_LOSS_WEIGHT} \
  train_dataloader.batch_size=256 \
  val_dataloader.batch_size=256 \
  train_dataloader.num_workers=6 \
  train_dataloader.num_batches_per_epoch=${BATCHES_PER_EPOCH} \
  trainer.max_epochs=${MAX_EPOCHS} \
  trainer.accumulate_grad_batches=${ACCUM_GRAD_BATCHES} \
  model.num_warmup_steps=${WARMUP_STEPS} \
  seed=1 \
  ${EXTRA}
