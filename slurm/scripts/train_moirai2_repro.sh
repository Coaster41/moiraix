#!/bin/bash
#SBATCH --job-name=moirai2_repro
#SBATCH --output=/srv/disk00/ctadler/moiraix/slurm/logs/moirai2_repro_%j.txt
#SBATCH --partition=datalab.p
#SBATCH --nodelist=markov-2
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=128gb
#
# Faithful Moirai-2.0-R-small reproduction (control run).
# All model/optimizer hyperparameters live in model=moiraix_moirai2_repro so this
# script only sets data + dataloader + seed. 100k steps (max_epochs=100 default x
# 1000 batches/epoch), batch 256, bf16 -> ~18-20h on 8x A5000.
#
# Expected: gmean-CRPS ~= 0.1096 (the released moirai2 score on the 55-dataset short
# split). If it lands there, your harness + data are sound; if not, the residual gap
# is data/harness, not your architecture ideas.

set -euo pipefail
cd /srv/disk00/ctadler/moiraix
PY=/srv/disk00/ctadler/moiraix/venv/bin/python

srun echo "Start of moirai2_repro_2 (faithful Moirai-2.0-R-small)"
srun ${PY} -u -m cli.train \
  -cp conf/pretrain \
  run_name=moirai2_repro_2 \
  model=moiraix_moirai2_repro \
  data=gift_eval_pretrain_moirai2 \
  val_data=gift_eval_test \
  train_dataloader.batch_size=256 \
  val_dataloader.batch_size=256 \
  train_dataloader.num_workers=6 \
  model.lr=1.5e-4 \
  seed=1
