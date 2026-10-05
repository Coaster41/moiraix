#!/bin/bash
#SBATCH --job-name=moirai2_repro_parity
#SBATCH --output=/srv/disk00/ctadler/moiraix/slurm/logs/moirai2_repro_parity_%j.txt
#SBATCH --partition=datalab.p
#SBATCH --nodelist=markov-2
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:6
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=6
#SBATCH --mem=196gb
#
# "Parity" run per HANDOFF_MOIRAI2_PARITY.md section 4 step 2: FIX A/B (loss
# clamp removal + boundary-leak mask, code-only, always on) + FIX C (50% iid
# patch masking) + FIX D (prefix-only scaler stats) + FIX E (z-score sample
# filter) + FIX F (effective batch size restored to the paper's 256), all on
# top of the existing model=moiraix_moirai2_repro config and the 20%-synthetic
# gift_eval_pretrain_moirai2 data mixture.
#
# FIX F: paper effective batch = 256. 6 GPUs * 43 per-rank batch = 258 (closest
# achievable to 256 with an integer per-rank batch on 6 devices).
#
# Expect CRPS to move from the earlier repro's 0.130 toward <= 0.12 (still
# short of the data-lever fixes G/H in the handoff, which are separate runs).

set -euo pipefail
cd /srv/disk00/ctadler/moiraix
PY=/srv/disk00/ctadler/moiraix/venv/bin/python

srun echo "Start of moirai2_repro_parity (FIX A-F)"
srun ${PY} -u -m cli.train \
  -cp conf/pretrain \
  run_name=moirai2_repro_parity_0 \
  model=moiraix_moirai2_repro \
  data=gift_eval_pretrain_moirai2 \
  val_data=gift_eval_test \
  train_dataloader.batch_size=43 \
  val_dataloader.batch_size=43 \
  train_dataloader.num_workers=6 \
  seed=1
