#!/bin/bash
#SBATCH --job-name=train_moiraix_moirai2
#SBATCH --output=/srv/disk00/ctadler/moiraix/slurm/logs/train_moiraix_moirai2_%j.txt
#SBATCH --partition=datalab.p
#SBATCH --nodelist=markov-2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:6
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=6
#SBATCH --mem=196gb

cd /srv/disk00/ctadler/moiraix
PY=/srv/disk00/ctadler/moiraix/venv/bin/python

# Moirai 2.0 recipe replication: moiraix causal decoder + paper-aligned optimizer
# (model=moiraix_moirai2) trained on the synthetic-heavy mixture
# (data=gift_eval_pretrain_moirai2).
#
# NOTE on batch size: the codebase batch_size is per-GPU, so 6 GPUs gives a global
# batch of ~6x the config value. The report uses a global batch of 256. To match
# it more closely, add e.g. `train_dataloader.batch_size=42` (256/6). Left at the
# default here to keep an apples-to-apples comparison with prior runs.

srun echo "Start of training moiraix_moirai2_1"
srun ${PY} -u -m cli.train \
  -cp conf/pretrain \
  run_name=moiraix_moirai2_1 \
  model=moiraix_moirai2 \
  data=gift_eval_pretrain_moirai2 \
  val_data=gift_eval_test
  # ckpt_path="last"  # uncomment to resume
