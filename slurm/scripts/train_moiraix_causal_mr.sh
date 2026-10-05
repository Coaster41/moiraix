#!/bin/bash
#SBATCH --job-name=train_moiraix_causal_mr
#SBATCH --output=/srv/disk00/ctadler/moiraix/slurm/logs/train_moiraix_causal_mr_%j.txt
#SBATCH --partition=datalab.p
#SBATCH --nodelist=markov-2
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:6
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=6
#SBATCH --mem=256gb

cd /srv/disk00/ctadler/moiraix
PY=/srv/disk00/ctadler/moiraix/venv/bin/python

srun echo "Start of training moiraix_causal_mr_0"
srun ${PY} -u -m cli.train \
  -cp conf/pretrain \
  run_name=moiraix_causal_mr_0 \
  model=moiraix_causal_mr \
  data=gift_eval_pretrain_weighted \
  val_data=gift_eval_test \
  train_dataloader.batch_size=256 \
  val_dataloader.batch_size=256 \
  trainer.gradient_clip_val=2.17524 \
  model.min_mask_ratio=0.107255 \
  model.max_mask_ratio=0.36345 \
  model.weight_decay=0.127608 \
  model.beta1=0.804699 \
  model.beta2=0.957911 \
  model.lr=0.00015 \
  seed=1 \
  model.module_kwargs.patch_size=16