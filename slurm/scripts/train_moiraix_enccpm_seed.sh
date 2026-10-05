#!/bin/bash
#SBATCH --job-name=moiraix_enccpm_seed
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
# Training-variance replicate of the ENC_CPM recipe (moiraix_enc_cpm_gift50_0):
# bidirectional + masked-recon + CPM (causal=false, mask_inputs=true,
# predict_next=false, npt1, patch16, CPM c<=5/p<=0.25, mask 0-0.2, lr 1.5e-4,
# tuned wd/betas/clip), 50ep, warmup 2500, 4xbs256xaccum2. ONLY the seed varies
# vs the seed=1 run — a true replicate for the seed-variance estimate.
#
# Usage:
#   sbatch --output=slurm/logs/moiraix_enccpm_%j.txt \
#     --export=ALL,RUN_NAME=moiraix_enc_cpm_gift50_s2,SEED=2 \
#     slurm/scripts/train_moiraix_enccpm_seed.sh

set -euo pipefail
cd /srv/disk00/ctadler/moiraix

: "${RUN_NAME:?set RUN_NAME}" "${SEED:?set SEED}"
DATA="${DATA:-gift_eval_pretrain_trackb_p30_nosde_gift}"
PY=/srv/disk00/ctadler/moiraix/venv/bin/python

srun echo "Start of ${RUN_NAME} (enc_cpm replicate, data=${DATA} seed=${SEED})"
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
  model.num_warmup_steps=2500 \
  trainer.gradient_clip_val=2.17524 \
  model.weight_decay=0.127608 \
  model.beta1=0.804699 \
  model.beta2=0.957911 \
  model.lr=0.00015 \
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
  model.val_max_mask_ratio=0.3
