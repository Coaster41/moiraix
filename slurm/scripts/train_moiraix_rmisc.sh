#!/bin/bash
#SBATCH --job-name=moiraix_rmisc
#SBATCH --output=/srv/disk00/ctadler/moiraix/slurm/logs/moiraix_rmisc_%j.txt
#SBATCH --partition=datalab.p
#SBATCH --nodelist=markov-2
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=128gb
#SBATCH --time=4-00:00:00
set -euo pipefail
cd /srv/disk00/ctadler/moiraix
PY=/srv/disk00/ctadler/moiraix/venv/bin/python

# exp-012: the exp-008 winner (moiraix_enc_cpm_gift_1, WQL 0.1118, job 29322)
# retrained UNCHANGED except the data config -> gift_eval_pretrain_rmisc (the
# p30+gift mix + the RMISC corpus at a 15% sampling share). Every hyperparameter
# below is copied verbatim from that run's .hydra/overrides.yaml so the ONLY
# variable vs the 0.1118 baseline is corpus breadth. Do not change anything else
# in this file if the intent is a clean single-variable read.
#
# 30M arch (d512/2048/8L) = moiraix.yaml defaults, so NO d_model/d_ff/num_layers
# overrides here (that is the ONLY structural difference from train_moiraix_100m.sh).
# Global batch = 4 ranks x 256 x accum 2 = 2048, matching the winner exactly
# (sacct job 29322: cpu=24 at 6 cpus/task = 4 tasks; the p20/p40 twins used the
# same 4-rank setup via train_moiraix_sweep.sh). Batch parity resolved.
RUN_NAME="${RUN_NAME:-moiraix_enc_cpm_rmisc_0}"
DATA="${DATA:-gift_eval_pretrain_rmisc}"
export RUN_NAME DATA

# PICK=N with --gres=gpu:7: use the N least-memory-used GPUs of the allocation,
# in case a non-slurm labmate process is squatting one. Leave unset to use all.
PICK="${PICK:-}"
if [ -n "${PICK}" ]; then
  PICKED_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' -v ids="${CUDA_VISIBLE_DEVICES}" \
        'BEGIN{n=split(ids,a,","); for(i=1;i<=n;i++) ok[a[i]]=1} ok[$1]{print $2, $1}' \
    | sort -n | head -n "${PICK}" | awk '{print $2}' | paste -sd,)
  export PICKED_GPUS
  echo "PICK=${PICK}: using GPUs ${PICKED_GPUS} of allocation ${CUDA_VISIBLE_DEVICES}"
fi

srun echo "Start of ${RUN_NAME} (data=${DATA})"
srun bash -c '
  if [ -n "${PICKED_GPUS:-}" ]; then export CUDA_VISIBLE_DEVICES="${PICKED_GPUS}"; fi
  exec ${PY} -u -m cli.train \
  -cp conf/pretrain \
  run_name=${RUN_NAME} \
  model=moiraix \
  data=${DATA} \
  val_data=gift_eval_test \
  train_dataloader.batch_size=256 \
  val_dataloader.batch_size=256 \
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
  model.loss_func.shift=false \
  model.min_mask_ratio=0 \
  model.max_mask_ratio=0.2 \
  model.cpm_pred_c_mask_max=5 \
  model.cpm_pred_p_mask_max=0.25 \
  model.val_cpm_pred_p_mask_max=0 \
  model.val_min_mask_ratio=0.3 \
  model.val_max_mask_ratio=0.3
'
