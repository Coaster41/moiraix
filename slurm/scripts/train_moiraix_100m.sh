#!/bin/bash
#SBATCH --job-name=moiraix_100m
#SBATCH --output=/srv/disk00/ctadler/moiraix/slurm/logs/moiraix_100m_%j.txt
#SBATCH --partition=datalab.p
#SBATCH --nodelist=markov-2
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:6
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=160gb
#SBATCH --time=4-00:00:00
set -euo pipefail
cd /srv/disk00/ctadler/moiraix
PY=/srv/disk00/ctadler/moiraix/venv/bin/python

# exp-009: scale the exp-008 winner. enc_cpm_2 recipe (see train_moiraix_sweep.sh)
# at ~100M params (d_model 768, d_ff 3072, 12 layers vs 512/2048/8) on the FIX G
# datamix. Total batch parity with the enc_cpm runs: 4 ranks x 128 x accum 4
# = 2048 (micro-batch halved for the larger activations). lr kept at the HPO'd
# 1.5e-4. PICK=4 with --gres=gpu:6: untracked labmate processes squat markov-2
# GPUs; use the 4 least-memory-used GPUs of the allocation.
RUN_NAME="${RUN_NAME:-moiraix_enc_cpm_gift_100m_0}"
DATA="${DATA:-gift_eval_pretrain_trackb_p30_nosde_gift}"
export RUN_NAME DATA

PICK="${PICK:-4}"
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
  train_dataloader.batch_size=128 \
  val_dataloader.batch_size=128 \
  train_dataloader.num_workers=6 \
  trainer.accumulate_grad_batches=4 \
  trainer.gradient_clip_val=2.17524 \
  model.weight_decay=0.127608 \
  model.beta1=0.804699 \
  model.beta2=0.957911 \
  model.lr=0.00015 \
  seed=1 \
  model.module_kwargs.d_model=768 \
  model.module_kwargs.d_ff=3072 \
  model.module_kwargs.num_layers=12 \
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
