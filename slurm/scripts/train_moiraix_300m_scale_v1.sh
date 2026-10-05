#!/bin/bash
#SBATCH --job-name=moiraix_300m_scale_v1
#SBATCH --output=/srv/disk00/ctadler/moiraix/slurm/logs/moiraix_300m_scale_v1_%j.txt
#SBATCH --partition=low.p
#SBATCH --nodelist=markov-2
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=200gb
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=SIGUSR1@120
#SBATCH --nice=1000000
set -euo pipefail
cd /srv/disk00/ctadler/moiraix
PY=/srv/disk00/ctadler/moiraix/venv/bin/python

# ---------------------------------------------------------------------------
# Track C arm 2 of 2: the ~400M scale-up, on the unthrottled corpus.
# SUBMIT ONLY AFTER the 30M control (train_moiraix_30m_scale_v1.sh) finishes --
# both want all 8 GPUs.
#
# Retry of moiraix_enc_cpm_gift_300m_0, which stopped at ep80 in a dead heat
# with the 30M model (WQL 0.1985 vs 0.1978 short-55 raw -- inside the +-2.8%
# seed-noise band). The audit said that run was data-throttled, not
# capacity-limited: it consumed 1.34T observations of which only ~50B were
# unique. `gift_eval_pretrain_scale_v1` raises the unique pool ~4x.
#
# Deltas vs the previous 400M script:
#   * data=gift_eval_pretrain_scale_v1        (the point of the rerun)
#   * +arcsinh_norm=true                      so the recipe matches the 30M
#     control exactly and the 30M-vs-400M delta is clean. This does mean the
#     comparison against the OLD 400M run (no arcsinh, old mix) is confounded on
#     two axes -- use the 30M control as the reference, not that run.
#   * 150 epochs, not 80. Epochs are ~15 min, and 80 epochs against the larger
#     pool would leave most of the new data unseen.
#
# Unchanged: dims 1024/4096/24 (~400M with our GLU FFN), lr 1e-4 (~1/sqrt(width)
# from the 30M's 1.5e-4; no muP, so HPs do not transfer for free), global batch
# 2048 = 64 x accum4 x 8 ranks, warmup 5000 steps, compile OFF (Triton JIT of
# cuda_utils.c fails on markov-2 -- missing multiarch pyconfig.h; job 29624).
#
# low.p is preemptible (any datalab.p job requeues this); the CKPT guard below
# resumes from the latest epoch checkpoint on requeue.
# ---------------------------------------------------------------------------
RUN_NAME="${RUN_NAME:-moiraix_enc_cpm_scale_v1_300m_0}"
DATA="${DATA:-gift_eval_pretrain_scale_v1}"
EPOCHS="${EPOCHS:-150}"
export RUN_NAME DATA EPOCHS

CKPT_DIR="outputs/pretrain/moiraix/${DATA}/${RUN_NAME}/checkpoints"
CKPT_ARG=""
if [ -f "${CKPT_DIR}/last.ckpt" ]; then
  CKPT_ARG="ckpt_path=last"
  echo "Resuming from ${CKPT_DIR}/last.ckpt"
else
  echo "No checkpoint in ${CKPT_DIR} -- starting fresh"
fi
export CKPT_ARG

srun echo "Start of ${RUN_NAME} (data=${DATA}, ${EPOCHS} epochs) on low.p"
srun bash -c '
  exec ${PY} -u -m cli.train \
  -cp conf/pretrain \
  run_name=${RUN_NAME} \
  model=moiraix \
  data=${DATA} \
  val_data=gift_eval_test \
  ${CKPT_ARG} \
  trainer.max_epochs=${EPOCHS} \
  train_dataloader.batch_size=64 \
  val_dataloader.batch_size=64 \
  train_dataloader.num_workers=6 \
  trainer.accumulate_grad_batches=4 \
  trainer.gradient_clip_val=2.17524 \
  model.weight_decay=0.127608 \
  model.beta1=0.804699 \
  model.beta2=0.957911 \
  model.lr=0.0001 \
  seed=1 \
  model.module_kwargs.d_model=1024 \
  model.module_kwargs.d_ff=4096 \
  model.module_kwargs.num_layers=24 \
  model.module_kwargs.patch_size=16 \
  model.module_kwargs.causal=false \
  model.module_kwargs.mask_inputs=true \
  model.module_kwargs.predict_next=false \
  model.module_kwargs.num_predict_token=1 \
  +model.module_kwargs.arcsinh_norm=true \
  model.loss_func.shift=false \
  model.min_mask_ratio=0 \
  model.max_mask_ratio=0.2 \
  model.cpm_pred_c_mask_max=5 \
  model.cpm_pred_p_mask_max=0.25 \
  model.val_cpm_pred_p_mask_max=0 \
  model.val_min_mask_ratio=0.3 \
  model.val_max_mask_ratio=0.3
'
