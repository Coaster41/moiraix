#!/bin/bash
#SBATCH --job-name=moiraix_300m_lowp
#SBATCH --output=/srv/disk00/ctadler/moiraix/slurm/logs/moiraix_300m_lowp_%j.txt
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
set -euo pipefail
cd /srv/disk00/ctadler/moiraix
PY=/srv/disk00/ctadler/moiraix/venv/bin/python

# ---------------------------------------------------------------------------
# "For fun" scale-up of the enc_cpm_gift recipe (former best, WQL 0.1118) from
# ~30M -> ~380-400M params. This is a curiosity/systems run, NOT expected to beat
# the 30M/100M WQL (no gain was seen 30M->100M; the plateau is data-bound, not
# capacity-bound -- see .claude memory "track-b-retrains" and Toto 2.0's finding
# that scaling needs data scaled in lockstep).
#
# Partition: low.p (PriorityTier=10, PreemptMode=REQUEUE). Any datalab.p job
# (PriorityTier=100) preempts this one; SLURM requeues it and it resumes from
# outputs/.../checkpoints/last.ckpt via the CKPT guard below. Expected to sit
# queued for ~a week while the cluster is busy -- that is fine.
#
# Scale-up deltas vs the 30M enc_cpm recipe (see "things to consider"):
#   * dims 1024/4096/24 (Toto-2.0 313M shape; ~400M here because our FFN is GLU)
#   * lr 1.5e-4 -> 1.0e-4  (~1/sqrt(width) down-scaling; no muP, so HPs don't
#     transfer for free -- lower LR + watch for spikes)
#   * micro-batch 128 -> 64, accum 4, 8 ranks => global batch 2048 (parity kept)
#   * torch.compile left OFF: Triton JIT-compiles a cuda_utils helper at runtime
#     which needs the multiarch pyconfig.h dev header, missing on markov-2 -> all
#     ranks crash at sanity-check (job 29624). No existing script uses compile.
#   * warmup left at 5000 steps: at accum=4 that is ~250 opt-steps/epoch => ~20
#     epochs / ~20% of optimizer steps, already generous -- pushing to 10k+ would
#     waste most of training on warmup.
# ---------------------------------------------------------------------------
RUN_NAME="${RUN_NAME:-moiraix_enc_cpm_gift_300m_0}"
DATA="${DATA:-gift_eval_pretrain_trackb_p30_nosde_gift}"
export RUN_NAME DATA

# Resume guard: pass ckpt_path=last only if a checkpoint already exists, so the
# first launch starts fresh and every requeue resumes from the latest epoch.
CKPT_DIR="outputs/pretrain/moiraix/${DATA}/${RUN_NAME}/checkpoints"
CKPT_ARG=""
if [ -f "${CKPT_DIR}/last.ckpt" ]; then
  CKPT_ARG="ckpt_path=last"
  echo "Resuming from ${CKPT_DIR}/last.ckpt"
else
  echo "No checkpoint found in ${CKPT_DIR} -- starting fresh"
fi
export CKPT_ARG

srun echo "Start of ${RUN_NAME} (data=${DATA}) on low.p"
srun bash -c '
  exec ${PY} -u -m cli.train \
  -cp conf/pretrain \
  run_name=${RUN_NAME} \
  model=moiraix \
  data=${DATA} \
  val_data=gift_eval_test \
  ${CKPT_ARG} \
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
  model.loss_func.shift=false \
  model.min_mask_ratio=0 \
  model.max_mask_ratio=0.2 \
  model.cpm_pred_c_mask_max=5 \
  model.cpm_pred_p_mask_max=0.25 \
  model.val_cpm_pred_p_mask_max=0 \
  model.val_min_mask_ratio=0.3 \
  model.val_max_mask_ratio=0.3
'
