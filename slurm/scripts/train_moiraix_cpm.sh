#!/bin/bash
#SBATCH --job-name=train_moiraix_dec_cpm
#SBATCH --output=/srv/disk00/ctadler/moiraix/slurm/logs/train_moiraix_cpm_%j.txt
#SBATCH --partition=datalab.p
#SBATCH --nodelist=markov-2
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:7
#SBATCH --ntasks=7
#SBATCH --ntasks-per-node=7
#SBATCH --mem=128gb

cd /srv/disk00/ctadler/moiraix
PY=/srv/disk00/ctadler/moiraix/venv/bin/python

srun echo "Start of training moiraix_dec_cpm_4"
srun ${PY} -u -m cli.train \
  -cp conf/pretrain \
  run_name=moiraix_dec_cpm_4 \
  model=moiraix \
  data=gift_eval_pretrain_moirai2 \
  val_data=gift_eval_test \
  train_dataloader.batch_size=256 \
  val_dataloader.batch_size=256 \
  trainer.gradient_clip_val=2.17524 \
  model.weight_decay=0.127608 \
  model.beta1=0.804699 \
  model.beta2=0.957911 \
  model.lr=0.00015 \
  seed=1 \
  model.module_kwargs.patch_size=16 \
  model.module_kwargs.causal=true \
  model.module_kwargs.mask_inputs=true \
  model.module_kwargs.predict_next=false \
  model.module_kwargs.num_predict_token=1 \
  model.loss_func.shift=false \
  model.min_mask_ratio=0 \
  model.max_mask_ratio=0 \
  model.cpm_pred_c_mask_max=5 \
  model.cpm_pred_p_mask_max=0.25 \
  model.val_cpm_pred_p_mask_max=0 \
  model.val_min_mask_ratio=0.3 \
  model.val_max_mask_ratio=0.3 
  # ckpt_path="/srv/disk00/ctadler/moiraix/outputs/uni2ts/moiraix/gift_eval_pretrain_weighted/moiraix_enc_cpm_0/checkpoints/last.ckpt" 

  # model.num_warmup_steps=10000
  # ckpt_path="/srv/disk00/ctadler/moiraix/outputs/uni2ts/moiraix/gift_eval_pretrain_weighted/moiraix_enc_cpm_0/checkpoints/last.ckpt" \
  # model.num_warmup_steps=0