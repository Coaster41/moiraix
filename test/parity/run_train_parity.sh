#!/bin/bash
#SBATCH --job-name=moiraix_train_parity
#SBATCH --output=/srv/disk00/ctadler/moiraix/slurm/logs/train_parity_%j.txt
#SBATCH --partition=datalab.p
#SBATCH --nodelist=markov-2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=96gb
#SBATCH --time=3:00:00
# Training parity: identical short run in the uni2ts and moiraix repos (same GPU,
# same seed), compare per-step losses; then resume the uni2ts Lightning .ckpt
# with the moiraix code to check old checkpoints still unpickle.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
# str hash() is randomized per process by default; pin it so runs are comparable.
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
OUT=/srv/disk00/ctadler/moiraix/outputs/parity
MODEL="${MODEL:-moiraix_encoder}"
rm -rf $OUT/train_uni2ts_$MODEL $OUT/train_moiraix_$MODEL $OUT/train_moiraix2_$MODEL $OUT/resume_$MODEL
ARGS=(-cp conf/pretrain run_name=parity model=$MODEL
  data=gift_eval_pretrain_trackb_p30_nosde_gift val_data=gift_eval_test
  trainer.devices=1 trainer.max_epochs=2 trainer.log_every_n_steps=1 trainer.limit_val_batches=5
  train_dataloader.batch_size=64 train_dataloader.num_batches_per_epoch=20
  val_dataloader.batch_size=64 val_dataloader.num_batches_per_epoch=5 model.num_warmup_steps=5)

echo "== uni2ts"
cd /srv/disk00/ctadler/uni2ts
/srv/disk00/ctadler/uni2ts/venv/bin/python -m cli.train "${ARGS[@]}" hydra.run.dir=$OUT/train_uni2ts_$MODEL
echo "== moiraix"
cd /srv/disk00/ctadler/moiraix
venv/bin/python -m cli.train "${ARGS[@]}" hydra.run.dir=$OUT/train_moiraix_$MODEL
echo "== moiraix (repeat, run-to-run control)"
venv/bin/python -m cli.train "${ARGS[@]}" hydra.run.dir=$OUT/train_moiraix2_$MODEL
echo "== compare uni2ts vs moiraix"
venv/bin/python test/parity/compare_train.py $OUT/train_uni2ts_$MODEL $OUT/train_moiraix_$MODEL || true
echo "== compare moiraix vs moiraix repeat"
venv/bin/python test/parity/compare_train.py $OUT/train_moiraix_$MODEL $OUT/train_moiraix2_$MODEL || true
echo "== resume uni2ts .ckpt in moiraix"
venv/bin/python -m cli.train "${ARGS[@]}" trainer.max_epochs=3 \
  ckpt_path=$OUT/train_uni2ts_$MODEL/checkpoints/last.ckpt hydra.run.dir=$OUT/resume_$MODEL
echo "RESUME OK"
