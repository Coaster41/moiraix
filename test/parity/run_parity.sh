#!/bin/bash
#SBATCH --job-name=moiraix_parity
#SBATCH --output=/srv/disk00/ctadler/moiraix/slurm/logs/parity_%j.txt
#SBATCH --partition=datalab.p
#SBATCH --nodelist=markov-2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=64gb
#SBATCH --time=3:00:00
# Inference parity: same GPU, uni2ts venv then moiraix venv, then compare.
set -euo pipefail
cd /srv/disk00/ctadler/moiraix
export PYTHONDONTWRITEBYTECODE=1
OUT=outputs/parity
echo "== CUDA check (moiraix venv)"
venv/bin/python -c "import torch; assert torch.cuda.is_available(); x=torch.randn(1024,1024,device='cuda',dtype=torch.bfloat16); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), float((x@x).float().abs().mean()))"
echo "== uni2ts"
/srv/disk00/ctadler/uni2ts/venv/bin/python test/parity/dump_outputs.py --backend uni2ts --out $OUT/uni2ts.pt
echo "== moiraix"
venv/bin/python test/parity/dump_outputs.py --backend moiraix --out $OUT/moiraix.pt
echo "== compare"
venv/bin/python test/parity/compare_outputs.py $OUT/uni2ts.pt $OUT/moiraix.pt
