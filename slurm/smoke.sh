#!/bin/bash
#SBATCH --job-name=fst_smoke
#SBATCH --partition=def
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/smoke_%j.log

set -euo pipefail

export HF_HOME=/scratch/user/fatemehdoudi_tamu.edu/.hf-cache
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN   # avoid flashinfer attention JIT

cd /scratch/user/fatemehdoudi_tamu.edu/fst

source .env
set -a; source /scratch/user/fatemehdoudi_tamu.edu/fst/.env; set +a


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONPATH=. /scratch/user/fatemehdoudi_tamu.edu/envs/fst/bin/python trainer.py --smoke
