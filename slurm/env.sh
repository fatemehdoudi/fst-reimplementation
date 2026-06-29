#!/bin/bash
#SBATCH --job-name=fst_build
#SBATCH --partition=def          # ← match your EGSPO partition
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/build_%j.log

PY=/scratch/user/fatemehdoudi_tamu.edu/envs/fst/bin/python

# vLLM FIRST so it pins compatible torch+CUDA
$PY -m pip install --no-input vllm
$PY -m pip install --no-input gepa litellm transformers

$PY -c "import torch, vllm, gepa, litellm, transformers; \
print('torch', torch.__version__); print('vllm', vllm.__version__); \
print('cuda', torch.cuda.is_available())"

$PY -m pip freeze > /scratch/user/fatemehdoudi_tamu.edu/fst/requirements.txt