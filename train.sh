#!/bin/bash

set -x

export TOKENIZERS_PARALLELISM=false
if [ -n "$ZE_AFFINITY_MASK" ]; then
  # 可见 Intel XPU 数量
  NPROC_PER_NODE=$(echo $ZE_AFFINITY_MASK | tr ',' '\n' | wc -l)
elif [ -z "$CUDA_VISIBLE_DEVICES" ] && ! command -v nvidia-smi >/dev/null; then
  # 没有 nvidia-smi 时按 Intel XPU 计数
  NPROC_PER_NODE=$(python -c "import torch; print(torch.xpu.device_count())")
elif [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
else
  # 可见 GPU 数量
  NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Using NPROC_PER_NODE=$NPROC_PER_NODE GPUs"
NNODES=${NNODES:=1}
NPROC_PER_NODE=${NPROC_PER_NODE:=$NPROC_PER_NODE}
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=62500}


torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT $@ 2>&1 | tee log.txt
