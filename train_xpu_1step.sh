#!/bin/bash
# 2 卡 B60 上跑 1 步并保存 checkpoint，用于产出部署所需的 hf_ckpt
cd /llm/zhuyong/vla/lingbot-vla
export PATH=/root/miniforge3/envs/lingbot/bin:$PATH
unset LD_LIBRARY_PATH
ulimit -n 524288
export PYTORCH_XPU_ALLOC_CONF=expandable_segments:True
export QWEN25_PATH=/llm/zhuyong/vla/lingbot-vla/Qwen2.5-VL-3B-Instruct
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robotwin_load20000h.yaml \
  --model.model_path ./lingbot-vla-4b \
  --model.tokenizer_path ./Qwen2.5-VL-3B-Instruct \
  --data.train_path /llm/zhuyong/RoboTwin/policy/pi0/lerobot_data/click_bell_aloha_repo \
  --train.output_dir ./output_click_bell_aloha \
  --train.micro_batch_size 1 \
  --train.global_batch_size 2 \
  --train.num_train_epochs 1 \
  --train.max_steps 1 \
  --train.save_steps 0 \
  --train.save_epochs 1 \
  --train.enable_resume false \
  --train.use_compile false \
  --train.enable_fsdp_offload true \
  --train.init_device cpu \
  --train.enable_full_shard true
