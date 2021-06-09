#!/bin/bash

export WANDB_ENTITY="USERNANE"
export WANDB_PROJECT="PROJECT"
export WANDB_MODE='dryrun'

DATA_PATH="PATH/TO/IMAGENET"

EXP_NAME="test"

if [ ! -d "checkpoints" ]; then
  mkdir "checkpoints"
fi

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$RANDOM" \
  --use_env train_trm.py \
  --exp "$EXP_NAME" \
  --model "tiny_patch16_224_hierarchical" \
  --lr 0.01 \
  --batch-size 256 \
  --start-epoch 0 \
  --epochs 12 \
  --data "$DATA_PATH" \
  --pretrained "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth" \
  --output_dir "checkpoints/$EXP_NAME"
