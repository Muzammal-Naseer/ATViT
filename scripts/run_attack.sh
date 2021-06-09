#!/bin/bash

DATA_DIR="PATH/TO/IMAGENET/val"

# run baseline attack
python test.py \
  --test_dir "$DATA_DIR" \
  --src_model deit_tiny_patch16_224 \
  --tar_model tnt_s_patch16_224  \
  --attack_type mifgsm \
  --eps 16 \
  --index "last" \
  --batch_size 128

# run self-ensemble attack
python test.py \
  --test_dir "$DATA_DIR" \
  --src_model deit_tiny_patch16_224 \
  --tar_model tnt_s_patch16_224  \
  --attack_type mifgsm \
  --eps 16 \
  --index "all" \
  --batch_size 128

# run self-ensemble attack with token-refinement
python test.py \
  --test_dir "$DATA_DIR" \
  --src_model tiny_patch16_224_hierarchical \
  --tar_model tnt_s_patch16_224  \
  --attack_type mifgsm \
  --eps 16 \
  --index "all" \
  --batch_size 128
