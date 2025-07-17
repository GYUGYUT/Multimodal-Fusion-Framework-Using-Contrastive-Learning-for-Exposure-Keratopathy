#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# broad dataset exel path
BROAD_TRAIN="grade_photo1_train.xlsx"
BROAD_VAL="grade_photo1_val.xlsx"
BROAD_TEST="grade_photo1_test.xlsx"
# slit dataset exel path
SLIT_TRAIN="grade_photo2_train.xlsx"
SLIT_VAL="grade_photo2_val.xlsx"
SLIT_TEST="grade_photo2_test.xlsx"
# scatter dataset exel path
SCATTER_TRAIN="grade_photo3_train.xlsx"
SCATTER_VAL="grade_photo3_val.xlsx"
SCATTER_TEST="grade_photo3_test.xlsx"
# blue dataset exel path
BLUE_TRAIN="grade_photo4_train.xlsx"
BLUE_VAL="grade_photo4_val.xlsx"
BLUE_TEST="grade_photo4_test.xlsx"

# dataset Image folder path
IMAGE_FOLDER="SMC_New_original"

# 공통 파라미터 (대조학습/분류기 학습)
EPOCHS=100
BATCH_SIZE=16
# 각 빔(모달리티)별 백본 지정
BROAD_BACKBONE="densenet121"   # broad 백본
SLIT_BACKBONE="densenet121"           # slit 백본
SCATTER_BACKBONE="densenet121"        # scatter 백본
BLUE_BACKBONE="densenet121"           # blue 백본
NUM_CLASSES=4
LR=0.0001
DEVICE="cuda"
PATIENCE=10
OUT_DIM=256  # SimpleEncoder projection head 출력 차원 (원하는 값으로 변경)

# 라벨 파일 (broad 기준)
TRAIN_LABEL="$BROAD_TRAIN"
VAL_LABEL="$BROAD_VAL"
TEST_LABEL="$BROAD_TEST"

# loss2 가중치(nt_xent_loss)
ALPHA=1.0
# loss3 가중치(supervised_contrastive_loss)
ALPHA2=0.0

CUDA_VISIBLE_DEVICES=0,2 python3 train_contrastive.py \
  --broad_train $BROAD_TRAIN --broad_val $BROAD_VAL --broad_test $BROAD_TEST \
  --slit_train $SLIT_TRAIN --slit_val $SLIT_VAL --slit_test $SLIT_TEST \
  --scatter_train $SCATTER_TRAIN --scatter_val $SCATTER_VAL --scatter_test $SCATTER_TEST \
  --blue_train $BLUE_TRAIN --blue_val $BLUE_VAL --blue_test $BLUE_TEST \
  --image_folder $IMAGE_FOLDER \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --broad_backbone $BROAD_BACKBONE \
  --slit_backbone $SLIT_BACKBONE \
  --scatter_backbone $SCATTER_BACKBONE \
  --blue_backbone $BLUE_BACKBONE \
  --num_classes $NUM_CLASSES \
  --lr $LR \
  --device $DEVICE \
  --patience $PATIENCE \
  --auto_finetune \
  --finetune_train_label $TRAIN_LABEL \
  --finetune_val_label $VAL_LABEL \
  --finetune_test_label $TEST_LABEL \
  --finetune_epochs $EPOCHS \
  --finetune_batch_size $BATCH_SIZE \
  --finetune_lr $LR \
  --finetune_patience $PATIENCE \
  --out_dim $OUT_DIM \
  --max_steps 0 \
  --eval_interval 100 \
  --alpha $ALPHA \
  --alpha2 $ALPHA2 