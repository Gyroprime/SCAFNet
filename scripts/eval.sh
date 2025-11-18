#!/usr/bin/env bash

gpus=0

data_name=CDD
net_G=hybrid_modelv3
split=test
project_name=CD_hybrid_modelv3_EncoderV3G9_DecocderV3G10_CDD_b8_lr0.0001_train_test_200_adamw_linear_pretrained_backbone
checkpoint_name=best_ckpt.pt

python eval_cd.py --split ${split} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


