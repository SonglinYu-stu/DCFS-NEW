#!/usr/bin/env bash
NET=$1
NUNMGPU=$2
EXPNAME=$3
SAVEDIR=workspace/DCFS/coco-det/${EXPNAME}  #<-- change it to you path
PRTRAINEDMODEL=pretrained_models/ImageNetPretrained                                     #<-- change it to you path


if [ "$NET"x = "r101"x ]; then
  IMAGENET_PRETRAIN=${PRTRAINEDMODEL}/MSRA/R-101.pkl
  IMAGENET_PRETRAIN_TORCH=${PRTRAINEDMODEL}/torchvision/resnet101-5d3b4d8f.pth
fi

if [ "$NET"x = "r50"x ]; then
  IMAGENET_PRETRAIN=${PRTRAINEDMODEL}/MSRA/R-50.pkl
  IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN}/torchvision/resnet50-19c8e357.pth
fi


# ------------------------------- Base Pre-train ---------------------------------- #
python3 main.py --num-gpus ${NUNMGPU} --config-file configs/coco/dcfs_det_r101_base-step8.yaml     \
    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                         \
           OUTPUT_DIR ${SAVEDIR}/dcfs_det_${NET}_base

