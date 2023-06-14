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


# # ------------------------------- Base Pre-train ---------------------------------- #
# python3 main.py --num-gpus ${NUNMGPU} --config-file configs/coco/dcfs_det_r101_base-step3.yaml     \
#     --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                         \
#            OUTPUT_DIR ${SAVEDIR}/dcfs_det_${NET}_base

# ----------------------------- Model Preparation --------------------------------- #
python3 tools/model_surgery.py --dataset coco --method randinit                        \
    --src-path ${SAVEDIR}/dcfs_det_${NET}_base/model_final.pth                         \
    --save-dir ${SAVEDIR}/dcfs_det_${NET}_base                                         \
    --param-name roi_heads.box_predictor.bbox_pred
BASE_WEIGHT=${SAVEDIR}/dcfs_det_${NET}_base/model_reset_surgery.pth

# ------------------------------ Base+Novel Fine-tuning ------------------------------- #
# --> 2. TFA-like, i.e. run seed0~9 (10 times) for gFSOD (80 classes)
# classloss="DC" # "CE"
for seed in 0 
do
    for shot in 10
    do
        TRAIN_ALL_NAME=coco14_trainval_all_${shot}shot_seed${seed}
        TEST_ALL_NAME=coco14_test_all
        CONFIG_PATH=configs/coco/dcfs_gfsod_${NET}_novel_${shot}shot_step9_seedx.yaml
        
        OUTPUT_DIR=${SAVEDIR}/dcfs_gfsod_${NET}_novel/tfa-like-${classloss}/${shot}shot_seed${seed}
        python3 main.py --num-gpus ${NUNMGPU} --config-file ${CONFIG_PATH}                      \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}               \
                   MODEL.ROI_BOX_HEAD.BBOX_CLS_LOSS_TYPE ${classloss} \
                   DATASETS.TRAIN "('"${TRAIN_ALL_NAME}"',)" \
                   DATASETS.TEST  "('"${TEST_ALL_NAME}"',)"  \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} \
                   TEST.PCB_MODELTYPE $NET
        #rm ${OUTPUT_DIR}/model_final.pth
    done
done