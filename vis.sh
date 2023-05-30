NET=$1
EXPNAME=$2
shot=10
seed=0
SAVEDIR=workspace/DCFS/coco-det/${EXPNAME}
NOVEL_WEIGHT=${SAVEDIR}/dcfs_gfsod_${NET}_novel/tfa-like-DC/${shot}shot_seed${seed}/model_final.pth
PRTRAINEDMODEL=pretrained_models/ImageNetPretrained
IMAGENET_PRETRAIN_TORCH=${PRTRAINEDMODEL}/torchvision/resnet101-5d3b4d8f.pth
classloss="DC" # "CE"
TRAIN_ALL_NAME=coco14_trainval_all_${shot}shot_seed${seed}
TEST_ALL_NAME=coco14_test_all
CONFIG_PATH=configs/coco/dcfs_gfsod_${NET}_novel_${shot}shot_step8_seedx.yaml

python3 visualize.py --config-file ${CONFIG_PATH}                      \
            --opts MODEL.WEIGHTS ${NOVEL_WEIGHT}             \
                   DATASETS.TEST  "('"${TEST_ALL_NAME}"',)"  \
                   MODEL.ROI_BOX_HEAD.BBOX_CLS_LOSS_TYPE ${classloss} \
                   DATASETS.TRAIN "('"${TRAIN_ALL_NAME}"',)" \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} \
                   TEST.PCB_MODELTYPE $NET