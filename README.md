# Learnable and Anchor-Based Sampling for 3D Shape Completion

Created By - Akash Bankar, MTP Guide - Prof. Ayan Chaudhury


[[Dataset]](./DATASET.md) [[Models]](#pretrained-models) [[supp]](https://yuxumin.github.io/files/PoinTr_supp.pdf)

This repository contains PyTorch implementation for Learnable and Anchor-Based Sampling for 3D Shape Completion

| dataset  | url| performance |
| --- | --- |  --- |
| ShapeNet-55 | Tsinghua Cloud / Google Drive / BaiDuYun  | CD = 0.81e-3|
| ShapeNet-34 | Tsinghua Cloud / Google Drive / BaiDuYun | CD = 1.23e-3| 
| Projected_ShapeNet-55 | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/41ed3a765c4b42d98d01/?dl=1) / Google Drive / [[BaiDuYun](https://pan.baidu.com/s/1Vx-E557-dOj7dLi132--Uw?pwd=dycc)](code:dycc)  | CD = 9.58e-3|
| Projected_ShapeNet-34 | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/71494f78cb694e45a448/?dl=1) / Google Drive / [[BaiDuYun](https://pan.baidu.com/s/1GQnfJuxtpV5Mchl-98BRBg?pwd=dycc)](code:dycc)  | CD = 9.12e-3|
| PCN |  [[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b822a5979762417ba75e/?dl=1)] / [[Google Drive](https://drive.google.com/file/d/17pE2U2T2k4w1KfmDbL6U-GkEwD-duTaF/view?usp=share_link)]  / [[BaiDuYun](https://pan.baidu.com/s/1KWccgcKXVIdVo4wJAmZ_8w?pwd=rc7p)](code:rc7p)  | CD = 6.53e-3|
## Usage

### Requirements

- PyTorch >= 1.7.0
- python >= 3.7
- CUDA >= 9.0
- GCC >= 4.9 
- torchvision
- timm
- open3d
- tensorboardX

```
pip install -r requirements.txt
```

#### Building Pytorch Extensions for Chamfer Distance, PointNet++ and kNN

*NOTE:* PyTorch >= 1.7 and GCC >= 4.9 are required.

```
# Chamfer Distance
bash install.sh
```
The solution for a common bug in chamfer distance installation can be found in Issue [#6](https://github.com/yuxumin/PoinTr/issues/6)
```
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

Note: If you still get `ModuleNotFoundError: No module named 'gridding'` or something similar then run these steps

```
    1. cd into extensions/Module (eg extensions/gridding)
    2. run `python setup.py install`
```

That will fix the `ModuleNotFoundError`.


### Dataset

The details of our new ***ShapeNet-55/34*** datasets and other existing datasets can be found in [DATASET.md](./DATASET.md).

### Inference

To inference sample(s) with pretrained model

```
python tools/inference.py \
${POINTR_CONFIG_FILE} ${POINTR_CHECKPOINT_FILE} \
[--pc_root <path> or --pc <file>] \
[--save_vis_img] \
[--out_pc_root <dir>] \
```

For example, inference all samples under `demo/` and save the results under `inference_result/`
```
python tools/inference.py \
cfgs/PCN_models/AdaPoinTr.yaml ckpts/AdaPoinTr_PCN.pth \
--pc_root demo/ \ 
--save_vis_img  \
--out_pc_root inference_result/ \
```

### Evaluation

To evaluate a pre-trained PoinTr model on the Three Dataset with single GPU, run:

```
bash ./scripts/test.sh <GPU_IDS>  \
    --ckpts <path> \
    --config <config> \
    --exp_name <name> \
    [--mode <easy/median/hard>]
```

####  Some examples:
Test the PoinTr (AdaPoinTr) pretrained model on the PCN benchmark or Projected_ShapeNet:
```
bash ./scripts/test.sh 0 \
    --ckpts ./pretrained/PoinTr_PCN.pth \
    --config ./cfgs/PCN_models/PoinTr.yaml \
    --exp_name example

bash ./scripts/test.sh 0 \
    --ckpts ./pretrained/PoinTr_ps55.pth \
    --config ./cfgs/Projected_ShapeNet55_models/AdaPoinTr.yaml \
    --exp_name example
```
Test the PoinTr pretrained model on ShapeNet55 benchmark (*easy* mode):
```
bash ./scripts/test.sh 0 \
    --ckpts ./pretrained/PoinTr_ShapeNet55.pth \
    --config ./cfgs/ShapeNet55_models/PoinTr.yaml \
    --mode easy \
    --exp_name example
```
Test the PoinTr pretrained model on the KITTI benchmark:
```
bash ./scripts/test.sh 0 \
    --ckpts ./pretrained/PoinTr_KITTI.pth \
    --config ./cfgs/KITTI_models/PoinTr.yaml \
    --exp_name example
CUDA_VISIBLE_DEVICES=0 python KITTI_metric.py \
    --vis <visualization_path> 
```

### Training

To train a point cloud completion model from scratch, run:

```
# Use DistributedDataParallel (DDP)
bash ./scripts/dist_train.sh <NUM_GPU> <port> \
    --config <config> \
    --exp_name <name> \
    [--resume] \
    [--start_ckpts <path>] \
    [--val_freq <int>]
# or just use DataParallel (DP)
bash ./scripts/train.sh <GPUIDS> \
    --config <config> \
    --exp_name <name> \
    [--resume] \
    [--start_ckpts <path>] \
    [--val_freq <int>]
```
####  Some examples:
Train a Gumbel - PoinTr model on PCN benchmark with 2 gpus:
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 \
    --config ./cfgs/PCN_models/PoinTr.yaml \
    --exp_name example
```
Resume a checkpoint:
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 \
    --config ./cfgs/PCN_models/PoinTr.yaml \
    --exp_name example --resume
```

Finetune a Gumbel - PoinTr on PCNCars
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 \
    --config ./cfgs/KITTI_models/PoinTr.yaml \
    --exp_name example \
    --start_ckpts ./weight.pth
```

Train a Gumbel - PoinTr model with a single GPU:
```
bash ./scripts/train.sh 0 \
    --config ./cfgs/KITTI_models/PoinTr.yaml \
    --exp_name example
```

We also provide the Pytorch implementation of several baseline models including GRNet, PCN, TopNet and FoldingNet. For example, to train a GRNet model on ShapeNet-55, run:
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 \
    --config ./cfgs/ShapeNet55_models/GRNet.yaml \
    --exp_name example
```
