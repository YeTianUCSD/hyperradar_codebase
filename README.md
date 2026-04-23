
# Introduction
:wave: This repository is a **Hyperdimensional Computing (HDC)** based 4D radar object detection repo built on top of L4DR and OpenPCDet.

## Installation

This code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). 

```shell
git clone https://github.com/YeTianUCSD/hyperradar_codebase.git
cd hyperradar_codebase

conda create --name l4dr python=3.8 cmake=3.22.1
conda activate l4dr
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install spconv-cu113
pip install torch-hd
pip install shapely

python setup.py develop
pip install -r requirements.txt
```

## Getting Started
The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs) (VoD related),
and the model configs are located within [VoD_models](https://github.com/ylwhxht/L4DR/tree/main/tools/cfgs/VoD_models).

### Dataset Preparation
#### VoD Dataset
##### 1. Download Dataset
Follow [VoD Dataset](https://github.com/tudelft-iv/view-of-delft-dataset/blob/main/docs/GETTING_STARTED.md) to download and arrange the dataset.

##### 2. Arrange Dataset Layout
Expected layout:

```
View-of-Delft-Dataset (root)
    ├── lidar
    │   └── ...
    ├── radar
    │   └── ...
    ├── radar_3_scans
    │   └── ...
    ├── radar_5_scans
    │   └── ...
```

##### 3. Update Dataset Path
Before generating infos, update `DATA_PATH` in:

- `tools/cfgs/dataset_configs/radar_5frames_as_kitti_dataset.yaml`
- `pcdet/datasets/kitti/kitti_dataset.py`

##### 4. Generate Dataset Infos
Generate dataset infos with:

```shell
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/radar_5frames_as_kitti_dataset.yaml
```

#### Training & Testing
```shell
cd tools
python train.py --cfg_file cfgs/kitti_models/pointpillar_vod.yaml

python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
sh scripts/dist_test.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```

You can optionally add `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to training.

Example multi-GPU evaluation:

```shell
CUDA_VISIBLE_DEVICES=2,3 bash scripts/dist_test.sh 2 --cfg_file cfgs/VoD_models/L4DR.yaml --extra_tag 'l4dr_demo' --ckpt /mnt/32THHD/hx/Outputs/output/VoD_models/PP_DF_OurGF/mf2048_re/ckpt/checkpoint_epoch_100.pth
```

#### HyperRadar Pipeline

##### 1. Pretrain pointpillar model

```shell
cd tools
python -u train.py \
  --cfg_file cfgs/kitti_models/pointpillar_vod_hd.yaml \
  --extra_tag run100_cls \
  --batch_size 16 \
  --epochs 100
```

##### 2. Evaluate pointpillar model

```shell
cd tools
python test.py \
  --cfg_file /home/code/hyperradar/hyperradar_codebase/tools/cfgs/kitti_models/pointpillar_vod_hd.yaml \
  --ckpt /home/code/hyperradar/hyperradar_codebase/output/kitti_models/pointpillar_vod_hd/run100_cls/ckpt/checkpoint_epoch_17.pth \
  --batch_size 16 \
  --workers 4 \
  --extra_tag pillar_vod_hd_retrain/run100_cls
```

##### 3. Retrain HD Branch
Freeze the pretrained CNN feature extractor and retrain only the HD branch.

```shell
bash /home/code/hyperradar/hyperradar_codebase/tools/bash/pretrain_model.sh
```

##### 4. Online Update HD
Use the pretrained CNN+HD model and update only the HD component online.

Unsupervised:

```shell
bash /home/code/hyperradar/hyperradar_codebase/tools/bash/online_update_hd_unsupervised.sh
```

Supervised:

```shell
bash /home/code/hyperradar/hyperradar_codebase/tools/bash/online_update_hd_supervised.sh
```

##### 5. Online Update NN
Use the pretrained model and update only the CNN classification head online.

Unsupervised:

```shell
bash /home/code/hyperradar/hyperradar_codebase/tools/bash/online_update_nn_unsupervised.sh
```

Supervised:

```shell
bash /home/code/hyperradar/hyperradar_codebase/tools/bash/online_update_nn_supervised.sh
```
