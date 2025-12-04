
# Introduction
:wave: This repo is based on L4DR -  **LiDAR-4D radar fusion** based 3D object detection on the VoD dataset!


# based on VoD dataset
## Installation

This code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). 

### 1. Clone (or download) the source code 
```
git clone https://github.com/sungeunmik/VoD_baseline_UCSD.git
cd L4DR
```
 
### 2. Create conda environment and set up the base dependencies
```
conda create --name l4dr python=3.8 cmake=3.22.1
conda activate l4dr
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install spconv-cu113
```

### 3. Install pcdet
```
python setup.py develop
```

### 4. Install required environment
```
pip install -r requirements.txt
```

## Getting Started
The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs) (vod related), 
and the model configs are located within [VoD_models](https://github.com/ylwhxht/L4DR/tree/main/tools/cfgs/VoD_models). 


### Dataset Preparation
#### 1. Dataset download
Please follow [VoD Dataset](https://github.com/tudelft-iv/view-of-delft-dataset/blob/main/docs/GETTING_STARTED.md) to download dataset.



After the preparation, the format of how the dataset is provided:

```
View-of-Delft-Dataset (root)
    ├── lidar (kitti dataset where velodyne contains the LiDAR point clouds)
      ...
    ├── radar (kitti dataset where velodyne contains the radar point clouds)
      ...
    ├── radar_3_scans (kitti dataset where velodyne contains the accumulated radar point clouds of 3 scans)
      ...
    ├── radar_5_scans (kitti dataset where velodyne contains the accumulated radar point clouds of 5 scans)
      ...
```



#### 3. Data infos generation
* Firstly, remember to change **DATA-PATH** in the tools/cfgs/dataset_configs/radar_5frames_as_kitti_dataset.yaml .

* Generate the data infos by running the following command: 
```
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/radar_5frames_as_kitti_dataset.yaml 
```

### Training & Testing
First, go to the tools folder:
```
cd tools
```


#### Train a model
You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters. 

* Train with a single GPU:
```shell script
python train.py --cfg_file cfgs/kitti_models/pointpillar_vod.yaml
```

#### Test and evaluate the pretrained models
* We can also provide our pretrained models. If you need it, please feel free to contact me

* Test with a pretrained model: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```

* To test with multiple GPUs:
```shell script
sh scripts/dist_test.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```

For example

```shell script
CUDA_VISIBLE_DEVICES=2,3 bash scripts/dist_test.sh 2 --cfg_file cfgs/VoD_models/L4DR.yaml --extra_tag 'l4dr_demo' --ckpt /mnt/32THHD/hx/Outputs/output/VoD_models/PP_DF_OurGF/mf2048_re/ckpt/checkpoint_epoch_100.pth
```




