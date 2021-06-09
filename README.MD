## Project description
The task is amodal scene layout as stated in [monolayout](https://github.com/hbutsuak95/monolayout).

Experiment details: [architecture](current_project.md).

## Installation
We recommend setting up a Python 3.7 Virtual Environment and installing all the dependencies listed in the requirements file.

```
git clone https://github.com/LeMondrok/monolayout_kr
cd monolayout_kr
pip install -r requirements.txt
```
## Data preparation
For KITTI RAW, KITTI Odometry, KITTI 3D Object, Argoverse follow instructions from the original repo [monolayout](https://github.com/hbutsuak95/monolayout).

For nuScenes download dataset from [nuScenes](https://www.nuscenes.org/) (only metadata, keyframes and map-expansion are needed). Then use following scripts:
```
python preprocessing/nuscenes/generate_static_gt.py --nusc_path path --nusc_version version 
python preprocessing/nuscenes/generate_bbox_dynamic_gt.py --nusc_path path --nusc_version version
```
![nuScenes gt image](nuscenes_GT.png)


## Training
For KITTI RAW, KITTI Odometry, KITTI 3D Object, Argoverse follow instructions from the original repo [monolayout](https://github.com/hbutsuak95/monolayout).

For nuScenes launch train script with following flags:
```
GAN:
python train.py --type static --split nuscenes --data_path 'path' --height 1024 --width 1024 --occ_map_size 256 --model_name 'checkpoint_subfolder_name' --use_wandb 0
python train.py --type dynamic --split nuscenes --data_path 'path' --height 1024 --width 1024 --occ_map_size 256 --model_name 'checkpoint_subfolder_name' --use_wandb 0

not GAN:
python train_plain.py --type static --split nuscenes --data_path 'path' --height 1024 --width 1024 --occ_map_size 256 --model_name 'checkpoint_subfolder_name' --use_wandb 0
python train_plain.py --type dynamic --split nuscenes --data_path 'path' --height 1024 --width 1024 --occ_map_size 256 --model_name 'checkpoint_subfolder_name' --use_wandb 0
```
## Test and eval
Same as in the parent repo

## Logging
Add your wandb settings in training script and set --use_wandb to 1.