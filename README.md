<p align="center">
<h1 align="center"><strong>Capturing Fine-Grained Details via Structure-View Collaborative Learning with Gaussian Splatting</strong></h1>

## Overview

In this work, we present Structure–View Collaborative Gaussian Splatting (SV‑GS), a unified framework designed to capture ultra‑fine details by jointly modeling global scene structure and enforcing multi‑view consistency. Specifically, we first fuse multi‑scale contextual cues into a global tri‑plane backbone via a novel three‑level compensation mechanism, seamlessly blending coarse layout with intricate local geometry and texture. To prevent any single camera from dominating the reconstruction, we then synchronize gradient updates across all overlapping views, ensuring that each Gaussian’s position and appearance remain coherent under varied viewpoints. Finally, our visibility‑aware optimization adaptively injects new Gaussians in sparsely observed regions while employing structure‑consistency‑based pruning of single‑view–overfitted Gaussians to eliminate overfitted primitives, delivering robust generalization across large‑scale, complex environments.

<p align="center">
<img src="fig_cover.png" width=100% height=100% 
class="center">
</p>

## Dataset

We performe experiments on eight scenes from four public datasets from the Mega-NeRF, MatrixCity, 3D Gaussian Splatting and WHU dataset.

Mill-19 dataset:
Please download the data from the [Mega-NeRF](https://github.com/cmusatyalab/mega-nerf)

MatrixCity-Aerial dataset:
Please download the data from the [MatrixCity](https://city-super.github.io/matrixcity/)

Tanks & Temples dataset:
Please download the data from the [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

WHU dataset:
Please download the data from the [WHU dataset](http://gpcv.whu.edu.cn/data/)

We performe experiments on five self-collected scenes(SCUT Campus and plateau regions).

Please contact us:

Haihong Xiao and Jianan Zou: auhhxiao@mail.scut.edu.cn; 202130450216@mail.scut.edu.cn

Prof. Wenxiong Kang: auwxkang@scut.edu.cn

## Installation

We tested on a server configured with Ubuntu 20.04, cuda 11.8 and gcc 9.4.0. Other similar configurations should also work, but we have not verified each one individually.

1. Clone this repo:

```
git clone https://github.com/SCUT-BIP-Lab/SV-GS.git
cd SV-GS
unzip ./submoudles.zip
```

2. Install dependencies

```
conda env create -f environment.yml
conda activate svgs
```

3、Data preparation

First, create a ```data/``` folder inside the project path by 

```
mkdir data
```

The data structure will be organised as follows:

```
data/
├── dataset_name
│   ├── scene1/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
│   ├── scene2/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
...
```

## Quick Start

You can quickly train the  dataset using the following command:

```
conda activate svgs
cd SV-GS
python train.py -s ./data/dataset/scene/  -m ./output/dataset/scene/ --mv 4 --num_channels 15 --plane_size 2800 --no_downsample --port 6555 --contractor --bbox_scale 0.3 --voxel_size 0 --update_init_factor 16 --appearance_dim 0 
```

## Training and Evaluation

You can run other scene datasets by either modifying or executing the following command. For specific file modifications, please contact us [Haihong Xiao](auhhxiao@mail.scut.edu.cn) and [Jianan Zou](202130450216@mail.scut.edu.cn), and we will provide assistance.

```
conda activate svgs
cd SV-GS
python train.py -s <path to COLMAP or NeRF Synthetic dataset>　--eval --mv 4 --num_channels 15 --plane_size 2800 --no_downsample --port 6555 --contractor --bbox_scale 0.3 --voxel_size 0 --update_init_factor 16 --appearance_dim 0 
python render.py -m <path to trained model>
python metrics.py -m <path to trained model>
```

## Results

Visual comparisons on Mill-19 and MatrixCity dataset:

<p align="center">
<img src="fig_rub.png" width=100% height=100% 
class="center">
</p>

Visual comparisons on tandt outdoor scenes:

<p align="center">
<img src="fig_tandt.png" width=100% height=100% 
class="center">
</p>

Visual comparisons on qualitative comparison of ablation study on cross-structure collaborated module:

<p align="center">
<img src="fig_ablation_s.png" width=100% height=100% 
class="center">
</p>

## Acknowledgements

Our code follows several awesome repositories. We appreciate them for making their codes available to public.

[Mega-NeRF](https://github.com/cmusatyalab/mega-nerf)

[Switch-NeRF](https://github.com/MiZhenxing/Switch-NeRF)

[3DGS](https://github.com/graphdeco-inria/gaussian-splatting) 

[GaMeS](https://github.com/waczjoan/gaussian-mesh-splatting) 

[Compact3DGS](https://github.com/maincold2/Compact-3DGS) 

[Scaffold-GS](https://github.com/city-super/Scaffold-GS) 
