<p align="center">
<h1 align="center"><strong>SplatCo: Structure-View Collaborative Gaussian Splatting for Detail-Preserving Large-Scale Unbounded Scene Rendering</strong></h1>

## Abstract

Neural Radiance Fields (NeRFs) have achieved impressive results in novel view synthesis but are less suited for large-scale scene reconstruction due to their reliance on dense per-ray sampling, which limits scalability and efficiency. In contrast, 3D Gaussian Splatting (3DGS) offers a more efficient alternative to computationally intensive volume rendering, enabling faster training and real-time rendering. Although recent efforts have extended 3DGS to large-scale settings, these methods often struggle to balance global structural coherence with local detail fidelity. Crucially, they also suffer from Gaussian redundancy due to a lack of effective geometric constraints, which further leads to rendering artifacts. To address these challenges, we present SplatCo, a structure–view collaborative Gaussian splatting framework for high-fidelity rendering of complex outdoor scenes. SplatCo builds upon three novel components: (1) a Cross-Structure Collaboration Module (CSCM) that combines global tri-plane representations, which capture coarse scene layouts, with local context grid features that represent fine details. This fusion is achieved through the proposed hierarchical compensation mechanism, ensuring both global spatial awareness and local detail preservation; (2) a Cross-View Pruning Mechanism (CVPM) that prunes overfitted or inaccurate Gaussians based on structural consistency, thereby improving storage efficiency while avoiding Gaussian rendering artifacts; (3) a Structure–View Co-learning (SVC) Module that aggregates structural gradients with view gradients, redirecting the Gaussian geometric and appearance attribute optimization more robustly guided by additional structural gradient flow. By combining these key components, SplatCo effectively achieves high-fidelity rendering for large-scale scenes. Comprehensive evaluations on 13 diverse large-scale scenes, including Mill19, MatrixCity, Tanks & Temples, WHU, and custom aerial captures, demonstrate that SplatCo establishes a new benchmark for high-fidelity rendering of large-scale unbounded scenes.

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
git clone https://github.com/SCUT-BIP-Lab/SplatCo.git
cd SplatCo
unzip ./submoudles.zip
```

2. Install dependencies

```
conda env create -f environment.yml
conda activate splatco
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

You can quickly train the dataset using the following command:

```
conda activate splatco
cd SplatCo
python train.py -s ./data/dataset/scene/  -m ./output/dataset/scene/ --mv 4 --num_channels 15 --plane_size 2800 --no_downsample --port 6555 --contractor --bbox_scale 0.3 --voxel_size 0 --update_init_factor 16 --appearance_dim 0 
```

## Training and Evaluation

You can run other scene datasets by either modifying or executing the following command. For specific file modifications, please contact us [Haihong Xiao](auhhxiao@mail.scut.edu.cn) and [Jianan Zou](202130450216@mail.scut.edu.cn), and we will provide assistance.

```
conda activate splatco
cd SplatCo
python train.py -s <path to COLMAP or NeRF Synthetic dataset>　--eval --mv 4 --num_channels 15 --plane_size 2800 --no_downsample --port 6555 --contractor --bbox_scale 0.3 --voxel_size 0 --update_init_factor 16 --appearance_dim 0 
python render.py -m <path to trained model>
python metrics.py -m <path to trained model>
```

## Results

Visual comparisons on Mill-19 and MatrixCity dataset:

<p align="center">
<img src="./results/fig_mill.png" width=100% height=100% 
class="center">
</p>

Visual comparisons on tandt outdoor scenes:

<p align="center">
<img src="./results/fig_tnt.png" width=100% height=100% 
class="center">
</p>

Visual comparisons on qualitative comparison of ablation study on cross-structure collaborated module:

<p align="center">
<img src="fig_ablation_s.png" width=100% height=100% 
class="center">
</p>

## Acknowledgements

Our code follows several awesome repositories. We appreciate them for making their codes available to public.

### Sincere Appreciation and Apology Regarding MVGS

Our structure–view colearning module builds on MVGS, and we gratefully acknowledge this inspiration. The earlier omission of this attribution in our preprint has now been fully corrected in the latest arXiv revision.

[CityGS-X](https://github.com/gyy456/CityGS-X)

[MVGS](https://github.com/xiaobiaodu/mvgs)

[Mega-NeRF](https://github.com/cmusatyalab/mega-nerf)

[Switch-NeRF](https://github.com/MiZhenxing/Switch-NeRF)

[3DGS](https://github.com/graphdeco-inria/gaussian-splatting) 

[GaMeS](https://github.com/waczjoan/gaussian-mesh-splatting) 

[Compact3DGS](https://github.com/maincold2/Compact-3DGS) 

[Scaffold-GS](https://github.com/city-super/Scaffold-GS)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SCUT-BIP-Lab/SplatCo&type=Date)](https://www.star-history.com/#SCUT-BIP-Lab/SplatCo&Date)
