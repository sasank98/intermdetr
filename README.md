# INTERMDETR

This model is constructed to work on encoders constructed for a different task and was implemented on "Unimatch". 
The model is made from deformable-DETR, but for 3D object detection
3D-object detection and stereo disparity on stereo cameras

## Installation and prerequisites
Note: Make sure you are in this current directory
Adding Unimatch to the directory, cloning this repo and unimatch

### Create a conda virtual environment before starting with the project
```
    conda create -n [YOUR_ENV_NAME] python=3.8.13 anaconda
```
### Install Pytorch based on your CUDA version 
This repository has been tested on CUDA 11.3 and Pytorch 1.11, the command I used is given below
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
You can find different versions here: https://pytorch.org/get-started/previous-versions/
### Install all other dependencies through the YAML file
```
    conda install -f intermdetr.yaml
```
### Install Multi scale Deformable Attention
This module is from Deformable-DETR, to install it run the following commands
```
    cd ./intermdetr/ops
    sh ./make.sh
    python test.py
```

## Kitti Dataset Preparation

Download [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) datasets and prepare the directory structure as:
```
    │stereo_cameras/
    ├──intermdetr/
    ├──unimatch/
    ├──data/KITTIDataset/
    │   ├──ImageSets/
    │   ├──training/
            ├──image_2/
            ├──image_3/
            ├──backbone/
    │   ├──testing/
    ├──...
```

This model is made to use intemediate layer from unimatch to reduce the computational over-head, we can use ```@torch.no_grad``` command to get the outputs from desired layers, instead of doing that we can save the intermediate layer outputs as .npz file and directly load them, this would be similar to runinng ```@torch.no_grad``` but we don't need to run unimatch model every time 



to prepeare the dataset accordingly run the following commands:
 
Make a parent directory and clone the 'unimatch' and 'intermdetr' repos into it, just like the above given tree

```
cd ~/stereo_cameras
git clone https://github.com/autonomousvision/unimatch.git
git clone https://github.com/sasank98/intermdetr.git
```
use the script in no_grad.py file to save the intermediate layers from unimatch and then train the deformable-detr model by considering intermediate layers as the input to it
```
    conda activate [YOUR_ENV_NAME]
    cd ./intermdetr 
    python no_grad.py
```

## Network Architecture

Made changes to the deformable-DETR architecture to accomodate for 3D object detection

![alt text](/logistics/Model_architecture.png)

Made only a Decoder and used the intermediate layer outputs from Unimatch, in this method we will be reducing the computational over head



## Getting Started

In cases of training the same architecture make changes to the 'cfg' variable in main.py file and run the following command in the virtual environment in this directory

```
    conda activate [YOUR_ENV_NAME]
    python main.py
```
NOTE: the Hyper-parameters for this model are controlled using the cfg variable in main.py file, currently working in progress to convert that to a .yaml file just like the [MonoDETR](https://github.com/ZrrSkywalker/MonoDETR/tree/main) repo

The model doesn't perform well for 3D object detection for using it parallel to unimatch, I got an AP3D of 9.87 for car on kitti evaluation dataset and the model seems to be underfitted. We can further try this technique on models built for other tasks like multi-view images

## Acknowlegment

This Repo used the training and testing pipelines from [MonoDETR](https://github.com/ZrrSkywalker/MonoDETR/tree/main)

