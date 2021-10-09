# GGHL
## A General Gaussian Heatmap Labeling for Arbitrary-Oriented Object Detection  
## This is the implementation of GGHL

审稿实在太慢了，三个月了快！！！先在arxiv上挂出来了。代码还没传完，最近会陆续修改和上传。如果熟悉我的NPMMR-Det代码的朋友，可以直接把dataloader的那个标签分配的代码放到那边去，稍微改改检测头的层数就可以跑出来了。正式版我争取一个月内更新完。论文里关于NPMMR-Det和LO-Det的实验代码会在那两个仓库里面更新，NPMMRDet的baseline目前已经更新完了，你们可以试试看能不能跑。LO-Det的正在更新中，可以看那边的说明。万一有AE或者审稿人大佬看到这个仓库，跪求千万别忘了审稿啊~求求，只希望顺利毕业
article{huang2021general,
  title = {A General Gaussian Heatmap Labeling for Arbitrary-Oriented Object Detection},
  author = {Huang, Zhanchao and Li, Wei and Xia, Xiang-Gen and Tao, Ran},
  year = {2021},
  journal = {arXiv preprint arXiv:2109.12848},
  eprint = {2109.12848},
  eprinttype = {arxiv},
  archiveprefix = {arXiv}
}

## Environments
Linux (Ubuntu 18.04, GCC>=5.4) & Windows (Win10, VS2019)   
CUDA 11.1, Cudnn 8.0.4

1. For RTX20/Titan RTX/V100 GPUs
cudatoolkit==10.0.130  
numpy==1.17.3  
opencv-python==3.4.2  
pytorch==1.2.0  
torchvision==0.4.0  
pycocotools==2.0 (In the ./lib folder)  
dcnv2==0.1 (In the ./lib folder)  
...  
The installation of other libraries can be carried out according to the prompts of pip/conda  
  
2. For RTX30 GPUs
cudatoolkit==11.0.221  
numpy==1.17.5  
opencv-python==4.4.0.46  
pytorch==1.7.0  
torchvision==0.8.1  
pycocotools==2.0 (In the ./lib folder)  
dcnv2==0.1 (In the ./lib folder)  
...

## Installation
1. git clone this repository    
2. Install the libraries in the ./lib folder  
(1) DCNv2  
cd ./GGHL/lib/DCNv2/  
sh make.sh  
(2) pycocotools  
cd ./GGHL/lib/cocoapi/PythonAPI/  
sh make.sh  

## Datasets
1. [DOTA dataset](https://captain-whu.github.io/DOTA/dataset.html) and its [devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)
(1) VOC Format  
You need to write a script to convert them into the train.txt file required by this repository and put them in the ./dataR folder.  
For the specific format of the train.txt file, see the example in the /dataR folder.  

## Usage Example
1. train  
python train.py  
2. test  
python test.py  

## To be continued 
