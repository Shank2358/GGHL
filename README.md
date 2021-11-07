# GGHL: A General Gaussian Heatmap Labeling for Arbitrary-Oriented Object Detection  
This is the implementation of GGHL  

If you use it, please give this project a star and consider citing:

article{huang2021general,
  title = {A General Gaussian Heatmap Labeling for Arbitrary-Oriented Object Detection},
  author = {Huang, Zhanchao and Li, Wei and Xia, Xiang-Gen and Tao, Ran},
  year = {2021},
  journal = {arXiv preprint arXiv:2109.12848},
  eprint = {2109.12848},
  eprinttype = {arxiv},
  archiveprefix = {arXiv}
}  

## 11.8：打算写一个数据预处理的教程和算法、代码的讲解，预计12月上线
## 11.8：I plan to write a tutorial on data preprocessing and explanation of algorithms and codes, which is expected to be launched in December

## 11.7：1.0版本全部更新完成了，欢迎使用，有任何问题可以在issue留言，谢谢。接下来会不断更新和完善  
## November 7: All updates of GGHL have been completed. Welcome to use it. If you have any questions, you can leave a message at the issue. Thank you.

![image](https://github.com/Shank2358/GGHL/blob/main/readme_imgs/GGHL_results.png)  

## Environments
Linux (Ubuntu 18.04, GCC>=5.4) & Windows (Win10, VS2019)   
CUDA > 11.1, Cudnn > 8.0.4

conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.1 -c pytorch -c conda-forge   
pip install -r requirements.txt  

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
(2) COCO Format...
to be continued


## Usage Example
1. train  
python train_GGHL.py  
2. For Distributed Training  
sh train_GGHL_dist.sh  
4. test  
python test.py  


## To be continued 

## 11.6更新了标签分配和dataload。更新了pytorch1.10版本的支持。预告一下，下周会更新分布式训练的内容。
（预训练权重的链接在NPMMR-Det和LO-Det的仓库说明里）

## 正文开始前的惯例的碎碎念（可以跳过直接看正文使用说明）
投稿排队实在太慢了，三个月了还在形式审查没分配AE,555~ 先在arxiv上挂出来了。代码还没传完，最近会陆续修改和上传。如果熟悉我的NPMMR-Det代码的朋友，可以直接把dataloader的那个标签分配的代码放到那边去，稍微改改检测头的层数就可以跑出来了。正式版我争取一个月内更新完。方法和代码的任何问题都欢迎大家批评指正，issues或者邮箱都可以联系到我，感谢各位大佬。 
等正式版出来以后，我会尽最大努力帮助大家跑通代码和复现出接近论文报道结果的实验，因为我自己也被坑多了，好多遥感领域的论文不开源代码或者根本复现不出来，或者就是模型复杂到眼花缭乱换个数据/参数就失灵，实在是太难了。论文里关于NPMMR-Det和LO-Det的实验代码会在那两个仓库里面更新，NPMMRDet的baseline目前已经更新完了，你们可以试试看能不能跑。LO-Det的正在更新中，可以看那边的说明(11.1也更新了)。 
万一有AE或者审稿人大佬看到这个仓库，跪求千万别忘了审稿啊~求求，我只希望顺利毕业就好。  
10.24 终于分配AE和审稿人了，不容易啊。  
关于论文超参数合实验的一些说明。论文里报道的训练超参数都没有精调，就选的对比方法一样的默认参数，也没有选最好的epoch的结果，直接固定了最大epoch，选择最后五个epoch的平均结果。精调学习率、训练策略合最好轮次还会涨点，最近有空闲的机器我试了一下。但是我觉得像很多论文那样为了state-of-the-art（SOTA）而SOTA没有必要，所以最后没那样做，后续如果审稿意见有这个建议我可能会再修改，如果没有我会把更多的实验结果在github和arxiv上展示出来。反思自己最近的工作，确实比不上各位大佬前辈的创新想法，这点还要继续努力。由于我也是自己一路磕磕绊绊摸索着进入科研领域的，也踩过很多坑，也被各种卷王卷的透不过气，所以我想追求的是想做一些踏实的、简单实用的工作，设计一个皮实、经得起折腾的模型，而不想去卷什么SOTA（实话是我也卷不过。。。。）。   
说一个我对目标检测的理解，请大家批评指正。在我看来，目标检测只是一个更庞大的视觉系统的入口任务而不是最终结果。我觉得大多数检测任务的目标是快速、粗略地在图像/视频中定位到目标候选区域，为后续更精细的比如分割、跟踪等其他任务服务，简化它们的输入。从这个视角来看，检测平均精度差距那么一两个点真的没论文里吹的那么重要，反而检测效率（速度）、模型的复杂度与鲁棒性、易用性（无论是对工程人员还是新入门的研究人员而言）的提升对于社区的贡献会更实际一些。最近几个月我也一直在反思自己，目标检测的初心是什么，目标检测完了然后呢，原来我写论文以为的终点很多时候只是我以为的，原来我想错了。深度学习火了这么些年，很多任务或许也是这样的吧，毕竟论文实验里的SOTA是有标准答案的考试，而它们的开花结果是一个开放性问题。这是接下来的努力方向，我相信哪怕道阻且长，行则将至，而且行而不辍，未来一定可期。


![image](https://github.com/Shank2358/GGHL/blob/main/readme_imgs/GGHL.png)   
