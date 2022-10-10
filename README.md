# GGHL: A General Gaussian Heatmap Label Assignment for Arbitrary-Oriented Object Detection  

  <a href="https://github.com/Shank2358/GGHL/">
    <img alt="Version" src="https://img.shields.io/badge/Version-1.3.0-blue" />
  </a>
  
  <a href="https://github.com/Shank2358/GGHL/blob/main/LICENSE">
    <img alt="GPLv3.0 License" src="https://img.shields.io/badge/License-GPLv3.0-blue" />
  </a>
  
  <a href="https://github.com/Shank2358" target="_blank">
  <img src="https://visitor-badge.glitch.me/badge?page_id=gghl.visitor-badge&right_color=blue"
  alt="Visitor" />
</a> 

<a href="mailto:zhanchao.h@outlook.com" target="_blank">
   <img alt="E-mail" src="https://img.shields.io/badge/To-Email-blue" />
</a> 

## This is the implementation of GGHL 👋👋👋
[[Arxiv](https://arxiv.org/abs/2109.12848)] [[IEEE TIP](https://ieeexplore.ieee.org/document/9709203)] [[Google Drive](https://drive.google.com/drive/folders/16k7JW-eb3jbga1xzq6B6r60gl2XniXfn?usp=sharing)][[Baidu Disk](https://pan.baidu.com/s/12MD7XAL6iwVUHMHRkEcLWA) (password: yn04)]  

#### IEEE的正式版排版丑死了（不得不吐槽），有的图还显示有问题，还是下载arxiv版本的吧，我已经更新了arxiv，内容和正式版一样。  

  ### Give a ⭐️ if this project helped you. If you use it, please consider citing:
  ```IEEE TIP
  @ARTICLE{9709203,
  author={Huang, Zhanchao and Li, Wei and Xia, Xiang-Gen and Tao, Ran},
  journal={IEEE Transactions on Image Processing}, 
  title={A General Gaussian Heatmap Label Assignment for Arbitrary-Oriented Object Detection}, 
  year={2022},
  volume={31},
  number={},
  pages={1895-1910},
  doi={10.1109/TIP.2022.3148874}}
  ```

### Clone不Star,都是耍流氓 🤡🤡🤡
  
<p algin="center">
<img src="https://github.com/Shank2358/GGHL/blob/main/readme_imgs/GGHL_results.png" width="380"><img src="https://github.com/Shank2358/GGHL/blob/main/readme_imgs/GGHL.png" width="430">
</p>
<p algin="center">
<img src="https://user-images.githubusercontent.com/33946139/147258790-8b0ddf42-fbd6-4372-8598-70d67d5de328.png" width="700">
</p>

## 0. Something Important 🦞 🦀 🦑 
* #### 💖💖💖 Thanks to [Crescent-Ao](https://github. com/Crescent-Ao) and [haohaoolalahao](https://github.com/haohaoolalahao) for contributions to the GGHL repository, thanks to [Crescent-Ao](https://github.com/Crescent-Ao) for the GGHL deployment Version. Relevant warehouses will continue to be updated, so stay tuned.  
    #### 打个广告，GGHL部署版本[GGHL-Deployment](https://github.com/Crescent-Ao/GGHL-Deployment)已经上线，欢迎大家使用~~ 感谢我最亲爱的师弟[Crescent-Ao](https://github.com/Crescent-Ao)和[haohaolalahao](https://github.com/haohaolalahao)对GGHL仓库的贡献，感谢[Crescent-Ao](https://github.com/Crescent-Ao)完成的GGHL部署版本。相关仓库还会持续更新中，敬请期待。

* #### 😺😺😺 Welcome everyone to pay attention to the MGAR completed by [haohaoolalahao](https://github.com/haohaoolalahao) in cooperation with me, which has been accepted by [IEEE TGRS](https://ieeexplore.ieee.org/document/9912396).  
    #### 再打个广告，欢迎大家关注[haohaolalahao](https://github.com/haohaolalahao)与我合作完成的遥感图像目标检测工作 MGAR: Multi-Grained Angle Representation for Remote Sensing Object Detection，论文已经正式接收[IEEE TGRS](https://ieeexplore.ieee.org/document/9912396) [Arxiv](https://arxiv.org/abs/2209.02884), 感谢大家引用：
  ```IEEE TGRS
    @ARTICLE{9912396,
      author={Wang, Hao and Huang, Zhanchao and Chen, Zhengchao and Song, Ying and Li, Wei},
      journal={IEEE Transactions on Geoscience and Remote Sensing}, 
      title={Multi-Grained Angle Representation for Remote Sensing Object Detection}, 
      year={2022},
      volume={},
      number={},
      pages={1-1},
      doi={10.1109/TGRS.2022.3212592}}
  ```  
* #### 🙆‍♂️🙆‍♂️Fixed multi-scale training bugs when torch>=1.7 and using distributed training. Please update the pytorch to version1.11. Thanks to [@haohaolalahao](https://github.com/haohaolalahao).
* #### 最近在学习MMRotate，后续有计划在MMRotate框架下写一版GGHL,先立个FLAG🤖🤖    
* #### 关于在去年年底前出GGHLv2的FLAG不出所料的倒掉了🤣🤣🤣，我是大鸽子🕊️🕊️🕊️🕊️咕咕咕。写论文对我来说好难啊啊啊啊，重新扶起这个FLAG，两个月后把论文写完吧（...实验早跑完了，现在每天憋出100个字）  
* #### 谢谢大家的反馈和ISSUES里面的各种意见，非常感谢🥰🥰🥰   
* #### 🧸 The OpenCV version needs to be >=4.5.3, the label conversion script is ./datasets_tools/[DOTA2Train.py](https://github.com/Shank2358/GGHL/blob/main/datasets_tools/DOTA2Train.py)
* #### 🚀 See [issues #4](https://github.com/Shank2358/GGHL/issues/4) for distributed training (pytorch==1.11).
* #### 🌟 I have updated the function of polyiou and polynms. The code of latest evaluator is in [./evalR/evaluatorGGHL_new.py](https://github.com/Shank2358/GGHL/blob/main/evalR/evaluatorGGHL_new.py)
* #### 🥂 Thanks [@trungpham2606](https://github.com/trungpham2606), [@lalalagogogo](https://github.com/lalalagogogochong), [@zhuyh1223](https://github.com/zhuyh1223) and [@aiboys](https://github.com/aiboys) for the suggestions and feedback.  
* #### 🦄 Thanks to my collaborator [@haohaolalahao](https://github.com/haohaolalahao) for his great contribution to this project. The discussion of “Refined approximation of OBBs” in the revised paper was proposed by [@haohaolalahao](https://github.com/haohaolalahao) and I when we participated in a remote sensing object detection competition in 2020.  
* #### 🤖 Thanks to [@Crescent-Ao](https://github.com/Crescent-Ao) for his suggestions on the revised manuscript and his great contribution to publicizing this work.  
* #### 😸 I am very grateful for every discussion between [@haohaolalahao](https://github.com/haohaolalahao), [@Crescent-Ao](https://github.com/Crescent-Ao) and me. Thank you for making this work grow. Although it is not mature yet, I believe it will get better and better with our efforts.  
* #### 中文版的说明可以看知乎上[@Crescent-Ao](https://github.com/Crescent-Ao)写的文章：[Gaussian heatmap label assignment](https://zhuanlan.zhihu.com/p/452595960)，[GGHL代码解读](https://zhuanlan.zhihu.com/p/465668926)和[Oriented-bounding-box representation component (ORC)](https://zhuanlan.zhihu.com/p/448339670)，还在陆续更新中。欢迎大家点赞、喜欢、收藏三连。

## 🌈 1.Environments
Linux (Ubuntu 18.04, GCC>=5.4) & Windows (Win10)   
CUDA > 11.1, Cudnn > 8.0.4

First, install CUDA, Cudnn, and Pytorch.
Second, install the dependent libraries in [requirements.txt](https://github.com/Shank2358/GGHL/blob/main/requirements.txt). 

```python
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch 
pip install -r requirements.txt  
```
    
## 🌟 2.Installation
1. git clone this repository    

2. Polygen NMS  
The poly_nms in this version is implemented using shapely and numpy libraries to ensure that it can work in different systems and environments without other dependencies. But doing so will slow down the detection speed in dense object scenes. If you want faster speed, you can compile and use the poly_iou library (C++ implementation version) in datasets_tools/DOTA_devkit. The compilation method is described in detail in [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) .

```bash
cd datasets_tools/DOTA_devkit
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace 
```   
  
## 🎃 3.Datasets
1. [DOTA dataset](https://captain-whu.github.io/DOTA/dataset.html) and its [devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)  

#### (1) Training Format  
You need to write a script to convert them into the train.txt file required by this repository and put them in the ./dataR folder.  
For the specific format of the train.txt file, see the example in the /dataR folder.   

```txt
image_path xmin,ymin,xmax,ymax,class_id,x1,y1,x2,y2,x3,y3,x4,y4,area_ratio,angle[0,180) xmin,ymin,xmax,ymax,class_id,x1,y1,x2,y2,x3,y3,x4,y4,area_ratio,angle[0,180)...
```  
The calculation method of angle is explained in [Issues #1](https://github.com/Shank2358/GGHL/issues/1) and our paper.

#### (2) Validation & Testing Format
The same as the Pascal VOC Format

#### (3) DataSets Files Structure
  ```
  cfg.DATA_PATH = "/opt/datasets/DOTA/"
  ├── ...
  ├── JPEGImages
  |   ├── 000001.png
  |   ├── 000002.png
  |   └── ...
  ├── Annotations (DOTA Dataset Format)
  |   ├── 000001.txt (class_idx x1 y1 x2 y2 x3 y3 x4 y4)
  |   ├── 000002.txt
  |   └── ...
  ├── ImageSets
      ├── test.txt (testing filename)
          ├── 000001
          ├── 000002
          └── ...
  ```  
There is a DOTA2Train.py file in the datasets_tools folder that can be used to generate training and test format labels.
First, you need to use [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) , the official tools of the DOTA dataset, for image and label splitting. Then, run DOTA2Train.py to convert them to the format required by GGHL. For the use of DOTA_devkit, please refer to the tutorial in the official repository.

## 🌠🌠🌠 4.Usage Example
#### (1) Training  
```python
python train_GGHL.py
```

#### (2) For Distributed Training  

```bash
sh train_GGHL_dist.sh
```

#### (3) Testing  
```python
python test.py
```
  
  
## ☃️❄️ 5.Weights
假装有表格...懒得画了  
1）The trained model for DOTA dataset is available from [Google Drive](https://drive.google.com/file/d/13yrGQTcA3xLf6TPsAA1cVTF0rAUk6Keg/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1aZ-bnNUAqJHqfOThK4tm5A) (password: 2dm8)  
Put them in. /weight folder

2）The trained model for SKU dataset is available from [Google Drive](https://drive.google.com/file/d/1l3FzZCUWpWL9adKIovQXi1EaHHzaOdAW/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1i8821aD0B-YNBkKXo-_CJg)(password: c3jv)   

3）The trained model for SKU dataset is available from [Google Drive](https://drive.google.com/file/d/19-dlqNaXJyKboJ5bH-UUiXEVJqrdNyAt/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1n7siqE0w49rkOqtTvkZaIA)(password: vdf5)  

4）The pre-trained weights of Darknet53 on ImageNet are available from [Google_Drive](https://drive.google.com/file/d/1vfUDVeI12cBCgSgKMBz39gxkD1Z7kmFd/view?usp=sharing) or [Baidu_Disk](https://pan.baidu.com/s/1DZhooaEClu6rOnC7lE0Aiw)(password:0blv)  

5）The trained model for DOTAv1.5 dataset is available from [Google Drive](https://drive.google.com/file/d/1PdRfWPbdT4mY6DYBOkhBGF6dMq4IB8PC/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1NRDjXeGixUhDm87-2DXBLQ)(password: wxlj)

6）The trained model for DOTAv2.0 dataset is available from [Google Drive](https://drive.google.com/file/d/1Gq0bKYg4DA9RpIE0XxAtMfxnetpb4dV8/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/12io6rkVUGptVaoGfAaI99g)(password: dmu7)  


## 💖💖💖 6.Reference
https://github.com/Peterisfar/YOLOV3  
https://github.com/argusswift/YOLOv4-pytorch  
https://github.com/ultralytics/yolov5  
https://github.com/jinfagang/DCNv2_latest  

  
## 📝 License  
Copyright © 2021 [Shank2358](https://github.com/Shank2358).<br />
This project is [GNU General Public License v3.0](https://github.com/Shank2358/GGHL/blob/main/LICENSE) licensed.

## 🤐 To be continued 

## 🎃 Update Log
* #### 👻👻 最近大家提问和反馈比较多的是论文里对ORC做refined approximaation（RA） 的问题，这里额外解释一下。  
ORC和Gliding Vertex有很多相似之处，对于Gliding Vertex，当凸四边形OBB的四个顶点不全部落在外接HBB上时是无法表示的，因此Gliding Vertex是先利用cv2.minRect函数将DOTA标注的凸四边形OBB转换为旋转矩形，然后在利用在HBB上的滑动顶点进行表示以保证OBB所有四个顶点都落在HBB上，具体见他们仓库的issue https://github.com/MingtaoFu/gliding_vertex/issues/21#issuecomment-647935655. 在此基础上有了Oriented R-CNN的进一步简化用delta alpha和delta beta表示（因为内接OBB都已经近似成矩形了，两边是轴对称的，只需要预测在内接矩形相邻两条边的滑动距离即可）。但是这样的近似实际预测的还是旋转矩形而不是任意凸四边形，近似本身就会在GT上带来更多误差，误差更大的GT作为学习目标自然会影响预测准确性。因此，GGHL的RA只对那些该表示无法处理的凸四边形（即顶点不全落在HBB上的四边形）进行近似，而其他凸四边形都可以直接用ORC表示无需再近似成最小矩形，论文中也提到根据统计“only 4.79% of the OBBs need to be approximated”，这样可以保证ORC表示法在生成GT时引入的误差可以最小，论文表5的实验也证明了其有效性。更具体地，这些4.79%无法用ORC直接表示的凸四边形我们也不是全部用cv2.minRect做近似，而是精细地把他们分成16种情况进行讨论和分别处理，这样做目的是尽可能地减小因为近似带来的误差。大家也可以把上述近似和cv2.minRect近似的四边形画出来对比一下它们俩的误差。具体方案见论文图7和代码。

* #### 👹👹 找了个动图，这样更好理解ORC和Gaussian的关系（外边是凸四边形的滑动，里面是高斯椭圆），很遗憾论文没能把完整的几何证明给出，后续版本会更新。 
![img1](https://user-images.githubusercontent.com/33946139/168231673-578b1fc6-3d92-4dcb-9313-b0b054ce17a0.gif)  
顺便再解释一下这个，这个动图是为了解释自适应调整高斯分布的，高斯根据ORC的顶点滑动（或者说凸四边形的变化）改变。根据OBB生成高斯通常是将OBB视为矩形，然后根据矩形的中心点坐标和长宽来得到高斯椭圆，但是如上RA问题的解释所述，GGHL的OBB不一定是矩形，而是任意凸四边形（有一部分对ORC无效的做了上述近似），那么要如何生成高斯？一种做法是仍然将凸四边形近为旋转举行计算高斯，另外一种就是我们给出的动图这样的做法。在我们最新的工作中会对这个进行更详细的讨论，快出来了（😒😒已经鸽了一年了），敬请期待。篇幅所限在GGHL论文中没地方写了，原文因为投稿TIP已经一再压缩😢😢，很多细节和几何证明放不下了，连作者照片都没放全🤣🤣🤣，我只能说我真是尽力且费劲心思把放不下的东西尽量在原文中留下一些线索了，有兴趣的可以慢慢去发现，我很乐意随时交流讨论。

* #### 12.31 I have updated the function of polyiou and polynms. Happy new year!!!  
* #### 12.28 I have updated the requirements.txt file because the distributed training may prompt that some dependent libraries are missing. 
* 更新了requirements.txt文件，因为DDP分布式训练时可能提示缺少一些依赖库。如果遇到这种情况，请根据提示pip安装补全相应的库即可。

* #### 12.25 🎅🎅🎅 Merry Christmas！
The latest and improved embedded version of GGHL will be launched soon. 嵌入式端的最新改进版本即将上线。

* #### 12.24 🏜️🏜️ The label assignment of FCOS-R and GGHL-FCOS are online.
<img src="https://user-images.githubusercontent.com/33946139/147256008-467d03f4-876e-4065-99b6-02198a7d7d93.png" width="520">

* #### 🐦🐦 12.23 Centernet and GGHL-CenterNet are online. FCOS will be launched soon. This GitHub repository is still being updated and optimized. Centernet和GGHL-CenterNet上线。FCOS即将上线。,最近会持续更新优化这个仓库。  
<p algin="center">
<img src="https://user-images.githubusercontent.com/33946139/147179756-2e4788d2-48d2-4fc6-bb1c-a35cbb7f377e.png" width="260"><img src="https://user-images.githubusercontent.com/33946139/147185286-69f8eff8-3e71-4b34-86d8-c9b6259d4b1e.png" width="520">
</p>
 
* #### 😺😺 12.23 因为论文还在审稿阶段的原因，消融实验还差一些代码没有完全更新完，稍安勿躁，持续更新中。  
有问题真的真的可以在issues里面留言不一定非得邮件，我都会回的，这里可能比邮件还快的。邮件最近又有被放到垃圾箱的情况了，真的非常抱歉。有谁知道怎么关闭邮件拦截，恳请教我一下。另外，请不要做伸手党直接要代码或者让我直接帮忙写代码，这真的让人心情不美丽😩，论文里该有的代码这个仓库里都会有的，其他的我也很乐意和大家一起讨论，不断更新这个仓库让它越来越好。在这里咱们定个约定，如果需要额外的代码和问题解答，请将遇到的问题具体准确的描述清楚，我也会尽己所能去解答和更新，如果只是伸手党要这要那的请恕我不打捞垃圾邮件了。真的求求了，issues里就可以说的问题没必要加微信吧，确实要加也行的吧至少告诉我你们是谁，谢谢。  

* #### 🙉🙉 12.23 我不喜欢过度的评价别人的方法，我觉得每个工作都有它的闪光点和值得学习的地方，我学习论文和审稿也都是这样要求自己的，所以我个人拒绝回答觉得和xxx工作比起来怎么样这种问题，请见谅。GGHL这个工作欢迎任何批评和评价，我都会虚心接受作为激励。  
 
* #### 12.17 🧑‍🚀 今天没有更新。感慨一句，对于一个深度学习任务而言，有一个成熟的benchmark是一件幸事也是最大的不幸，当大家乐此不疲于此，这个领域就死掉了。

* #### 12.15 🤪 The trained models for DOTAv1.5 and DOTAv2.0 dataset are available. [Google Drive](https://drive.google.com/drive/folders/16k7JW-eb3jbga1xzq6B6r60gl2XniXfn?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/12MD7XAL6iwVUHMHRkEcLWA)(password: yn04)  *
🐾 🐾 🐾 DOTAv1.5和DOTAv2.0的权重可以下载啦。这版本的结果是没调参，没数据增强，没多尺度测试的，后续有空会再精调和加tricks，应该还会涨点。  
😝😝😝 其实每天事儿挺多的，做科研都是见缝插针，github这边就更顾不上了，使用教程和代码注释更新慢还请见谅，过年期间会加油更新。另外，有问题可以在issues里面留言，为什么都喜欢发邮件啊，邮件经常会莫名其妙的跑到垃圾邮件里，因此可能会有延迟，实在抱歉，我打捞出来就会立即回复的。 
 
* #### 12.13 😭 改论文改的头昏脑胀，补了一堆实验和解释，改论文比写论文难产多了~/(ㄒoㄒ)/~我可以选择剖腹产吗...

* #### 12.11 😁 修复了两个索引的bug。调整了学习率重新训练了，conf_thresh调到0.005，DOTA数据集精度能到79+了。顺便回复一句，总是有人问area normalization那个公式设计怎么来的，我睡觉梦到的。 

* #### 12.9 😳 终于收到一审的审稿意见了，感谢审稿人大大。

* #### 11.22 👺 Notice. Due to a bug in the cv2.minAreaRect() function of different versions of opencv, I updated datasets_obb.py, datasets_obb_pro.py, augmentations.py, and DOTA2Train.py. Opencv supports version 4.5.3 and above. Please note the update. Thank you. Thanks @Fly-dream12 for the feedback.  
不同版本opencv的cv2.minAreaRect()函数不一致且存在一些角度转换的bug (我用的低版本角度是(0,-90]，新版的是[0,90]，所以可能有一些bug，我全部更新统一到新版了现在。还有就是cv2.minAreaRect()函数本身的一些bug，有很多博客介绍过了我就不重复了，由于我的原版为了解决这些bug做的判断函数和新版cv2.minAreaRect()的输出不太一样，这里也有一些问题，我也修改了)，我更新了datasets_obb.py, datasets_obb_pro.py, augmentations.py, DOTA2Train.py文件，全部按长边表示法计算（角度范0,180)），请大家及时更新，opencv版本也请更新到4.5.3及以上。谢谢。

* #### 🧐 广告：招募可以一起维护更新这个仓库的小伙伴。或者大家fork更新了以后麻烦推上来一下啊，感激不尽。  

* #### 11.21 😸😸 Thanks @trungpham2606 for the suggestions and feedback. 

* #### 11.20 ❤️ 修复了一些bug，谢谢大家的建议。大家有啥问题可以在issues里面详细描述，我会及时回复，你的问题也可能帮助到其他人。

* #### 11.19 😶 During label conversion, it should be noted that the vertices in the paper are in order (see the paper for details).
11.19-11.20 更新修复了标签转换脚本的一些bug (对于custom data的顶点顺序可能与DOTA不一致的问题)

<p algin="center">
<img src="https://user-images.githubusercontent.com/33946139/142638611-39a20148-ce04-49fc-be19-2b6ffff0f9fa.png" width="320">
</p>

* #### 11.18 😺 Fixed some bugs, please update the codes

* #### 🙏🙏🙏 11.17 Release Notes
There are still some uncompleted content that is being continuously updated. Thank you for your feedback and suggestions. 

* #### 🐟 🐡 11.16 The script for generating datasets in the format required by GGHL is added in ./datasets_tools/DOTA2Train.py
更新了用于生成GGHL所需格式数据集的工具(./datasets_tools/DOTA2Train.py)
论文里的Refine Approx.在代码里面有详细的分类讨论。

* #### 👾 11.15 The models for the SKU dataset are available 
其他数据的权重近期会陆续上传和更新

* #### 🤖 11.14 更新预告 
即将更新更多的backbone和模型，以及mosaic数据增强,一周内更完。下周会更新第一版的代码注释和教程，即dataloadR/datasets_obb.py文件，主要是GGHL中最重要的标签分配策略。
另外GGHLv2.0正在准备和实验中，立个flag今年更新完。

* #### 🎅 11.10 Add DCNv2 for automatic mixed precision (AMP) training. 
增加了DCNv2的混合精度训练和onnx转换 (推理阶段要记得把offsets改成FP16)

* #### 🐣 🐤 🐥 11.9: The model weight has been released. You can download it and put it in the ./weight folder, and then modify the weight path in test.py to test and get the results reported in the paper. The download link is given in the introduction later.  
论文结果对应的模型权重可以下载了（终于发工资把网盘续上了~）

* #### 🐞 11.8：I plan to write a tutorial on data preprocessing and explanation of algorithms and codes, which is expected to be launched in December   
打算写一个数据预处理的教程和算法、代码的讲解，预计12月上线  

* #### 🦄 11.7: All updates of GGHL have been completed. Welcome to use it. If you have any questions, you can leave a message at the issue. Thank you.
1.0版本全部更新完成了，欢迎使用，有任何问题可以在issue留言，谢谢。接下来会不断更新和完善  

#### 💣 11.6 更新了标签分配和dataload。更新了pytorch1.10版本的支持。预告一下，下周会更新分布式训练的内容。
（预训练权重的链接在NPMMR-Det和LO-Det的仓库说明里）
  
#### 🙈 正文开始前的惯例的碎碎念（可以跳过直接看正文使用说明）
投稿排队实在太慢了，三个月了还在形式审查没分配AE,555~ 先在arxiv上挂出来了。  
我会尽最大努力帮助大家跑通代码和复现出接近论文报道结果的实验，因为我自己也被坑多了，好多遥感领域的论文不开源代码或者根本复现不出来，或者就是模型复杂到眼花缭乱换个数据/参数就失灵，实在是太难了。论文里关于NPMMR-Det和LO-Det的实验代码会在那两个仓库里面更新，NPMMRDet的baseline目前已经更新完了，你们可以试试看能不能跑。LO-Det的正在更新中，可以看那边的说明(11.1也更新了)。 
万一有AE或者审稿人大佬看到这个仓库，跪求千万别忘了审稿啊~ 求求，希望能顺利毕业😭😭😭 

#### 😸😸 10.24 终于分配AE和审稿人了🐌🐌🐌，不容易啊。这投稿流程可太慢了，担心能不能赶上毕业，真的是瑟瑟发抖😭😭😭  

#### 🙉🙉 关于论文超参数和实验的一些说明。
🐛 论文里报道的训练超参数都没有精调，就选的对比方法一样的默认参数，也没有选最好的epoch的结果，直接固定了最大epoch，选择最后五个epoch的平均结果。精调学习率、训练策略合最好轮次还会涨点，最近有空闲的机器我试了一下。但是我觉得像很多论文那样为了state-of-the-art（SOTA）而SOTA没有必要，所以最后没那样做，后续如果审稿意见有这个建议我可能会再修改，如果没有我会把更多的实验结果在github和arxiv上展示出来。反思自己最近的工作，确实比不上各位大佬前辈的创新想法，这点还要继续努力。由于我也是自己一路磕磕绊绊摸索着进入科研领域的，也踩过很多坑，也被各种卷王卷的透不过气，所以我想追求的是想做一些踏实的、简单实用的工作，设计一个皮实、经得起折腾的模型，而不想去卷什么SOTA（😭😭😭 实话是我也卷不过。。。。）。   
🐰🐰 说一个我对目标检测的理解，请大家批评指正。在我看来，目标检测只是一个更庞大的视觉系统的入口任务而不是最终结果。我觉得大多数检测任务的目标是快速、粗略地在图像/视频中定位到目标候选区域，为后续更精细的比如分割、跟踪等其他任务服务，简化它们的输入。从这个视角来看，检测平均精度差距那么一两个点真的没论文里吹的那么重要，反而检测效率（速度）、模型的复杂度与鲁棒性、易用性（无论是对工程人员还是新入门的研究人员而言）的提升对于社区的贡献会更实际一些。最近几个月我也一直在反思自己，目标检测的初心是什么，目标检测完了然后呢，原来我写论文以为的终点很多时候只是我以为的，原来我想错了。深度学习火了这么些年，很多任务或许也是这样的吧，毕竟论文实验里的SOTA是有标准答案的考试，而它们的开花结果是一个开放性问题。这是接下来的努力方向，我相信哪怕道阻且长，行则将至，而且行而不辍，未来一定可期。

另外，请不要做伸手党，如果你们想训练自己的数据集，以下已经详细描述了GGHL的数据格式和使用说明，在tools文件夹中提供了转换脚本。我也在许多论文以外的数据集和大家提供的数据集上进行了实验，都可以正常工作，请花些时间阅读说明和issues #1中的一些解释，如果还有疑问可以在issues中留言给我，都会得到回复。我没有义务直接帮你们改代码和训练你们的数据。
