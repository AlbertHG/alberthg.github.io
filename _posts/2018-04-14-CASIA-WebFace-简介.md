---
layout:     post
title:      CASIA WebFaces 数据集
subtitle:    "\"Deep Learning 数据集\""
date:       2018-04-16
author:     ATuk
header-img: img/dataset.jpg
catalog: true
tags:
    - dataset
    - 数据集
    - 简介
    - 人脸数据库
---

## 前言

> 发现服务器里边有一个非常多照片的文件夹 CASIA-WebFace，上网探索之，现做简要记录。

## 简介

在大数据和深度卷积神经网络（CNN）的推动下，人脸识别的性能快速接近人类的识别效率和准确率。使用私人大规模训练数据集，几个小组在LFW上取得了很高的成绩。虽然CNN网络的开源实现有很多，大规模的人脸数据集却没有一个是开源的。所以，目前在人脸识别领域，高质量的数据比算法更加重要。为了解决这个问题，CASIA（中国科学院自动化研究所）提出了一个半自动的方式来从互联网收集人脸图像的方法，并建立了一个大型的人脸数据集：该数据集包含`10575`个类别和总共`494414`张图片。

所提出的CASIA-WebFaces数据集的统计如表所示：

|Dataset|Subjects|Images|Availability|
|------|-------|------|-------|
|LFW[1]|5,749|13,233|Public|
|WDRef [2] |  2,995 |  99,773  |Public (feature only)|
|CelebFaces [3]  |10,177 | 202,599 |Private|
|SFC [4] |4,030  | 4,400,000 |  Private|
|CACD [5]  |  2,000  | 163,446 |Public (partial annotated)|
|CASIA-WebFace|   10,575  |494,414 |Public|

为了说明CASIA-WebFace的质量，我们对它进行了大量的CNN训练，并将其准确性与最先进的方法（如DeepFace和DeepID2）进行比较。有关详细信息，请参阅以下技术报告。

[Dong Yi, Zhen Lei, Shengcai Liao and Stan Z. Li, “Learning Face Representation from Scratch”. arXiv preprint arXiv:1411.7923. 2014](https://arxiv.org/abs/1411.7923)

免责声明：数据库是为了研究和教育目的而发布的。对于使用数据库的任何不良后果，我们不承担任何责任。

## 参考文献：

[1] LFW，http : //vis-www.cs.umass.edu/lfw/

[2] D. Chen，X. Cao，L. Wang，F. Wen，and J. Sun。“重访贝叶斯面孔：一种联合表述”。在ECCV 2012，第566-579页。Springer，2012.

[3] Y. Sun，X. Wang，and X. Tang。“通过联合身份验证进行深度学习的人脸表征”。arXiv预印本arXiv：1406.4773,2014。

[4] Y.泰格曼，M.杨，M. Ranzato和L.沃尔夫。“深层面：缩小面部验证中人类表现的差距”。在计算机视觉和模式识别（CVPR），2014年IEEE会议上，第1701-1708页。IEEE，2014.

[5] CARC，http://bcsiriuschen.github.io/CARC/
