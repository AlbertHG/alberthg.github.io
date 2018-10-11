---
layout:     post
title:      TensorFlow GPU Windows 配置
subtitle:    "\"TensorFlow-GPU for Windows10\""
date:       2018-05-26
author:     Canary
header-img: img/tensorflow.jpg
catalog: true
tags:
    - tensorflow
    - win10
    - python3
    - windows
    - 记录
    - 环境搭建
---

## 前言

> 好机器不能浪费，遂在本人笔记本 DELL 5530 上安装 TensorFlow-GPU 。

## 环境

- Windows10家庭版
- GPU Nvidia M1200
- Python 3.5.3
- CUDA 8.0
- cuDNN 5.0.5

## 资源地址

1. cuda_8.0.61_win10.exe下载地址。[cuda_8.0.61_win10](https://developer.nvidia.com/cuda-80-ga2-download-archive)

    安装完成后，在系统环境变量会有如下环境变量:
    ```
    CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0
    CUDA_PATH_V8_0=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0
    NVCUDASAMPLES8_0_ROOT=C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0
    NVCUDASAMPLES_ROOT=C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0
    NVTOOLSEXT_PATH=C:\Program Files\NVIDIA Corporation\NvToolsExt\
    ```

2. cuDNN 下载需要注册账号。就不提供地址了，认准版本号就可以。下载完之后，将压缩包解压到CUDA安装目录中，比如`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0`；或者定义CUADD_ROOT缓存变量指向解压缩cuDNN文件的位置。

3. Python 3.5.3 我使用了Anaconda3，通过创建虚拟环境安装，因为本人电脑已经安装有Caffe，如果两个框架安装在同一环境中，生怕出事，于是新建环境。

    ```
    conda create -n tensorflow python=3.5.3
    ```

    然后将新创建的虚拟环境添加到Path环境变量：

    ```
    C:\ProgramData\Anaconda3\envs\tensorflow
    C:\ProgramData\Anaconda3\envs\tensorflow\Scripts
    ```

    两个常用指令：

    ```
    打开环境：activate tensorflow
    关闭环境：deactivate tensorflow
    ```

## 使用 Anaconda navigator 安装 TensorFlow-GPU

使用搜索框检索 “tensorflow-gpu” 然后点击安装即可。

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180526tensorflowgpu/1.png)

## 验证安装

打开anaconda prompt，激活环境并进入python：

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180526tensorflowgpu/2.png)

测试代码如下：

```python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

运行结果：

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180526tensorflowgpu/3.png)

至于中间那一串警告，只是框架的一个建议而已：

```
The TensorFlow library wasn't compiled to use SSE instructions, but these are available on your machine and could speed up CPU computations.
```

忽略之即可！