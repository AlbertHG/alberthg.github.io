---
layout:     post
title:      Caffe Windows 配置
subtitle:    "\"Caffe for Windows10\""
date:       2018-04-12
author:     ATuk
header-img: img/caffe.jpg
catalog: true
tags:
    - caffe
    - win10
    - python3
    - python
    - matlab
    - windows
    - cmake
    - vs2015
    - 教程
    - 环境搭建
---

## 前言

> 学校服务器的资源已经被兄弟们占满了，没有办法，只好在自己的笔记本下尝试安装单机版Caffe，以备不时之需！

## 环境

- Windows10家庭版
- GPU Nvidia M1200
- Visual Srudio Community 2015 Update 1
- Cmake 3.11.0
- Git
- Python 3.5.3
- Matlab 2014a
- CUDA 8.0
- cuDNN 5.0.5

## 资源地址

1. 伯克利BVLC(Berkeley Vision And Learning Center) 版。 [https://github.com/BVLC/caffe](https://github.com/BVLC/caffe)
    下载源码,在这里我使用`D:\GitHub Repository`用作根文件夹：
    ```
    git clone https://github.com/BVLC/caffe.git
    cd caffe
    git branch -a
    git checkout windows
    ```

2. cuda_8.0.61_win10.exe下载地址。[cuda_8.0.61_win10](https://developer.nvidia.com/cuda-80-ga2-download-archive)

    安装完成后，在系统环境变量会有如下环境变量:
    ```
    CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0
    CUDA_PATH_V8_0=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0
    NVCUDASAMPLES8_0_ROOT=C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0
    NVCUDASAMPLES_ROOT=C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0
    NVTOOLSEXT_PATH=C:\Program Files\NVIDIA Corporation\NvToolsExt\
    ```

3. cuDNN 下载需要注册账号。就不提供地址了，认准版本号就可以。下载完之后，将压缩包解压到CUDA安装目录中，比如`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0`；或者定义CUADD_ROOT缓存变量指向解压缩cuDNN文件的位置，比如`D:/caffe/cudnn-8.0-windows10-x64-v5.1/cuda`,使用这种方法，则需要修改`build\_win.cmd`文件，待会再说。

4. Python 3.5.3 我使用了Anaconda3，通过创建虚拟环境安装。

    ```
    conda create -n Caffe_3_5 python=3.5.3
    ```

    然后将新创建的虚拟环境添加到Path环境变量：

    ```
    C:\ProgramData\Anaconda3\envs\Caffe_3_5
    C:\ProgramData\Anaconda3\envs\Caffe_3_5\Scripts
    ```

    要成功构建python界面，您需要添加以下conda通道：

    ```
    conda config --add channels conda-forge
    conda config --add channels willyd
    ```

    同时使用conda在新建的Caffe_3_5中安装下列软件包：

    ```
    conda install -n Caffe_3_5 --yes cmake ninja numpy scipy protobuf==3.1.0 six scikit-image pyyaml pydotplus graphviz
    ```

4. Matlab 2014a 的安装不赘述，没压力！

    添加Path环境变量
    ```
    C:\Program Files\MATLAB\R2014a\runtime\win64
    C:\Program Files\MATLAB\R2014a\bin
    C:\Program Files\MATLAB\R2014a\polyspace\bin
    ```
    添加环境变量`MATLAB_ROOT_DIR`:
    ```
    C:\Program Files\MATLAB\R2014a
    ```

5. Cmake安装也不赘述。

    添加Path环境变量
    ```
    C:\Program Files\CMake\bin
    ```

## 开始编译Caffe之前的准备工作

> 在这里我是用`Cmake.gui`这个工具进行生成vs工程文件的。

- 设置 源代码路径：`D:/GitHub Repository/caffe-windows`
- 设置 生成build路径：`D:/GitHub Repository/caffe-windows/build/x64`

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180412caffewin/1.jpg)

- 点击[Configure]，弹窗：
    - 选择生产vs工程的版本：Visual Sutdio 14 2015 Win64
    - tooset 默认值，空
    - 勾选[Use default native compilers]
*过程有下载依赖，时间较长。下载后压缩包放在camke-gui指定的编译目录下，我的在`C:\Users\ATuk\.caffe\dependencies\download`。如果速度慢或者没有科学上网的道友可以去 [https://github.com/willyd/caffe-builder/releases/download/v1.1.0/libraries_v140_x64_py35_1.1.0.tar.bz2](https://github.com/willyd/caffe-builder/releases/download/v1.1.0/libraries_v140_x64_py35_1.1.0.tar.bz2) 下载，放在camke-gui指定的编译目录下。*  

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180412caffewin/2.jpg)

- 下载完之后，或者再次点击[Configure]之后，出现了报错，**通过把BLAS设置为OPEN即可** ，当然根据需要,我在面板继续修改参数，我勾选Build_python，使caffe支持python接口；勾选Build_python_layer，使caffe支持python语言自定义层，设置python_version属性为3，指定python版本是3.0+；勾选Build_matlab，使caffe支持Mallab接口：

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180412caffewin/3.jpg)

- 然后再次点击[Configure]即可，可能最后还会出现几个Warning，关于Boost的，好像没有影响。

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180412caffewin/4.jpg)

- 最后点击[Generate]按钮，生成vs工程文件。

去到生成build的路径里，发现`Caffe.sln`，真是顺眼。

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180412caffewin/5.jpg)

## 使用VS2015编译Caffe

用vs2015打开 `D:\GitHub Repository\caffe-windows\build\x64\Caffe.sln`。编译整个Caffe.sln下面的工程

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180412caffewin/6.jpg)

等待若干时刻（检查你CPU强劲程度的时候到了），编译成功后，生成INSTALL工程，这样完整的debug和release版本就安装到D:\GitHub Repository\caffe-windows\build\x64\install目录下了。

在D:\GitHub Repository\caffe-windows\build\x64\install\bin目录下`shift+右键`调出Windows PorweShell，敲入`.\caffe.exe -version`，cooooool。

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180412caffewin/7.jpg)

如果要运行caffe自己提供的测试用例，项目入口是runtest工程。运行runtest工程。跑全部测试用例。
具体测试内容在test.testbin工程。

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180412caffewin/8.jpg)

## 官方使用`build\_win.cmd`编译Caffe

[传送门](https://github.com/BVLC/caffe/tree/windows)

因为我们没有定义APPVEYOR，所以直接拉到else（大约69行）以后,修改：

```
@echo off
@setlocal EnableDelayedExpansion

:: Default values
if DEFINED APPVEYOR (

    :: ----这里有很多代码，在这里忽略了，因为不需要改动。

) else (
    :: Change the settings here to match your setup
    :: ----Change MSVC_VERSION to 12 to use VS 2013
    if NOT DEFINED MSVC_VERSION set MSVC_VERSION=14
    :: ----Change to 1 to use Ninja generator (builds much faster)
    if NOT DEFINED WITH_NINJA set WITH_NINJA=0
    :: ----Change to 1 to build caffe without CUDA support
    if NOT DEFINED CPU_ONLY set CPU_ONLY=0
    :: ----Change to generate CUDA code for one of the following GPU architectures
    :: ----[Fermi  Kepler  Maxwell  Pascal  All]
    if NOT DEFINED CUDA_ARCH_NAME set CUDA_ARCH_NAME=Auto
    :: ----Change to Debug to build Debug. This is only relevant for the Ninja generator the Visual Studio generator will generate both Debug and Release configs
    if NOT DEFINED CMAKE_CONFIG set CMAKE_CONFIG=Release
    :: ----Set to 1 to use NCCL
    if NOT DEFINED USE_NCCL set USE_NCCL=0
    :: ----Change to 1 to build a caffe.dll
    if NOT DEFINED CMAKE_BUILD_SHARED_LIBS set CMAKE_BUILD_SHARED_LIBS=0
    :: ----Change to 3 if using python 3.5 (only 2.7 and 3.5 are supported)
    if NOT DEFINED PYTHON_VERSION set PYTHON_VERSION=3
    :: ----Change these options for your needs.
    :: ----caffe支持python接口
    if NOT DEFINED BUILD_PYTHON set BUILD_PYTHON=1
    :: ----使caffe支持python语言自定义层
    if NOT DEFINED BUILD_PYTHON_LAYER set BUILD_PYTHON_LAYER=1
    :: ----使caffe支持Mallab接口
    if NOT DEFINED BUILD_MATLAB set BUILD_MATLAB=1
    :: ----If python is on your path leave this alone
    if NOT DEFINED PYTHON_EXE set PYTHON_EXE=python
    :: ----Run the tests
    if NOT DEFINED RUN_TESTS set RUN_TESTS=0
    :: ----Run lint
    if NOT DEFINED RUN_LINT set RUN_LINT=0
    :: ----Build the install target
    if NOT DEFINED RUN_INSTALL set RUN_INSTALL=0


)

::----这里有很多代码，在这里忽略了，因为不需要改动。

echo INFO: ============================================================
echo INFO: Summary:
echo INFO: ============================================================
echo INFO: MSVC_VERSION               = !MSVC_VERSION!
echo INFO: WITH_NINJA                 = !WITH_NINJA!
echo INFO: CMAKE_GENERATOR            = "!CMAKE_GENERATOR!"
echo INFO: CPU_ONLY                   = !CPU_ONLY!
echo INFO: CUDA_ARCH_NAME             = !CUDA_ARCH_NAME!
echo INFO: CMAKE_CONFIG               = !CMAKE_CONFIG!
echo INFO: USE_NCCL                   = !USE_NCCL!
echo INFO: CMAKE_BUILD_SHARED_LIBS    = !CMAKE_BUILD_SHARED_LIBS!
echo INFO: PYTHON_VERSION             = !PYTHON_VERSION!
echo INFO: BUILD_PYTHON               = !BUILD_PYTHON!
echo INFO: BUILD_PYTHON_LAYER         = !BUILD_PYTHON_LAYER!
echo INFO: BUILD_MATLAB               = !BUILD_MATLAB!
echo INFO: PYTHON_EXE                 = "!PYTHON_EXE!"
echo INFO: RUN_TESTS                  = !RUN_TESTS!
echo INFO: RUN_LINT                   = !RUN_LINT!
echo INFO: RUN_INSTALL                = !RUN_INSTALL!
echo INFO: ============================================================

::----这里有很多代码，在这里忽略了，因为不需要改动。

:: ----Configure using cmake and using the caffe-builder dependencies，好好核对，下列代码
:: Add -DCUDNN_ROOT=C:/Projects/caffe/cudnn-8.0-windows10-x64-v5.1/cuda ^
:: below to use cuDNN
cmake -G"!CMAKE_GENERATOR!" ^
      -DBLAS=Open ^
      -DCMAKE_BUILD_TYPE:STRING=%CMAKE_CONFIG% ^
      -DBUILD_SHARED_LIBS:BOOL=%CMAKE_BUILD_SHARED_LIBS% ^
      -DBUILD_python:BOOL=%BUILD_PYTHON% ^
      -DBUILD_python_layer:BOOL=%BUILD_PYTHON_LAYER% ^
      -DBUILD_matlab:BOOL=%BUILD_MATLAB% ^
      -DCPU_ONLY:BOOL=%CPU_ONLY% ^
      -DCOPY_PREREQUISITES:BOOL=1 ^
      -DINSTALL_PREREQUISITES:BOOL=1 ^
      -DUSE_NCCL:BOOL=!USE_NCCL! ^
      -DCUDA_ARCH_NAME:STRING=%CUDA_ARCH_NAME% ^
	  -DCUDNN_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\cuda ^
      "%~dp0\.."

::----这里有很多代码，在这里忽略了，因为不需要改动。

popd
@endlocal
```
不出意外，双击`build_win.cmd`运行,检查系统打印出来的配置是不是和自己设置的一致。

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180412caffewin/9.jpg)

开始编译，一切顺利的话，等待若干时间（这就得看CPU给不给力了）就编译好了。

## 测试Caffe编译结果

接下来运行一下caffe项目自带的examples里的00-classification的代码来验证一下caffe是否能够正常运行

打开anaconda的命令行，进入caffe的examples目录，运行`jupyter-notebook`

我这边是一路顺畅！！

## 将pyCaffe加入系统变量

在当前用户的变量和系统变量我都新添加`PYTHONPATH`变量，并将路径指向caffe的根目录的python文件夹上，比如我的`D:\GitHub Repository\caffe\python`。

我其实不太了解，当我只在系统变量添加`PYTHONPATH`时，并没有用。而当我在当前用户的变量添加`PYTHONPATH`时，成功`import caffe`。真是怪哉！

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180412caffewin/10.jpg)

## caffe的Matlab接口部署

在`D:\GitHub Repository\caffe\matlab\+caffe\private`的Debug和Release文件夹里都有`caffe_.mexw64`文件，将其中一个（我使用的是Release里边的）拷贝到`D:\GitHub Repository\caffe\matlab\+caffe\private`

打开Matlab 2014a，点击面板`设置路径`，添加路径：

```
D:\GitHub Repository\caffe\matlab
```
运行以下命令：
```
caffe.run_tests()
```

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180412caffewin/11.jpg)

好像第一次运行还是会报错，然后我通过在matlab里，将路径定位到`D:\GitHub Repository\caffe\matlab\+caffe\private`之后，在运行`caffe.run_tests()`就成功了。然后第二次之后，就不需要在特定的文件夹都可以运行成功，不明白为什么，不过不想纠结了，反正我不用Matlab（个人觉得Matlab的代码一点美感都没）。

可以参考该博客[微软官方caffe之 matlab接口配置](#https://blog.csdn.net/zb1165048017/article/details/51702686)。
