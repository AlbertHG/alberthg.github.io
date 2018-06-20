---
layout:     post
title:      《机器学习要领》 比较人类水平表现（中文翻译版）
subtitle:   Machine Learning Yearning Chapter6 Comparing to human-level performance(Chinese ver)
date:       2018-05-30
author:     Canary
header-img: img/ML_yearning1.jpg
catalog: true
tags:
    - maching learning yearning
    - 机器学习
    - Andrew NG
    - 翻译
---

## 前言

> 本篇博客是 Andrew NG 《Machine Learning Yearning》 的“第六章：比较人类水平表现”翻译。本章内容将提出通过和人类表现水平的比较来加快机器学习发展的策略。学习算法的性能表现在越来越多的领域超越了人类水平表现，从语音识别到图像识别（狭义领域）。在深度学习领域，与人类水平表现竞争已然成为一项新兴的运动，当你的算法表现超越人类的时候会发生什么呢？开启本章内容，出发！   
👉[官网传送门](http://www.mlyearning.org/)<br>
👉[GitHub项目传送门](https://github.com/AlbertHG/Machine-Learning-Yearning-Chinese-ver)，欢迎Star

## 33. 为什么我们要比较人类表现水平

许多机器学习系统旨在自动化人类已经做的很完美的那些事情，例子包括：图像识别、语言识别、邮箱垃圾邮件分类等。经过发展，学习算法也有了长足的进步，现在我们的算法在越来越多类似的任务中的表现都超过了人类的表现水平(Human-Level Performance)。

此外，如果你使用机器学习正在努力完成一项人类非常擅长的工作，那么建立机器学习系统相对容易的原因有如下几个：

1. 能够轻松地获取人类标签数据：例如：由于人们可以很轻易的识别出猫咪的图片来，那么，人们可以很轻易的为你的学习算法提供高精度的标签。
2. 能够借鉴人类直觉来进行误差分析：假设一个语音识别算法远比不上人类的表现。比如，算法将音频片段转录拼写为 "This recipe calls for a pear of apples" 其实算法将 "pair" 错误转录为 "pear"。你可以利用人类的直觉，试图去理解一个人的话，他是用什么信息来去获得正确的转录信息的，并将其用来修改学习算法。
3. 能够利用人类水平表现来估计最佳误差率并设置好“期望误差率”：假设你的学习算法在目标任务中达到了 10% 的误差水平，但在同样的任务中，人的误差只有 2% 或者更低。然后我们就知道了最优误差率是 2% 或者更低，也就是说可避免偏差至少有 8% 。因此，你应该尝试那些能够减少偏差的技巧。

尽管第三项听起来好像不是很重要，但我发现有一个合理且可实现的目标误差率有助于加速团队的进度。知道你的算法具有高可避免偏差是非常有价值的，因为它为我们提供了一个可尝试去优化的选项。

当然，对于那些人类来说本来就不是很擅长的任务。比如，为你推荐书籍、或者在网页上为特定用户显示特定的广告、或者预测股市走势。计算机的表现已经远远超过了大部分人的能力表现，对于这些任务来说，我们在建立机器学习系统时有以下问题：

1. 获取标签数据相对困难：比如，负责为样本贴标签的人很难用“最佳”书籍标签去标注用户数据集。如果你利用销售图书的网站和应用的话，那还可以通过向用户展示图书并统计他们购买的内容来获取数据。如果你不经营类似的网站，那么你可能需要找到更多有创意的方法才能获取到数据了。
2. 无法指望人类的直觉：例如，几乎没人能够准确预测股票市场。因此，如果我们的股票预测算法没能比随机猜测来的更好，那么我们也很难去搞清楚应该如何改进它。
3. 很难去定义最佳误差率和期望误差率：假设你已经设计出一个不错的图书推荐系统。你怎么知道在没有人类基准表现水平的情况下它的性能还能提高多少？

## 34. 如何定义人类水平表现

假设你正在研究一种医疗影像系统，该系统能够自动根据X射线图像进行诊断。下列的哪个误差率能够作为人类水平表现基准呢？

- 除了一些基本的医学训练之外没有其他专业医学背景的志愿者在这项任务中能够达到 15% 的误差；
- 初级医生的误差是 10%；
- 一位有经验的医生的误差是 5%；
- 通过讨论，一个由医生组成的小组的误差是 2%。

在这种情况下，我会使用 2% 代表人类性能表现作为我们的最优误差率。同时，你还可以将 2% 设置为期望性能水平，因为在上一节“为什么我们要比较人类表现水平”中所提到的三个原因在此都适用：

1. 能够轻松地获取人类标签数据：你可以让医生小组为您提供数据标签，误差率只有 2%。
2. 能够借鉴人类直觉来进行误差分析：通过与医生小组讨论图像，你可以借鉴他们的专业直觉。
3. 能够利用人类水平表现来估计最佳误差率并设置好“期望误差率”：使用 2% 来作为我们对最优误差率的估计是合理的。在本例中，最优误差率只可能比 2% 更低而不会更高，因为这是专业的医生团队所能做到的最好的表现了（代表了人类最高水平）。相反，使用 5% 和 10% 来作为对最优误差率的估计的话是不合理的，因为我们知道这些估计值偏高了（人类能做的更好）。

如果您的系统目前有 40％ 的错误，那么纠结于使用初级医生（10% 的误差）还是一个有经验的医生（5% 的误差）来标记数据和提供直觉并不是那么重要。但是如果你的系统误差已经优化到 10% 了，那么将人类水平参考定义为 2% 将为你的系统优化带来更有用的帮助。

## 35. 超越人类表现水平

你正在进行语音识别的研究并有一个音频剪辑的数据集。假设你的数据集有很多的噪声音频片段，以至于人能达到的最好表现的误差都有 10%。 这个时候假设你的系统的误差有 8% 的话。那能否使用第 33 节的描述的三种技术中的任何一种来继续加速研发速度呢？

如果你可以确认在一些数据子集中，人类表现水平要远好于你的系统表现水平，例如，在你的系统中，识别嘈杂环境中的音频性能表现要好于人类表现，然而在转录那些语速非常快的音频片段中人类表现则比机器表现要好。那么你依然可以使用这些技术手段去推动研究进度。

对于上述表述的语速非常快的音频数据子集：

1. 你仍然可以从人类中获得质量高于你算法的输出的转录文本；
2. 你可以借鉴人类的直觉来理解为什么你的系统没能够正确听出语速非常快的音频片段；
3. 你可以将快速语音音频片段下的人类水平表现作为期望表现目标。

更一般的说，只要在开发集中，你是正确的而算法是错误的，那么前面描述的许多技术都能适用。这是真话，即使在整个开发/测试集的平均水平上，你的算法的表现已经超过了人类的性能表现。

有很多重要的机器学习应用程序，其中机器的表现超过了人类的表现。例如，机器在预测电影评级、对送货车在某地驾驶时长的预测、或者是评断是否批准申贷人的贷款申请等方面的效果都要比人类表现更好。一旦人类很难识别出算法明显出错的例子，那么只有一部分技术能用上。因此，在机器表现已经超越人类表现的问题上，研发进展通常较慢。而在当机器的表现不能够和人类表现媲美的情况下，进展较快。

------