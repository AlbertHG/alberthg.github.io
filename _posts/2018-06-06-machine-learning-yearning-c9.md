---
layout:     post
title:      《机器学习要领》 端到端的深度学习（中文翻译版）
subtitle:   Machine Learning Yearning Chapter9 End-to-end deep learning(Chinese ver)
date:       2018-06-27
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

> 本篇博客是 Andrew NG 《Machine Learning Yearning》 的「第九章：端到端的深度学习」翻译。Andrew NG 提到他曾经负责开发过一个大型端到端语音识别系统，并取得的很好的效果，但是他同时表示盲目使用该技术并不是好事。本章内容将探讨什么是端到端的深度学习？ 什么时候应该使用它，什么时候应该避免它？。开启本章内容，出发！   
👉[官网传送门](http://www.mlyearning.org/)<br>
👉[GitHub 项目传送门](https://github.com/AlbertHG/Machine-Learning-Yearning-Chinese-ver)，欢迎 Star

## 47. 端到端学习技术的兴起

假设您希望构建一个系统来检索关于在线购物网站上的产品评论，并自动通知你，用户是否喜欢或不喜欢该产品。例如，您希望识别出以下评论是属于积极评论：

- This is a great mop!

识别出以下评论是属于消极评论：

- This mop is low quality--I regret buying it.

识别正面和负面意见的机器学习问题被称为「情感分类」问题。为了建立这个系统，你可以构建一个由两部分组成的「管道（Pipeline）」：

1. 解析器（Parser）：一种用来标注文本内容中最重要的单词信息[^1]的系统，例如，你可以使用解析器来标记文本中所有的形容词和名词，因此例子变成了下列所示的那样：
    - This is a $great_{Adjectve}$  $mop_{Noun}$!
2. 情感分类器（Sentiment classifier）：一种将带注释的文本作为输入，并预测整体情绪的学习算法。解析器的注释可以极大地帮助这个学习算法：通过给予形容词一个更高的权重，你的算法将能够快速地识别出那些重要的单词（如「great」），同时忽略不那么重要的单词，例如「this」。

[^1]:解析器给出的文本注释要比这个丰富得多，但是这个简化的描述足以解释端到端的深度学习

我们将两个组件组成的「管道」可视化如下：

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180422mlyearning/20.png)

近期出现了用单一学习算法替代管道系统的趋势。这项任务的端到端学习算法尝试在仅输入原始文本「This is a great mop!」的情况下直接识别情绪的可能性：

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180422mlyearning/21.png)

神经网络通常用于端到端学习系统。术语「端到端（End-to-end）」指的是我们要求学习算法直接从输入到所需的输出。即，学习算法直接将系统的「输入端」连接到「输出端」。
在处理一些数据丰富的应用中，端到端学习算法非常成功。但这并不意味着它们总是一个好的选择。接下来的几章将给出更多的端到端系统的例子，并给出关于何时应该使用它们和不应该使用它们的建议。

## 48. 更多的端到端学习的例子

假设你想建立一个语音识别系统。那么你可以构建一个包含三个组件的系统：

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180422mlyearning/22.png)

这些组件的职能如下：

1. 计算特征：提取手工设计的特征，例如 MFCC 特征（梅尔频率倒谱系数），它尝试捕捉话语的内容，同时忽略不太相关的属性，例如说话者的音调；
2. 音素识别器：一些语言学家认为声音的基本单位是「音素」。例如，「keep」中最初的「k」音与「cake」中的「c」音相同。这个识别器试图识别音频片段中的音素；
3. 最终识别器：取出识别的音素序列，并尝试将它们串成一个输出抄本。

相反，端到端的系统可能会输入一个音频剪辑，并尝试直接输出文本:

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180422mlyearning/23.png)

到目前为止，我们只描述了完全线性的机器学习「管道」——输出顺序从一个阶段传递到下一个阶段。然而，管道可能更复杂，例如，这里有一个自动驾驶汽车的简单架构：

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180422mlyearning/24.png)

它有三个组成部分：一个负责使用相机图像检测其他车辆； 一个负责检测行人； 最终的组件则为我们的汽车出规划一条避开汽车和行人的道路来。

然而，并不是说管道中的每个组件都需要训练和学习。例如，有关「机器人运动规划」的文献中就有大量关于汽车最终路径规划步骤的算法。这里边的许多算法都不涉及学习。

相反，端到端的方法可能尝试接受传感器输入并直接输出转向方向。

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180422mlyearning/25.png)

尽管端到端学习已经取得了许多成功，但它并不总是最好的方法。例如，我们认为端到端的语音识别效果很好，但对自主驾驶的端到端学习则持怀疑态度。接下来的几章我会解释原因。

## 49. 端到端学习的优点和缺点

考虑与我们之前的例子相同的语音系统管道：

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180422mlyearning/22.png)

这条管道的许多部分都是「手工设计的」:

- MFCCs 属于手工设计的音频特征集。虽然它们提供了一个合理的音频输入的摘要，但它们也因为丢弃一些信息使得输入信号简化了。
- 音素是语言学家的发明。它们是语音的不完美表现。如果「音素」对现实声音的近似很差，那么迫使一个算法使用音素来表示则会限制语音系统的性能。

虽然这些手工设计的组件限制了语音系统的潜在性能。但是，允许手工设计的组件也有一些优点:

- MFCC 提取的特征能对与内容无关的某些语音属性（如扬声器音高）具有鲁棒性。因此，它们有助于简化学习算法的问题；
- 音素在一定程度上也是一种合理的语音表达，也可以帮助学习算法理解基本的语音成分，从而提高其性能。

拥有更多手工设计的组件通常可以让语音系统以更少的数据进行学习。MFCCs 和基于音素捕获的手工工程知识「补充」了我们的算法从数据中获得的知识。当我们没有太多的数据时，这些知识是有用的。

现在，考虑端到端系统：

![](https://raw.githubusercontent.com/AlbertHG/alberthg.github.io/master/makedown_img/20180422mlyearning/23.png)

该系统缺乏手工设计的知识。因此，当训练集很小时，它可能比手工设计的管道更糟糕。

然而，当训练集很大时，它不会受到 MFCC 或基于音素表示的限制。如果学习算法是一个足够大的神经网络，同时经过了足够多的训练数据的训练，那么它就有可能做得很好，甚至可能接近最优错误率。

当有大量的「两端」（输入端和输出端）标记数据时，端到端学习系统往往会做得很好。在本例中，我们需要一个大数据集（音频-文本对）。因此，当这种类型的数据不可用时，请小心谨慎地进行端到端的学习。

如果你正在研究一个机器学习问题，而训练集非常小，那么你的算法的大部分知识都必须来自于你的人类洞察力——即，来自您的「手工工程」组件。

如果您选择不使用端到端系统，那么您必须决定管道中的每一个步骤，以及它们连接次序。在接下来的几章中，我们将对此类管道的设计提出一些建议。

------

🚧🚧未完待续🚧🚧