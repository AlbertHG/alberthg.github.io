---
layout:     post
title:      序列模型和注意力机制
subtitle:    "\"deeplearning.ai-Class5-Week3\""
date:       2019-01-21
author:     Canary
header-img: img/deeplearning_c4_w1.jpg
catalog: true
tags:
    - 深度学习
    - 笔记
    - deeplearning.ai
    - 网易云课堂
    - Andrew NG
---

## [GitHub项目传送门](https://github.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai)

> 欢迎Star

*本篇博客有大量公式演示，不推荐使用手机查看*

## 基础模型

**Seq2Seq（Sequence-to-Sequence）模型能够应用于机器翻译、语音识别等各种序列到序列的转换问题。一个 Seq2Seq 模型包含编码器（Encoder）和解码器（Decoder）** 两部分，它们通常是两个不同的 RNN。如下图所示，将编码器的输出作为解码器的输入，由解码器负责输出正确的翻译结果。

假如这里我们要将法语「Jane visite I'Afrique en septembre.」翻译成英文：

- 输入： $x^{<1>}，x^{<2>}，\cdots，x^{<T_{x}>}$ ；这里每个 $x^{<t>}$ 均为对应法语句子中的每个单词；
- 输出： $y^{<1>}，y^{<2>}，\cdots，y^{<T_{y}>}$ ；这里每个 $y^{<t>}$ 均为对应英语句子中的每个单词；
- 网络结构：many-to-many RNN 网络结构。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week3/md_images/01.png)

相关论文：[Sutskever et al., 2014. Sequence to sequence learning with neural networks](https://arxiv.org/pdf/1409.3215.pdf)

相关论文：[Cho et al., 2014. Learning phrase representaions using RNN encoder-decoder for statistical machine translation](https://arxiv.org/abs/1406.1078)

这种编码器-解码器的结构也可以用于图像描述（Image captioning）。将 AlexNet 作为编码器，最后一层的 Softmax 换成一个 RNN 作为解码器，网络的输出序列就是对图像的一个描述。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week3/md_images/02.png)

相关论文：[Mao et. al., 2014. Deep captioning with multimodal recurrent neural networks](https://arxiv.org/pdf/1412.6632.pdf)

相关论文：[Vinyals et. al., 2014. Show and tell: Neural image caption generator](https://arxiv.org/pdf/1411.4555.pdf)

相关论文：[Karpathy and Fei Fei, 2015. Deep visual-semantic alignments for generating image descriptions](https://arxiv.org/pdf/1412.2306.pdf)

## 选择最可能的句子

对于机器翻译来说和之前几节介绍的语言模型有很大的相似性但也有不同之处。

- 在语言模型中，我们通过估计句子的可能性，来生成新的句子。语言模型总是以零向量开始，也就是其第一个时间步的输入可以直接为零向量；
- 在机器翻译中，包含了编码网络和解码网络，其中解码网络的结构与语言模型的结构是相似的。机器翻译以句子中每个单词的一系列向量作为输入，所以相比语言模型来说，机器翻译可以称作条件语言模型，其输出的句子概率是相对于输入的条件概率。

对于各种可能的翻译结果，我们并不是要从得到的分布中进行随机取样，而是需要找到能使条件概率最大化的翻译，即：

$$arg \ max_{y^{⟨1⟩}, ..., y^{⟨T_y⟩}}P(y^{⟨1⟩}, ..., y^{⟨T_y⟩} | x)$$

所以在设计机器翻译模型的时候，一个重要的步骤就是设计一个合适的算法，找到使得条件概率最大化的的结果。目前最通用的算法就是：集束搜索（Beam Search）。

不使用贪心搜索的原因：

对于我们的机器翻译模型来说，使用贪心搜索算法，在生成第一个词的分布后，贪心搜索会根据我们的条件语言模型挑选出最有可能输出的第一个词语，然后再挑选出第二个最有可能的输出词语，依次给出所有的输出。

但是对于我们建立的机器翻译模型来说，我们真正需要的是通过模型一次性地挑选出整个输出序列： $y^{<1>}，y^{<2>}，\cdots，y^{<T_{y}>}$ ，来使得整体的概率最大化。所以对于贪心搜索来说对于解决机器翻译问题而言不可用。

另外对于贪心搜索算法来说，我们的单词库中有成百到千万的词汇，去计算每一种单词的组合的可能性是不可行的。所以我们使用近似的搜索办法，使得条件概率最大化或者近似最大化的句子，而不是通过单词去实现，虽然不能保证我们得到的就是条件概率最大化的结果，但是往往这已经足够了。

## 集束搜索

集束搜索（Beam Search）会考虑每个时间步多个可能的选择。这里我们还是以法语翻译成英语的机器翻译为例：

第一步：设定一个集束宽（Beam Width）$B$，代表了解码器中每个时间步的预选单词数量。例如 $B=3$，则将第一个时间步最可能的三个预选单词及其概率值 $P(y^⟨1⟩|x)$ 保存到计算机内存。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week3/md_images/03.jpg)

第二步：在第一步中得到的集束宽度的单词数作为预选词，当作第二个时间步的输入，计算其与单词表中的所有单词组成词对的概率，并与第一步的概率相乘，得到第一和第二两个词对的概率： $P(\hat y^{⟨2⟩}|x, \hat y^{⟨1⟩})$，有 $3\times 10000$ 个选择：

$$P(\hat y^{⟨1⟩}, \hat y^{⟨2⟩}|x) = P(\hat y^{⟨1⟩}|x) P(\hat y^{⟨2⟩}|x, \hat y^{⟨1⟩})$$

假设词汇表有 $10000$ 个词，则当 $B=3$ 时，有 $3 × 10000$ 个 $P(y^⟨1⟩,y^⟨2⟩|x)$。仍然取其中概率值最大的 $3$ 个，作为对应第一个词条件下的第二个词的预选词。以此类推，最后输出一个最优的结果，即结果符合公式：

$$arg \ max \prod^{T_y}_{t=1} P(\hat y^{⟨t⟩} | x, \hat y^{⟨1⟩}, ..., \hat y^{⟨t-1⟩})$$

可以看到，当 $B=1$ 时，集束搜索就变为贪心搜索。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week3/md_images/04.png)

第三步 - 第 T 步：与第二步的过程是相似的，直到遇到句尾符号结束。

## 改进集束搜索

**长度归一化（Length Normalization）** ，对于集束搜索算法，我们的目标就是最大化下面的概率：

$$
arg \ max \prod^{T_y}_{t=1} P(\hat y^{⟨t⟩} | x, \hat y^{⟨1⟩}, ..., \hat y^{⟨t-1⟩})= \\
arg \underset{y}{\ max} P(y^{<1>},...,y^{<T_y>}|x)=\\
P(y^{<1>}|x)P(y^{<2>}|x,y^{<1>}) ... P(y^{<T_y>}|x,y^{<1>},...,y^{<T_y-1>})
$$

当多个小于 1 的概率值相乘后，会造成数值下溢（Numerical Underflow），即得到的结果将会是一个电脑不能精确表示的极小浮点数。因此，我们会取 $log$ 值，并进行标准化：

$$arg \ max \sum^{T_y}_{t=1} logP(\hat y^{⟨t⟩} | x, \hat y^{⟨1⟩}, ..., \hat y^{⟨t-1⟩})$$

另外，我们还可以通过对上面的目标进行归一化，使其得到更好的效果。相比直接除以输出单词长度的值，可以使用更加柔和的方式：在 $T_{y}$ 上加上一个指数 $\alpha$ ，如 $\alpha = 0.7$ ，通过调整其大小获得更加好的效果。如果这里不使用长度归一化，模型会倾向于输出更短的翻译句子，对结果产生影响。

$$arg \ max \frac{1}{T_y^{\alpha}} \sum^{T_y}_{t=1} logP(\hat y^{⟨t⟩} | x, \hat y^{⟨1⟩}, ..., \hat y^{⟨t-1⟩})$$

其中，$T_y$ 是翻译结果的单词数量，$α$ 是一个需要根据实际情况进行调节的超参数。标准化用于减少对输出长的结果的惩罚（因为翻译结果一般没有长度限制）。

通过上面的目标，选取得分最大的句子，即为我们的模型最后得到的输出结果。

对于超参数 $B$ 的选择，$B$ 越大考虑的情况越多，但是所需要进行的计算量也就相应的越大。在常见的产品系统中，一般设置 $B = 10$，而更大的值（如 100，1000，...）则需要对应用的领域和场景进行选择。

相比于算法范畴中的搜索算法像 BFS 或者 DFS 这些精确的搜索算法，Beam Search 算法运行的速度很快，但是不能保证找到目标准确的最大值。

## 集束搜索的误差分析

集束搜索算法是一种近似搜索算法，也被称为启发式搜索算法。它的输出不能保证总是可能性最大的句子，因为其每一步中仅记录着 Beam width 为 3 或者 10 或者 100 种的可能的句子。

我们如何确定是算法出现了错误还是模型出现了错误呢？此时集束搜索算法的误差分析就显示出了作用。

例如，对于下述两个由人工翻译和在已经完成学习的 RNN 模型中运行集束搜索算法时算法得到的翻译结果：

- Human: Jane visits Africa in September. $(y^{* })$
- Algorithm: Jane visited Africa last September. $(\hat y)$

通过我们的 RNN 模型，我们分别计算人类翻译的概率 
$P(y^{* }|x)$ 
以及模型翻译的概率 
$P(\hat y|x)$ 
，比较两个概率的大小：

- 如果 
$P(y^{∗ }|x)>P(\hat y|x)$
，说明是集束搜索算法出现错误，没有选择到概率最大的词；
- 
如果 $P(y^{∗ }|x)≤P(\hat y|x)$，
说明是 RNN 模型的效果不佳，因为，根据人类经验， $y^{* }$ 翻译比 $\hat y$ 要好，然而 RNN 模型却认为 $P(y^∗ |x)≤P(\hat y|x)$，所以这里是RNN模型出现了错误。

建立一个如下图所示的表格，记录对每一个错误的分析，有助于判断错误出现在 RNN 模型还是集束搜索算法中。如果错误出现在集束搜索算法中，可以考虑增大集束宽 B；否则，需要进一步分析，看是需要正则化、更多数据或是尝试一个不同的网络结构。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week3/md_images/05.jpg)

## Bleu得分（选修）

对于机器翻译系统来说，一种语言对于另外一种语言的翻译常常有多种正确且合适的翻译，我们无法做到像图像识别一样有固定准确度答案，所以针对不同的翻译结果，往往很难评估那一个结果是更好的，哪一个翻译系统是更加有效的。这里引入Bleu score 用来评估翻译系统的准确性。（Bleu, bilingual evaluation understudy）

评估机器翻译：

最原始的 Bleu 将机器翻译结果中每个单词在人工翻译中出现的次数作为分子，机器翻译结果总词数作为分母得到。但是容易出现错误。

我们假设机器翻译的结果是：「the the the the the the the」。按照上述的评判标准，机器翻译输出了七个单词并且这七个词中的每一个都出现在了参考 1 或是参考 2。单词 the 在两个参考中都出现了，所以看上去每个词都是很合理的。因此这个输出的精确度就是 7/7，看起来是一个极好的精确度。然而，这个输入烂的一比。

因此，我们需要改良的精确度评估方法，我们把每一个单词的记分上限定为它在参考句子中出现的最多次数。在参考 1 中，单词 the 出现了 2 次，在参考 2 中，单词 the 只出现了 1 次。而 2 比 1 大，所以我们会说，单词the的得分上限为2。有了这个改良后的精确度，我们就说，这个输出句子的得分为2/7，

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week3/md_images/06.png)

上面的方法是一个词一个词进行统计，这种以一个单词为单位的集合统称为 unigram（一元组）。以 uni-gram 统计得到的精度 $p_1$ 体现了翻译的充分性，也就是逐字逐句地翻译能力。而以成对的词为单位的集合称为 bi-gram（二元组）。对每个二元组，可以统计其在机器翻译结果（count）和人工翻译结果（countclip）出现的次数，计算 Bleu 得分。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week3/md_images/07.jpg)

例如对以上机器翻译结果（count）及参考翻译（countclip）以二元组统计有：

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week3/md_images/08.png)

以此类推，以 $n$ 个单词为单位的集合称为 n-gram（多元组），对应的 Blue（即翻译精确度）得分计算公式为：

$$p_n = \frac{\sum_{\text{n-gram} \in \hat y}count_{clip}(\text{n-gram})}{\sum_{\text{n-gram} \in \hat y}count(\text{n-gram})}$$

对 $N$ 个 $p_n$ 进行几何加权平均得到：

$$p_{ave} = exp(\frac{1}{N}\sum^N_{i=1}log^{p_n})$$

有一个问题是，当机器翻译结果短于人工翻译结果时，比较容易能得到更大的精确度分值，因为输出的大部分词可能都出现在人工翻译结果中。改进的方法是设置一个最佳匹配长度（Best Match Length），如果机器翻译的结果短于该最佳匹配长度，则需要接受简短惩罚（Brevity Penalty，BP）：

$$
BP =
\begin{cases}
1, &MT\_length \ge BM\_length \\
exp(1 - \frac{MT\_length}{BM\_length}), &MT\_length \lt BM\_length
\end{cases}
$$

因此，最后得到的 Bleu 得分为：

$$Blue = BP \times exp(\frac{1}{N}\sum^N_{i=1}log^{p_n})$$

Bleu 得分的贡献是提出了一个表现不错的单一实数评估指标，因此加快了整个机器翻译领域以及其他文本生成领域的进程。

相关论文：[Papineni et. al., 2002. A method for automatic evaluation of machine translation](http://www.aclweb.org/anthology/P02-1040.pdf)

## 注意力模型

人工翻译一大段文字时，一般都是阅读其中的一小部分后翻译出这一部分，在一小段时间里注意力只能集中在一小段文字上，而很难做到把整段读完后一口气翻译出来。用Seq2Seq模型构建的机器翻译系统中，输出结果的BLEU评分会随着输入序列长度的增加而下降，其中的道理就和这个差不多。

实际上，我们也并不希望神经网络每次去「记忆」很长一段文字，而是想让它像人工翻译一样工作。因此，**注意力模型（Attention Model）** 被提出。目前，其思想已经成为深度学习领域中最有影响力的思想之一。

注意力模型中，网络的示例结构如下所示：

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week3/md_images/09.jpg)

底层是一个双向循环神经网络，需要处理的序列作为它的输入。该网络中每一个时间步的激活 $a^{⟨t^{′}⟩}$ 中，都包含前向传播产生的和反向传播产生的激活：

$$a^{\langle t’ \rangle} = ({\overrightarrow a}^{\langle t’ \rangle}, {\overleftarrow a}^{\langle t’ \rangle})$$

顶层是一个「多对多」结构的循环神经网络，第 $t$ 个时间步的输入包含该网络中前一个时间步的激活 $s^{\langle t-1 \rangle}$、输出 $y^{\langle t-1 \rangle}$ 以及底层的 BRNN 中多个时间步的激活 $c$，其中 $c$ 有（注意分辨 $\alpha$ 和 $a$）：

$$c^{\langle t \rangle} = \sum\_{t’}\alpha^{\langle t,t’ \rangle}a^{\langle t’ \rangle}$$

其中，参数 $\alpha^{\langle t,t’ \rangle}$ 即代表着 $y^{\langle t \rangle}$ 对 $a^{\langle t' \rangle}$ 的「注意力」，总有：

$$\sum\_{t’}\alpha^{\langle t,t’ \rangle} = 1$$

我们使用 Softmax 来确保上式成立，因此有：

$$\alpha^{\langle t,t’ \rangle} = \frac{exp(e^{\langle t,t’ \rangle})}{\sum^{T\_x}\_{t'=1}exp(e^{\langle t,t’ \rangle})}$$

而对于 $e^{\langle t,t’ \rangle}$，我们通过神经网络学习得到。输入为 $s^{\langle t-1 \rangle}$ 和 $a^{\langle t’ \rangle}$，如下图所示：

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week3/md_images/10.png)

注意力模型的一个缺点是时间复杂度为 $O(n^3)$。

相关论文：

* [Bahdanau et. al., 2014. Neural machine translation by jointly learning to align and translate](https://arxiv.org/pdf/1409.0473.pdf)
* [Xu et. al., 2015. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)

## 语音识别

在语音识别中，要做的是将输入的一段语音 $x$ 转换为一段文字副本作为输出。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week3/md_images/11.jpg)

曾经的语音识别系统都是采用人工设计出的音素（Phonemes）识别单元来构建，音素指的是一种语言中能区别两个词的最小语音单位。现在有了端对端深度学习，已经完美没有必要采用这种识别音素的方法实现语音识别。

采用深度学习方法训练语音识别系统的前提条件是拥有足够庞大的训练数据集。在学术界的研究中，3000小时的长度被认为是训练一个语音识别系统时，需要的较为合理的音频数据大小。而训练商用级别的语音识别系统，需要超过一万小时甚至十万小时以上的音频数据。

语音识别系统可以采用注意力模型来构建：

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week3/md_images/12.jpg)

用 **CTC（Connectionist Temporal Classification）** 损失函数来做语音识别的效果也不错。由于输入是音频数据，使用 RNN 所建立的系统含有很多个时间步，且输出数量往往小于输入。因此，不是每一个时间步都有对应的输出。CTC 允许 RNN 生成下图红字所示的输出，并将两个空白符（blank）中重复的字符折叠起来，再将空白符去掉，得到最终的输出文本。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week3/md_images/13.png)

相关论文：[Graves et al., 2006. Connectionist Temporal Classification: Labeling unsegmented sequence data with recurrent neural networks](http://people.idsia.ch/~santiago/papers/icml2006.pdf)

## 触发字检测

触发词检测（Trigger Word Detection）现在已经被应用在各种语音助手以及智能音箱上。例如在Windows 10上能够设置微软小娜用指令「你好，小娜」进行唤醒，安卓手机上的Google Assistant则可以通过「OK，Google」唤醒。

想要训练一个触发词检测系统，同样需要有大量的标记好的训练数据。使用RNN训练语音识别系统实现触发词词检测的功能时，可以进行如下图所示的工作：

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week3/md_images/14.jpg)