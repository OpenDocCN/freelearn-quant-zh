# 前言

多样化数据的可用性增加了对算法交易策略专业知识的需求。通过本书，您将选择并应用**机器学习**（**ML**）到广泛的数据源，并创建强大的算法策略。

本书将首先介绍一些基本要素，如评估数据集、使用 Python 访问数据 API、使用 Quandl 访问金融数据以及管理预测误差。然后我们将涵盖各种机器学习技术和算法，这些技术和算法可用于使用 pandas、Seaborn、StatsModels 和 sklearn 构建和训练算法模型。然后我们将使用 StatsModels 构建、估计和解释 AR(p)、MA(q) 和 ARIMA(p, d, q) 模型。您将应用贝叶斯先验、证据和后验的概念，以区分使用 PyMC3 的不确定性概念。然后我们将利用 NLTK、sklearn 和 spaCy 为财经新闻分配情感得分，并对文档进行分类以提取交易信号。我们将学习设计、构建、调整和评估前馈神经网络、**循环神经网络**（**RNNs**）和**卷积神经网络**（**CNNs**），使用 Keras 设计复杂的算法。您将应用迁移学习到卫星图像数据，以预测经济活动。最后，我们将应用强化学习来获得最佳交易结果。

通过本书，您将能够采用算法交易来实现智能投资策略。

# 本书的受众

本书适用于数据分析师、数据科学家和 Python 开发人员，以及在金融和投资行业工作的投资分析师和投资组合经理。如果您想通过开发智能调查策略使用 ML 算法来执行高效的算法交易，那么这就是您需要的书！对 Python 和 ML 技术的一些理解是必需的。

# 本书内容

第一章，*用于交易的机器学习*，通过概述 ML 在生成和评估信号以设计和执行交易策略中的重要性来确定本书的重点。它从假设生成和建模、数据选择和回测到评估和执行在投资组合背景下的策略过程进行了概述，包括风险管理。

第二章，*市场与基础数据*，介绍了数据来源以及如何处理原始交易所提供的 tick 和财务报告数据，以及如何访问许多本书中将依赖的开源数据提供商。

第三章，*金融替代数据*，提供了评估不断增加的数据来源和供应商的分类和标准。它还演示了如何通过网站爬取创建替代数据集，例如收集用于第二部分书籍中的**自然语言处理**（**NLP**）和情感分析算法的收益电话转录。

第四章，*Alpha 因子研究*，提供了理解因子工作原理以及如何衡量其绩效的框架，例如使用**信息系数**（**IC**）。它演示了如何使用 Python 库离线和在 Quantopian 平台上工程化数据生成 alpha 因子。它还介绍了使用`zipline`库对因子进行回测和使用`alphalens`库评估其预测能力。

第五章，*战略评估*，介绍了如何利用历史数据使用`zipline`离线和在 Quantopian 平台上建立、测试和评估交易策略。它展示并演示了如何使用`pyfolio`库计算投资组合绩效和风险指标。它还讨论了如何处理策略回测的方法论挑战，并介绍了从投资组合风险角度优化策略的方法。

第六章，*机器学习工作流*，通过概述如何构建、训练、调整和评估 ML 模型的预测性能作为系统化工作流程，为后续章节做好铺垫。

第七章，*线性模型*，展示了如何使用线性和逻辑回归进行推断和预测，以及如何使用正则化来管理过拟合的风险。它介绍了 Quantopian 交易平台，并演示了如何构建因子模型并预测资产价格。

第八章，*时间序列模型*，涵盖了单变量和多变量时间序列，包括向量自回归模型和协整检验，以及它们如何应用于配对交易策略。

第九章，*贝叶斯机器学习*，介绍了如何制定概率模型以及如何使用**马尔可夫链蒙特卡罗**（**MCMC**）采样和变分贝叶斯来进行近似推断。它还说明了如何使用 PyMC3 进行概率编程以深入了解参数和模型的不确定性。

第十章，*决策树和随机森林*，展示了如何构建、训练和调整非线性基于树的模型以进行洞察和预测。它介绍了基于树的集成模型，并展示了随机森林如何使用自举聚合来克服决策树的一些弱点。第十一章，*梯度提升机*，展示了如何使用库`xgboost`、`lightgbm`和`catboost`进行高性能训练和预测，并深入审查了如何调整众多超参数。

第十一章，*梯度提升机*，演示了如何使用库`xgboost`、`lightgbm`和`catboost`进行高性能训练和预测，并深入审查了如何调整众多超参数。

第十二章，*无监督学习*，介绍了如何使用降维和聚类进行算法交易。它使用主成分和独立成分分析来提取数据驱动的风险因素。它提出了几种聚类技术，并演示了如何使用层次聚类进行资产配置。

第十三章，*处理文本数据*，演示了如何将文本数据转换为数值格式，并将*第二部分*中的分类算法应用于大型数据集的情感分析。

第十四章，*主题建模*，应用贝叶斯无监督学习来提取能够总结大量文档的潜在主题，并提供更有效地探索文本数据或将主题用作分类模型特征的方法。它演示了如何将这一技术应用于第三章，*金融替代数据*，中来源的盈利电话交易摘要和向**证券交易委员会**（**SEC**）提交的年度报告。

第十五章，*词嵌入*，使用神经网络学习形式的最新语言特征，即捕获语义上下文比传统文本特征更好的词向量，并代表从文本数据中提取交易信号的一个非常有前途的途径。

第十六章，*下一步*，是对所有前面章节的总结

[第十七章](https://www.packtpub.com/sites/default/files/downloads/Deep_Neural_Networks.pdf)，*深度学习*，介绍了 Keras、TensorFlow 和 PyTorch，这是我们将在第四部分中使用的最流行的深度学习框架。它还介绍了训练和调整的技术，包括正则化，并提供了常见架构的概述。要阅读此章节，请访问链接[`www.packtpub.com/sites/default/files/downloads/Deep_Learning.pdf`](https://www.packtpub.com/sites/default/files/downloads/Deep_Learning.pdf)。

[第十八章](https://www.packtpub.com/sites/default/files/downloads/Recurrent_Neural_Networks.pdf)，*循环神经网络*，展示了 RNN 在序列到序列建模中的用途，包括用于时间序列。它演示了 RNN 如何在较长时间段内捕捉非线性模式。要阅读此章节，请访问链接[`www.packtpub.com/sites/default/files/downloads/Recurrent_Neural_Networks.pdf`](https://www.packtpub.com/sites/default/files/downloads/Recurrent_Neural_Networks.pdf)。

[第十九章](https://www.packtpub.com/sites/default/files/downloads/Convolutions_Neural_Networks.pdf)，*卷积神经网络*，涵盖了 CNN 在大规模非结构化数据分类任务中的强大性能。我们将介绍成功的架构设计，例如在卫星数据上训练 CNN，以预测经济活动，并使用迁移学习加速训练。要阅读此章节，请访问链接[`www.packtpub.com/sites/default/files/downloads/Convolutions_Neural_Networks.pdf`](https://www.packtpub.com/sites/default/files/downloads/Convolutions_Neural_Networks.pdf)。

[第二十章](https://www.packtpub.com/sites/default/files/downloads/Unsupervised_Deep_Learning.pdf)，*自编码器和生成对抗网络*，介绍了无监督深度学习，包括用于高维数据非线性压缩的自编码器和**生成对抗网络**（**GANs**），这是生成合成数据的最重要的最近创新之一。要阅读此章节，请访问链接[`www.packtpub.com/sites/default/files/downloads/Autoencoders_and_Generative_Adversarial_Nets.pdf`](https://www.packtpub.com/sites/default/files/downloads/Autoencoders_and_Generative_Adversarial_Nets.pdf)。

[第二十一章](https://www.packtpub.com/sites/default/files/downloads/Reinforcement_Learning.pdf)，*强化学习*，介绍了允许设计和训练代理程序以随着时间响应其环境优化决策的强化学习。您将看到如何构建一个通过 Open AI gym 响应市场信号的代理。要阅读此章节，请访问链接[`www.packtpub.com/sites/default/files/downloads/Reinforcement_Learning.pdf`](https://www.packtpub.com/sites/default/files/downloads/Reinforcement_Learning.pdf)。

# 要充分利用本书

本书所需的全部内容只需要基本的 Python 和机器学习技术的理解。

# 下载示例代码文件

您可以从 [www.packt.com](http://www.packt.com) 的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，可以访问 [www.packt.com/support](http://www.packt.com/support) 并注册，以便直接将文件发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  在 [www.packt.com](http://www.packt.com) 登录或注册。

1.  选择 **支持** 选项卡。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书名并按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的软件解压缩文件夹：

+   Windows 系统使用 WinRAR/7-Zip

+   Mac 系统使用 Zipeg/iZip/UnRarX

+   Linux 系统使用 7-Zip/PeaZip

本书的代码包也托管在 GitHub 上，网址为 [`github.com/PacktPublishing/Hands-On-Machine-Learning-for-Algorithmic-Trading`](https://github.com/PacktPublishing/Hands-On-Machine-Learning-for-Algorithmic-Trading)。如果代码有更新，将在现有的 GitHub 仓库上进行更新。

我们还提供了来自我们丰富的图书和视频目录的其他代码包，可在 **[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)** 上获取。快去看看吧！

# 下载彩色图片

我们还提供了一个 PDF 文件，其中包含本书中使用的截图/图表的彩色图像。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789346411_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781789346411_ColorImages.pdf)。

# 使用的约定

本书中使用了一些文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 句柄。例如：“调用`run_algorithm()`函数后，算法继续执行，并返回相同的回测性能`DataFrame`。”

代码块设置如下：

```py
interesting_times = extract_interesting_date_ranges(returns=returns)
interesting_times['Fall2015'].to_frame('pf') \
    .join(benchmark_rets) \
    .add(1).cumprod().sub(1) \
    .plot(lw=2, figsize=(14, 6), title='Post-Brexit Turmoil')
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。

警告或重要说明显示如下。

提示和技巧显示如下。
