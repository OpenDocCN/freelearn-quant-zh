# 前言

本书的第二版*Mastering Python for Finance*将指导您使用下一代方法在金融行业中进行复杂的金融计算。您将通过利用公开可用的工具来掌握 Python 生态系统，成功进行研究和建模，并学习如何使用高级示例来管理风险。

您将首先设置一个 Jupyter 笔记本，以实现本书中的任务。您将学习如何使用流行的库（如 TensorFlow、Keras、NumPy、SciPy、scikit-learn 等）做出高效而强大的数据驱动金融决策。您还将学习如何通过掌握股票、期权、利率及其衍生品以及使用计算方法进行风险分析等概念来构建金融应用程序。有了这些基础，您将学习如何对时间序列数据进行统计分析，并了解如何利用高频数据来设计交易策略，从而构建算法交易平台。您将学习通过实施事件驱动的回测系统来验证您的交易策略，并衡量其性能。最后，您将探索在金融领域应用的机器学习和深度学习技术。

通过本书，您将学会如何将 Python 应用于金融行业中的不同范式，并进行高效的数据分析。

# 这本书是为谁准备的

如果您是金融或数据分析师，或者是金融行业的软件开发人员，有兴趣使用高级 Python 技术进行量化方法，那么这本书就是您需要的！如果您想要使用智能机器学习技术扩展现有金融应用程序的功能，您也会发现这本书很有用。

# 充分利用本书

需要有 Python 的先验经验。

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，文件将直接发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  登录或注册[www.packt.com](http://www.packt.com)。

1.  选择“支持”选项卡。

1.  单击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的软件解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Mastering-Python-for-Finance-Second-Edition`](https://github.com/PacktPublishing/Mastering-Python-for-Finance-Second-Edition)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还有其他代码包，来自我们丰富的图书和视频目录，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**上找到。去看看吧！

# 下载彩色图像

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图像。您可以在这里下载：`www.packtpub.com/sites/default/files/downloads/9781789346466_ColorImages.pdf`。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。例如：“默认情况下，pandas 的`.plot()`命令使用`matplotlib`库来显示图形。”

代码块设置如下：

```py
In [ ]:
     %matplotlib inline
     import quandl

     quandl.ApiConfig.api_key = QUANDL_API_KEY
     df = quandl.get('EURONEXT/ABN.4')
     daily_changes = df.pct_change(periods=1)
     daily_changes.plot();
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```py
2015-02-26 TICK WIKI/AAPL open: 128.785 close: 130.415
2015-02-26 FILLED BUY 1 WIKI/AAPL at 128.785
2015-02-26 POSITION value:-128.785 upnl:1.630 rpnl:0.000
2015-02-27 TICK WIKI/AAPL open: 130.0 close: 128.46
```

任何命令行输入或输出都是按照以下格式编写的：

```py
$ cd my_project_folder
$ virtualenv my_env
```

**粗体**：表示一个新术语，一个重要词或屏幕上看到的词。例如，菜单或对话框中的单词会以这种方式出现在文本中。这是一个例子：“要启动你的第一个笔记本，选择**新建**，然后**Python 3**。”

警告或重要提示会显示为这样。提示和技巧会显示为这样。
