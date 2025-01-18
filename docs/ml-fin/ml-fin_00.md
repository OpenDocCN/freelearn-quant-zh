# 前言

在大量数据计算资源的支持下，**机器学习**（**ML**）取得了重大进展。金融行业作为一个以信息处理为核心的企业，蕴藏着大量应用这些新技术的机会。

本书是一本现代机器学习在金融行业应用的实用指南。采用代码优先的方法，它将教你最有用的机器学习算法是如何工作的，以及如何利用它们解决现实世界的问题。

# 本书适合的人群

有三类人群将从本书中受益最大：

+   希望进入金融行业并了解可能的应用范围和相关问题的数据科学家

+   希望提升技能并将高级机器学习方法应用于建模过程的任何金融科技公司开发人员或量化金融专业人士

+   希望为进入劳动力市场做好准备并学习一些雇主重视的实用技能的学生

本书假设你具备一定的线性代数、统计学、概率论和微积分基础知识。然而，你不必是这些领域的专家。

为了跟随代码示例，你应该熟悉 Python 和常见的数据科学库，如 pandas、NumPy 和 Matplotlib。本书的示例代码是在 Jupyter Notebooks 中呈现的。

不需要显式的金融知识。

# 为了最大限度地利用本书

所有代码示例托管在 Kaggle 上。你可以免费使用 Kaggle 并获得 GPU，这将使你能够更快地运行示例代码。如果你的机器没有非常强大的 GPU，使用 Kaggle 运行代码将更为舒适。你可以在本书的 GitHub 页面找到所有笔记本的链接：[`github.com/PacktPublishing/Machine-Learning-for-Finance`](https://github.com/PacktPublishing/Machine-Learning-for-Finance)。

本书假设你具备一定的数学概念基础，如线性代数、统计学、概率论和微积分。但你不必是这些领域的专家。

同样，假设你了解 Python 和一些流行的数据科学库，如 pandas 和 Matplotlib。

## 下载示例代码文件

你可以从[`www.packt.com`](http://www.packt.com)的账户中下载本书的示例代码文件。如果你是从其他地方购买的本书，可以访问[`www.packt.com/support`](http://www.packt.com/support)，注册后即可直接通过邮件收到代码文件。

你可以按照以下步骤下载代码文件：

1.  在[`www.packt.com`](http://www.packt.com)登录或注册。

1.  选择**支持**标签。

1.  点击**代码下载与勘误**。

1.  在**搜索**框中输入书名，并按照屏幕上的指示操作。

文件下载后，请确保使用最新版本的工具解压或提取文件夹：

+   WinRAR / 7-Zip for Windows

+   Zipeg / iZip / UnRarX for Mac

+   7-Zip / PeaZip for Linux

我们还提供了来自我们丰富书籍和视频目录的其他代码包，访问[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)了解更多！

## 下载彩色图像

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图像。你可以在这里下载：`www.packtpub.com/sites/default/files/downloads/9781789136364_ColorImages.pdf`。

## 使用的约定

本书中使用了许多文本约定。

`CodeInText`：指文中出现的代码词汇、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 账号。例如：“将下载的`WebStorm-10*.dmg`磁盘映像文件挂载为系统中的另一个磁盘。”

一段代码块的设置方式如下：

```py
import numpy as np
x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)
x_train.shape
```

当我们希望引起你对代码块中特定部分的注意时，相关的行或项目会以粗体显示：

```py
from keras.models import Sequential
img_shape = (28,28,1)
model = Sequential()
model.add(Conv2D(6,3,input_shape=img_shape))
```

任何命令行输入或输出如下所示：

```py
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
60000/60000 [==============================] - 22s 374us/step - loss: 7707.2773 - acc: 0.6556 - val_loss: 55.7280 - val_acc: 0.7322

```

**粗体**：表示新术语、重要词汇，或在屏幕上看到的词语，例如在菜单或对话框中，也以此格式出现在文本中。例如：“从**系统信息**中选择**管理面板**。”

### 注意

警告或重要的说明会以此形式呈现。

### 提示

提示和技巧将以此形式出现。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：如果你对本书的任何方面有疑问，请在邮件主题中提到书名，并通过`customercare@packtpub.com`联系我们。

**勘误**：尽管我们已尽力确保内容的准确性，但错误仍然会发生。如果你发现了本书中的错误，我们将非常感激你向我们报告。请访问[`www.packt.com/submit-errata`](http://www.packt.com/submit-errata)，选择你的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果你在互联网上发现我们作品的非法复制版本，我们将非常感谢你提供具体的地址或网站名称。请通过`copyright@packt.com`与我们联系，并附上链接。

**如果你有兴趣成为作者**：如果你在某个主题上有专长，并且有兴趣写书或为书籍贡献内容，请访问[`authors.packtpub.com`](http://authors.packtpub.com)。

## 评价

请留下评论。当你阅读并使用完本书后，为什么不在你购买本书的网站上留下评论呢？潜在的读者可以看到并参考你的公正意见来做出购买决定，我们 Packt 也能了解你对我们产品的看法，作者们也能看到你对他们书籍的反馈。谢谢！

想了解更多关于 Packt 的信息，请访问[packt.com](http://packt.com)。
