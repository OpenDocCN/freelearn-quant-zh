# 第十章：金融机器学习

机器学习正在迅速被金融服务行业广泛采用。金融服务业对机器学习的采用受到供应因素的推动，如数据存储、算法和计算基础设施的技术进步，以及需求因素的推动，如盈利需求、与其他公司的竞争，以及监管和监督要求。金融中的机器学习包括算法交易、投资组合管理、保险承保和欺诈检测等多个领域。

有几种类型的机器学习算法，但在机器学习文献中你通常会遇到的两种主要算法是监督学习和无监督学习。我们本章的讨论重点在监督学习上。监督学习涉及提供输入和输出数据来帮助机器预测新的输入数据。监督学习可以是基于回归或基于分类的。基于回归的机器学习算法预测连续值，而基于分类的机器学习算法预测类别或标签。

在本章中，我们将介绍机器学习，研究其在金融领域的概念和应用，并查看一些应用机器学习来辅助交易决策的实际例子。我们将涵盖以下主题：

+   探索金融中的机器学习应用

+   监督学习和无监督学习

+   基于分类和基于回归的机器学习

+   使用 scikit-learn 实现机器学习算法

+   应用单资产回归机器学习来预测价格

+   了解风险度量标准以衡量回归模型

+   应用多资产回归机器学习来预测回报

+   应用基于分类的机器学习来预测趋势

+   了解风险度量标准以衡量分类模型

# 机器学习简介

在机器学习算法成熟之前，许多软件应用决策都是基于规则的，由一堆`if`和`else`语句组成，以生成适当的响应以交换一些输入数据。一个常见的例子是电子邮箱收件箱中的垃圾邮件过滤器功能。邮箱可能包含由邮件服务器管理员或所有者定义的黑名单词。传入的电子邮件内容会被扫描以检查是否包含黑名单词，如果黑名单条件成立，邮件将被标记为垃圾邮件并发送到“垃圾邮件”文件夹。随着不受欢迎的电子邮件的性质不断演变以避免被检测，垃圾邮件过滤机制也必须不断更新自己以做得更好。然而，通过机器学习，垃圾邮件过滤器可以自动从过去的电子邮件数据中学习，并在收到新的电子邮件时计算分类新邮件是否为垃圾邮件的可能性。

面部识别和图像检测背后的算法基本上是相同的。存储在位和字节中的数字图像被收集、分析和分类，根据所有者提供的预期响应。这个过程被称为“训练”，使用“监督学习”方法。经过训练的数据随后可以用于预测下一组输入数据作为某种输出响应，并带有一定的置信水平。另一方面，当训练数据不包含预期的响应时，机器学习算法被期望从训练数据中学习，这个过程被称为“无监督学习”。

# 金融中的机器学习应用

机器学习在金融领域的许多领域中越来越多地发挥作用，如数据安全、客户服务、预测和金融服务。许多使用案例也利用了大数据和人工智能，它们并不仅限于机器学习。在本节中，我们将探讨机器学习如何改变金融行业的一些方式。

# 算法交易

机器学习算法研究高度相关资产价格的统计特性，在回测期间测量它们对历史数据的预测能力，并预测价格在一定精度范围内。机器学习交易算法可能涉及对订单簿、市场深度和成交量、新闻发布、盈利电话或财务报表的分析，分析结果转化为价格变动可能性，并纳入生成交易信号的考虑。

# 投资组合管理

近年来，“机器顾问”这一概念越来越受欢迎，作为自动化对冲基金经理。它们帮助进行投资组合构建、优化、配置和再平衡，甚至根据客户的风险承受能力和首选投资工具建议客户投资的工具。这些咨询服务作为与数字财务规划师互动的平台，提供财务建议和投资组合管理。

# 监管和监管职能

金融机构和监管机构正在采用人工智能和机器学习来分析、识别和标记需要进一步调查的可疑交易。像证券交易委员会这样的监管机构采取数据驱动的方法，利用人工智能、机器学习和自然语言处理来识别需要执法的行为。全球范围内，中央机构正在开发监管职能的机器学习能力。

# 保险和贷款承销

保险公司积极利用人工智能和机器学习来增强一些保险行业的功能，改善保险产品的定价和营销，减少理赔处理时间和运营成本。在贷款承销方面，单个消费者的许多数据点，如年龄、收入和信用评分，与候选人数据库进行比较，以建立信用风险概况，确定信用评分，并计算贷款违约的可能性。这些数据依赖于金融机构的交易和支付历史。然而，放贷人越来越多地转向社交媒体活动、手机使用和消息活动，以捕捉对信用价值的更全面的观点，加快放贷决策，限制增量风险，并提高贷款的评级准确性。

# 新闻情绪分析

自然语言处理，作为机器学习的一个子集，可以用于分析替代数据、财务报表、新闻公告，甚至是 Twitter 动态，以创建由对冲基金、高频交易公司、社交交易和投资平台使用的投资情绪指标，用于实时分析市场。政治家的演讲，或者重要的新发布，比如中央银行发布的，也正在实时分析，每个字都在被审查和计算，以预测资产价格可能会如何变动以及变动的幅度。机器学习不仅能理解股价和交易的波动，还能理解社交媒体动态、新闻趋势和其他数据来源。

# 金融之外的机器学习

机器学习越来越多地应用于面部识别、语音识别、生物识别、贸易结算、聊天机器人、销售推荐、内容创作等领域。随着机器学习算法的改进和采用速度的加快，使用案例的列表变得更加长。

让我们通过了解一些术语来开始我们的机器学习之旅，这些术语在机器学习文献中经常出现。

# 监督和无监督学习

有许多类型的机器学习算法，但你通常会遇到的两种主要类型是监督和无监督机器学习。

# 监督学习

监督学习从给定的输入中预测特定的输出。这些输入到输出数据的配对被称为**训练数据**。预测的质量完全取决于训练数据；不正确的训练数据会降低机器学习模型的有效性。例如，一个带有标签的交易数据集，标识哪些是欺诈交易，哪些不是。然后可以构建模型来预测新交易是否是欺诈交易。

监督学习中一些常见的算法包括逻辑回归、支持向量机和随机森林。

# 无监督学习

无监督学习是基于给定的不包含标签的输入数据构建模型，而是要求检测数据中的模式。这可能涉及识别具有相似基本特征的观察值的聚类。无监督学习旨在对新的、以前未见过的数据进行准确预测。

例如，无监督学习模型可以通过寻找具有相似特征的证券群来定价不流动的证券。常见的无监督学习算法包括 k 均值聚类、主成分分析和自动编码器。

# 监督机器学习中的分类和回归

有两种主要类型的监督机器学习算法，主要是分类和回归。分类机器学习模型试图从预定义的可能性列表中预测和分类响应。这些预定义的可能性可能是二元分类（例如对问题的“是这封电子邮件是垃圾吗？”的*是*或*否*回答）或多类分类。

回归机器学习模型试图预测连续的输出值。例如，预测房价或温度都期望连续范围的输出值。常见的回归形式有普通最小二乘（OLS）回归、LASSO 回归、岭回归和弹性网络正则化。

# 过度拟合和欠拟合模型

机器学习模型的性能不佳可能是由于过度拟合或欠拟合造成的。过度拟合的机器学习模型是指在训练数据上训练得太好，导致在新数据上表现不佳。这是因为训练数据适应了每一个微小的变化，包括噪音和随机波动。无监督学习算法非常容易过度拟合，因为模型从每个数据中学习，包括好的和坏的。

欠拟合的机器学习模型预测准确性差。这可能是由于可用于构建准确模型的训练数据太少，或者数据不适合提取其潜在趋势。欠拟合模型很容易检测，因为它们始终表现不佳。要改进这样的模型，可以提供更多的训练数据或使用另一个机器学习算法。

# 特征工程

特征是定义数据特征的属性。通过使用数据的领域知识，可以创建特征来帮助机器学习算法提高其预测性能。这可以是简单的将现有数据的相关部分分组或分桶以形成定义特征。甚至删除不需要的特征也是特征工程。

例如，假设我们有以下时间序列价格数据：

| **编号** | **日期和时间** | **价格** | **价格行动** |
| --- | --- | --- | --- |
| 1 | 2019-01-02 09:00:01 | 55.00 | 上涨 |
| 2 | 2019-01-02 10:03:42 | 45.00 | 下跌 |
| 3 | 2019-01-02 10:31:23 | 48.00 | 上涨 |
| 4 | 2019-01-02 11:14:02 | 33.00 | DOWN |

通过一天中的小时将时间序列分组，并在每个时间段内采取最后的价格行动，我们得到了这样一个特征：

| **No.** | **Hour of Day** | **Last Price Action** |
| --- | --- | --- |
| 1 | 9 | UP |
| 2 | 10 | UP |
| 3 | 11 | DOWN |

特征工程的过程包括以下四个步骤：

1.  构思要包括在训练模型中的特征

1.  创建这些特征

1.  检查特征如何与模型配合

1.  从步骤 1 重复，直到特征完美工作

在构建特征方面，没有绝对的硬性规则。特征工程被认为更像是一门艺术而不是科学。

# 用于机器学习的 Scikit-learn

Scikit-learn 是一个专为科学计算设计的 Python 库，包含一些最先进的机器学习算法，用于分类、回归、聚类、降维、模型选择和预处理。其名称源自 SciPy 工具包，这是 SciPy 模块的扩展。有关 scikit-learn 的详细文档可以在[`scikit-learn.org`](https://scikit-learn.org)找到。

SciPy 是用于科学计算的 Python 模块集合，包含一些核心包，如 NumPy、Matplotlib、IPython 等。

在本章中，我们将使用 scikit-learn 的机器学习算法来预测证券的走势。Scikit-learn 需要安装 NumPy 和 SciPy。使用以下命令通过`pip`包管理器安装 scikit-learn：

```py
 pip install scikit-learn
```

# 使用单一资产回归模型预测价格

配对交易是一种常见的统计套利交易策略，交易者使用一对协整和高度正相关的资产，尽管也可以考虑负相关的配对。

在本节中，我们将使用机器学习来训练基于回归的模型，使用一对可能用于配对交易的证券的历史价格。给定某一天某一证券的当前价格，我们每天预测另一证券的价格。以下示例使用了**纽约证券交易所**（**NYSE**）上交易的**高盛**（**GS**）和**摩根大通**（**JPM**）的历史每日价格。我们将预测 2018 年 JPM 股票价格。

# 通过 OLS 进行线性回归

让我们从一个简单的线性回归模型开始我们的基于回归的机器学习调查。一条直线的形式如下：

![](img/f7b4925f-5179-47a2-bf43-8b887bf19c6f.png)

这尝试通过 OLS 拟合数据：

+   *a*是斜率或系数

+   *c*是*y*截距的值

+   *x*是输入数据集

+   ![](img/b25ba907-2ae5-48c0-a41b-538f32b462f6.png)是直线的预测值

系数和截距由最小化成本函数确定：

![](img/2b7ae4b3-76ee-4adb-8a94-272eef594751.png)

*y*是用于执行直线拟合的观察实际值的数据集。换句话说，我们正在执行最小化平方误差和，以找到系数*a*和*c*，从中我们可以预测当前时期。

在开发模型之前，让我们下载并准备所需的数据集。

# 准备自变量和目标变量

让我们使用以下代码从 Alpha Vantage 获取 GS 和 JPM 的价格数据集：

```py
In [ ]:
    from alpha_vantage.timeseries import TimeSeries

    # Update your Alpha Vantage API key here...
    ALPHA_VANTAGE_API_KEY = 'PZ2ISG9CYY379KLI'

    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    df_jpm, meta_data = ts.get_daily_adjusted(
        symbol='JPM', outputsize='full')
    df_gs, meta_data = ts.get_daily_adjusted(
        symbol='GS', outputsize='full')
```

`pandas` DataFrame 对象`df_jpm`和`df_gs`包含了 JPM 和 GS 的下载价格。我们将从每个数据集的第五列中提取调整后的收盘价。

让我们使用以下代码准备我们的自变量：

```py
In [ ]:
    import pandas as pd

    df_x = pd.DataFrame({'GS': df_gs['5\. adjusted close']})
```

从 GS 的调整收盘价中提取到一个新的 DataFrame 对象`df_x`。接下来，使用以下代码获取我们的目标变量：

```py
In [ ]: 
    jpm_prices = df_jpm['5\. adjusted close']
```

JPM 的调整收盘价被提取到`jpm_prices`变量中，作为一个`pandas` Series 对象。准备好我们的数据集以用于建模后，让我们继续开发线性回归模型。

# 编写线性回归模型

我们将创建一个类，用于使用线性回归模型拟合和预测值。这个类还用作在本章中实现其他模型的基类。以下步骤说明了这个过程。

1.  声明一个名为`LinearRegressionModel`的类，如下所示：

```py
from sklearn.linear_model import LinearRegression

class LinearRegressionModel(object):
    def __init__(self):
        self.df_result = pd.DataFrame(columns=['Actual', 'Predicted'])

    def get_model(self):
        return LinearRegression(fit_intercept=False)

    def get_prices_since(self, df, date_since, lookback):
        index = df.index.get_loc(date_since)
        return df.iloc[index-lookback:index]        
```

在我们新类的构造函数中，我们声明了一个名为`df_result`的 pandas DataFrame，用于存储之后绘制图表时的实际值和预测值。`get_model()`方法返回`sklearn.linear_model`模块中`LinearRegression`类的一个实例，用于拟合和预测数据。`set_intercept`参数设置为`True`，因为数据没有居中（即在*x*和*y*轴上都围绕 0）。

有关 scikit-learn 的`LinearRegression`的更多信息可以在[`scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)找到。

`get_prices_since()`方法使用`iloc`命令从给定的日期索引`date_since`开始，获取由`lookback`值定义的较早期间的子集。

1.  在`LinearRegressionModel`类中添加一个名为`learn()`的方法，如下所示：

```py
def learn(self, df, ys, start_date, end_date, lookback_period=20):
     model = self.get_model()

     for date in df[start_date:end_date].index:
         # Fit the model
         x = self.get_prices_since(df, date, lookback_period)
         y = self.get_prices_since(ys, date, lookback_period)
         model.fit(x, y.ravel())

         # Predict the current period
         x_current = df.loc[date].values
         [y_pred] = model.predict([x_current])

         # Store predictions
         new_index = pd.to_datetime(date, format='%Y-%m-%d')
         y_actual = ys.loc[date]
         self.df_result.loc[new_index] = [y_actual, y_pred]
```

`learn()`方法作为运行模型的入口点。它接受`df`和`ys`参数作为我们的自变量和目标变量，`start_date`和`end_date`作为对应于我们将要预测的数据集索引的字符串，以及`lookback_period`参数作为用于拟合当前期间模型的历史数据点的数量。

`for`循环模拟了每日的回测。调用`get_prices_since()`获取数据集的子集，用于在*x*和*y*轴上拟合模型。`ravel()`命令将给定的`pandas` Series 对象转换为用于拟合模型的目标值的扁平列表。

`x_current`变量表示指定日期的自变量值，输入到`predict()`方法中。预测的输出是一个`list`对象，我们从中提取第一个值。实际值和预测值都保存到`df_result` DataFrame 中，由当前日期作为`pandas`对象的索引。

1.  让我们实例化这个类，并通过以下命令运行我们的机器学习模型：

```py
In [ ]:
    linear_reg_model = LinearRegressionModel()
    linear_reg_model.learn(df_x, jpm_prices, start_date='2018', 
                           end_date='2019', lookback_period=20)
```

在`learn()`命令中，我们提供了我们准备好的数据集`df_x`和`jpm_prices`，并指定了 2018 年的预测。在这个例子中，我们假设一个月有 20 个交易日。使用`lookback_period`值为`20`，我们使用过去一个月的价格来拟合我们的模型以进行每日预测。

1.  让我们从模型中检索结果的`df_result` DataFrame，并绘制实际值和预测值：

```py
In [ ]:
    %matplotlib inline

    linear_reg_model.df_result.plot(
        title='JPM prediction by OLS', 
        style=['-', '--'], figsize=(12,8));
```

在`style`参数中，我们指定实际值绘制为实线，预测值绘制为虚线。这给我们以下图表：

![](img/13a145be-ebca-4b73-90da-a02f4b8fd62c.png)

图表显示我们的预测结果在一定程度上紧随实际值。我们的模型实际上表现如何？在下一节中，我们将讨论用于衡量基于回归的模型的几种常见风险指标。

# 用于衡量预测性能的风险指标

`sklearn.metrics`模块实现了几种用于衡量预测性能的回归指标。我们将在接下来的部分讨论平均绝对误差、均方误差、解释方差得分和 R²得分。

# 作为风险指标的平均绝对误差

**平均绝对误差**（**MAE**）是一种风险度量，衡量了平均绝对预测误差，可以表示如下：

![](img/9ba0fced-2749-4951-93ce-76bea007fb68.png)

这里，*y*和![](img/0b48f0bb-7913-4104-a460-d7f2da9a005c.png)分别是实际值和预测值的列表，长度相同，为*n*。![](img/9f3fb134-6d08-4e20-b86e-90a5de21fdf5.png)和*y[i]*分别是索引*i*处的预测值和实际值。取绝对值意味着我们的输出结果为正小数。希望 MAE 的值尽可能低。完美的分数为 0 表示我们的预测能力与实际值完全一致，因为两者之间没有差异。

使用`sklearn.metrics`模块的`mean_abolute_error`函数获得我们预测的 MAE 值，以下是代码：

```py
In [ ]:
    from sklearn.metrics import mean_absolute_error

    actual = linear_reg_model.df_result['Actual']
    predicted = linear_reg_model.df_result['Predicted']

    mae = mean_absolute_error(actual, predicted)
    print('mean absolute error:', mae)
Out[ ]:
    mean absolute error: 2.4581692107823367
```

我们的线性回归模型的 MAE 为 2.458。

# 均方误差作为风险度量

与 MAE 类似，**均方误差**（**MSE**）也是一种风险度量，衡量了预测误差的平方的平均值，可以表示如下：

![](img/e43040d0-6942-4890-be73-193e3ac3b317.png)

平方误差意味着 MSE 的值始终为正，并且希望 MSE 的值尽可能低。完美的 MSE 分数为 0 表示我们的预测能力与实际值完全一致，这些差异的平方可以忽略不计。虽然 MSE 和 MAE 的应用有助于确定模型预测能力的强度，但 MSE 通过对偏离均值较远的错误进行惩罚而胜过 MAE。平方误差对风险度量施加了更重的偏见。

使用以下代码通过`sklearn.metrics`模块的`mean_squared_error`函数获得我们预测的 MSE 值：

```py
In [ ]:
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(actual, predicted)
    print('mean squared error:', mse)
Out[ ]:
    mean squared error: 12.156835196436589
```

我们的线性回归模型的 MSE 为 12.156。

# 解释方差分数作为风险度量

解释方差分数解释了给定数据集的误差分散，公式如下：

![](img/24beb2c6-1765-419a-b919-cdf3608a9789.png)

这里，*![](img/186aff43-e401-4808-b8d8-8b6adc1bd9b0.png)*和*Var*(*y*)分别是预测误差和实际值的方差。接近 1.0 的分数是非常理想的，表示误差标准差的平方更好。

使用`sklearn.metrics`模块的`explained_variance_score`函数获得我们预测的解释方差分数的值，以下是代码：

```py
In [ ]:
    from sklearn.metrics import explained_variance_score
    eva = explained_variance_score(actual, predicted)
    print('explained variance score:', eva)
Out[ ]:
    explained variance score: 0.5332235487812286
```

我们线性回归模型的解释方差分数为 0.533。

# R²作为风险度量

R²分数也被称为**确定系数**，它衡量了模型对未来样本的预测能力。它的表示如下：

![](img/9c360c02-0496-4e6f-a56b-5c31879971cb.png)

这里，![](img/40eec259-3732-44f6-ba5e-f5edf10bd6a2.png)是实际值的均值，可以表示如下：

![](img/bdb47b9a-bf50-47c2-a325-2f1cd33d08b8.png)

R²分数的范围从负值到 1.0。R²分数为 1 表示回归分析没有误差，而分数为 0 表示模型总是预测目标值的平均值。负的 R²分数表示预测表现低于平均水平。

使用`sklearn.metrics`模块的`r2_score`函数获得我们预测的 R²分数，以下是代码：

```py
In [ ]:
    from sklearn.metrics import r2_score
    r2 = r2_score(actual, predicted) 
    print('r2 score:', r2)
Out[ ]:
    r2 score: 0.41668246393290576
```

我们的线性回归模型的 R²为 0.4167。这意味着 41.67%的目标变量的可变性已经被解释。

# 岭回归

岭回归，或 L2 正则化，通过惩罚模型系数的平方和来解决 OLS 回归的一些问题。岭回归的代价函数可以表示如下：

![](img/7918e635-6bd5-4727-b68b-81d215aaa5f3.png)

在这里，α参数预期是一个控制收缩量的正值。较大的 alpha 值会产生更大的收缩，使得系数对共线性更加稳健。

`sklearn.linear_model`模块的`Ridge`类实现了岭回归。要实现这个模型，创建一个名为`RidgeRegressionModel`的类，扩展`LinearRegressionModel`类，并运行以下代码：

```py
In [ ]:
    from sklearn.linear_model import Ridge

    class RidgeRegressionModel(LinearRegressionModel): 
        def get_model(self):
            return Ridge(alpha=.5)

    ridge_reg_model = RidgeRegressionModel()
    ridge_reg_model.learn(df_x, jpm_prices, start_date='2018', 
                          end_date='2019', lookback_period=20)
```

在新类中，重写`get_model()`方法以返回 scikit-learn 的岭回归模型，同时重用父类中的其他方法。将`alpha`值设为 0.5，其余模型参数保持默认。`ridge_reg_model`变量表示我们的岭回归模型的一个实例，并使用通常的参数值运行`learn()`命令。

创建一个名为`print_regression_metrics()`的函数，以打印之前介绍的各种风险指标：

```py
In [ ]:
    from sklearn.metrics import (
        accuracy_score, mean_absolute_error, 
        explained_variance_score, r2_score
    )
    def print_regression_metrics(df_result):
        actual = list(df_result['Actual'])
        predicted = list(df_result['Predicted'])
        print('mean_absolute_error:', 
            mean_absolute_error(actual, predicted))
        print('mean_squared_error:', mean_squared_error(actual, predicted))
        print('explained_variance_score:', 
            explained_variance_score(actual, predicted))
        print('r2_score:', r2_score(actual, predicted)) 
```

将`df_result`变量传递给此函数，并在控制台显示风险指标：

```py
In [ ]:
    print_regression_metrics(ridge_reg_model.df_result)
Out[ ]:
    mean_absolute_error: 1.5894879428144535
    mean_squared_error: 4.519795633665941
    explained_variance_score: 0.7954229624785825
    r2_score: 0.7831280913202121
```

岭回归模型的平均误差得分都低于线性回归模型，并且更接近于零。解释方差得分和 R²得分都高于线性回归模型，并且更接近于 1。这表明我们的岭回归模型在预测方面比线性回归模型做得更好。除了性能更好外，岭回归计算成本也比原始线性回归模型低。

# 其他回归模型

`sklearn.linear_model`模块包含了我们可以考虑在模型中实现的各种回归模型。其余部分简要描述了它们。线性模型的完整列表可在[`scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)找到。

# Lasso 回归

与岭回归类似，**最小绝对值收缩和选择算子**（**LASSO**）回归也是正则化的另一种形式，涉及对回归系数的绝对值之和进行惩罚。它使用 L1 正则化技术。LASSO 回归的成本函数可以写成如下形式：

![](img/fd0a41c4-b24d-43b6-bcfa-7e8970f31cae.png)

与岭回归类似，alpha 参数α控制惩罚的强度。然而，由于几何原因，LASSO 回归产生的结果与岭回归不同，因为它强制大多数系数被设为零。它更适合估计稀疏系数和具有较少参数值的模型。

`sklearn.linear_model`的`Lasso`类实现了 LASSO 回归。

# 弹性网络

弹性网络是另一种正则化回归方法，结合了 LASSO 和岭回归方法的 L1 和 L2 惩罚。弹性网络的成本函数可以写成如下形式：

![](img/310632bc-d7e6-4ea2-b5a5-df69bc56c116.png)

这里解释了 alpha 值：

![](img/689be77f-a002-41e5-97fa-b52f59a26e5b.png)

![](img/4b0169a9-8d38-4ad6-8371-c4cb9a12fe28.png)

在这里，`alpha`和`l1_ratio`是`ElasticNet`函数的参数。当`alpha`为零时，成本函数等同于 OLS。当`l1_ratio`为零时，惩罚是岭或 L2 惩罚。当`l1_ratio`为 1 时，惩罚是 LASSO 或 L1 惩罚。当`l1_ratio`在 0 和 1 之间时，惩罚是 L1 和 L2 的组合。

`sklearn.linear_model`的`ElasticNet`类实现了弹性网络回归。

# 结论

我们使用了单资产的趋势跟随动量策略通过回归来预测使用 GS 的 JPM 价格，假设这对是协整的并且高度相关。我们也可以考虑跨资产动量来从多样化中获得更好的结果。下一节将探讨用于预测证券回报的多资产回归。

# 使用跨资产动量模型预测回报

在本节中，我们将通过拥有四种多样化资产的价格来预测 2018 年 JPM 的每日回报，创建一个跨资产动量模型。我们将使用 S&P 500 股票指数、10 年期国库券指数、美元指数和黄金价格的先前 1 个月、3 个月、6 个月和 1 年的滞后回报来拟合我们的模型。这给我们总共 16 个特征。让我们开始准备我们的数据集来开发我们的模型。

# 准备独立变量

我们将再次使用 Alpha Vantage 作为我们的数据提供者。由于这项免费服务没有提供我们调查所需的所有数据集，我们将考虑其他相关资产作为代理。标准普尔 500 股票指数的股票代码是 SPX。我们将使用 SPDR Gold Trust（股票代码：GLD）来表示黄金价格的代理。Invesco DB 美元指数看涨基金（股票代码：UUP）将代表美元指数。iShares 7-10 年期国库券 ETF（股票代码：IEF）将代表 10 年期国库券指数。运行以下代码来下载我们的数据集：

```py
In [ ]:
    df_spx, meta_data = ts.get_daily_adjusted(
        symbol='SPX', outputsize='full')
    df_gld, meta_data = ts.get_daily_adjusted(
        symbol='GLD', outputsize='full')
    df_dxy, dxy_meta_data = ts.get_daily_adjusted(
        symbol='UUP', outputsize='full')
    df_ief, meta_data = ts.get_daily_adjusted(
        symbol='IEF', outputsize='full')
```

`ts`变量是在上一节中创建的 Alpha Vantage 的`TimeSeries`对象。使用以下代码将调整后的收盘价合并到一个名为`df_assets`的`pandas` DataFrame 中，并使用`dropna()`命令删除空值：

```py
In [ ]:
    import pandas as pd

    df_assets = pd.DataFrame({
        'SPX': df_spx['5\. adjusted close'],
        'GLD': df_gld['5\. adjusted close'],
        'UUP': df_dxy['5\. adjusted close'],
        'IEF': df_ief['5\. adjusted close'],
    }).dropna()
```

使用以下代码计算我们的`df_assets`数据集的滞后百分比回报：

```py
IN [ ]:
    df_assets_1m = df_assets.pct_change(periods=20)
    df_assets_1m.columns = ['%s_1m'%col for col in df_assets.columns]

    df_assets_3m = df_assets.pct_change(periods=60)
    df_assets_3m.columns = ['%s_3m'%col for col in df_assets.columns]

    df_assets_6m = df_assets.pct_change(periods=120)
    df_assets_6m.columns = ['%s_6m'%col for col in df_assets.columns]

    df_assets_12m = df_assets.pct_change(periods=240)
    df_assets_12m.columns = ['%s_12m'%col for col in df_assets.columns]
```

在`pct_change()`命令中，`periods`参数指定要移动的周期数。在计算滞后回报时，我们假设一个月有 20 个交易日。使用`join()`命令将四个`pandas` DataFrame 对象合并成一个 DataFrame：

```py
In [ ]:
    df_lagged = df_assets_1m.join(df_assets_3m)\
        .join(df_assets_6m)\
        .join(df_assets_12m)\
        .dropna()
```

使用`info()`命令查看其属性：

```py
In [ ]:
    df_lagged.info()
Out[ ]:
    <class 'pandas.core.frame.DataFrame'>
    Index: 2791 entries, 2008-02-12 to 2019-03-14
    Data columns (total 16 columns):
    ...
```

输出被截断，但您可以看到 16 个特征作为我们的独立变量，跨越 2008 年至 2019 年。让我们继续获取我们目标变量的数据集。

# 准备目标变量

JPM 的收盘价早些时候已经下载到`pandas` Series 对象`jpm_prices`中，只需使用以下代码计算实际百分比收益：

```py
In [ ]:
    y = jpm_prices.pct_change().dropna()
```

我们获得一个`pandas` Series 对象作为我们的目标变量`y`。

# 多资产线性回归模型

在上一节中，我们使用了单一资产 GS 的价格来拟合我们的线性回归模型。这个相同的模型`LinearRegressionModel`可以容纳多个资产。运行以下命令来创建这个模型的实例并使用我们的新数据集：

```py
In [ ]:
    multi_linear_model = LinearRegressionModel()
    multi_linear_model.learn(df_lagged, y, start_date='2018', 
                             end_date='2019', lookback_period=10)
```

在线性回归模型实例`multi_linear_model`中，`learn()`命令使用具有 16 个特征的`df_lagged`数据集和`y`作为 JPM 的百分比变化。考虑到有限的滞后回报数据，`lookback_period`值被减少。让我们绘制 JPM 的实际与预测百分比变化：

```py
In [ ]:
    multi_linear_model.df_result.plot(
        title='JPM actual versus predicted percentage returns',
        style=['-', '--'], figsize=(12,8));
```

这将给我们以下图表，其中实线显示了 JPM 的实际百分比回报，而虚线显示了预测的百分比回报：

![](img/c417dd6d-9ece-4f2a-b9bd-964481cb61ab.png)

我们的模型表现如何？让我们在前一节中定义的`print_regression_metrics()`函数中运行相同的性能指标：

```py
In [ ]:
    print_regression_metrics(multi_linear_model.df_result)
Out[ ]:
    mean_absolute_error: 0.01952328066607389
    mean_squared_error: 0.0007225502867195044
    explained_variance_score: -2.729798588246765
    r2_score: -2.738404583097052

```

解释的方差分数和 R²分数都在负数范围内，表明模型表现低于平均水平。我们能做得更好吗？让我们探索更复杂的用于回归的树模型。

# 决策树的集成

决策树是用于分类和回归任务的广泛使用的模型，就像二叉树一样，其中每个节点代表一个问题，导致对相应左右节点的是或否答案。目标是通过尽可能少的问题得到正确答案。

可以在[`arxiv.org/pdf/1806.06988.pdf`](https://arxiv.org/pdf/1806.06988.pdf)找到描述深度神经决策树的论文。

深入决策树很快会导致给定数据的过拟合，而不是从中抽取的分布的整体特性。为了解决这个过拟合问题，数据可以分成子集，并在不同的树上进行训练，每个子集上进行训练。这样，我们最终得到了不同决策树模型的集成。当用替换抽取预测的随机样本子集时，这种方法称为**装袋**或**自举聚合**。我们可能会或可能不会在这些模型中获得一致的结果，但通过对自举模型进行平均得到的最终模型比使用单个决策树产生更好的结果。使用随机化决策树的集成被称为**随机森林**。

让我们在 scikit-learn 中访问一些决策树模型，我们可能考虑在我们的多资产回归模型中实施。

# 装袋回归器

`sklearn.ensemble`的`BaggingRegressor`类实现了装袋回归器。我们可以看看装袋回归器如何对 JPM 的百分比收益进行多资产预测。以下代码说明了这一点：

```py
In [ ]:
    from sklearn.ensemble import BaggingRegressor

    class BaggingRegressorModel(LinearRegressionModel):
        def get_model(self):
            return BaggingRegressor(n_estimators=20, random_state=0) 
In [ ]:
 bagging = BaggingRegressorModel()
    bagging.learn(df_lagged, y, start_date='2018', 
                  end_date='2019', lookback_period=10) 
```

我们创建了一个名为`BaggingRegressorModel`的类，它扩展了`LinearRegressionModel`，并且`get_model()`方法被重写以返回装袋回归器。`n_estimators`参数指定了集成中的`20`个基本估计器或决策树，`random_state`参数作为随机数生成器使用的种子为`0`。其余参数为默认值。我们使用相同的数据集运行这个模型。

运行相同的性能指标，看看我们的模型表现如何：

```py
In [ ]:
    print_regression_metrics(bagging.df_result)
Out[ ]:
    mean_absolute_error: 0.0114699264723
    mean_squared_error: 0.000246352185742
    explained_variance_score: -0.272260304849
    r2_score: -0.274602137956
```

MAE 和 MSE 值表明，决策树的集成产生的预测误差比简单线性回归模型少。此外，尽管解释方差分数和 R²分数为负值，但它表明数据的方差向均值的方向比简单线性回归模型提供的更好。

# 梯度树提升回归模型

梯度树提升，或简单地说梯度提升，是一种利用梯度下降过程来最小化损失函数，从而改善或提升弱学习器性能的技术。树模型，通常是决策树，一次添加一个，并以分阶段的方式构建模型，同时保持模型中现有的树不变。由于梯度提升是一种贪婪算法，它可以很快地过拟合训练数据集。然而，它可以受益于对各个部分进行惩罚的正则化方法，并减少过拟合以改善其性能。

`sklearn.ensemble`模块提供了一个梯度提升回归器，称为`GradientBoostingRegressor`。

# 随机森林回归

随机森林由多个决策树组成，每个决策树都基于训练数据的随机子样本，并使用平均化来提高预测准确性和控制过拟合。随机选择无意中引入了某种形式的偏差。然而，由于平均化，它的方差也减小，有助于补偿偏差的增加，并被认为产生了一个整体更好的模型。

`sklearn.ensemble`模块提供了一个随机森林回归器，称为`RandomForestRegressor`。

# 更多集成模型

`sklearn.ensemble`模块包含各种其他集成回归器，以及分类器模型。更多信息可以在[`scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)找到。

# 使用基于分类的机器学习预测趋势

基于分类的机器学习是一种监督机器学习方法，模型从给定的输入数据中学习，并根据新的观察结果进行分类。分类可以是双类别的，比如识别一个期权是否应该行使，也可以是多类别的，比如价格变化的方向，可以是上升、下降或不变。

在这一部分，我们将再次看一下通过四种多元资产的价格来预测 2018 年 JPM 每日趋势的动量模型。我们将使用 S&P 500 股票指数、10 年期国债指数、美元指数和黄金价格的前 1 个月和 3 个月滞后收益来拟合预测模型。我们的目标变量包括布尔指示器，其中`True`值表示与前一个交易日收盘价相比的增加或不变，而`False`值表示减少。

让我们开始准备我们模型的数据集。

# 准备目标变量

我们已经在之前的部分将 JPM 数据集下载到了`pandas` DataFrame `df_jpm`中，而`y`变量包含了 JPM 的每日百分比变化。使用以下代码将这些值转换为标签：

```py
In [ ]:
    import numpy as np
    y_direction = y >= 0
    y_direction.head(3)
Out[ ]:
    date
    1998-01-05     True
    1998-01-06    False
    1998-01-07     True
    Name: 5\. adjusted close, dtype: bool
```

使用`head()`命令，我们可以看到`y_direction`变量成为了一个`pandas` Series 对象，其中包含布尔值。百分比变化为零或更多的值被分类为`True`标签，否则为`False`。让我们使用`unique()`命令提取唯一值作为以后使用的列名：

```py
In [ ]:
    flags = list(y_direction.unique())
    flags.sort()
    print(flags)
Out[ ]:    
    [False, True]
```

列名被提取到一个名为`flags`的变量中。有了我们的目标变量，让我们继续获取我们的独立多资产变量。

# 准备多个资产作为输入变量的数据集

我们将重用前一部分中包含四种资产的滞后 1 个月和 3 个月百分比收益的`pandas` DataFrame 变量`df_assets_1m`和`df_assets_3m`，并将它们合并为一个名为`df_input`的单个变量，使用以下代码：

```py
In [ ]:    
    df_input = df_assets_1m.join(df_assets_3m).dropna()
```

使用`info()`命令查看其属性：

```py
In [ ]:
    df_input.info()
Out[ ]:
    <class 'pandas.core.frame.DataFrame'>
    Index: 2971 entries, 2007-05-25 to 2019-03-14
    Data columns (total 8 columns):
    ...
```

输出被截断了，但是你可以看到我们有八个特征作为我们的独立变量，跨越了 2007 年到 2019 年。有了我们的输入和目标变量，让我们探索 scikit-learn 中可用的各种分类器模型。

# 逻辑回归

尽管其名称是逻辑回归，但实际上它是用于分类的线性模型。它使用逻辑函数，也称为**S 形**函数，来模拟描述单次试验可能结果的概率。逻辑函数有助于将任何实值映射到 0 和 1 之间的值。标准的逻辑函数如下所示：

![](img/c47c9d05-fbda-41d4-85d1-563d19452828.png)

*e*是自然对数的底，*x*是 S 形函数中点的*X*值。![](img/3da6b246-a810-490a-ae25-c0bb3259dafd.png)是预测的实际值，介于 0 和 1 之间，可以通过四舍五入或截断值转换为 0 或 1 的二进制等价值。

`sklean.linear_model`模块的`LogisticRegression`类实现了逻辑回归。让我们通过编写一个名为`LogisticRegressionModel`的新类来实现这个分类器模型，该类扩展了`LinearRegressionModel`，使用以下代码：

```py
In [ ]:
    from sklearn.linear_model import LogisticRegression

    class LogisticRegressionModel(LinearRegressionModel):
        def get_model(self):
            return LogisticRegression(solver='lbfgs')
```

我们的新分类器模型使用了相同的基本线性回归逻辑。`get_model()`方法被重写以返回一个使用 LBFGS 求解器算法的`LogisticRegression`分类器模型的实例。

有关用于机器学习的**有限内存** **Broyden**–**Fletcher**–**Goldfarb**–**Shanno** (**LBFGS**)算法的论文可以在[`arxiv.org/pdf/1802.05374.pdf`](https://arxiv.org/pdf/1802.05374.pdf)上阅读。

创建这个模型的实例并提供我们的数据：

```py
In [ ]:
    logistic_reg_model = LogisticRegressionModel()
    logistic_reg_model.learn(df_input, y_direction, start_date='2018', 
                             end_date='2019', lookback_period=100)
```

再次，参数值表明我们有兴趣对 2018 年进行预测，并且在拟合模型时将使用`lookback_period`值为`100`作为每日历史数据点的数量。让我们使用`head()`命令检查存储在`df_result`中的结果：

```py
In [ ]:
    logistic_reg_model.df_result.head()
```

这产生了以下表格：

| **日期** | **实际** | **预测** |
| --- | --- | --- |
| **2018-01-02** | True | True |
| **2018-01-03** | True | True |
| **2018-01-04** | True | True |
| **2018-01-05** | False | True |
| **2018-01-08** | True | True |

由于我们的目标变量是布尔值，模型输出也预测布尔值。但我们的模型表现如何？在接下来的部分中，我们将探讨用于测量我们预测的风险指标。这些指标与前面部分用于基于回归的预测的指标不同。基于分类的机器学习采用另一种方法来测量输出标签。

# 用于测量基于分类的预测的风险指标

在本节中，我们将探讨用于测量基于分类的机器学习预测的常见风险指标，即混淆矩阵、准确度分数、精度分数、召回率分数和 F1 分数。

# 混淆矩阵

混淆矩阵，或错误矩阵，是一个帮助可视化和描述分类模型性能的方阵，其中真实值是已知的。`sklearn.metrics`模块的`confusion_matrix`函数帮助我们计算这个矩阵，如下面的代码所示：

```py
In [ ]:
    from sklearn.metrics import confusion_matrix

    df_result = logistic_reg_model.df_result 
    actual = list(df_result['Actual'])
    predicted = list(df_result['Predicted'])

    matrix = confusion_matrix(actual, predicted)
In [ ]:
    print(matrix)
Out[ ]:
    [[60 66]
     [55 70]]
```

我们将实际值和预测值作为单独的列表获取。由于我们有两种类标签，我们获得一个二乘二的矩阵。`seaborn`库的`heatmap`模块帮助我们理解这个矩阵。

Seaborn 是一个基于 Matplotlib 的数据可视化库。它提供了一个高级接口，用于绘制引人注目和信息丰富的统计图形，是数据科学家的流行工具。如果您没有安装 Seaborn，只需运行命令：`pip install seaborn`

运行以下 Python 代码生成混淆矩阵：

```py
In [ ]:
    %matplotlib inline
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.subplots(figsize=(12,8))
    sns.heatmap(matrix.T, square=True, annot=True, fmt='d', cbar=False, 
                xticklabels=flags, yticklabels=flags)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('JPM percentage returns 2018');
```

这将产生以下输出：

![](img/e8dcd93d-44a8-4d51-aaaf-28b98491978a.png)

不要让混淆矩阵让你困惑。让我们以一种逻辑的方式分解数字，看看混淆矩阵是如何工作的。从左列开始，我们有 126 个样本被分类为 False，分类器正确预测了 60 次，这些被称为真负例（TNs）。然而，分类器错误预测了 66 次，这些被称为假负例（FNs）。在右列，我们有 125 个属于 True 类的样本。分类器错误预测了 55 次，这些被称为假正例（FPs）。分类器虽然有 70 次预测正确，这些被称为真正例（TPs）。这些计算出的比率在其他风险指标中使用，我们将在接下来的部分中发现。

# 准确度分数

准确度分数是正确预测的观测值与总观测值的比率。默认情况下，它表示为 0 到 1 之间的分数。当准确度分数为 1.0 时，意味着样本中的整个预测标签集与真实标签集匹配。准确度分数可以写成如下形式：

![](img/3a76457f-d1b9-4343-963b-e13f2919c63a.png)

这里，*I(x)*是指示函数，对于正确的预测返回 1，否则返回 0。`sklearn.metrics`模块的`accuracy_score`函数使用以下代码为我们计算这个分数：

```py
In [ ]:
    from sklearn.metrics import accuracy_score
    print('accuracy_score:', accuracy_score(actual, predicted))
Out[ ]:
    accuracy_score: 0.5179282868525896
```

准确度分数表明我们的模型有 52%的正确率。准确度分数非常适合测量对称数据集的性能，其中假正例和假负例的值几乎相同。为了充分评估我们模型的性能，我们需要查看其他风险指标。

# 精度分数

精度分数是正确预测的正例观测值与总预测的正例观测值的比率，可以写成如下形式：

![](img/c729d639-a8bb-4c08-a8c6-6877e0fc1b97.png)

这给出了一个介于 0 和 1 之间的精度分数，1 表示模型一直正确分类。`sklearn.metrics`模块的`precision_score`函数使用以下代码为我们计算这个分数：

```py
In [ ]:
    from sklearn.metrics import precision_score
    print('precision_score:', precision_score(actual, predicted))
Out[ ]:
    precision_score: 0.5147058823529411
```

精度分数表明我们的模型能够正确预测分类的 52%的时间。

# 召回分数

召回分数是正确预测的正样本占实际类别中所有样本的比率，可以写成如下形式：

![](img/5a9ad0ce-d532-48ba-9666-4151d3f5e5cd.png)

这给出了一个介于 0 和 1 之间的召回分数，1 是最佳值。`sklearn.metrics`模块的`recall_score`函数使用以下代码为我们计算这个分数：

```py
In [ ]:
    from sklearn.metrics import recall_score
    print('recall_score:', recall_score(actual, predicted))
Out[ ]:
    recall_score: 0.56
```

召回分数表明我们的逻辑回归模型正确识别正样本的时间为 56%。

# F1 分数

F1 分数，或 F-度量，是精度分数和召回分数的加权平均值，可以写成如下形式：

![](img/e1661497-9810-45da-b138-0665a041a5c4.png)

这给出了介于 0 和 1 之间的 F1 分数。当精度分数或召回分数为 0 时，F1 分数将为 0。但是，当精度分数和召回分数都为正时，F1 分数对两个度量给予相等的权重。最大化 F1 分数可以创建一个具有最佳召回和精度平衡的平衡分类模型。

`sklearn.metrics`模块的`f1_score`函数使用以下代码为我们计算这个分数：

```py
In [ ]:
    from sklearn.metrics import f1_score
    print('f1_score:', f1_score(actual, predicted))
Out[ ]:
    f1_score: 0.5363984674329502
```

我们的逻辑回归模型的 F1 分数为 0.536。

# 支持向量分类器

**支持向量分类器**（**SVC**）是使用支持向量对数据集进行分类的**支持向量机**（**SVM**）的概念。

有关 SVM 的更多信息可以在[`www.statsoft.com/textbook/support-vector-machines`](http://www.statsoft.com/textbook/support-vector-machines)找到。

`SVC`类的`sklean.svm`模块实现了 SVM 分类器。编写一个名为`SVCModel`的类，并使用以下代码扩展`LogisticRegressionModel`：

```py
In [ ]:
    from sklearn.svm import SVC

    class SVCModel(LogisticRegressionModel):
        def get_model(self):
            return SVC(C=1000, gamma='auto')
In [ ]:
    svc_model = SVCModel()
    svc_model.learn(df_input, y_direction, start_date='2018', 
                    end_date='2019', lookback_period=100)
```

在这里，我们重写`get_model()`方法以返回 scikit-learn 的`SVC`类。指定了高惩罚`C`值为`1000`。`gamma`参数是具有默认值`auto`的核系数。使用我们通常的模型参数执行`learn()`命令。有了这些，让我们在这个模型上运行风险指标：

```py
In [ ]:
    df_result = svc_model.df_result
    actual = list(df_result['Actual'])
    predicted = list(df_result['Predicted'])
In [ ]:
    print('accuracy_score:', accuracy_score(actual, predicted))
    print('precision_score:', precision_score(actual, predicted))
    print('recall_score:', recall_score(actual, predicted))
    print('f1_score:', f1_score(actual, predicted)) 
Out[ ]:
    accuracy_score: 0.5577689243027888
    precision_score: 0.5538461538461539
    recall_score: 0.576
    f1_score: 0.5647058823529412
```

我们获得的分数比逻辑回归分类器模型更好。默认情况下，线性 SVM 的`C`值为 1.0，这在实践中通常会给我们与逻辑回归模型相当的性能。选择`C`值没有固定的规则，因为它完全取决于训练数据集。可以通过向`SVC()`模型提供`kernel`参数来考虑非线性 SVM 核。有关 SVM 核的更多信息，请访问[`scikit-learn.org/stable/modules/svm.html#svm-kernels`](https://scikit-learn.org/stable/modules/svm.html#svm-kernels)。

# 其他类型的分类器

除了逻辑回归和 SVC，scikit-learn 还包含许多其他类型的分类器用于机器学习。以下部分讨论了一些我们也可以考虑在我们的基于分类的模型中实现的分类器。

# 随机梯度下降

**随机梯度下降**（**SGD**）是一种使用迭代过程来估计梯度以最小化目标损失函数的**梯度下降**形式，例如线性支持向量机或逻辑回归。随机项是因为样本是随机选择的。当使用较少的迭代时，会采取更大的步骤来达到解决方案，模型被认为具有**高学习率**。同样，使用更多的迭代，会采取更小的步骤，导致具有**小学习率**的模型。SGD 是从业者中机器学习算法的流行选择，因为它已经在大规模文本分类和自然语言处理模型中得到有效使用。

`sklearn.linear_model`模块的`SGDClassifier`类实现了 SGD 分类器。

# 线性判别分析

**线性判别分析**（**LDA**）是一种经典的分类器，它使用线性决策面，其中估计了数据每个类的均值和方差。它假设数据是高斯分布的，每个属性具有相同的方差，并且每个变量的值都在均值附近。LDA 通过使用贝叶斯定理为每个观测计算*判别分数*，以确定它属于哪个类。

`sklearn.discriminant_analysis`模块的`LinearDiscriminantAnalysis`类实现了 LDA 分类器。

# 二次判别分析

**二次判别分析**（**QDA**）与 LDA 非常相似，但使用二次决策边界，每个类使用自己的方差估计。运行风险指标显示，QDA 模型不一定比 LDA 模型表现更好。必须考虑所需模型的决策边界类型。QDA 更适用于具有较低偏差和较高方差的大型数据集。另一方面，LDA 适用于具有较低偏差和较高方差的较小数据集。

`sklearn.discriminant_analysis`模块的`QuadraticDiscriminantAnalysis`类实现了 QDA 模型。

# KNN 分类器

**k-最近邻**（**k-NN**）分类器是一种简单的算法，它对每个点的最近邻进行简单的多数投票，并将该点分配给在该点的最近邻中具有最多代表的类。虽然不需要为泛化训练模型，但预测阶段在时间和内存方面较慢且成本较高。

`sklearn.neighbors`模块的`KNeighborsClassifier`类实现了 KNN 分类器。

# 对机器学习算法的使用结论

您可能已经注意到，我们模型的预测值与实际值相差甚远。本章旨在展示 scikit-learn 提供的机器学习功能的最佳特性，这可能用于预测时间序列数据。迄今为止，没有研究表明机器学习算法可以预测价格接近 100%的时间。构建和有效运行机器学习系统需要付出更多的努力。

# 总结

在本章中，我们介绍了金融领域的机器学习。我们讨论了人工智能和机器学习如何改变金融行业。机器学习可以是监督的或无监督的，监督算法可以是基于回归或基于分类的。Python 的 scikit-learn 库提供了各种机器学习算法和风险指标。

我们讨论了基于回归的机器学习模型的使用，如 OLS 回归、岭回归、LASSO 回归和弹性网络正则化，用于预测证券价格等连续值。还讨论了决策树的集成，如装袋回归器、梯度树提升和随机森林。为了衡量回归模型的性能，我们讨论了均方误差、平均绝对误差、解释方差分数和 R²分数。

基于分类的机器学习将输入值分类为类别或标签。这些类别可以是二元或多元的。我们讨论了逻辑回归、支持向量机、LDA 和 QDA 以及 k-NN 分类器用于预测价格趋势。为了衡量分类模型的性能，我们讨论了混淆矩阵、准确率、精确度和召回率，以及 F1 分数。

在下一章中，我们将探讨在金融领域中使用深度学习。
