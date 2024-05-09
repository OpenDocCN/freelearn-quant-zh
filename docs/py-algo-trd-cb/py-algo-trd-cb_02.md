股票市场-交易入门

在构建算法交易系统时，有必要与提供用于程序化下单和查询交易的 API 的现代经纪人开设账户。这样我们就可以通过我们的 Python 脚本控制经纪账户，而传统上是通过经纪人的网站手动操作经纪账户，这个 Python 脚本将成为我们更大型算法交易系统的一部分。本章演示了一些基本的配方，介绍了开发完整的算法交易系统所需的基本经纪 API 调用。

本章涵盖以下配方：

+   设置 Python 与经纪人的连接

+   查询工具列表

+   获取工具

+   查询交易所列表

+   查询段列表

+   了解经纪人支持的其他属性

+   下单普通订单

+   下单布林订单

+   下单交割订单

+   下单当日订单

+   查询保证金和资金

+   计算经纪费

+   计算政府征收的税费

让我们开始吧！

# 技术要求

要成功执行本章中的示例，您将需要以下内容：

+   Python 3.7+

+   Python 的`pyalgotrading`包（`$ pip install pyalgotrading`）

本章的最新 Jupyter 笔记本可以在 GitHub 上找到：[`github.com/PacktPublishing/Python-Algorithmic-Trading-Cookbook/tree/master/Chapter02`](https://github.com/PacktPublishing/Python-Algorithmic-Trading-Cookbook/tree/master/Chapter02)。

本章演示了现代经纪人`ZERODHA`的 API，它受到`pyalgotrading`的支持。您也可以选择其他由`pyalgotrading`支持的经纪人。本章中的配方对于任何其他经纪人应该大致相同。`pyalgotrading`包将经纪 API 抽象为统一的接口，因此您不需要担心底层经纪 API 调用。

要设置与`ZERODHA`的经纪账户，请参阅*附录 I*中提供的详细步骤。

# 设置 Python 与经纪人的连接

设置与经纪人的连接的第一步是获取 API 密钥。经纪人通常为每个客户提供唯一的密钥，通常是作为`api-key`和`api-secret`密钥对。这些 API 密钥通常是收费的，通常按月订阅。在开始本配方之前，您需要从经纪人的网站获取`api-key`和`api-secret`的副本。有关更多详细信息，请参阅*附录 I*。

## 怎么做…

我们执行以下步骤来完成此配方：

1.  导入必要的模块：

```py
>>> from pyalgotrading.broker.broker_connection_zerodha import BrokerConnectionZerodha
```

1.  从经纪人那里获取`api_key`和`api_secret`密钥。这些对您是唯一的，并将由经纪人用于识别您的 Demat 账户：

```py
>>> api_key = "<your-api-key>"
>>> api_secret = "<your-api-secret>"
>>> broker_connection = BrokerConnectionZerodha(api_key, api_secret)
```

您将获得以下结果：

```py
Installing package kiteconnect via pip...
Please login to this link to generate your request token: https://kite.trade/connect/login?api_key=<your-api-key>&v=3
```

1.  从上述网址获取请求令牌：

```py
>>> request_token = "<your-request-token>"
>>> broker_connection.set_access_token(request_token)
```

## 工作原理… 

在 *第一步* 中，你从 `pyalgotrading` 导入 `BrokerConnectionZerodha` 类。`BrokerConnectionZerodha` 类提供了围绕特定经纪人 API 的抽象。对于 *第二步*，你需要经纪人的 API 密钥和 API 秘密。如果你没有它们，请参考 *附录 I* 获取详细的说明和带有截图的指导。在 *第二步* 中，你将你的 API 密钥和 API 秘密分配给新的 `api_key` 和 `api_secret` 变量，并使用它们创建 `broker_connection`，`BrokerConnectionZerodha` 类的一个实例。如果你是第一次运行这个程序并且没有安装 `kiteconnect`，`pyalgotrading` 将自动为你安装它。（`kiteconnect` 是官方的 Python 包，用于与 Zerodha 后端通信；`BrokerConnectionZerodha` 是在 `kiteconnect` 之上的一个封装。）*第二步* 生成一个登录链接。在这里，你需要点击链接并使用你的 Zerodha 凭据登录。如果认证过程成功，你将在浏览器地址栏中看到一个类似以下的链接：

```py
https://127.0.0.1/?request_token=&action=login&status=success
```

例如，完整的链接将如下所示：

```py
https://127.0.0.1/?request_token=H06I6Ydv95y23D2Dp7NbigFjKweGwRP7&action=login&status=success
```

复制字母数字令牌，`H06I6Ydv95y23D2Dp7NbigFjKweGwRP7`，并将其粘贴到 *第三步* 的 `request_token` 中。`broker_connection` 实例现在已准备好执行 API 调用。

# 查询一组工具

一旦 `broker_connection` 句柄准备好，它就可以用来查询包含经纪人提供的所有金融工具的列表。

## 准备工作

确保 `broker_connection` 对象在你的 Python 命名空间中可用。请参考本章中前一个配方设置此对象。

## 如何做…

我们执行以下步骤来完成这个配方：

1.  显示所有工具：

```py
>>> instruments = broker_connection.get_all_instruments()
>>> instruments
```

你将会得到类似以下的输出。对你来说，确切的输出可能会有所不同：

```py
  instrument_token exchange_token tradingsymbol name last_price expiry strike tick_size lot_size instrument_type segment exchange
0 267556358 1045142 EURINR20AUGFUT EURINR 0.0 2020-08-27 0.0 0.0025 1 FUT BCD-FUT BCD
1 268660998 1049457 EURINR20DECFUT EURINR 0.0 2020-12-29 0.0 0.0025 1 FUT BCD-FUT BCD
2 266440966 1040785 EURINR20JULFUT EURINR 0.0 2020-07-29 0.0 0.0025 1 FUT BCD-FUT BCD
3 266073606 1039350 EURINR20JUNFUT EURINR 0.0 2020-06-26 0.0 0.0025 1 FUT BCD-FUT BCD
4 265780742 1038206 EURINR20MAYFUT EURINR 0.0 2020-05-27 0.0 0.0025 1 FUT BCD-FUT BCD
... ... ... ... ... ... ... ... ... ... ... ... ...
64738 978945 3824 ZODJRDMKJ ZODIAC JRD-MKJ 0.0 0.0 0.0500 1 EQ NSE NSE
64739 2916865 11394 ZOTA ZOTA HEALTH CARE 0.0 0.0 0.0500 1 EQ NSE NSE
64740 7437825 29054 ZUARI-BE ZUARI AGRO CHEMICALS 0.0 0.0 0.0500 1 EQ NSE NSE
64741 979713 3827 ZUARIGLOB ZUARI GLOBAL 0.0 0.0 0.0500 1 EQ NSE NSE
64742 4514561 17635 ZYDUSWELL ZYDUS WELLNESS 0.0 0.0 0.0500 1 EQ NSE NSE

64743 rows × 12 columns
```

1.  打印工具总数：

```py
>>> print(f'Total instruments: {len(instruments)}')
```

我们得到以下输出（你的输出可能会有所不同）：

```py
Total instruments: 64743
```

## 工作原理…

第一步使用 `broker_connection` 的 `get_all_instruments()` 方法获取所有可用的金融工具。此方法返回一个 `pandas.DataFrame` 对象。此对象分配给一个新变量 `instruments`，显示在 *第一步* 的输出中。由于经常添加新的金融工具并定期过期现有的工具，这个输出可能对你来说有所不同。最后一步显示了经纪人提供的工具总数。

关于前面的 API 调用返回的数据的解释将在第三章中深入讨论，*分析金融数据*。对于这个配方，知道如何获取工具列表的方法就足够了。

# 获取一个工具

**工具**，也称为**金融工具**或**证券**，是可以在交易所交易的资产。在交易所中，可以有数万种工具。本示例演示了如何根据其**交易所**和**交易符号**获取工具。

## 准备就绪

确保 `broker_connection` 对象在你的 Python 命名空间中可用。请参考本章第一个示例设置此对象。

## 怎么做…

获取特定交易符号和交易所的工具：

```py
>>> broker_connection.get_instrument(segment='NSE', tradingsymbol='TATASTEEL')
```

你将得到以下输出：

```py
segment: NSE
exchange: NSE
tradingsymbol: TATASTEEL
broker_token: 895745
tick_size: 0.05
lot_size: 1
expiry: 
strike_price: 0.0
```

## 工作原理…

`broker_connection` 对象提供了一个方便的方法，`get_instrument`，用于获取任何金融工具。在返回工具之前，它以 `segment` 和 `tradingsymbol` 为属性。返回对象是 `Instrument` 类的一个实例。

# 查询交易所列表

**交易所** 是一个交易工具交易的市场。交易所确保交易过程公平且始终按规则进行。通常，经纪人支持多个交易所。本示例演示了如何查找经纪人支持的交易所列表。

## 准备就绪

确保 `instruments` 对象在你的 Python 命名空间中可用。请参考本章第二个示例以了解如何设置此对象。

## 怎么做…

显示经纪人支持的交易所：

```py
>>> exchanges = instruments.exchange.unique()
>>> print(exchanges)
```

你将得到以下输出：

```py
['BCD' 'BSE' 'NSE' 'CDS' 'MCX' 'NFO']
```

## 工作原理…

`instruments.exchange` 返回一个 `pandas.Series` 对象。其 `unique()` 方法返回一个由经纪人支持的唯一交易所组成的 `numpy.ndarray` 对象。

# 查询分段列表

一个分段实质上是根据其类型对工具进行分类。在交易所中常见的各种分段类型包括现金/股票、期货、期权、大宗商品和货币。每个分段可能有不同的运营时间。通常，经纪人支持多个交易所内的多个分段。本示例演示了如何查找经纪人支持的分段列表。

## 准备就绪

确保 `instruments` 对象在你的 Python 命名空间中可用。请参考本章第二个示例以了解如何设置此对象。

## 怎么做…

显示经纪人支持的分段：

```py
>>> segments = instruments.segment.unique()
>>> print(segments)
```

你将得到以下输出：

```py
['BCD-FUT' 'BCD' 'BCD-OPT' 'BSE' 'INDICES' 'CDS-FUT' 'CDS-OPT' 'MCX-FUT' 'MCX-OPT' 'NFO-OPT' 'NFO-FUT' 'NSE']
```

## 工作原理…

`instruments.segment` 返回一个 `pandas.Series` 对象。它的 `unique` 方法返回一个由经纪人支持的唯一分段组成的 `numpy.ndarray` 对象。

# 了解经纪人支持的其他属性

为了下订单，需要以下属性：订单交易类型、订单种类、订单类型和订单代码。不同的经纪人可能支持不同类型的订单属性。例如，一些经纪人可能仅支持普通订单，而其他经纪人可能支持普通订单和止损订单。可以使用 `pyalgotrading` 包提供的经纪人特定常量查询经纪人支持的每个属性的值。

## 如何做…

我们执行以下步骤来完成此配方：

1.  从`pyalgotrading`模块中导入必要的类：

```py
>>> from pyalgotrading.broker.broker_connection_zerodha import BrokerConnectionZerodha
```

1.  列出订单交易类型：

```py
>>> list(BrokerConnectionZerodha.ORDER_TRANSACTION_TYPE_MAP.keys())
```

我们将得到以下输出：

```py
[<BrokerOrderTransactionTypeConstants.BUY: 'BUY'>,
 <BrokerOrderTransactionTypeConstants.SELL: 'SELL'>]
```

1.  列出订单品种：

```py
>>> list(BrokerConnectionZerodha.ORDER_VARIETY_MAP.keys())
```

我们将得到以下输出：

```py
[<BrokerOrderVarietyConstants.MARKET: 'ORDER_VARIETY_MARKET'>,
 <BrokerOrderVarietyConstants.LIMIT: 'ORDER_VARIETY_LIMIT'>,
 <BrokerOrderVarietyConstants.STOPLOSS_LIMIT: 'ORDER_VARIETY_STOPLOSS_LIMIT'>,
 <BrokerOrderVarietyConstants.STOPLOSS_MARKET: 'ORDER_VARIETY_STOPLOSS_MARKET'>]
```

1.  列出订单类型：

```py
>>> list(BrokerConnectionZerodha.ORDER_TYPE_MAP.keys())
```

我们将得到以下输出：

```py
[<BrokerOrderTypeConstants.REGULAR: 'ORDER_TYPE_REGULAR'>,
 <BrokerOrderTypeConstants.BRACKET: 'ORDER_TYPE_BRACKET'>,
 <BrokerOrderTypeConstants.COVER: 'ORDER_TYPE_COVER'>,
 <BrokerOrderTypeConstants.AMO: 'ORDER_TYPE_AFTER_MARKET_ORDER'>]
```

1.  列出订单代码：

```py
>>> list(BrokerConnectionZerodha.ORDER_CODE_MAP.keys())
```

我们将得到以下输出：

```py
[<BrokerOrderCodeConstants.INTRADAY: 'ORDER_CODE_INTRADAY'>,
 <BrokerOrderCodeConstants.DELIVERY: 'ORDER_CODE_DELIVERY_T0'>]
```

## 它是如何工作的…

在*步骤 1* 中，我们从`pyalgotrading`导入`BrokerConnectionZerodha`类。此类保存了`pyalgotrading`和特定经纪人常量之间的订单属性映射，作为字典对象。接下来的步骤获取并打印这些映射。步骤 2 显示您的经纪人支持`BUY`和`SELL`订单交易类型。

*步骤 3* 显示您的经纪人支持`MARKET`、`LIMIT`、`STOPLOSS_LIMIT`和`STOPLOSS_MARKET`订单品种。*步骤 4* 显示您的经纪人支持`REGULAR`、`BRACKET`、`COVER`和`AFTER_MARKET`订单类型。*步骤 5* 显示您的经纪人支持`INTRADAY`和`DELIVERY`订单代码。

输出可能因经纪人而异，因此如果您使用不同的经纪人，请查阅您的经纪人文档。所有这些类型参数的详细解释将在第六章 *在交易所下订单* 中涵盖。本配方仅概述这些参数，因为它们在本章后续配方中需要。

# 放置一个简单的常规订单

本配方演示了如何通过经纪人在交易所上放置`REGULAR`订单。`REGULAR`订单是最简单的订单类型。尝试完此配方后，通过登录经纪人网站检查您的经纪人账户；您会发现一个订单已经被放置在那里。您可以将订单 ID 与本配方中显示的最后一个代码片段中返回的订单 ID 匹配。

## 准备就绪

确保`broker_connection`对象在你的 Python 命名空间中可用。参考本章第一个配方，了解如何设置此对象。

## 如何做…

我们执行以下步骤来完成此配方：

1.  从`pyalgotrading`中导入必要的常量：

```py
>>> from pyalgotrading.constants import *
```

1.  获取特定交易符号和交易所的金融工具：

```py
>>> instrument = broker_connection.get_instrument(segment='NSE', 
                                        tradingsymbol='TATASTEEL')
```

1.  放置一个简单的常规订单 - 一个`BUY`、`REGULAR`、`INTRADAY`、`MARKET`订单：

```py
>>> order_id = broker_connection.place_order(
                   instrument=instrument, 
                   order_transaction_type= \
                       BrokerOrderTransactionTypeConstants.BUY,
                   order_type=BrokerOrderTypeConstants.REGULAR, 
                   order_code=BrokerOrderCodeConstants.INTRADAY,
                   order_variety= \
                       BrokerOrderVarietyConstants.MARKET, 
                   quantity=1)
>>> order_id
```

我们将得到以下输出：

```py
191209000001676
```

## 它是如何工作的…

在*步骤 1* 中，您从`pyalgotrading`导入常量。在*步骤 2* 中，您使用`broker_connection`的`get_instrument()`方法以`segment = 'NSE'`和`tradingsymbol = 'TATASTEEL'`获取金融工具。在*步骤 3* 中，您使用`broker_connection`的`place_order()`方法放置一个`REGULAR`订单。`place_order()`方法接受的参数描述如下：

+   `instrument`：必须放置订单的金融工具。应该是`Instrument`类的实例。您在这里传递`instrument`。

+   `order_transaction_type`: 订单交易类型。应为`BrokerOrderTransactionTypeConstants`类型的枚举。在这里，你传递了`BrokerOrderTransactionTypeConstants.BUY`。

+   `order_type`: 订单类型。应为`BrokerOrderTypeConstants`类型的枚举。在这里，你传递了`BrokerOrderTypeConstants.REGULAR`。

+   `order_code`: 订单代码。应为`BrokerOrderCodeConstants`类型的枚举。在这里，你传递了`BrokerOrderCodeConstants.INTRADAY`。

+   `order_variety`: 订单种类。应为`BrokerOrderVarietyConstants`类型的枚举。在这里，你传递了`BrokerOrderVarietyConstants.MARKET`。

+   `quantity`: 要交易的股票数量。应为正整数。我们在这里传递了`1`。

如果订单放置成功，该方法将返回一个订单 ID，您可以随时以后用于查询订单状态。

不同类型参数的详细解释将在第六章中介绍，*在交易所上下订单*。这个配方旨在让你了解如何下达`REGULAR`订单，这是各种可能订单类型之一的想法。

# 下达一个简单的 BRACKET 订单

这个配方演示了如何通过经纪人在交易所上下达一个`BRACKET`订单。`BRACKET`订单是两腿订单。一旦第一个订单执行完毕，经纪人会自动下达两个新订单 – 一个`STOPLOSS`订单和一个`TARGET`订单。在任何时候只有一个订单被执行；当第一个订单完成时，另一个订单将被取消。在尝试了此配方后，通过登录经纪人的网站，您可以在您的经纪账户中找到已下达的订单。您可以将订单 ID 与本配方中显示的最后一个代码片段中返回的订单 ID 进行匹配。

## 准备就绪

确保`broker_connection`对象在你的 Python 命名空间中可用。参考本章的第一个配方，学习如何设置此对象。

## 如何操作...

我们执行以下步骤完成此配方：

1.  导入必要的模块：

```py
>>> from pyalgotrading.constants import *
```

1.  获取特定交易符号和交易所的工具：

```py
>>> instrument = broker_connection.get_instrument(segment='NSE', 
                                        tradingsymbol='ICICIBANK')
```

1.  获取工具的最新交易价格：

```py
>>> ltp = broker_connection.get_ltp(instrument)
```

1.  下达一个简单的`BRACKET`订单 – 一个`BUY`，`BRACKET`，`INTRADAY`，`LIMIT`订单：

```py
>>> order_id = broker_connection.place_order(
                   instrument=instrument,
                   order_transaction_type= \
                       BrokerOrderTransactionTypeConstants.BUY,
                   order_type=BrokerOrderTypeConstants.BRACKET, 
                   order_code=BrokerOrderCodeConstants.INTRADAY, 
                   order_variety=BrokerOrderVarietyConstants.LIMIT,
                   quantity=1, price=ltp-1, 
                   stoploss=2, target=2)
>>> order_id
```

我们将得到以下输出：

```py
191212001268839
```

如果在执行此代码时收到以下错误，则意味着由于市场波动性较高，经纪人阻止了 Bracket 订单：

`InputException: 由于市场预期波动性较高，Bracket 订单暂时被阻止。`

当经纪人开始允许 Bracket 订单时，你应该稍后尝试该配方。你可以不时地查看经纪人网站以了解 Bracket 订单何时被允许。

## 工作原理...

在*步骤 1*中，您从`pyalgotrading`导入常量。在*步骤 2*中，您使用`broker_connection`的`get_instrument()`方法获取`segment = 'NSE'`和`tradingsymbol = 'ICICBANK'`的金融工具。在*步骤 3*中，您获取工具的**最后交易价格**或**LTP**。（LTP 将在第三章的*分析金融数据*中更详细地解释。）在*步骤 4*中，您使用`broker_connection`的`place_order()`方法放置一个`BRACKET`订单。`place_order()`方法接受的参数的描述如下：

+   `instrument`: 必须放置订单的金融工具。应该是`Instrument`类的实例。你在这里传递`instrument`。

+   `order_transaction_type`: 订单交易类型。应该是`BrokerOrderTransactionTypeConstants`类型的枚举。你在这里传递`BrokerOrderTransactionTypeConstants.BUY`。

+   `order_type`: 订单类型。应该是`BrokerOrderTypeConstants`类型的枚举。你在这里传递`BrokerOrderTypeConstants.BRACKET`。

+   `order_code`: 订单代码。应该是`BrokerOrderCodeConstants`类型的枚举。你在这里传递`BrokerOrderCodeConstants.INTRADAY`。

+   `order_variety`: 订单种类。应该是`BrokerOrderVarietyConstants`类型的枚举。你在这里传递`BrokerOrderVarietyConstants.LIMIT`。

+   `quantity`: 给定工具要交易的股份数量。应为正整数。你在这里传递`1`。

+   `price`: 应该放置订单的限价。你在这里传递`ltp-1`，这意味着低于`ltp`值的 1 个单位价格。

+   `stoploss`: 初始订单价格的价格差，应该放置止损订单的价格。应为正整数或浮点值。你在这里传递`2`。

+   `target`: 初始价格的价格差，应该放置目标订单的价格。应为正整数或浮点值。你在这里传递`2`。

如果订单放置成功，该方法会返回一个订单 ID，您可以随时稍后用于查询订单的状态。

对不同类型参数的详细解释将在第六章中进行，*在交易所上放置交易订单*。本示例旨在向您展示如何放置`BRACKET`订单，这是各种可能订单类型之一。

# 放置一个简单的 DELIVERY 订单

此示例演示了如何通过经纪人在交易所下达 `DELIVERY` 订单。`DELIVERY` 订单将传递到用户的 Demat 账户，并存在直到用户明确平仓为止。在交易会话结束时由交货订单创建的仓位将转移到下一个交易会话。它们不会由经纪人明确平仓。尝试完这个示例后，通过登录经纪人的网站检查你的经纪账户；你会发现已经有一个订单被下达了。你可以将订单 ID 与此示例中最后显示的代码片段返回的订单 ID 进行匹配。

## 准备工作

确保在你的 Python 命名空间中可用 `broker_connection` 对象。请参考本章第一个示例来学习如何设置此对象。

## 操作方法…

我们执行以下步骤来完成此示例：

1.  导入必要的模块：

```py
>>> from pyalgotrading.constants import *
```

1.  获取特定交易符号和交易所的金融工具：

```py
>>> instrument = broker_connection.get_instrument(segment='NSE', 
                                        tradingsymbol='AXISBANK')
```

1.  下达一个简单的 `DELIVERY` 订单 - 一个 `SELL`、`REGULAR`、`DELIVERY`、`MARKET` 订单：

```py
>>> order_id = broker_connection.place_order(
                   instrument=instrument,
                   order_transaction_type= \
                       BrokerOrderTransactionTypeConstants.SELL,
                   order_type=BrokerOrderTypeConstants.REGULAR,
                   order_code=BrokerOrderCodeConstants.DELIVERY,
                   order_variety= \
                       BrokerOrderVarietyConstants.MARKET, 
                    quantity=1)
>>> order_id
```

我们将得到以下输出：

```py
191212001268956
```

## 工作原理…

在 *第 1 步* 中，你从 `pyalgotrading` 导入常量。在 *第 2 步* 中，你使用 `broker_connection` 的 `get_instrument()` 方法，通过 `segment = 'NSE'` 和 `tradingsymbol = 'AXISBANK'` 获取金融工具。在 *第 3 步* 中，你使用 `broker_connection` 的 `place_order()` 方法下达 `DELIVERY` 订单。此方法接受以下参数：

+   `instrument`：必须下订单的金融工具。应该是 `Instrument` 类的实例。你在这里传递 `instrument`。

+   `order_transaction_type`：订单交易类型。应该是 `BrokerOrderTransactionTypeConstants` 类型的枚举。你在这里传递 `BrokerOrderTransactionTypeConstants.SELL`。

+   `order_type`：订单类型。应该是 `BrokerOrderTypeConstants` 类型的枚举。你在这里传递 `BrokerOrderTypeConstants.REGULAR`。

+   `order_code`：订单代码。应该是 `BrokerOrderCodeConstants` 类型的枚举。你在这里传递 `BrokerOrderCodeConstants.DELIVERY`。

+   `order_variety`：订单类型。应该是 `BrokerOrderVarietyConstants` 类型的枚举。你在这里传递 `BrokerOrderVarietyConstants.MARKET`。

+   `quantity:` 要为给定金融工具交易的股票数量。应该是正整数。我们在这里传递 `1`。

如果订单下达成功，该方法会返回一个订单 ID，你随时可以使用它查询订单的状态。

关于不同类型参数的详细解释将在 第六章 *在交易所上下达交易订单* 中介绍。此示例旨在让你了解如何下达 `DELIVERY` 订单，这是各种可能订单中的一种。

# 下达一个简单的 INTRADAY 订单

此配方演示如何通过经纪人 API 下达 `INTRADAY` 订单。`INTRADAY` 订单不会传送到用户的 Demat 账户。由日内订单创建的头寸具有一天的生命周期。这些头寸在交易会话结束时由经纪人明确平仓，并不转入下一个交易会话。尝试完此配方后，通过登录经纪人网站查看您的经纪账户；您会发现已经有了一个订单。您可以将订单 ID 与此配方中显示的最后一个代码片段返回的订单 ID 进行匹配。

## 准备工作

确保 `broker_connection` 对象在您的 Python 命名空间中可用。请参考本章的第一个配方，了解如何设置此对象。

## 如何操作…

我们执行以下步骤来完成此配方：

1.  导入必要的模块：

```py
>>> from pyalgotrading.constants import *
```

1.  获取特定交易符号和交易所的工具：

```py
>>> instrument = broker_connection.get_instrument(segment='NSE', 
                                        tradingsymbol='HDFCBANK')
```

1.  获取工具的最近成交价：

```py
>>> ltp = broker_connection.get_ltp(instrument)
```

1.  下达一个简单的 `INTRADAY` 订单 —— 一个 `SELL`，`BRACKET`，`INTRADAY`，`LIMIT` 订单：

```py
>>> order_id = broker_connection.place_order(
                   instrument=instrument,
                   order_transaction_type= \
                       BrokerOrderTransactionTypeConstants.SELL,
                   order_type=BrokerOrderTypeConstants.BRACKET,
                   order_code=BrokerOrderCodeConstants.INTRADAY, 
                   order_variety=BrokerOrderVarietyConstants.LIMIT,
                   quantity=1, price=ltp+1, stoploss=2, target=2)
>>> order_id
```

我们将获得以下输出：

```py
191212001269042
```

如果在执行此代码时出现以下错误，则意味着经纪人由于市场波动性较高而阻止了 Bracket 订单：

`InputException: 由于市场预期波动率较高，Bracket 订单暂时被阻止。`

当经纪人开始允许 Bracket 订单时，您应该稍后尝试此配方。您可以不时地在经纪人网站上查看更新，了解何时允许 Bracket 订单。

## 工作原理…

在 *步骤 1* 中，您从 `pyalgotrading` 导入常量。在 *步骤 2* 中，您使用 `broker_connection` 的 `get_instrument()` 方法通过 `segment = 'NSE'` 和 `tradingsymbol = 'HDFCBANK'` 获取金融工具。在 *步骤 3* 中，您获取该工具的 LTP。（LTP 将在 第三章 的 *金融工具的最近成交价* 配方中详细解释。）在 *步骤 4* 中，您使用 `broker_connection` 的 `place_order()` 方法下达 `BRACKET` 订单。`place_order()` 方法接受的参数描述如下：

+   `instrument`：必须下达订单的金融工具。应为 `Instrument` 类的实例。在这里传递 `instrument`。

+   `order_transaction_type`：订单交易类型。应为 `BrokerOrderTransactionTypeConstants` 类型的枚举。在这里传递 `BrokerOrderTransactionTypeConstants.SELL`。

+   `order_type`：订单类型。应为 `BrokerOrderTypeConstants` 类型的枚举。在这里传递 `BrokerOrderTypeConstants.BRACKET`。

+   `order_code`：订单代码。应为 `BrokerOrderCodeConstants` 类型的枚举。在这里传递 `BrokerOrderCodeConstants.INTRADAY`。

+   `order_variety`：订单种类。应为 `BrokerOrderVarietyConstants` 类型的枚举。在这里传递 `BrokerOrderVarietyConstants.LIMIT`。

+   `quantity`：给定工具要交易的股票数量。应该是正整数。这里你传递了`1`。

+   `price`：应该下订单的限价。这里你传递了`ltp+1`，表示高于`ltp`值的 1 个单位价格。

+   `stoploss`：与初始订单价格的价格差，应在该价格处放置止损订单。应该是正整数或浮点数值。这里你传递了`2`。

+   `target`：与初始订单价格的价格差，应在该价格处放置目标订单。应该是正整数或浮点数值。这里你传递了`2`。

如果下单成功，该方法将返回一个订单 ID，您随时可以在以后的任何时间使用它来查询订单的状态。

不同类型参数的详细解释将在第六章，*在交易所下订单* 中介绍。本示例旨在让您了解如何下达 `INTRADAY` 订单，这是各种可能订单类型之一。

# 查询保证金和资金

在下单之前，重要的是要确保您的经纪账户中有足够的保证金和资金可用以成功下单。资金不足会导致经纪拒绝任何下单，这意味着其他人将永远不会在交易所下单。本示例向您展示了如何随时查找您的经纪账户中可用的保证金和资金。

## 准备就绪

确保 `broker_connection` 对象在您的 Python 命名空间中可用。请参考本章的第一个示例来学习如何设置它。

## 如何操作…

我们执行以下步骤完成此示例：

1.  显示股票保证金：

```py
>>> equity_margins = broker_connection.get_margins('equity')
>>> equity_margins
```

我们将得到以下输出（您的输出可能有所不同）：

```py
{'enabled': True,
 'net': 1623.67,
 'available': {'adhoc_margin': 0,
  'cash': 1623.67,
  'opening_balance': 1623.67,
  'live_balance': 1623.67,
  'collateral': 0,
  'intraday_payin': 0},
 'utilised': {'debits': 0,
  'exposure': 0,
  'm2m_realised': 0,
  'm2m_unrealised': 0,
  'option_premium': 0,
  'payout': 0,
  'span': 0,
  'holding_sales': 0,
  'turnover': 0,
  'liquid_collateral': 0,
  'stock_collateral': 0}}
```

1.  显示股票资金：

```py
>>> equity_funds = broker_connection.get_funds('equity')
>>> equity_funds
```

我们将得到以下输出（您的输出可能有所不同）：

```py
1623.67
```

1.  显示商品保证金：

```py
>>> commodity_margins = get_margins(commodity')
>>> commodity_margins
```

我们将得到以下输出（您的输出可能有所不同）：

```py
{'enabled': True,
 'net': 16215.26,
 'available': {'adhoc_margin': 0,
  'cash': 16215.26,
  'opening_balance': 16215.26,
  'live_balance': 16215.26,
  'collateral': 0,
  'intraday_payin': 0},
 'utilised': {'debits': 0,
  'exposure': 0,
  'm2m_realised': 0,
  'm2m_unrealised': 0,
  'option_premium': 0,
  'payout': 0,
  'span': 0,
  'holding_sales': 0,
  'turnover': 0,
  'liquid_collateral': 0,
  'stock_collateral': 0}}
```

1.  显示商品资金：

```py
>>> commodity_funds = broker_connection.get_funds('commodity')
>>> commodity_funds
```

我们将得到以下输出（您的输出可能有所不同）：

```py
0
```

## 工作原理…

`broker_connection`对象提供了用于获取经纪账户可用保证金和资金的方法：

+   `get_margins()`

+   `get_funds()`

经纪公司 Zerodha 分别跟踪 `equity` 和 `commodity` 产品的保证金和资金。如果您使用的是 `pyalgotrading` 支持的其他经纪公司，则可能会将资金和保证金分别跟踪 `equity` 和 `commodity`。

*步骤 1* 展示了如何使用`broker_connection`对象的`get_margins()`方法查询`equity`产品的保证金，参数为`equity`。*步骤 2* 展示了如何使用`broker_connection`对象的`get_funds()`方法查询`equity`产品的资金，参数为`equity`字符串。

*步骤 3* 和 *4* 展示了如何查询以`commodity`字符串为参数的`commodity`产品的保证金和资金情况。

# 计算收取的佣金

每次成功完成的订单，经纪人可能会收取一定的费用，这通常是买卖工具价格的一小部分。虽然金额看似不大，但重要的是要跟踪佣金，因为它最终可能会吃掉你一天结束时的可观利润的一大部分。

收取的佣金因经纪人而异，也因交易段而异。针对这个方案，我们将考虑佣金为 0.01%。

## 如何做…

我们执行以下步骤完成这个方案：

1.  计算每笔交易收取的佣金：

```py
>>> entry_price = 1245
>>> brokerage = (0.01 * 1245)/100
>>> print(f'Brokerage charged per trade: {brokerage:.4f}')
```

我们将获得以下输出：

```py
Brokerage charged per trade: 0.1245
```

1.  计算 10 笔交易的总佣金：

```py
>>> total_brokerage = 10 * (0.01 * 1245) / 100
>>> print(f'Total Brokerage charged for 10 trades: \
            {total_brokerage:.4f}')
```

我们将获得以下输出：

```py
Total Brokerage charged for 10 trades: 1.2450
```

## 工作原理…

在 *第 1 步* 中，我们从交易买入或卖出的价格`entry_price`开始。对于这个方案，我们使用了`1245`。接下来，我们计算价格的 0.01%，即`0.1245`。然后，我们计算 10 笔这样的交易的总佣金，结果为`10 * 0.1245 = 1.245`。

每个订单，佣金都会收取两次。第一次是当订单进入持仓时，而第二次是当订单退出持仓时。要获取所收取的佣金的确切细节，请参考您的经纪人提供的费用清单。

# 计算收取的政府税费

对于每个成功完成的订单，政府可能会收取一定的费用，这是买卖工具价格的一小部分。虽然金额看似不大，但重要的是要跟踪政府税费，因为它最终可能会吃掉你一天结束时的可观利润的一大部分。

政府的收费取决于交易所的位置，并且从一个交易段到另一个交易段都有所不同。针对这个方案，我们将考虑政府税费的费率为 0.1%。

## 如何做…

我们执行以下步骤完成这个方案：

1.  计算每笔交易收取的政府税费：

```py
>>> entry_price = 1245
>>> brokerage = (0.1 * 1245)/100
>>> print(f'Government taxes charged per trade: {brokerage:.4f}')
```

我们将获得以下输出：

```py
Government taxes charged per trade: 1.2450
```

1.  计算 10 笔交易收取的总政府税费：

```py
>>> total_brokerage = 10 * (0.1 * 1245) / 100
>>> print(f'Total Government taxes charged for 10 trades: \
            {total_brokerage:.4f}')
```

我们将获得以下输出：

```py
Total Government taxes charged for 10 trades: 12.4500
```

## 工作原理…

在 *第 1 步* 中，我们从交易买入或卖出的价格`entry_price`开始。对于这个方案，我们使用了`1245`。接下来，我们计算价格的 0.1%，即`1.245`。然后，我们计算 10 笔这样的交易的总佣金，结果为`10 * 1.245 = 12.245`。

对于每个订单，政府税费会收取两次。第一次是当订单进入持仓时，而第二次是当订单退出持仓时。要获取所收取的政府税费的确切细节，请参考交易所提供的政府税费清单。
