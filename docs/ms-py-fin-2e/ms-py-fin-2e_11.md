# 第八章：构建算法交易平台

算法交易自动化系统交易流程，根据定价、时机和成交量等多种因素以尽可能最佳价格执行订单。经纪公司可能会为希望部署自己交易算法的客户提供**应用程序编程接口**（**API**）作为其服务提供的一部分。算法交易系统必须非常健壮，以处理订单执行过程中的任何故障点。网络配置、硬件、内存管理、速度和用户体验是设计执行订单系统时需要考虑的一些因素。设计更大的系统不可避免地会给框架增加更多复杂性。

一旦在市场上开立头寸，就会面临各种风险，如市场风险、利率风险和流动性风险。为了尽可能保护交易资本，将风险管理措施纳入交易系统非常重要。金融行业中最常用的风险度量可能是**风险价值**（**VaR**）技术。我们将讨论 VaR 的优点和缺点，以及如何将其纳入我们将在本章开发的交易系统中。

在本章中，我们将涵盖以下主题：

+   算法交易概述

+   具有公共 API 的经纪人和系统供应商列表

+   为交易系统选择编程语言

+   设计算法交易平台

+   在 Oanda v20 Python 模块上设置 API 访问

+   实施均值回归算法交易策略

+   实施趋势跟踪算法交易策略

+   在我们的交易系统中引入 VaR 进行风险管理

+   在 Python 上对 AAPL 进行 VaR 计算

# 介绍算法交易

上世纪 90 年代，交易所已经开始使用电子交易系统。到 1997 年，全球 44 个交易所使用自动化系统进行期货和期权交易，更多交易所正在开发自动化技术。芝加哥期货交易所（**CBOT**）和伦敦国际金融期货和期权交易所（**LIFFE**）等交易所将他们的电子交易系统用作传统的公开喊价交易场所之外的交易补充，从而使交易者可以全天候访问交易所的风险管理工具。随着技术的改进，基于技术的交易变得更加廉价，推动了更快更强大的交易平台的增长。订单执行的可靠性更高，消息传输错误率更低，这加深了金融机构对技术的依赖。大多数资产管理人、专有交易者和做市商已经从交易场所转移到了电子交易场所。

随着系统化或计算机化交易变得更加普遍，速度成为决定交易结果的最重要因素。通过利用复杂的基本模型，量化交易者能够动态重新计算交易产品的公平价值并执行交易决策，从而能够以牺牲使用传统工具的基本交易者的利润。这催生了**高频交易**（**HFT**）这一术语，它依赖快速计算机在其他人之前执行交易决策。事实上，高频交易已经发展成为一个价值数十亿美元的行业。

算法交易是指对系统化交易流程的自动化，其中订单执行被大大优化以获得最佳价格。它不是投资组合配置过程的一部分。

银行、对冲基金、经纪公司、结算公司和交易公司通常会将他们的服务器放置在电子交易所旁边，以接收最新的市场价格，并在可能的情况下执行最快的订单。他们给交易所带来了巨大的交易量。任何希望参与低延迟、高交易量活动（如复杂事件处理或捕捉瞬息的价格差异）的人，可以通过获得交易所连接的方式来进行，可以选择共同定位的形式，他们的服务器硬件可以放置在交易所旁边的机架上，需要支付一定费用。

**金融信息交换**（**FIX**）协议是与交易所进行电子通信的行业标准，从私人服务器实现**直接市场访问**（**DMA**）到实时信息。C++是在 FIX 协议上进行交易的常见选择，尽管其他语言，如.NET Framework 公共语言和 Java 也可以使用。**表述性状态转移**（**REST**）API 提供正在变得越来越普遍，供零售投资者使用。在创建算法交易平台之前，您需要评估各种因素，如学习的速度和便捷性，然后才能决定特定的语言用于此目的。

经纪公司将为他们的客户提供某种交易平台，以便他们可以在选定的交易所上执行订单，作为佣金费用的回报。一些经纪公司可能会提供 API 作为他们的服务提供的一部分，以满足技术倾向的客户希望运行自己的交易算法。在大多数情况下，客户也可以从第三方供应商提供的多个商业交易平台中进行选择。其中一些交易平台也可能提供 API 访问以将订单电子路由到交易所。在开发算法交易系统之前，重要的是事先阅读 API 文档，了解经纪人提供的技术能力，并制定开发算法交易系统的方法。

# 具有公共 API 的交易平台

以下表格列出了一些经纪人和交易平台供应商，他们的 API 文档是公开可用的：

| **经纪人/供应商** | **网址** | **支持的编程语言** |
| --- | --- | --- |
| CQG | [`www.cqg.com`](https://www.cqg.com) | REST, FIX, C#, C++, and VB/VBA |
| Cunningham Trading Systems | [`www.ctsfutures.com`](http://www.ctsfutures.com) | Microsoft .Net Framework 4.0 and FIX |
| E*Trade | [`developer.etrade.com/home`](https://developer.etrade.com/home) | Python, Java, and Node.js |
| Interactive Brokers | [`www.interactivebrokers.com/en/index.php?f=5041`](https://www.interactivebrokers.com/en/index.php?f=5041) | Java, C++, Python, C#, C++, and DDE |
| IG | [`labs.ig.com/`](https://labs.ig.com/) | REST, Java, JavaScript, .NET, Clojure, and Node.js |
| Tradier | [`developer.tradier.com/`](https://developer.tradier.com/) | REST |
| Trading Technologies | [`www.tradingtechnologies.com/trading/apis/`](https://www.tradingtechnologies.com/trading/apis/) | REST, .NET, and FIX |
| OANDA | [`developer.oanda.com/`](https://developer.oanda.com/) | REST, Java, and FIX |
| FXCM | [`www.fxcm.com/uk/algorithmic-trading/api-trading/`](https://www.fxcm.com/uk/algorithmic-trading/api-trading/) | REST, Java, and FIX |

# 选择编程语言

对于与经纪人或供应商进行接口的多种编程语言选择，对于刚开始进行算法交易平台开发的人来说，自然而然会产生一个问题：*我应该使用哪种语言？*

在回答这个问题之前，重要的是要弄清楚你的经纪人是否提供开发者工具。RESTful API 正变得越来越普遍，与 FIX 协议访问并列。少数经纪人支持 Java 和 C#。使用 RESTful API，几乎可以在支持**超文本传输协议**（**HTTP**）的任何编程语言中搜索或编写其包装器。

请记住，每个工具选项都有其自身的限制。你的经纪人可能会对价格和事件更新进行速率限制。产品的开发方式、要遵循的性能指标、涉及的成本、延迟阈值、风险度量以及预期的用户界面都是需要考虑的因素。风险管理、执行引擎和投资组合优化器是会影响系统设计的一些主要组件。你现有的交易基础设施、操作系统的选择、编程语言编译器的能力以及可用的软件工具对系统设计、开发和部署提出了进一步的限制。

# 系统功能

定义交易系统的结果非常重要。结果可能是一个基于研究的系统，涉及从数据供应商获取高质量数据，执行计算或运行模型，并通过信号生成评估策略。研究组件的一部分可能包括数据清理模块或回测界面，以在历史数据上使用理论参数运行策略。在设计我们的系统时，CPU 速度、内存大小和带宽是需要考虑的因素。

另一个结果可能是一个更关注风险管理和订单处理功能以确保多个订单及时执行的执行型系统。系统必须非常健壮，以处理订单执行过程中的任何故障点。因此，在设计执行订单的系统时需要考虑网络配置、硬件、内存管理和速度以及用户体验等因素。

一个系统可能包含一个或多个这些功能。设计更大的系统不可避免地会给框架增加复杂性。建议选择一个或多个编程语言，可以解决和平衡交易系统的开发速度、开发便捷性、可扩展性和可靠性。

# 构建算法交易平台

在这一部分，我们将使用 Python 设计和构建一个实时算法交易系统。由于每个经纪人的开发工具和服务都不同，因此需要考虑与我们自己的交易系统集成所需的不同编程实现。通过良好的系统设计，我们可以构建一个通用服务，允许配置不同经纪人的插件，并与我们的交易系统良好地协同工作。

# 设计经纪人接口

在设计交易平台时，以下三个功能对于实现任何给定的交易计划都是非常理想的：

+   **获取价格**：定价数据是交易所提供的最基本信息之一。它代表了市场为购买或出售交易产品所做的报价价格。经纪人可能会重新分发交易所的数据，以自己的格式传递给你。可用的价格数据的最基本形式是报价的日期和时间、交易产品的符号以及交易产品的报价买入和卖出价格。通常情况下，这些定价数据对基于交易的决策非常有用。

最佳的报价和询价价格被称为**Level 1**报价。在大多数情况下，可以向经纪人请求 Level 2、3 甚至更多的报价级别。

+   **向市场发送订单：**当向市场发送订单时，它可能会被您的经纪人或交易所执行，也可能不会。如果订单得到执行，您将在交易产品中开立一个持仓，并使自己承担各种风险以及回报。最简单的订单形式指定了要交易的产品（通常用符号表示）、要交易的数量、您想要采取的持仓（即您是买入还是卖出），以及对于非市价订单，要交易的价格。根据您的需求，有许多不同类型的订单可用于帮助管理您的交易风险。

您的经纪人可能不支持所有订单类型。最好与您的经纪人核实可用的订单类型以及哪种订单类型可以最好地管理您的交易风险。市场参与者最常用的订单类型是市价订单、限价订单和长期有效订单。**市价订单**是立即在市场上买入或卖出产品的订单。由于它是根据当前市场价格执行的，因此不需要为此类型的订单指定执行价格。**限价订单**是以特定或更好的价格买入或卖出产品的订单。**长期有效**（**GTC**）订单是一种保持在交易所队列中等待执行的订单，直到规定的到期时间。除非另有规定，大多数订单都是在交易日结束时到期的长期有效订单。您可以在[`www.investopedia.com/university/how-start-trading/how-start-trading-order-types.asp`](https://www.investopedia.com/university/how-start-trading/how-start-trading-order-types.asp)找到更多有关各种订单类型的信息。

+   **跟踪持仓：**一旦您的订单执行，您将进入一个持仓。跟踪您的持仓将有助于确定您的交易策略表现如何（好坏皆有可能！），并管理和规划您的风险。您的持仓盈亏根据市场波动而变化，并被称为**未实现盈利和亏损**。关闭持仓后，您将获得**实现盈利和亏损**，这是您交易策略的最终结果。

有了这三个基本功能，我们可以设计一个通用的`Broker`类，实现这些功能，并可以轻松扩展到任何特定经纪人的配置。

# Python 库要求

在本章中，我们将使用公开可用的 v20 模块，Oanda 作为我们的经纪人。本章中提到的所有方法实现都使用`v20` Python 库作为示例。

# 安装 v20

OANDA v20 REST API 的官方存储库位于[`github.com/oanda/v20-python`](https://github.com/oanda/v20-python)。使用终端命令使用 pip 进行安装。

```py
pip install v20 
```

有关 OANDA v20 REST API 的详细文档可以在[`developer.oanda.com/rest-live-v20/introduction/`](http://developer.oanda.com/rest-live-v20/introduction/)找到。API 的使用因经纪人而异，因此在编写交易系统实现之前，请务必与您的经纪人咨询适当的文档。

# 编写基于事件驱动的经纪人类

无论是获取价格、发送订单还是跟踪持仓，基于事件驱动的系统设计将以多线程方式触发我们系统的关键部分，而不会阻塞主线程。

让我们开始编写我们的 Python`Broker`类，如下所示：

```py
from abc import abstractmethod

class Broker(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port

        self.__price_event_handler = None
        self.__order_event_handler = None
        self.__position_event_handler = None
```

在构造函数中，我们可以为继承子类提供我们的经纪人的`host`和`port`公共连接配置。分别声明了三个变量，用于存储价格、订单和持仓更新的事件处理程序。在这里，我们设计了每个事件只有一个监听器。更复杂的交易系统可能支持同一事件处理程序上的多个监听器。

# 存储价格事件处理程序

在`Broker`类内部，分别添加以下两个方法作为价格事件处理程序的 getter 和 setter：

```py
@property
def on_price_event(self):
    """
    Listeners will receive: symbol, bid, ask
    """
    return self.__price_event_handler

@on_price_event.setter
def on_price_event(self, event_handler):
    self.__price_event_handler = event_handler
```

继承的子类将通过`on_price_event`方法调用通知监听器有关符号、买价和卖价的信息。稍后，我们将使用这些基本信息做出我们的交易决策。

# 存储订单事件处理程序

分别添加以下两个方法作为订单事件处理程序的获取器和设置器：

```py
@property
def on_order_event(self):
    """
    Listeners will receive: transaction_id
    """
    return self.__order_event_handler

@on_order_event.setter
def on_order_event(self, event_handler):
    self.__order_event_handler = event_handler
```

在订单被路由到您的经纪人之后，继承的子类将通过`on_order_event`方法调用通知监听器，同时附带订单交易 ID。

# 存储持仓事件处理程序

添加以下两个方法作为持仓事件处理程序的获取器和设置器：

```py
@property
def on_position_event(self):
    """
    Listeners will receive:
    symbol, is_long, units, unrealized_pnl, pnl
    """
    return self.__position_event_handler

@on_position_event.setter
def on_position_event(self, event_handler):
    self.__position_event_handler = event_handler
```

当从您的经纪人接收到持仓更新事件时，继承的子类将通过`on_position_event`方法通知监听器，其中包含符号信息、表示多头或空头持仓的标志、交易单位数、未实现的盈亏和已实现的盈亏。

# 声明一个用于获取价格的抽象方法

由于从数据源获取价格是任何交易系统的主要要求，创建一个名为`get_prices()`的抽象方法来执行这样的功能。它期望一个`symbols`参数，其中包含一个经纪人定义的符号列表，将用于从我们的经纪人查询数据。继承的子类应该实现这个方法，否则会抛出`NotImplementedError`异常：

```py
@abstractmethod
def get_prices(self, symbols=[]):
    """
    Query market prices from a broker
    :param symbols: list of symbols recognized by your broker
    """
    raise NotImplementedError('Method is required!')
```

请注意，`get_prices()`方法预计执行一次获取当前市场价格的操作。这给我们提供了特定时间点的市场快照。对于一个持续运行的交易系统，我们将需要实时流式传输市场价格来满足我们的交易逻辑，接下来我们将定义这一点。

# 声明一个用于流式传输价格的抽象方法

添加一个`stream_prices()`抽象方法，使用以下代码接受一个符号列表来流式传输价格：

```py
@abstractmethod
def stream_prices(self, symbols=[]):
    """"
    Continuously stream prices from a broker.
    :param symbols: list of symbols recognized by your broker
    """
    raise NotImplementedError('Method is required!')
```

继承的子类应该在从您的经纪人流式传输价格时实现这个方法，否则会抛出`NotImplementedError`异常消息。

# 声明一个用于发送订单的抽象方法

为继承的子类添加一个`send_market_order()`抽象方法，用于在向您的经纪人发送市价订单时实现：

```py
@abstractmethod
def send_market_order(self, symbol, quantity, is_buy):
    raise NotImplementedError('Method is required!')
```

使用我们的`Broker`基类中编写的前述方法，我们现在可以在下一节中编写特定于经纪人的类。

# 实现经纪人类

在本节中，我们将实现特定于我们的经纪人 Oanda 的`Broker`类的抽象方法。这需要使用`v20`库。但是，您可以轻松地更改配置和任何特定于您选择的经纪人的实现方法。

# 初始化经纪人类

编写以下`OandaBroker`类，它是特定于我们经纪人的类，扩展了通用的`Broker`类：

```py
import v20

class OandaBroker(Broker):
    PRACTICE_API_HOST = 'api-fxpractice.oanda.com'
    PRACTICE_STREAM_HOST = 'stream-fxpractice.oanda.com'

    LIVE_API_HOST = 'api-fxtrade.oanda.com'
    LIVE_STREAM_HOST = 'stream-fxtrade.oanda.com'

    PORT = '443'

    def __init__(self, accountid, token, is_live=False):
        if is_live:
            host = self.LIVE_API_HOST
            stream_host = self.LIVE_STREAM_HOST
        else:
            host = self.PRACTICE_API_HOST
            stream_host = self.PRACTICE_STREAM_HOST

        super(OandaBroker, self).__init__(host, self.PORT)

        self.accountid = accountid
        self.token = token

        self.api = v20.Context(host, self.port, token=token)
        self.stream_api = v20.Context(stream_host, self.port, token=token)
```

请注意，Oanda 使用两个不同的主机用于常规 API 端点和流式 API 端点。这些端点对于他们的模拟和实盘交易环境是不同的。所有端点都连接在标准的**安全套接字层**（**SSL**）端口 440 上。在构造函数中，`is_live`布尔标志选择适合所选交易环境的适当端点，以保存在父类中。`is_live`的`True`值表示实盘交易环境。构造函数参数还保存了账户 ID 和令牌，这些信息是用于验证用于交易的账户的。这些信息可以从您的经纪人那里获取。

`api`和`stream_api`变量保存了`v20`库的`Context`对象，通过调用方法向您的经纪人发送指令时使用。

# 实现获取价格的方法

以下代码实现了`OandaBroker`类中的父`get_prices()`方法，用于从您的经纪人获取价格：

```py
def get_prices(self, symbols=[]):
    response = self.api.pricing.get(
        self.accountid,
        instruments=",".join(symbols),
        snapshot=True,
        includeUnitsAvailable=False
    )
    body = response.body
    prices = body.get('prices', [])
    for price in prices:
        self.process_price(price)
```

响应主体包含一个 `prices` 属性和一个对象列表。列表中的每个项目都由 `process_price()` 方法处理。让我们也在 `OandaBroker` 类中实现这个方法：

```py
def process_price(self, price):
    symbol = price.instrument

    if not symbol:
        print('Price symbol is empty!')
        return

    bids = price.bids or []
    price_bucket_bid = bids[0] if bids and len(bids) > 0 else None
    bid = price_bucket_bid.price if price_bucket_bid else 0

    asks = price.asks or []
    price_bucket_ask = asks[0] if asks and len(asks) > 0 else None
    ask = price_bucket_ask.price if price_bucket_ask else 0

    self.on_price_event(symbol, bid, ask)
```

`price` 对象包含一个字符串对象的 `instrument` 属性，以及 `bids` 和 `asks` 属性中的 `list` 对象。通常，Level 1 报价是可用的，所以我们读取每个列表的第一项。列表中的每个项目都是一个 `price_bucket` 对象，我们从中提取买价和卖价。

有了这些提取的信息，我们将其传递给 `on_price_event()` 事件处理程序方法。请注意，在这个例子中，我们只传递了三个值。在更复杂的交易系统中，您可能希望考虑提取更详细的信息，比如成交量、最后成交价格或多级报价，并将其传递给价格事件监听器。

# 实现流动价格的方法

在 `OandaBroker` 类中添加以下 `stream_prices()` 方法，以从经纪人那里开始流动价格：

```py
def stream_prices(self, symbols=[]):
    response = self.stream_api.pricing.stream(
        self.accountid,
        instruments=",".join(symbols),
        snapshot=True
    )

    for msg_type, msg in response.parts():
        if msg_type == "pricing.Heartbeat":
            continue
        elif msg_type == "pricing.ClientPrice":
            self.process_price(msg)
```

由于主机连接期望连续流，`response` 对象有一个 `parts()` 方法来监听传入的数据。`msg` 对象本质上是一个 `price` 对象，我们可以重复使用它来通知监听器有一个传入的价格事件。

# 实现发送市价订单的方法

在 `OandaBroker` 类中添加以下 `send_market_order()` 方法，它将向您的经纪人发送一个市价订单：

```py
def send_market_order(self, symbol, quantity, is_buy):
    response = self.api.order.market(
        self.accountid,
        units=abs(quantity) * (1 if is_buy else -1),
        instrument=symbol,
        type='MARKET',
    )
    if response.status != 201:
        self.on_order_event(symbol, quantity, is_buy, None, 'NOT_FILLED')
        return

    body = response.body
    if 'orderCancelTransaction' in body:
        self.on_order_event(symbol, quantity, is_buy, None, 'NOT_FILLED')
        return transaction_id = body.get('lastTransactionID', None) 
    self.on_order_event(symbol, quantity, is_buy, transaction_id, 'FILLED')
```

当调用 v20 `order` 库的 `market()` 方法时，预期响应的状态为 `201`，表示成功连接到经纪人。建议进一步检查响应主体，以查看我们订单执行中的错误迹象。在成功执行的情况下，交易 ID 和订单的详细信息通过调用 `on_order_event()` 事件处理程序传递给监听器。否则，订单事件将以空的交易 ID 触发，并带有 `NOT_FILLED` 状态，表示订单不完整。

# 实现获取头寸的方法

在 `OandaBroker` 类中添加以下 `get_positions()` 方法，它将为给定账户获取所有可用的头寸信息：

```py
def get_positions(self):
    response = self.api.position.list(self.accountid)
    body = response.body
    positions = body.get('positions', [])
    for position in positions:
        symbol = position.instrument
        unrealized_pnl = position.unrealizedPL
        pnl = position.pl
        long = position.long
        short = position.short

        if short.units:
            self.on_position_event(
                symbol, False, short.units, unrealized_pnl, pnl)
        elif long.units:
            self.on_position_event(
                symbol, True, long.units, unrealized_pnl, pnl)
        else:
            self.on_position_event(
                symbol, None, 0, unrealized_pnl, pnl)
```

在响应主体中，`position` 属性包含一个 `position` 对象列表，每个对象都有合同符号、未实现和已实现的盈亏、多头和空头头寸的数量属性。这些信息通过 `on_position_event()` 事件处理程序传递给监听器。

# 获取价格

现在定义了来自我们经纪人的价格事件监听器的方法，我们可以通过阅读当前市场价格来测试与我们经纪人之间建立的连接。可以使用以下 Python 代码实例化 `Broker` 类：

```py
# Replace these 2 values with your own!
ACCOUNT_ID = '101-001-1374173-001'
API_TOKEN = '6ecf6b053262c590b78bb8199b85aa2f-d99c54aecb2d5b4583a9f707636e8009'

broker = OandaBroker(ACCOUNT_ID, API_TOKEN)
```

用您的经纪人提供的自己的凭据替换两个常量变量 `ACCOUNT_ID` 和 `API_TOKEN`，这些凭据标识了您自己的交易账户。`broker` 变量是 `OandaBroker` 的一个实例，我们可以使用它来执行各种特定于经纪人的调用。

假设我们有兴趣了解 EUR/USD 货币对的当前市场价格。让我们定义一个常量变量来保存这个工具的符号，这个符号被我们的经纪人所认可：

```py
SYMBOL = 'EUR_USD'
```

接下来，使用以下代码定义来自我们经纪人的价格事件监听器：

```py
import datetime as dt

def on_price_event(symbol, bid, ask):
   print(
        dt.datetime.now(), '[PRICE]',
        symbol, 'bid:', bid, 'ask:', ask
    )

broker.on_price_event = on_price_event
```

`on_price_event()` 函数被定义为监听器，用于接收价格信息，并分配给 `broker.on_price_event` 事件处理程序。我们期望从定价事件中获得三个值 - 合同符号、买价和卖价 - 我们只是简单地将它们打印到控制台。

调用 `get_prices()` 方法来从我们的经纪人那里获取当前市场价格：

```py
broker.get_prices(symbols=[SYMBOL])
```

我们应该在控制台上得到类似的输出：

```py
2018-11-19 21:29:13.214893 [PRICE] EUR_USD bid: 1.14361 ask: 1.14374
```

输出是一行，显示 EUR/USD 货币对的买价和卖价分别为 `1.14361` 和 `1.14374`。

# 发送一个简单的市价订单

与获取价格时一样，我们可以重用`broker`变量向我们的经纪人发送市价订单。

现在假设我们有兴趣购买一单位相同的 EUR/USD 货币对；以下代码执行此操作：

```py
def on_order_event(symbol, quantity, is_buy, transaction_id, status):
    print(
        dt.datetime.now(), '[ORDER]',
        'transaction_id:', transaction_id,
        'status:', status,
        'symbol:', symbol,
        'quantity:', quantity,
        'is_buy:', is_buy,
    )

broker.on_order_event = on_order_event
broker.send_market_order(SYMBOL, 1, True)
```

`on_order_event()`函数被定义为监听来自我们经纪人的订单更新的函数，并分配给`broker.on_order_event`事件处理程序。例如，执行的限价订单或取消的订单将通过此方法调用。最后，`send_market_order()`方法表示我们有兴趣购买一单位 EUR/USD 货币对。

如果在运行上述代码时货币市场开放，您应该会得到以下结果，交易 ID 不同：

```py
2018-11-19 21:29:13.484685 [ORDER] transaction_id: 754 status: FILLED symbol: EUR_USD quantity: 1 is_buy: True
```

输出显示订单成功填写，购买一单位 EUR/USD 货币对，交易 ID 为`754`。

# 获取持仓更新

通过发送市价订单进行开仓，我们应该能够查看当前的 EUR/USD 头寸。我们可以在`broker`对象上使用以下代码来实现：

```py
def on_position_event(symbol, is_long, units, upnl, pnl):
    print(
        dt.datetime.now(), '[POSITION]',
        'symbol:', symbol,
        'is_long:', is_long,
        'units:', units,
        'upnl:', upnl,
        'pnl:', pnl
    )

broker.on_position_event = on_position_event
broker.get_positions()
```

`on_position_event()`函数被定义为监听来自我们经纪人的持仓更新的函数，并分配给`broker.on_position_event`事件处理程序。当调用`get_positions()`方法时，经纪人返回持仓信息并触发以下输出：

```py
2018-11-19 21:29:13.752886 [POSITION] symbol: EUR_USD is_long: True units: 1.0 upnl: -0.0001 pnl: 0.0
```

我们的头寸报告目前是 EUR/USD 货币对的一个多头单位，未实现损失为$0.0001。由于这是我们的第一笔交易，我们还没有实现任何利润或损失。

# 构建均值回归算法交易系统

现在我们的经纪人接受订单并响应我们的请求，我们可以开始设计一个完全自动化的交易系统。在本节中，我们将探讨如何设计和实施一个均值回归算法交易系统。

# 设计均值回归算法

假设我们相信在正常的市场条件下，价格会波动，但往往会回归到某个短期水平，例如最近价格的平均值。在这个例子中，我们假设 EUR/USD 货币对在近期短期内表现出均值回归特性。首先，我们将原始的 tick 级数据重新采样为标准时间序列间隔，例如一分钟间隔。然后，取最近几个周期来计算短期平均价格（例如，使用五个周期），我们认为 EUR/USD 价格将向前五分钟的价格平均值回归。

一旦 EUR/USD 货币对的出价价格超过短期平均价格，以五分钟为例，我们的交易系统将生成一个卖出信号，我们可以选择通过卖出市价订单进入空头头寸。同样，当 EUR/USD 的询价价格低于平均价格时，将生成买入信号，我们可以选择通过买入市价订单进入多头头寸。

一旦开仓，我们可以使用相同的信号来平仓。当开多头头寸时，我们在卖出信号时通过输入市价卖出订单来平仓。同样，当开空头头寸时，我们在买入信号时通过输入市价买入订单来平仓。

您可能会观察到我们交易策略中存在许多缺陷。平仓并不保证盈利。我们对市场的看法可能是错误的；在不利的市场条件下，信号可能会在一个方向上持续一段时间，并且有很高的可能性以巨大的损失平仓！作为交易员，您应该找出适合自己信念和风险偏好的个人交易策略。

# 实施均值回归交易员类

我们交易系统需要的两个重要参数是重新取样间隔和计算周期数。首先，创建一个名为`MeanReversionTrader`的类，我们可以实例化并作为我们的交易系统运行：

```py
import time
import datetime as dt
import pandas as pd

class MeanReversionTrader(object):
    def __init__(
        self, broker, symbol=None, units=1,
        resample_interval='60s', mean_periods=5
    ):
        """
        A trading platform that trades on one side
            based on a mean-reverting algorithm.

        :param broker: Broker object
        :param symbol: A str object recognized by the broker for trading
        :param units: Number of units to trade
        :param resample_interval: 
            Frequency for resampling price time series
        :param mean_periods: Number of resampled intervals
            for calculating the average price
        """
        self.broker = self.setup_broker(broker)

        self.resample_interval = resample_interval
        self.mean_periods = mean_periods
        self.symbol = symbol
        self.units = units

        self.df_prices = pd.DataFrame(columns=[symbol])
        self.pnl, self.upnl = 0, 0

        self.mean = 0
        self.bid_price, self.ask_price = 0, 0
        self.position = 0
        self.is_order_pending = False
        self.is_next_signal_cycle = True
```

构造函数中的五个参数初始化了我们交易系统的状态 - 使用的经纪人、要交易的标的、要交易的单位数、我们价格数据的重新取样间隔，以及我们均值计算的周期数。这些值只是存储为类变量。

`setup_broker()`方法调用设置我们的类来处理我们即将定义的`broker`对象的事件。当我们接收到价格数据时，这些数据存储在一个`pandas` DataFrame 变量`df_prices`中。最新的买入和卖出价格存储在`bid_price`和`ask_price`变量中，用于计算信号。`mean`变量将存储先前`mean_period`价格的计算均值。`position`变量将存储我们当前持仓的单位数。负值表示空头持仓，正值表示多头持仓。

`is_order_pending`布尔标志指示是否有订单正在等待经纪人执行，`is_next_signal_cycle`布尔标志指示当前交易状态周期是否开放。请注意，我们的系统状态可以如下：

1.  等待买入或卖出信号。

1.  在买入或卖出信号上下订单。

1.  当持仓被打开时，等待卖出或买入信号。

1.  在卖出或买入信号上下订单。

1.  当持仓被平仓时，转到步骤 1。

在步骤 1 到 5 的每个周期中，我们只交易一个单位。这些布尔标志作为锁，防止多个订单同时进入系统。

# 添加事件监听器

让我们在我们的`MeanReversionTrader`类中连接价格、订单和持仓事件。

将`setup_broker()`方法添加到这个类中，如下所示：

```py
def setup_broker(self, broker):
    broker.on_price_event = self.on_price_event
    broker.on_order_event = self.on_order_event
    broker.on_position_event = self.on_position_event
    return broker
```

我们只是将三个类方法分配为经纪人生成的任何事件的监听器，以监听价格、订单和持仓更新。

将`on_price_event()`方法添加到这个类中，如下所示：

```py
def on_price_event(self, symbol, bid, ask):
    print(dt.datetime.now(), '[PRICE]', symbol, 'bid:', bid, 'ask:', ask)

    self.bid_price = bid
    self.ask_price = ask
    self.df_prices.loc[pd.Timestamp.now(), symbol] = (bid + ask) / 2.

    self.get_positions()
    self.generate_signals_and_think()

    self.print_state()
```

当收到价格事件时，我们将它们存储在我们的`bid_price`、`ask_price`和`df_prices`类变量中。随着价格的变化，我们的持仓和信号值也会发生变化。`get_position()`方法调用将检索我们持仓的最新信息，`generate_signals_and_think()`调用将重新计算我们的信号并决定是否进行交易。使用`print_state()`命令将系统的当前状态打印到控制台。

编写`get_position()`方法来从我们的经纪人中检索持仓信息，如下所示：

```py
def get_positions(self):
    try:
        self.broker.get_positions()
    except Exception as ex:
        print('get_positions error:', ex)
```

将`on_order_event()`方法添加到我们的类中，如下所示：

```py
def on_order_event(self, symbol, quantity, is_buy, transaction_id, status):
    print(
        dt.datetime.now(), '[ORDER]',
        'transaction_id:', transaction_id,
        'status:', status,
        'symbol:', symbol,
        'quantity:', quantity,
        'is_buy:', is_buy,
    )
    if status == 'FILLED':
        self.is_order_pending = False
        self.is_next_signal_cycle = False

        self.get_positions()  # Update positions before thinking
        self.generate_signals_and_think()
```

当接收到订单事件时，我们将它们打印到控制台上。在我们的经纪人的`on_order_event`实现中，成功执行的订单将传递`status`值为`FILLED`或`UNFILLED`。只有在成功的订单中，我们才能关闭我们的布尔锁，检索我们的最新持仓，并进行决策以平仓我们的持仓。

将`on_position_event()`方法添加到我们的类中，如下所示：

```py
def on_position_event(self, symbol, is_long, units, upnl, pnl):
    if symbol == self.symbol:
        self.position = abs(units) * (1 if is_long else -1)
        self.pnl = pnl
        self.upnl = upnl
        self.print_state()
```

当接收到我们预期交易标的的持仓更新事件时，我们存储我们的持仓信息、已实现收益和未实现收益。使用`print_state()`命令将系统的当前状态打印到控制台。

将`print_state()`方法添加到我们的类中，如下所示：

```py
def print_state(self):
    print(
        dt.datetime.now(), self.symbol, self.position_state, 
        abs(self.position), 'upnl:', self.upnl, 'pnl:', self.pnl
    )
```

一旦我们的订单、持仓或市场价格有任何更新，我们就会将系统的最新状态打印到控制台。

# 编写均值回归信号生成器

我们希望我们的决策算法在每次价格或订单更新时重新计算交易信号。让我们在`MeanReversionTrader`类中创建一个`generate_signals_and_think()`方法来做到这一点：

```py
def generate_signals_and_think(self):
    df_resampled = self.df_prices\
        .resample(self.resample_interval)\
        .ffill()\
        .dropna()
    resampled_len = len(df_resampled.index)

    if resampled_len < self.mean_periods:
        print(
            'Insufficient data size to calculate logic. Need',
            self.mean_periods - resampled_len, 'more.'
        )
        return

    mean = df_resampled.tail(self.mean_periods).mean()[self.symbol]

    # Signal flag calculation
    is_signal_buy = mean > self.ask_price
    is_signal_sell = mean < self.bid_price

    print(
        'is_signal_buy:', is_signal_buy,
        'is_signal_sell:', is_signal_sell,
        'average_price: %.5f' % mean,
        'bid:', self.bid_price,
        'ask:', self.ask_price
    )

    self.think(is_signal_buy, is_signal_sell)
```

由于价格数据存储在`df_prices`变量中作为 pandas DataFrame，我们可以按照构造函数中给定的`resample_interval`变量的定义，定期对其进行重新采样。`ffill()`方法向前填充任何缺失的数据，`dropna()`命令在重新采样后移除第一个缺失值。必须有足够的数据可用于计算均值，否则此方法将简单退出。`mean_periods`变量表示必须可用的重新采样数据的最小长度。

`tail(self.mean_periods)`方法获取最近的重新采样间隔并使用`mean()`方法计算平均值，从而得到另一个 pandas DataFrame。平均水平通过引用 DataFrame 的列来获取，该列简单地是工具符号。

使用均值回归算法可用的平均价格，我们可以生成买入和卖出信号。在这里，当平均价格超过市场要价时，会生成买入信号，当平均价格超过市场竞价时，会生成卖出信号。我们的短期信念是市场价格将回归到平均价格。

在将这些计算出的值打印到控制台以便更好地调试后，我们现在可以利用买入和卖出信号来执行实际交易，这在同一类中的名为`think()`的方法中完成：

```py
def think(self, is_signal_buy, is_signal_sell):
    if self.is_order_pending:
        return

    if self.position == 0:
        self.think_when_flat_position(is_signal_buy, is_signal_sell)
    elif self.position > 0:
        self.think_when_position_long(is_signal_sell)
    elif self.position < 0: 
        self.think_when_position_short(is_signal_buy)       
```

如果订单仍处于待处理状态，我们只需不做任何操作并退出该方法。由于市场条件可能随时发生变化，您可能希望添加自己的逻辑来处理待处理状态已经过长时间的订单，并尝试另一种策略。

这三个 if-else 语句分别处理了当我们的仓位是平的、多头的或空头的交易逻辑。当我们的仓位是平的时，将调用`think_when_position_flat()`方法，写成如下：

```py
def think_when_position_flat(self, is_signal_buy, is_signal_sell):
    if is_signal_buy and self.is_next_signal_cycle:
        print('Opening position, BUY', 
              self.symbol, self.units, 'units')
        self.is_order_pending = True
        self.send_market_order(self.symbol, self.units, True)
        return

    if is_signal_sell and self.is_next_signal_cycle:
        print('Opening position, SELL', 
              self.symbol, self.units, 'units')
        self.is_order_pending = True
        self.send_market_order(self.symbol, self.units, False)
        return

    if not is_signal_buy and not is_signal_sell:
        self.is_next_signal_cycle = True
```

第一个`if`语句处理的是，在买入信号时，当前交易周期处于开放状态时，我们通过发送市价订单来买入并将该订单标记为待处理的条件。相反，第二个`if`语句处理的是在卖出信号时进入空头仓位的条件。否则，由于仓位是平的，既没有买入信号也没有卖出信号，我们只需将`is_next_signal_cycle`设置为`True`，直到有信号可用为止。

当我们处于多头仓位时，将调用`think_when_position_long()`方法，写成如下：

```py
def think_when_position_long(self, is_signal_sell):
    if is_signal_sell:
        print('Closing position, SELL', 
              self.symbol, self.units, 'units')
        self.is_order_pending = True
        self.send_market_order(self.symbol, self.units, False)
```

在卖出信号时，我们将订单标记为待处理，并立即通过发送市价订单来卖出来平仓我们的多头仓位。

同样，当我们处于空头仓位时，将调用`think_when_position_short()`方法，写成如下：

```py
def think_when_position_short(self, is_signal_buy):
    if is_signal_buy:
        print('Closing position, BUY', 
              self.symbol, self.units, 'units')
        self.is_order_pending = True
        self.send_market_order(self.symbol, self.units, True)
```

在买入信号时，我们将订单标记为待处理，并立即通过发送市价订单来买入来平仓我们的空头仓位。

为了执行订单路由功能，将以下`send_market_order()`类方法添加到我们的`MeanReversionTrader`类中：

```py
def send_market_order(self, symbol, quantity, is_buy):
    self.broker.send_market_order(symbol, quantity, is_buy)
```

订单信息简单地转发给我们的`Broker`类进行执行。

# 运行我们的交易系统

最后，为了开始运行我们的交易系统，我们需要一个入口点。将以下`run()`类方法添加到`MeanReversionTrader`类中：

```py
def run(self):
    self.get_positions()
    self.broker.stream_prices(symbols=[self.symbol])
```

在我们的交易系统的第一次运行期间，我们读取我们当前的仓位并使用该信息来初始化所有与仓位相关的信息。然后，我们请求我们的经纪人开始为给定的符号流式传输价格，并保持连接直到程序终止。

有了入场点的定义，我们只需要初始化我们的`MeanReversionTrader`类，并使用以下代码调用`run()`命令：

```py
trader = MeanReversionTrader(
    broker, 
    symbol='EUR_USD', 
    units=1
    resample_interval='60s', 
    mean_periods=5,
)
trader.run()
```

请记住，`broker`变量包含了前面*获取价格*部分定义的`OandaBroker`类的实例，我们可以重复使用它。我们的交易系统将使用这个经纪人对象来执行与经纪人相关的调用。我们对 EUR/USD 货币对感兴趣，每次交易一单位。`resample_interval`变量的值为`60s`表示我们的存储价格将以一分钟的间隔重新采样。`mean_periods`变量的值为`5`表示我们将取最近五个间隔的平均值，或者过去五分钟的平均价格。

要启动我们的交易系统，请调用`run()`；定价更新将开始涓涓流入，使我们的系统能够自行交易。您应该在控制台上看到类似以下的输出：

```py
...
2018-11-21 15:19:34.487216 [PRICE] EUR_USD bid: 1.1393 ask: 1.13943
2018-11-21 15:19:35.686323 EUR_USD FLAT 0 upnl: 0.0 pnl: 0.0
Insufficient data size to calculate logic. Need 5 more.
2018-11-21 15:19:35.694619 EUR_USD FLAT 0 upnl: 0.0 pnl: 0.0
...
```

从输出中看，我们的头寸目前是平的，并且没有足够的定价数据来计算我们的交易信号。

五分钟后，当有足够的数据进行交易信号计算时，我们应该能够观察到以下结果：

```py
...
2018-11-21 15:25:07.075883 EUR_USD FLAT 0 upnl: 0.0 pnl: -0.3246
is_signal_buy: False is_signal_sell: True average_price: 1.13934 bid: 1.13936 ask: 1.13949
Opening position, SELL EUR_USD 1 units
2018-11-21 15:25:07.356520 [ORDER] transaction_id: 2848 status: FILLED symbol: EUR_USD quantity: 1 is_buy: False
2018-11-21 15:25:07.688082 EUR_USD SHORT 1.0 upnl: -0.0001 pnl: 0.0
is_signal_buy: False is_signal_sell: True average_price: 1.13934 bid: 1.13936 ask: 1.13949
2018-11-21 15:25:07.692292 EUR_USD SHORT 1.0 upnl: -0.0001 pnl: 0.0

...
```

过去五分钟的平均价格为`1.13934`。由于 EUR/USD 的当前市场竞价价格为`1.13936`，高于平均价格，生成了一个卖出信号。生成一个卖出市价订单，开设 EUR/USD 的一个单位的空头头寸。这导致了 0.0001 美元的未实现损失。

让系统自行运行一段时间，它应该能够自行平仓。要停止交易，请使用*Ctrl* + *Z*或类似的方法终止运行的进程。请记住，一旦程序停止运行，手动平仓任何剩余的交易头寸。现在您拥有一个完全功能的自动交易系统了！

这里的系统设计和交易参数仅作为示例，并不一定会产生积极的结果！您应该尝试不同的交易参数，并改进事件处理，以找出您交易计划的最佳策略。

# 构建趋势跟踪交易平台

在前一节中，我们按照构建均值回归交易平台的步骤进行了操作。相同的功能可以很容易地扩展到包括任何其他交易策略。在本节中，我们将看看如何重用`MeanReversionTrader`类来实现一个趋势跟踪交易系统。

# 设计趋势跟踪算法

假设这一次，我们相信当前的市场条件呈现出趋势跟踪的模式，可能是由于季节性变化、经济预测或政府政策。随着价格的波动，短期平均价格水平穿过平均长期价格水平的某个阈值，我们生成买入或卖出信号。

首先，我们将原始的 tick 级数据重新采样为标准的时间序列间隔，例如，每分钟一次。其次，我们取最近的若干个周期，例如，五个周期，计算过去五分钟的短期平均价格。最后，取最近的较大数量的周期，例如，十个周期，计算过去十分钟的长期平均价格。

在没有市场波动的市场中，短期平均价格应该与长期平均价格相同，比率为一 - 这个比率也被称为贝塔。当短期平均价格增加超过长期平均价格时，贝塔大于一，市场可以被视为处于上升趋势。当短期价格下降超过长期平均价格时，贝塔小于一，市场可以被视为处于下降趋势。

在上升趋势中，一旦 beta 穿过某个价格阈值水平，我们的交易系统将生成买入信号，我们可以选择以买入市价单进入多头头寸。同样，在下降趋势中，当 beta 跌破某个价格阈值水平时，将生成卖出信号，我们可以选择以卖出市价单进入空头头寸。

一旦开仓，相同的信号可以用来平仓。当开多头头寸时，我们在卖出信号时平仓，通过以市价卖出单进入卖出订单。同样，当开空头头寸时，我们在买入信号时平仓，通过以市价买入单进入买入订单。

上述机制与均值回归交易系统设计非常相似。请记住，该算法不能保证任何利润，只是对市场的简单看法。您应该有一个与此不同（更好）的观点。

# 编写趋势跟踪交易员类

让我们为我们的趋势跟踪交易系统编写一个新的名为`TrendFollowingTreader`的类，它简单地扩展了`MeanReversionTrader`类，使用以下 Python 代码：

```py
class TrendFollowingTrader(MeanReversionTrader):
    def __init__(
        self, *args, long_mean_periods=10,
        buy_threshold=1.0, sell_threshold=1.0, **kwargs
    ):
        super(TrendFollowingTrader, self).__init__(*args, **kwargs)

        self.long_mean_periods = long_mean_periods
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
```

在我们的构造函数中，我们定义了三个额外的关键字参数，`long_mean_periods`，`buy_threshold`和`sell_threshold`，保存为类变量。`long_mean_periods`变量定义了我们的时间序列价格的重新采样间隔数量，用于计算长期平均价格。请注意，父构造函数中现有的`mean_periods`变量用于计算短期平均价格。`buy_threshold`和`sell_threshold`变量包含确定生成买入或卖出信号的 beta 边界值。

# 编写趋势跟踪信号生成器

因为只有决策逻辑需要从我们的父类`MeanReversionTrader`类中进行修改，而其他所有内容，包括订单、下单和流动价格，都保持不变，我们只需覆盖`generate_signals_and_think()`方法，并使用以下代码实现我们的新趋势跟踪信号生成器：

```py
def generate_signals_and_think(self):
    df_resampled = self.df_prices\
        .resample(self.resample_interval)\
        .ffill().dropna()
    resampled_len = len(df_resampled.index)

    if resampled_len < self.long_mean_periods:
        print(
            'Insufficient data size to calculate logic. Need',
            self.mean_periods - resampled_len, 'more.'
        )
        return

    mean_short = df_resampled\
        .tail(self.mean_periods).mean()[self.symbol]
    mean_long = df_resampled\
        .tail(self.long_mean_periods).mean()[self.symbol]
    beta = mean_short / mean_long

    # Signal flag calculation
    is_signal_buy = beta > self.buy_threshold
    is_signal_sell = beta < self.sell_threshold

    print(
        'is_signal_buy:', is_signal_buy,
        'is_signal_sell:', is_signal_sell,
        'beta:', beta,
        'bid:', self.bid_price,
        'ask:', self.ask_price
    )

    self.think(is_signal_buy, is_signal_sell)
```

与以前一样，在每次调用`generate_signals_and_think()`方法时，我们以`resample_interval`定义的固定间隔重新采样价格。现在，用于计算信号的最小间隔由`long_mean_periods`而不是`mean_periods`定义。`mean_short`变量指的是短期平均重新采样价格，`mean_long`变量指的是长期平均重新采样价格。

`beta`变量是短期平均价格与长期平均价格的比率。当 beta 上升到`buy_threshold`值以上时，将生成买入信号，并且`is_signal_buy`变量为`True`。同样，当 beta 跌破`sell_threshold`值时，将生成卖出信号，并且`is_signal_sell`变量为`True`。

交易参数被打印到控制台以进行调试，并且对父类`think()`类方法的调用会触发使用市价订单进行买入和卖出的通常逻辑。

# 运行趋势跟踪交易系统

通过实例化`TrendFollowingTrader`类并使用以下代码运行我们的趋势跟踪交易系统：

```py
trader = TrendFollowingTrader(
    broker,
    resample_interval='60s',
    symbol='EUR_USD',
    units=1,
    mean_periods=5,
    long_mean_periods=10,
    buy_threshold=1.000010,
    sell_threshold=0.99990,
)
trader.run()
```

第一个参数`broker`与上一节中为我们的经纪人创建的对象相同。同样，我们以一分钟间隔重新取样我们的时间序列价格，并且我们对交易 EUR/USD 货币对感兴趣，在任何给定时间最多进入一单位的头寸。使用`mean_periods`值为`5`，我们对最近的五个重新取样间隔感兴趣，以计算过去五分钟的平均价格作为我们的短期平均价格。使用`long_mean_period`值为`10`，我们对最近的 10 个重新取样间隔感兴趣，以计算过去 10 分钟的平均价格作为我们的长期平均价格。

短期平均价格与长期平均价格的比率被视为贝塔。当贝塔上升到超过`buy_threshold`定义的值时，将生成买入信号。当贝塔下降到低于`sell_threshold`定义的值时，将生成卖出信号。

设置好交易参数后，调用`run()`方法启动交易系统。我们应该在控制台上看到类似以下的输出：

```py
...
2018-11-23 08:51:12.438684 [PRICE] EUR_USD bid: 1.14018 ask: 1.14033
2018-11-23 08:51:13.520880 EUR_USD FLAT 0 upnl: 0.0 pnl: 0.0
Insufficient data size to calculate logic. Need 10 more.
2018-11-23 08:51:13.529919 EUR_USD FLAT 0 upnl: 0.0 pnl: 0.0
... 
```

在交易开始时，我们获得了当前市场价格，保持平仓状态，既没有盈利也没有损失。没有足够的数据可用于做出任何交易决策，我们将不得不等待 10 分钟，然后才能看到计算参数生效。

如果您的交易系统依赖于更长时间的过去数据，并且不希望等待所有这些数据被收集，考虑使用历史数据对您的交易系统进行引导。

过一段时间后，您应该会看到类似以下的输出：

```py
...
is_signal_buy: True is_signal_sell: False beta: 1.0000333228980047 bid: 1.14041 ask: 1.14058
Opening position, BUY EUR_USD 1 units
2018-11-23 09:01:01.579208 [ORDER] transaction_id: 2905 status: FILLED symbol: EUR_USD quantity: 1 is_buy: True
2018-11-23 09:01:01.844743 EUR_USD LONG 1.0 upnl: -0.0002 pnl: 0.0
...
```

让系统自行运行一段时间，它应该能够自行平仓。要停止交易，请使用*Ctrl* + *Z*或类似的方法终止运行进程。记得在程序停止运行后手动平仓任何剩余的交易头寸。采取措施改变您的交易参数和决策逻辑，使您的交易系统成为盈利性的！

请注意，作者对您的交易系统的任何结果概不负责！在实时交易环境中，需要更多的控制参数、订单管理和头寸跟踪来有效管理风险。

在接下来的部分，我们将讨论一个可以应用于我们交易计划的风险管理策略。

# VaR 用于风险管理

一旦我们在市场上开仓，就会面临各种风险，如波动风险和信用风险。为了尽可能保护我们的交易资本，将风险管理措施纳入我们的交易系统是非常重要的。

也许金融行业中最常用的风险度量是 VaR 技术。它旨在简单回答以下问题：*在特定概率水平（例如 95%）和一定时间段内，预期的最坏损失金额是多少？* VaR 的美妙之处在于它可以应用于多个层次，从特定头寸的微观层面到基于组合的宏观层面。例如，对于 1 天的时间范围，95%的置信水平下的 100 万美元 VaR 表明，平均而言，你只有 20 天中的 1 天可能会因市场波动而损失超过 100 万美元。

以下图表说明了一个均值为 0%的正态分布组合收益率，VaR 是分布中第 95 百分位数对应的损失：

![](img/467de0d4-9a4e-4bb6-9de3-575a4c3515a2.png)

假设我们在一家声称具有与标普 500 指数基金相同风险的基金中管理了 1 亿美元，预期收益率为 9%，标准偏差为 20%。使用方差-协方差方法计算 5%风险水平或 95%置信水平下的每日 VaR，我们将使用以下公式：

![](img/aa4ac48a-9c35-4eb0-a1a2-9c96d4ad6015.png)

![](img/170e85cd-3453-44fd-ae6a-9f3c5eeb9f53.png)

![](img/d9180d47-0c28-42d6-82d3-2079ac546b80.png)

在这里，*P*是投资组合的价值，*N^(−1)(α,u,σ)*是具有风险水平*α*、平均值*u*和标准差*σ*的逆正态概率分布。每年的交易日数假定为 252 天。结果表明，5%水平的每日 VaR 为$2,036,606.50。

然而，VaR 的使用并非没有缺陷。它没有考虑正态分布曲线尾端发生极端事件的损失概率。超过一定 VaR 水平的损失规模也很难估计。我们调查的 VaR 使用历史数据和假定的恒定波动率水平 - 这些指标并不代表我们未来的表现。

让我们采取一种实际的方法来计算股票价格的每日 VaR；我们将通过从数据源下载 AAPL 股票价格来调查：

```py
"""
Download the all-time AAPL dataset
"""
from alpha_vantage.timeseries import TimeSeries

# Update your Alpha Vantage API key here...
ALPHA_VANTAGE_API_KEY = 'PZ2ISG9CYY379KLI'

ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
df, meta_data = ts.get_daily_adjusted(symbol='AAPL', outputsize='full')
```

数据集将作为 pandas DataFrame 下载到`df`变量中：

```py
df.info()
```

这给我们以下输出：

```py
<class 'pandas.core.frame.DataFrame'>
Index: 5259 entries, 1998-01-02 to 2018-11-23
Data columns (total 8 columns):
1\. open                 5259 non-null float64
2\. high                 5259 non-null float64
3\. low                  5259 non-null float64
4\. close                5259 non-null float64
5\. adjusted close       5259 non-null float64
6\. volume               5259 non-null float64
7\. dividend amount      5259 non-null float64
8\. split coefficient    5259 non-null float64
dtypes: float64(8)
memory usage: 349.2+ KB
```

我们的 DataFrame 包含八列，价格从 1998 年开始到现在的交易日。感兴趣的列是调整后的收盘价。假设我们有兴趣计算 2017 年的每日 VaR；让我们使用以下代码获取这个数据集：

```py
import datetime as dt
import pandas as pd

# Define the date range
start = dt.datetime(2017, 1, 1)
end = dt.datetime(2017, 12, 31)

# Cast indexes as DateTimeIndex objects
df.index = pd.to_datetime(df.index)
closing_prices = df['5\. adjusted close']
prices = closing_prices.loc[start:end]
```

`prices`变量包含了我们 2017 年的 AAPL 数据集。

使用前面讨论的公式，您可以使用以下代码实现`calculate_daily_var()`函数：

```py
from scipy.stats import norm

def calculate_daily_var(
    portfolio, prob, mean, 
    stdev, days_per_year=252.
):
    alpha = 1-prob
    u = mean/days_per_year
    sigma = stdev/np.sqrt(days_per_year)
    norminv = norm.ppf(alpha, u, sigma)
    return portfolio - portfolio*(norminv+1)
```

假设我们持有$100 百万的 AAPL 股票，并且有兴趣找到 95%置信水平下的每日 VaR。我们可以使用以下代码定义 VaR 参数：

```py
import numpy as np

portfolio = 100000000.00
confidence = 0.95

daily_returns = prices.pct_change().dropna()
mu = np.mean(daily_returns)
sigma = np.std(daily_returns)
```

`mu`和`sigma`变量分别代表每日平均百分比收益和每日收益的标准差。

我们可以通过调用`calculate_daily_var()`函数获得 VaR，如下所示：

```py
VaR = calculate_daily_var(
    portfolio, confidence, mu, sigma, days_per_year=252.)
print('Value-at-Risk: %.2f' % VaR)
```

我们将得到以下输出：

```py
Value-at-Risk: 114248.72
```

假设每年有 252 个交易日，2017 年 AAPL 股票的每日 VaR 在 95%的置信水平下为$114,248.72。

# 摘要

在本章中，我们介绍了交易从交易场到电子交易平台的演变，并了解了算法交易的产生过程。我们看了一些经纪人提供 API 访问其交易服务。为了帮助我们开始开发算法交易系统，我们使用 Oanda `v20`库来实现一个均值回归交易系统。

在设计一个事件驱动的经纪人接口类时，我们为监听订单、价格和持仓更新定义了事件处理程序。继承`Broker`类的子类只需用经纪人特定的函数扩展这个接口类，同时保持底层交易函数与我们的交易系统兼容。我们通过获取市场价格、发送市价订单和接收持仓更新成功测试了与我们经纪人的连接。

我们讨论了一个简单的均值回归交易系统的设计，该系统根据历史平均价格的波动以及开仓和平仓市价订单来生成买入或卖出信号。由于这个交易系统只使用了一个交易逻辑来源，因此需要更多的工作来构建一个健壮、可靠和盈利的交易系统。

我们还讨论了一个趋势跟随交易系统的设计，该系统根据短期平均价格与长期平均价格的波动来生成买入或卖出信号。通过一个设计良好的系统，我们看到了通过简单地扩展均值回归父类并覆盖决策方法来修改现有交易逻辑是多么容易。

交易的一个关键方面是有效地管理风险。在金融行业，VaR 是用来衡量风险的最常见的技术。使用 Python，我们采取了一种实际的方法来计算 AAPL 过去数据集的每日 VaR。

一旦我们建立了一个有效的算法交易系统，我们可以探索其他衡量交易策略表现的方式。其中之一是回测；我们将在下一章讨论这个话题。
