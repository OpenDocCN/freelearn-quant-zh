# 第一章：算法交易简介

在本章中，我们将带您走过交易的简要历史，并解释在哪些情况下手动交易和算法交易各自有意义。此外，我们将讨论金融资产类别，这是对不同类型金融资产的分类。您将了解现代电子交易所的组成部分，最后，我们将概述算法交易系统的关键组成部分。

在本章中，我们将涵盖以下主题：

+   穿越算法交易的演变

+   了解金融资产类别

+   穿越现代电子交易所

+   了解算法交易系统的组成部分

# 穿越算法交易的演变

从交换一种财产换另一种财产的概念自古以来就存在。在其最早期形式中，交易对于交换较不理想的财产以换取较理想的财产很有用。随着时间的流逝，交易演变为参与者试图找到一种以低于公平价值的价格购买和持有交易工具（即产品）的方式，希望能够在未来以高于购买价格的价格出售它们。这个**低买高卖的原则**成为迄今所有盈利交易的基础；当然，如何实现这一点就是复杂性和竞争的关键所在。

市场由**供求基本经济力量**驱动。当需求增加而供应没有相应增加，或者供应减少而需求没有减少时，商品变得稀缺并增值（即，其市场价格）。相反，如果需求下降而供应没有减少，或者供应增加而需求没有增加，商品变得更容易获得并且价值更低（市场价格更低）。因此，商品的市场价格应该反映基于现有供应（卖方）和现有需求（买方）的平衡价格。

**手动交易方法**有许多缺点，如下所示：

+   人类交易员天生处理新市场信息的速度较慢，使他们很可能错过信息或在解释更新的市场数据时出错。这导致了糟糕的交易决策。

+   人类总体上也容易受到分心和偏见的影响，从而降低利润和/或产生损失。例如，对于失去金钱的恐惧和赚钱的喜悦也会导致我们偏离理论上理解但在实践中无法执行的最佳系统化交易方法。此外，人们也天生且非均匀地偏向于盈利交易与亏损交易；例如，人类交易员往往会在盈利交易后迅速增加风险金额，并在亏损交易后减缓降低风险金额的速度。

+   人类交易员通过经历市场条件来学习，例如通过参与和交易实时市场。因此，他们无法从历史市场数据条件中学习和进行**回测**——这是自动化策略的一个重要优势，我们稍后将会看到。

随着技术的发展，交易已经从通过大声呼喊和示意购买和出售订单进行的交易演变为使用复杂、高效和快速的计算机硬件和软件执行交易，通常几乎没有人为干预。复杂的算法交易软件系统已经取代了人类交易员和工程师，而构建、运行和改进这些系统的数学家，即**量化交易员**，已经崛起。

特别是，**自动化、计算机驱动的系统化/算法交易方法**的主要优势如下：

+   计算机非常擅长执行明确定义和重复的基于规则的任务。它们可以非常快速地执行这些任务，并且可以处理大规模的吞吐量。

+   此外，计算机不会分心、疲倦或犯错误（除非存在软件错误，从技术上讲，这算是软件开发者的错误）。

+   算法交易策略在交易过程中也没有情绪上的损失或利润，因此它们可以坚持系统性的交易计划。

所有这些优势使得系统性算法交易成为建立低延迟、高吞吐量、可扩展和稳健交易业务的完美选择。

然而，算法交易并不总是比手动交易更好：

+   手动交易更擅长处理极其复杂的思想和现实交易运营的复杂性，有时很难将其表达为自动化软件解决方案。

+   自动交易系统需要大量的时间和研发成本投入，而手动交易策略通常能更快地进入市场。

+   算法交易策略也容易受到软件开发/操作错误的影响，这可能会对交易业务产生重大影响。整个自动交易操作在几分钟内被清除并不罕见。

+   通常，自动量化交易系统不擅长处理被称为**黑天鹅**事件的极不可能发生的事件，例如 LTCM 崩盘、2010 年闪崩、Knight Capital 崩盘等。

在本节中，我们了解了交易历史以及何时自动化/算法交易优于手动交易。现在，让我们继续前往下一节，在那里我们将了解被分类为金融资产类别的实际交易主题。

# 了解金融资产类别

算法交易涉及金融资产的交易。金融资产是一种价值来源于合同协议的非实物资产。

主要的金融资产类别如下：

+   **股票**（**股票**）：这允许市场参与者直接投资于公司并成为公司的所有者。

+   **固定收益**（**债券**）：这些代表投资者向借款人（例如政府或公司）提供的贷款。每张债券都有其到期日，到期日时贷款本金应偿还，并且通常由借款人在债券寿命期间支付固定或可变的利息。

+   **房地产投资信托**（**REITs**）：这些是拥有、经营或融资产生收入的房地产的上市公司。这些可以被用作直接投资于房地产市场的代理，比如通过购买一处房产。

+   **大宗商品**：例如金属（银、金、铜等）和农产品（小麦、玉米、牛奶等）。它们是跟踪基础大宗商品价格的金融资产。

+   **交易所交易基金**（**ETFs**）：ETF 是一个在交易所上市的安全性，跟踪其他证券的集合。ETFs，例如 SPY、DIA 和 QQQ，持有股票来跟踪更大型的著名标准普尔 500、道琼斯工业平均指数和纳斯达克股票指数。ETFs，如**美国石油基金**（**USO**），通过投资于短期 WTI 原油期货来跟踪油价。ETFs 是投资者以相对较低成本投资于广泛资产类别的便利投资工具。

+   **外汇**（**FX**）在不同货币对之间交易，主要货币包括**美元**（**USD**）、**欧元**（**EUR**）、**英镑**（**GBP**）、**日元**（**JPY**）、**澳大利亚元**（**AUD**）、**新西兰元**（**NZD**）、**加拿大元**（**CAD**）、**瑞士法郎**（**CHF**）、**挪威克朗**（**NOK**）和**瑞典克朗**（**SEK**）。它们通常被称为 G10 货币。

+   主要的**金融衍生品**包括期权和期货——这些是复杂的杠杆衍生产品，可以放大风险和回报：

    a) **期货**是金融合同，以预定的未来日期和价格购买或出售资产。

    b) **期权**是金融合同，赋予其所有者权利，但不是义务，以在规定的价格（行权价）之前或之后的特定日期买入或卖出基础资产。

在本节中，我们了解了金融资产类别及其独特属性。现在，让我们讨论现代电子交易交易所的订单类型和交易匹配算法。

# 通过现代电子交易交易所进行交易

第一个交易所是阿姆斯特丹证券交易所，始于 1602 年。在这里，交易是面对面进行的。将技术应用于交易的方式包括使用信鸽、电报系统、莫尔斯电码、电话、计算机终端，以及如今的高速计算机网络和先进的计算机。随着时间的推移，交易微观结构已经发展成为我们今天所熟悉的订单类型和匹配算法。

对于算法策略的设计，现代电子交易所微观结构的了解至关重要。

## 订单类型

金融交易策略采用各种不同的订单类型，其中一些最常见的包括市价订单、带价格保护的市价订单、**立即取消**（**IOC**）订单、**填写和取消**（**FAK**）订单、**有效至当天**（**GTD**）订单、**长效**（**GTC**）订单、止损订单和冰山订单。

对于我们将在本书中探讨的策略，我们将专注于市价订单、IOC 和 GTC。

### 市价订单

市价订单是需要立即以当前市场价格执行的买入或卖出订单，当执行的即时性优于执行价格时使用。

这些订单将以订单价格执行对立方的所有可用订单，直到要求的所有数量被执行。如果没有可用的流动性可以匹配，它可以被配置为**停留在订单簿中**或**到期**。停留在订单簿中意味着订单变为待定订单，被添加到订单簿中供其他参与者进行交易。到期意味着剩余订单数量被取消，而不是被添加到订单簿中，因此新订单无法与剩余数量匹配。

因此，例如，买入市价订单将与订单簿中从最佳价格到最差价格的所有卖出订单匹配，直到整个市价订单被执行。

这些订单可能会遭受极端的**滑点**，滑点被定义为已执行订单价格与发送订单时市场价格之间的差异。

### IOC 订单

IOC 订单无法以比发送价格更差的价格执行，这意味着买入订单无法以高于订单价格的价格执行，卖出订单也无法以低于订单价格的价格执行。这个概念被称为**限价**，因为价格受限于订单可以执行的最差价格。

IOC 订单将继续与订单方的订单进行匹配，直到出现以下情况之一：

+   IOC 订单的全部数量被执行。

+   对方的被动订单价格比 IOC 订单的价格差。 

+   IOC 订单部分执行，剩余数量到期。

如果 IOC 订单的价格优于另一方的最佳可用订单（即，买单低于最佳卖出价，或卖单高于最佳买入价），则根本不会执行，而只会过期。

### GTC 订单

GTC 订单可以无限期存在，并需要特定的取消订单。

## 限价订单簿

交易所接受来自所有市场参与者的订单请求，并将其保存在**限价订单簿**中。限价订单簿是交易所在任何时间点上所有可见订单的视图。

**买单**（或**竞价**）按照从最高价格（即，最佳价格）到最低价格（即，最差价格）的顺序排列，而**卖单**（即**卖出**或**报价**）则按照从最低价格（即，最佳价格）到最高价格（即，最低价格）的顺序排列。

最高竞价价格被认为是最佳竞价价格，因为具有最高买价的买单首先被匹配，而对于卖价，情况相反，即具有最低卖价的卖单首先匹配。

相同方向、相同价格水平的订单按照**先进先出**（**FIFO**）的顺序排列，也被称为优先顺序 - 优先级更高的订单排在优先级较低的订单前面，因为优先级更高的订单比其他订单先到达了交易所。其他条件相同（即，订单方向、价格和数量相同）的情况下，优先级更高的订单将在优先级较低的订单之前执行。

## 交易所撮合引擎

电子交易所的撮合引擎使用**交易所撮合算法**执行订单的**撮合**。撮合过程包括检查市场参与者输入的所有活跃订单，并将价格交叉的订单进行匹配，直到没有可以匹配的未匹配订单为止 - 因此，价格在或高于其他卖单的买单与之匹配，反之亦然，即价格在或低于其他买单的卖单与之匹配。剩余订单保留在交易所撮合簿中，直到新的订单流入，如果可能的话，进行新的匹配。

在 FIFO 匹配算法中，订单首先按照价格从最佳价格到最差价格进行匹配。因此，来自最佳价格的买单会尝试与摆放在最低价格到最高价格的卖单（即要价/出价）匹配，而来自最高价格的卖单会尝试与摆放在最高价格到最低价格的买单匹配。新到达的订单将根据特定的规则进行匹配。对于具有更好价格的主动订单（价格优于另一侧的最佳价格水平的订单），它们将按照先到先服务的原则进行匹配，即首先出现的订单会提取流动性，因此首先匹配。对于坐在订单簿中的被动挂单，因为它们不会立即执行，它们将根据先到先服务的优先级进行分配。这意味着同一方和相同价格的订单将根据它们到达匹配引擎的时间进行排列；时间较早的订单将获得更好的优先级，因此有资格首先匹配。

在本节中，我们了解了现代电子交易所的订单类型和交易匹配引擎。现在，让我们继续前往下一节，我们将了解算法交易系统的组件。

# 了解算法交易系统的组件

客户端算法交易基础设施大致可以分为两类：**核心基础设施**和**量化基础设施**。

## 算法交易系统的核心基础设施

核心基础设施负责使用市场数据和订单输入协议与交易所进行通信。它负责在交易所和算法交易策略之间传递信息。

它的组件还负责捕获、时间戳和记录历史市场数据，这是算法交易策略研究和开发的重中之重。

核心基础设施还包括一层风险管理组件，以防止交易系统受到错误或失控的交易策略的影响，以防止灾难性结果发生。

最后，算法交易业务中涉及的一些不太光彩的任务，如后勤协调任务、合规性等，也由核心基础设施处理。

### 交易服务器

交易服务器涉及一个或多个计算机接收和处理市场和其他相关数据，并处理交易所信息（例如订单簿），并发出交易订单。

从限价订单簿中，交易所匹配簿的更新通过**市场数据协议**传播给所有市场参与者。

市场参与者拥有接收这些市场数据更新的**交易服务器**。尽管技术上，这些交易服务器可以位于世界任何地方，但现代算法交易参与者将其交易服务器放置在离交易所匹配引擎非常近的数据中心。这称为**共同定位**或**直接市场访问**（**DMA**）设置，这保证参与者尽可能快地收到市场数据更新，因为它们尽可能接近匹配引擎。

一旦市场数据更新通过交易所提供的市场数据协议通信到每个市场参与者，它们就使用称为**市场数据接收处理程序**的软件应用程序解码市场数据更新并将其馈送到客户端上的算法交易策略。

一旦算法交易策略消化了市场数据更新，根据策略中开发的智能，它生成外向订单流。这可以是在特定价格和数量上添加、修改或取消订单。

订单请求通常由一个名为**订单录入网关**的单独客户端组件接收。订单录入网关组件使用**订单录入协议**与交易所通信，将策略对交易所的请求进行转换。电子交易所对这些订单请求的响应通知被发送回订单录入网关。再次，针对特定市场参与者的订单流动，匹配引擎生成市场数据更新，因此回到此信息流循环的开头。

## 算法交易系统的量化基础设施

量化基础设施构建在核心基础设施提供的平台之上，并尝试在其上构建组件，以研究、开发和有效利用平台以产生收入。

研究框架包括回测、**交易后分析**（**PTA**）和信号研究组件等组件。

其他在研究中使用并部署到实时市场的组件包括限价订单簿、预测信号和信号聚合器，将单个信号组合成复合信号。

执行逻辑组件使用交易信号并完成管理活动，管理各种策略和交易工具之间的活动订单、持仓和**损益**（**PnL**）。

最后，交易策略本身有一个风险管理组件，用于管理和减轻不同策略和工具之间的风险。

### 交易策略

有利可图的交易理念始终是由人类直觉驱动的，这种直觉是从观察市场条件的模式和不同市场条件下各种策略的结果中发展起来的。

例如，历史上观察到，大规模的市场上涨会增强投资者信心，导致更多的市场参与者进入市场购买更多；因此，反复造成更大规模的上涨。相反，市场价格大幅下跌会吓跑投资于交易工具的参与者，导致他们抛售持有的资产并加剧价格下跌。市场观察到的这些直观观念导致了**趋势跟随策略**的想法。

还观察到，短期内价格的波动往往倾向于恢复到其之前的市场价格，导致了**均值回归**为基础的投机者和交易策略。同样，历史观察到类似产品价格的移动会相互影响，这也是直觉的合理性所在，这导致了相关性和共线性为基础的交易策略的产生，如**统计套利**和**配对交易**策略。

由于每个市场参与者使用不同的交易策略，最终的市场价格反映了大多数市场参与者的观点。与大多数市场参与者观点一致的交易策略在这些条件下是有利可图的。单一的交易策略通常不可能 100%的盈利，所以复杂的参与者有一个交易策略组合。

#### 交易信号

交易信号也被称为特征、计算器、指标、预测器或阿尔法。

交易信号是驱动算法交易策略决策的因素。信号是从市场数据、另类数据（如新闻、社交媒体动态等）甚至我们自己的订单流中获得的明确的情报，旨在预测未来某些市场条件。

信号几乎总是源自对某些市场条件和/或策略表现的直觉观察。通常，大多数量化开发人员花费大部分时间研究和开发新的交易信号，以改善在不同市场条件下的盈利能力，并全面提高算法交易策略。

#### 交易信号研究框架

大量的人力投入到研究和发现新信号以改善交易表现。为了以系统化、高效、可扩展和科学的方式做到这一点，通常第一步是建立一个良好的**信号研究框架**。

这个框架有以下子组件：

+   数据生成是基于我们试图构建的信号和我们试图捕捉/预测的市场条件/目标。在大多数现实世界的算法交易中，我们使用 tick 数据，这是代表市场上每个事件的数据。正如你可以想象的那样，每天都会有大量的事件发生，这导致了大量的数据，因此您还需要考虑对接收到的数据进行子抽样。**子抽样**有几个优点，例如减少数据规模，消除噪音/虚假数据片段，并突出显示有趣/重要的数据。

+   对与其尝试捕捉/预测的市场目标相关的特征的预测能力或有用性进行评估。

+   在不同市场条件下维护信号的历史结果，并调整现有信号以适应不断变化的市场条件。

#### 信号聚合器

**信号聚合器**是可选组件，它们从各个信号中获取输入，并以不同的方式对其进行聚合，以生成新的复合信号。

一个非常简单的聚合方法是取所有输入信号的平均值，并将平均值作为复合信号值输出。

熟悉统计学习概念的读者 - bagging 和 boosting 的集成学习 - 可能能够发现这些学习模型与信号聚合器之间的相似之处。通常，信号聚合器只是统计模型（回归/分类），其中输入信号只是用于预测相同最终市场目标的特征。

### 策略执行

策略执行涉及根据交易信号的输出有效地管理和执行订单，以最小化交易费用和滑点。

**滑点**是市场价格和执行价格之间的差异，由于订单经历了延迟才能到达市场，价格在变化之前发生了变化，以及订单的大小在达到市场后引起价格变化所致。

在算法交易策略中使用的执行策略的质量可以显著改善/降低有利交易信号的表现。

### 限价订单簿

限价订单簿既在交易所撮合引擎中构建，也在算法交易策略期间构建，尽管并不一定所有算法交易信号/策略都需要整个限价订单簿。

复杂的算法交易策略可以将更多的智能集成到其限价订单簿中。我们可以在限价订单簿中检测和跟踪自己的订单，并了解根据我们的优先级，我们的订单被执行的概率是多少。我们还可以利用这些信息在交易所的订单录入网关收到执行通知之前甚至执行我们自己的订单，并利用这种能力为我们谋利。通过限价订单簿和许多电子交易所的市场数据更新，还可以实现更复杂的微观结构特征，例如检测冰山订单、检测止损订单、检测大量买入/卖出订单流入或流出等。

### 头寸和损益管理

让我们探讨交易策略通过执行交易开仓和平仓多头和空头头寸时，头寸和损益如何演变。

当策略没有市场头寸时，即价格变动不影响交易账户价值时，这被称为持平头寸。

从持平头寸开始，如果执行买单，则被称为持有多头头寸。如果策略持有多头头寸且价格上涨，则头寸从价格上涨中获利。在这种情况下，损益也增加，即利润增加（或亏损减少）。相反，如果策略持有多头头寸且价格下跌，则头寸从价格下跌中损失。在这种情况下，损益减少，例如，利润减少（或亏损增加）。

从持平头寸开始，如果执行卖单，则被称为持有空头头寸。如果策略持有空头头寸且价格下跌，则头寸从价格下跌中获利。在这种情况下，损益增加。相反，如果策略持有空头头寸且价格上涨，则损益减少。仍然未平仓头寸的损益被称为**未实现损益（unrealized PnL）**，因为只要头寸保持未平仓状态，损益就会随着价格变动而变化。

通过卖出等量的工具来关闭多头头寸。这被称为平仓，此时损益被称为**实现损益（realized PnL）**，因为价格变动不再影响损益，因为头寸已关闭。

类似地，空头头寸通过买入与头寸规模相同的数量来关闭。

在任何时刻，**总损益（total PnL）**是所有已平仓头寸的实现损益和所有未平仓头寸的未实现损益的总和。

当多头或空头头寸由以不同价格和不同大小进行多次买入或卖出时，则通过计算**成交量加权平均价格（Volume Weighted Average Price，VWAP）**来计算头寸的平均价格，即根据每个价格上执行的数量加权平均。按市价计价是指获取头寸的 VWAP，并将其与当前市场价格进行比较，以了解某个多头/空头头寸的盈利或亏损情况。

### 回测

回测器使用历史记录的市场数据和模拟组件来模拟算法交易策略的行为和性能，就好像它在过去被部署到实时市场中一样。直到策略的表现符合预期，才会开发和优化算法交易策略。

回测器是需要模拟市场数据流、客户端和交易所端延迟的复杂组件在软件和网络组件中、准确的 FIFO 优先级、滑点、费用和市场影响来自策略订单流（即其他市场参与者将如何对策略的订单流作出反应添加到市场数据流）以生成准确的策略和投资组合绩效统计数据。

### PTA

PTA 是在模拟或实时市场运行的算法交易策略生成的交易上执行的。

PTA 系统用于从历史回测策略生成性能统计，目的是了解历史策略性能期望。

当应用于由实时交易策略生成的交易时，PTA 可用于了解实时市场中的策略性能，并比较和确认实时交易性能是否符合模拟策略性能期望。

### 风险管理

良好的风险管理原则确保策略以最佳 PnL 表现运行，并采取措施防止失控 / 错误策略。

不良的风险管理不仅可以将有利可图的交易策略变成无利可图的策略，而且还可能由于无法控制的策略损失、失灵的策略和可能的监管后果而使投资者的整个资本面临风险。

# 概要

在本章中，我们学习了什么时候算法交易比手动交易更具优势，金融资产类别是什么，最常用的订单类型是什么，限价订单簿是什么，以及订单是如何由金融交易所匹配的。

我们还讨论了算法交易系统的关键组成部分 - 核心基础设施和量化基础设施，包括交易策略、执行、限价订单簿、持仓、PnL 管理、回测、交易后分析和风险管理。

在下一章中，我们将讨论 Python 在算法交易中的价值。
