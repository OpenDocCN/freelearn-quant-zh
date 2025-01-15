# 12

# 回测交易策略

在前面的章节中，我们获得了创建交易策略所需的知识。一方面，我们可以利用技术分析来识别交易机会；另一方面，我们可以使用书中已覆盖的其他一些技术。我们可以尝试使用因子模型或波动率预测的知识，或者使用投资组合优化技术来确定我们投资的最佳资产数量。仍然缺少的一个关键点是评估如果我们在过去实施这样的策略，它会如何表现。这就是回测的目标，本章将深入探讨这一点。

**回测**可以被描述为对我们的交易策略进行的现实模拟，通过使用历史数据来评估其表现。其基本思想是，回测的表现应能代表未来当策略真正应用于市场时的表现。当然，这种情况并不总是如此，我们在实验时应该牢记这一点。

回测有多种方法，但我们应该始终记住，回测应该真实地反映市场如何运作、交易如何执行、可用的订单是什么等等。例如，忘记考虑交易成本可能会迅速将一个“有利可图”的策略变成一个失败的实验。

我们已经提到过，在不断变化的金融市场中，预测的普遍不确定性。然而，还有一些实施方面的因素可能会影响回测结果，增加将样本内表现与可推广的模式混淆的风险。我们简要提到以下一些因素：

+   **前瞻偏差**：这种潜在的缺陷出现在我们使用历史数据开发交易策略时，数据在实际使用之前就已知或可用。一些例子包括在财务报告发布后进行的修正、股票拆分或反向拆分。

+   **存活偏差**：这种偏差出现在我们仅使用当前仍然活跃/可交易的证券数据进行回测时。通过这样做，我们忽略了那些随着时间的推移消失的资产（例如破产、退市、收购等）。大多数时候，这些资产表现不佳，而我们的策略可能会因未能包括这些资产而发生偏差，因为这些资产在过去仍然可在市场中被选择。

+   **异常值检测与处理**：主要的挑战是辨别那些不代表分析期的异常值，而不是那些市场行为中不可或缺的一部分。

+   **代表性样本周期**：由于回测的目标是提供对未来表现的指示，因此样本数据应该反映当前的市场行为，并且可能还要反映未来的市场行为。如果在这一部分花费的时间不够，我们可能会错过一些关键的市场环境特征，比如波动性（极端事件过少/过多）或交易量（数据点过少）。

+   **随着时间推移实现投资目标和约束**：有时，一个策略可能在评估期的最后阶段表现良好。然而，在它活跃的某些阶段，可能会导致不可接受的高损失或高波动性。我们可以通过使用滚动绩效/风险指标来跟踪这些情况，例如风险价值（value-at-risk）或夏普比率/索提诺比率（Sharpe/Sortino ratio）。

+   **现实的交易环境**：我们已经提到过，忽略交易成本可能会极大地影响回测的最终结果。更重要的是，现实中的交易还涉及更多的复杂因素。例如，可能无法在任何时候或以目标价格执行所有交易。需要考虑的一些因素包括**滑点**（交易预期价格与实际执行价格之间的差异）、空头头寸的对手方可用性、经纪费用等。现实环境还考虑到一个事实，即我们可能基于某一日的收盘价做出交易决策，但交易可能（有可能）会基于下一交易日的开盘价执行。由于价格差异较大，我们准备的订单可能无法执行。

+   **多重测试**：在进行多个回测时，我们可能会发现虚假的结果，或者某个策略过度拟合了测试样本，导致产生不太可能在实际交易中有效的异常正面结果。此外，我们可能会在策略设计中泄露出关于什么有效、什么无效的先验知识，这可能导致进一步的过度拟合。我们可以考虑的一些方法包括：报告试验次数、计算最小回测长度、使用某种最优停止规则，或者计算考虑多重测试影响的指标（例如，通货膨胀的夏普比率）。

在本章中，我们展示了如何使用两种方法——向量化和事件驱动——对各种交易策略进行回测。我们稍后会详细介绍每种方法，但现在我们可以说，第一种方法适合于快速测试，以了解策略是否有潜力。另一方面，第二种方法更适合于进行彻底且严格的测试，因为它试图考虑上述提到的许多潜在问题。

本章的关键学习内容是如何使用流行的Python库来设置回测。我们将展示一些基于流行技术指标构建的策略示例，或使用均值方差投资组合优化的策略。掌握这些知识后，你可以回测任何自己想到的策略。

本章介绍了以下几个回测示例：

+   使用pandas进行矢量化回测

+   使用backtrader进行事件驱动回测

+   回测基于RSI的多空策略

+   回测基于布林带的买卖策略

+   使用加密数据回测移动平均交叉策略

+   回测均值方差投资组合优化

# 使用pandas进行矢量化回测

正如我们在本章的介绍中提到的，回测有两种方法。较简单的一种叫做**矢量化回测**。在这种方法中，我们将信号向量/矩阵（包含我们是开仓还是平仓的指标）与回报向量相乘。通过这样做，我们可以计算某一时间段内的表现。

由于其简单性，这种方法无法处理我们在介绍中提到的许多问题，例如：

+   我们需要手动对齐时间戳，以避免未来数据偏倚。

+   没有明确的头寸大小控制。

+   所有的性能度量都在回测的最后手动计算。

+   像止损这样的风险管理规则不容易纳入。

因此，如果我们处理的是简单的交易策略并希望用少量代码探索其初步潜力，我们应该主要使用矢量化回测。

在这个示例中，我们回测了一个非常简单的策略，规则集如下：

+   当收盘价高于20日简单移动平均线（SMA）时，我们会开仓做多。

+   当收盘价跌破20日SMA时，我们会平仓。

+   不允许卖空。

+   该策略与单位无关（我们可以持有1股或1000股），因为我们只关心价格的百分比变化

我们使用苹果公司股票及其2016到2021年的历史价格对该策略进行回测。

## 如何实现…

执行以下步骤以使用矢量化方法回测一个简单的策略：

1.  导入相关库：

    ```py
    import pandas as pd
    import yfinance as yf
    import numpy as np 
    ```

1.  下载2016到2021年间苹果公司的股票价格，并只保留调整后的收盘价：

    ```py
    df = yf.download("AAPL",
                     start="2016-01-01",
                     end="2021-12-31",
                     progress=False)
    df = df[["Adj Close"]] 
    ```

1.  计算收盘价的对数收益率和20日SMA：

    ```py
    df["log_rtn"] = df["Adj Close"].apply(np.log).diff(1)
    df["sma_20"] = df["Adj Close"].rolling(window=20).mean() 
    ```

1.  创建一个头寸指标：

    ```py
    df["position"] = (df["Adj Close"] > df["sma_20"]).astype(int) 
    ```

    使用以下代码片段，我们计算了进入多头头寸的次数：

    ```py
    sum((df["position"] == 1) & (df["position"].shift(1) == 0)) 
    ```

    答案是56。

1.  可视化2021年的策略：

    ```py
    fig, ax = plt.subplots(2, sharex=True)
    df.loc["2021", ["Adj Close", "sma_20"]].plot(ax=ax[0])
    df.loc["2021", "position"].plot(ax=ax[1])
    ax[0].set_title("Preview of our strategy in 2021") 
    ```

    执行该代码片段会生成以下图形：

    ![](../Images/B18112_12_01.png)

    图12.1：基于简单移动平均线的交易策略预览

    在 *图 12.1* 中，我们可以清楚地看到策略的运作——当收盘价高于 20 日 SMA 时，我们确实持有仓位。这由包含持仓信息的列中的值 1 所表示。

1.  计算该策略的每日和累计收益：

    ```py
    df["strategy_rtn"] = df["position"].shift(1) * df["log_rtn"]
    df["strategy_rtn_cum"] = (
        df["strategy_rtn"].cumsum().apply(np.exp)
    ) 
    ```

1.  添加买入并持有策略进行比较：

    ```py
    df["bh_rtn_cum"] = df["log_rtn"].cumsum().apply(np.exp) 
    ```

1.  绘制策略的累计收益：

    ```py
    (
        df[["bh_rtn_cum", "strategy_rtn_cum"]]
        .plot(title="Cumulative returns")
    ) 
    ```

    执行该代码片段生成了如下图表：

![](../Images/B18112_12_02.png)

图 12.2：我们的策略和买入并持有基准的累计收益

在 *图 12.2* 中，我们可以看到两种策略的累计收益。初步结论可能是，简单策略在考虑的时间段内表现优于买入并持有策略。然而，这种简化的回测形式并未考虑许多关键方面（例如，使用收盘价交易、假设没有滑点和交易成本等），这些因素可能会显著改变最终结果。在 *更多内容...* 部分，我们将看到当我们仅考虑交易成本时，结果如何迅速发生变化。

## 它是如何工作的……

一开始，我们导入了相关库并下载了 2016 到 2021 年间苹果公司的股票价格。我们只保留了调整后的收盘价用于回测。

在 *第 3 步* 中，我们计算了对数收益和 20 日 SMA。为了计算该技术指标，我们使用了 `pandas` DataFrame 的 `rolling` 方法。然而，我们也可以使用之前探讨过的 `TA-Lib` 库来实现。

我们计算了对数收益，因为它具有随着时间累加的便利性。如果我们持有仓位 10 天并关注该仓位的最终收益，我们可以简单地将这 10 天的对数收益相加。如需更多信息，请参见 *第 2 章*，*数据预处理*。

在 *第 4 步* 中，我们创建了一个列，用于标明我们是否有开仓（仅多头）或者没有开仓。正如我们决定的那样，当收盘价高于 20 日 SMA 时，我们开仓。当收盘价低于 SMA 时，我们平仓。我们还将该列编码为整数。在 *第 5 步* 中，我们绘制了收盘价、20 日 SMA 和包含持仓标志的列。为了使图表更加易读，我们只绘制了 2021 年的数据。

*第 6 步* 是向量化回测中最重要的一步。在这一部分，我们计算了策略的每日和累计收益。为了计算每日收益，我们将当天的对数收益与平移后的持仓标志相乘。为了避免未来数据偏差，持仓向量被平移了 1 天。换句话说，标志是利用截至并包括时间 *t* 的所有信息生成的。我们只能利用这些信息在下一个交易日（即时间 *t+1*）开仓。

一位好奇的读者可能已经发现我们回测中出现的另一种偏差。我们正确地假设只能在下一个交易日买入，然而，日志回报是按我们在 *t+1* 日使用 *t* 日的收盘价买入来计算的，这在某些市场条件下可能非常不准确。我们将在接下来的示例中看到如何通过事件驱动回测来克服这个问题。

然后，我们使用了 `cumsum` 方法计算日志回报的累计和，这对应于累计回报。最后，我们通过 `apply` 方法应用了指数函数。

在 *步骤 7* 中，我们计算了买入持有策略的累计回报。对于这个策略，我们只是使用了日志回报进行计算，省略了将回报与持仓标志相乘的步骤。

在最后一步，我们绘制了两种策略的累计回报。

## 还有更多...

从最初的回测结果来看，简单策略的表现优于买入持有策略。但我们也看到，在这 6 年中，我们已经进场 56 次。总交易次数翻倍，因为我们还退出了这些仓位。根据经纪商的不同，这可能导致相当可观的交易成本。

由于交易成本通常以固定百分比报价，我们可以简单地计算投资组合在连续时间步之间的变化量，基于此计算交易成本，然后直接从策略回报中减去这些成本。

在下面的步骤中，我们展示了如何在向量化回测中考虑交易成本。为了简单起见，我们假设交易成本为 1%。

执行以下步骤以在向量化回测中考虑交易成本：

1.  计算每日交易成本：

    ```py
    TRANSACTION_COST = 0.01
    df["tc"] = df["position"].diff(1).abs() * TRANSACTION_COST 
    ```

    在这个代码片段中，我们计算了投资组合是否发生变化（绝对值，因为我们可能会进场或退出仓位），然后将该值乘以以百分比表示的交易成本。

1.  计算考虑交易成本后的策略表现：

    ```py
    df["strategy_rtn_cum_tc"] = (
        (df["strategy_rtn"] - df["tc"]).cumsum().apply(np.exp)
    ) 
    ```

1.  绘制所有策略的累计回报：

    ```py
    STRATEGY_COLS = ["bh_rtn_cum", "strategy_rtn_cum", 
                     "strategy_rtn_cum_tc"]
    (
        df
        .loc[:, STRATEGY_COLS]
        .plot(title="Cumulative returns")
    ) 
    ```

    执行代码片段会生成以下图形：

![](../Images/B18112_12_03.png)

图 12.3：所有策略的累计回报，包括考虑交易成本的策略

在考虑交易成本后，表现明显下降，甚至不如买入持有策略。而且为了公平起见，我们也应该考虑买入持有策略中的初始和终端交易成本，因为我们必须进行一次买入和一次卖出。

# 使用 backtrader 进行事件驱动回测

回测的第二种方法称为**事件驱动回测**。在这种方法中，回测引擎模拟交易环境的时间维度（你可以把它看作一个遍历时间的 `for` 循环，顺序执行所有操作）。这对回测施加了更多结构，包括使用历史日历来定义交易实际执行的时间、价格何时可用等。

基于事件驱动的回测旨在模拟执行某个策略时遇到的所有操作和约束，同时比向量化方法提供更多灵活性。例如，这种方法允许模拟订单执行中的潜在延迟、滑点成本等。在理想情况下，为事件驱动回测编写的策略可以轻松转换为适用于实时交易引擎的策略。

目前，有相当多的事件驱动回测库可供 Python 使用。本章介绍了其中一个最流行的库——`backtrader`。该框架的主要特性包括：

+   提供大量可用的技术指标（`backtrader` 还提供了对流行的 TA-Lib 库的封装）和绩效衡量标准。

+   容易构建和应用新的指标。

+   提供多个数据源（包括 Yahoo Finance 和 Nasdaq Data Link），并支持加载外部文件。

+   模拟许多真实经纪商的方面，如不同类型的订单（市价单、限价单、止损单）、滑点、佣金、做多/做空等。

+   对价格、技术指标、交易信号、绩效等进行全面和互动式的可视化。

+   与选定的经纪商进行实时交易。

对于这个食谱，我们考虑了一个基于简单移动平均的基础策略。实际上，它几乎与我们在前一个食谱中使用向量化方法回测的策略完全相同。该策略的逻辑如下：

+   当收盘价高于 20 日均线时，买入一股。

+   当收盘价低于 20 日均线且我们持有股票时，卖出它。

+   我们在任何给定时间只能持有最多一股。

+   不允许卖空。

我们使用 2021 年的苹果股票价格进行该策略的回测。

## 准备就绪

在这个食谱中（以及本章的其余部分），我们将使用两个用于打印日志的辅助函数——`get_action_log_string` 和 `get_result_log_string`。此外，我们将使用一个自定义的 `MyBuySell` 观察者来以不同颜色显示持仓标记。你可以在 GitHub 上的 `strategy_utils.py` 文件中找到这些辅助函数的定义。

在编写本文时，PyPI（Python 包索引）上提供的 `backtrader` 版本并非最新版本。通过简单的 `pip install backtrader` 命令安装时，会安装一个包含一些问题的版本，例如，无法正确加载来自 Yahoo Finance 的数据。为了解决这个问题，您应该从 GitHub 安装最新版本。您可以使用以下代码片段来完成安装：

```py
pip install git+https://github.com/mementum/backtrader.git#egg=backtrader 
```

## 如何实现...

执行以下步骤来使用事件驱动的方法回测一个简单的策略：

1.  导入库：

    ```py
    from datetime import datetime
    import backtrader as bt
    from backtrader_strategies.strategy_utils import * 
    ```

1.  从 Yahoo Finance 下载数据：

    ```py
    data = bt.feeds.YahooFinanceData(dataname="AAPL",
                                     fromdate=datetime(2021, 1, 1),
                                     todate=datetime(2021, 12, 31)) 
    ```

    为了使代码更具可读性，我们首先展示定义交易策略的类的一般轮廓，然后在以下子步骤中介绍各个方法。

1.  策略的模板如下所示：

    ```py
    class  SmaStrategy(bt.Strategy):
        params = (("ma_period", 20), )
        def  __init__(self):
            # some code

        def  log(self, txt):
            # some code
        def  notify_order(self, order):
            # some code
        def  notify_trade(self, trade):
            # some code
        def  next(self):
            # some code
        def  start(self):
            # some code
        def  stop(self):
            # some code 
    ```

    1.  `__init__` 方法定义如下：

        ```py
        def  __init__(self):
            # keep track of close price in the series
            self.data_close = self.datas[0].close
            # keep track of pending orders
            self.order = None
            # add a simple moving average indicator
            self.sma = bt.ind.SMA(self.datas[0],
                                  period=self.params.ma_period) 
        ```

    1.  `log` 方法定义如下：

        ```py
        def  log(self, txt):
            dt = self.datas[0].datetime.date(0).isoformat()
            print(f"{dt}: {txt}") 
        ```

    1.  `notify_order` 方法定义如下：

        ```py
        def  notify_order(self, order):
            if order.status in [order.Submitted, order.Accepted]:
                # order already submitted/accepted
         # no action required
                return
            # report executed order
            if order.status in [order.Completed]:
                direction = "b" if order.isbuy() else "s"
                log_str = get_action_log_string(
                    dir=direction,
                    action="e",
                    price=order.executed.price,
                    size=order.executed.size,
                    cost=order.executed.value,
                    commission=order.executed.comm
                )
                self.log(log_str)
            # report failed order
            elif order.status in [order.Canceled, order.Margin,
                                  order.Rejected]:
                self.log("Order Failed")
            # reset order -> no pending order
            self.order = None 
        ```

    1.  `notify_trade` 方法定义如下：

        ```py
        def  notify_trade(self, trade): 
            if not trade.isclosed: 
                return 

            self.log( 
                get_result_log_string(
                    gross=trade.pnl, net=trade.pnlcomm
                ) 
            ) 
        ```

    1.  `next` 方法定义如下：

        ```py
        def  next(self):
            # do nothing if an order is pending
            if self.order:
                return

            # check if there is already a position
            if not self.position:
                # buy condition
                if self.data_close[0] > self.sma[0]:
                    self.log(
                        get_action_log_string(
                            "b", "c", self.data_close[0], 1
                        )
                    )
                    self.order = self.buy()
            else:
                # sell condition
                if self.data_close[0] < self.sma[0]:
                    self.log(
                        get_action_log_string(
                            "s", "c", self.data_close[0], 1
                        )
                    )      
                    self.order = self.sell() 
        ```

    1.  `start` 和 `stop` 方法定义如下：

        ```py
        def  start(self):
            print(f"Initial Portfolio Value: {self.broker.get_value():.2f}")
        def  stop(self):
            print(f"Final Portfolio Value: {self.broker.get_value():.2f}") 
        ```

1.  设置回测：

    ```py
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(data)
    cerebro.broker.setcash(1000.0)
    cerebro.addstrategy(SmaStrategy)
    cerebro.addobserver(MyBuySell)
    cerebro.addobserver(bt.observers.Value) 
    ```

1.  运行回测：

    ```py
    cerebro.run() 
    ```

    运行该代码片段会生成以下（简化的）日志：

    ```py
    Initial Portfolio Value: 1000.00
    2021-02-01: BUY CREATED - Price: 133.15, Size: 1.00
    2021-02-02: BUY EXECUTED - Price: 134.73, Size: 1.00, Cost: 134.73, Commission: 0.00
    2021-02-11: SELL CREATED - Price: 134.33, Size: 1.00
    2021-02-12: SELL EXECUTED - Price: 133.56, Size: -1.00, Cost: 134.73, Commission: 0.00
    2021-02-12: OPERATION RESULT - Gross: -1.17, Net: -1.17
    2021-03-16: BUY CREATED - Price: 124.83, Size: 1.00
    2021-03-17: BUY EXECUTED - Price: 123.32, Size: 1.00, Cost: 123.32, Commission: 0.00
    ...
    2021-11-11: OPERATION RESULT - Gross: 5.39, Net: 5.39
    2021-11-12: BUY CREATED - Price: 149.80, Size: 1.00
    2021-11-15: BUY EXECUTED - Price: 150.18, Size: 1.00, Cost: 150.18, Commission: 0.00
    Final Portfolio Value: 1048.01 
    ```

    日志包含关于所有已创建和执行的交易的信息，以及在头寸被关闭时的操作结果。

1.  绘制结果：

    ```py
    cerebro.plot(iplot=True, volume=False) 
    ```

    运行该代码片段会生成以下图表：

![](../Images/B18112_12_04.png)

图 12.4：回测期间我们策略行为/表现的总结

在 *图 12.4* 中，我们可以看到苹果公司的股票价格、20日简单移动平均线（SMA）、买卖订单，以及我们投资组合价值随时间的变化。正如我们所见，该策略在回测期间赚取了 48 美元。在考虑绩效时，请记住，该策略仅操作单一股票，同时将大部分可用资源保持为现金。

## 它是如何工作的...

使用 `backtrader` 的关键概念是，回测的核心大脑是 `Cerebro`，通过使用不同的方法，我们为它提供历史数据、设计的交易策略、我们希望计算的附加指标（例如投资期内的投资组合价值，或者整体的夏普比率）、佣金/滑点信息等。

创建策略有两种方式：使用信号（`bt.Signal`）或定义完整的策略（`bt.Strategy`）。这两种方式会产生相同的结果，然而，较长的方式（通过 `bt.Strategy` 创建）会提供更多关于实际发生的操作的日志记录。这使得调试更容易，并且能够跟踪所有操作（日志的详细程度取决于我们的需求）。因此，我们在本篇中首先展示这种方法。

您可以在本书的 GitHub 仓库中找到使用信号方法构建的等效策略。

在 *步骤 1* 中导入库和辅助函数之后，我们使用 `bt.feeds.YahooFinanceData` 函数从 Yahoo Finance 下载了价格数据。

你还可以从CSV文件、`pandas` DataFrame、纳斯达克数据链接（Nasdaq Data Link）以及其他来源添加数据。有关可用选项的列表，请参考`bt.feeds`的文档。我们在GitHub的Notebook中展示了如何从`pandas` DataFrame加载数据。

在*步骤3*中，我们将交易策略定义为继承自`bt.Strategy`的类。在类中，我们定义了以下方法（我们实际上是覆盖了这些方法，以便根据我们的需求量身定制）：

+   `__init__`：在此方法中，我们定义了希望跟踪的对象。在我们的示例中，这些对象包括收盘价、订单的占位符和技术分析指标（SMA）。

+   `log`：此方法用于日志记录，记录日期和提供的字符串。我们使用辅助函数`get_action_log_string`和`get_result_log_string`来创建包含各种订单相关信息的字符串。

+   `notify_order`：此方法报告订单（仓位）的状态。通常，在第*t*天，指标可以根据收盘价建议开盘/平仓（假设我们使用的是日数据）。然后，市场订单将在下一个交易日（使用*t+1*时刻的开盘价）执行。然而，无法保证订单会被执行，因为它可能会被取消或我们可能没有足够的现金。此方法还通过设置`self.order = None`来取消任何挂单。

+   `notify_trade`：此方法报告交易结果（在仓位关闭后）。

+   `next`：此方法包含了交易策略的逻辑。首先，我们检查是否已有挂单，如果有则不做任何操作。第二个检查是查看是否已有仓位（由我们的策略强制执行，这不是必须的），如果没有仓位，我们检查收盘价是否高于移动平均线。如果结果是正面的，我们将进入日志并使用`self.order = self.buy()`下达买入订单。这也是我们选择购买数量（我们想要购买的资产数量）的地方。默认值为1（等同于使用`self.buy(size=1)`）。

+   `start`/`stop`：这些方法在回测的开始/结束时执行，可以用于报告投资组合的价值等操作。

在*步骤4*中，我们设置了回测，即执行了一系列与Cerebro相关的操作：

+   我们创建了`bt.Cerebro`的实例并设置`stdstats=False`，以抑制许多默认的图表元素。这样，我们避免了输出的冗余，而是手动选择了感兴趣的元素（观察者和指标）。

+   我们使用`adddata`方法添加了数据。

+   我们使用`broker`的`setcash`方法设置了可用资金的数量。

+   我们使用`addstrategy`方法添加了策略。

+   我们使用 `addobserver` 方法添加了观察者。我们选择了两个观察者：自定义的 `BuySell` 观察者，用于在图表上显示买入/卖出决策（由绿色和红色三角形表示），以及 `Value` 观察者，用于跟踪投资组合价值随时间的变化。

最后一步是通过 `cerebro.run()` 运行回测，并通过 `cerebro.plot()` 绘制结果。在这一步骤中，我们禁用了显示交易量图表，以避免图表杂乱。

关于使用 `backtrader` 进行回测的几点补充说明：

+   根据设计，`Cerebro` 应该只使用一次。如果我们想要运行另一个回测，应该创建一个新的实例，而不是在开始计算后再往其中添加内容。

+   通常，使用 `bt.Signal` 构建的策略只使用一个信号。然而，我们可以通过使用 `bt.SignalStrategy` 来基于不同的条件组合多个信号。

+   如果我们没有特别指定，所有订单都将以一个单位的资产进行。

+   `backtrader` 会自动处理热身期。在此期间，无法进行交易，直到有足够的数据点来计算 20 天的简单移动平均线（SMA）。当同时考虑多个指标时，`backtrader` 会自动选择最长的必要周期。

## 还有更多...

值得一提的是，`backtrader` 具有参数优化功能，以下代码展示了这一功能。该代码是本策略的修改版本，我们优化了用于计算 SMA 的天数。

在调整策略参数值时，您可以创建一个简化版的策略，不记录过多信息（例如起始值、创建/执行订单等）。您可以在 `sma_strategy_optimization.py` 脚本中找到修改后策略的示例。

以下列表提供了代码修改的详细信息（我们只展示相关部分，因为大部分代码与之前使用的代码相同）：

+   我们不使用 `cerebro.addstrategy`，而是使用 `cerebro.optstrategy`，并提供定义的策略对象和参数值范围：

    ```py
    cerebro.optstrategy(SmaStrategy, ma_period=range(10, 31)) 
    ```

+   我们修改了 `stop` 方法，使其也记录 `ma_period` 参数的考虑值。

+   在运行扩展回测时，我们增加了 CPU 核心数：

    ```py
    cerebro.run(maxcpus=4) 
    ```

我们在以下总结中展示了结果（请记住，当使用多个核心时，参数的顺序可能会被打乱）：

```py
2021-12-30: (ma_period = 10) --- Terminal Value: 1018.82
2021-12-30: (ma_period = 11) --- Terminal Value: 1022.45
2021-12-30: (ma_period = 12) --- Terminal Value: 1022.96
2021-12-30: (ma_period = 13) --- Terminal Value: 1032.44
2021-12-30: (ma_period = 14) --- Terminal Value: 1027.37
2021-12-30: (ma_period = 15) --- Terminal Value: 1030.53
2021-12-30: (ma_period = 16) --- Terminal Value: 1033.03
2021-12-30: (ma_period = 17) --- Terminal Value: 1038.95
2021-12-30: (ma_period = 18) --- Terminal Value: 1043.48
2021-12-30: (ma_period = 19) --- Terminal Value: 1046.68
2021-12-30: (ma_period = 20) --- Terminal Value: 1048.01
2021-12-30: (ma_period = 21) --- Terminal Value: 1044.00
2021-12-30: (ma_period = 22) --- Terminal Value: 1046.98
2021-12-30: (ma_period = 23) --- Terminal Value: 1048.62
2021-12-30: (ma_period = 24) --- Terminal Value: 1051.08
2021-12-30: (ma_period = 25) --- Terminal Value: 1052.44
2021-12-30: (ma_period = 26) --- Terminal Value: 1051.30
2021-12-30: (ma_period = 27) --- Terminal Value: 1054.78
2021-12-30: (ma_period = 28) --- Terminal Value: 1052.75
2021-12-30: (ma_period = 29) --- Terminal Value: 1045.74
2021-12-30: (ma_period = 30) --- Terminal Value: 1047.60 
```

我们发现，当使用 27 天计算 SMA 时，策略表现最佳。

我们应该始终牢记，调整策略的超参数会带来更高的过拟合风险！

## 另见

您可以参考以下书籍，以获取有关算法交易和构建成功交易策略的更多信息：

+   Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale* (第 625 卷)。John Wiley & Sons 出版社。

# 基于 RSI 的多空策略回测

**相对强弱指数** (**RSI**) 是一个利用资产的收盘价来识别超买/超卖状态的指标。通常，RSI使用14天的时间段进行计算，范围从0到100（它是一个振荡器）。交易者通常在RSI低于30时买入资产（超卖），在RSI高于70时卖出资产（超买）。较极端的高/低水平，如80-20，使用得较少，并且通常意味着更强的市场动能。

在这个示例中，我们构建了一个遵循以下规则的交易策略：

+   我们可以同时做多仓和空仓。

+   计算RSI时，我们使用14个周期（交易日）。

+   当RSI突破下限（标准值为30）向上时，进入多仓；当RSI大于中位数（值为50）时，退出仓位。

+   当RSI突破上限（标准值为70）向下时，进入空仓；当RSI小于50时，退出仓位。

+   一次只能开一个仓位。

我们在2021年对Meta的股票进行策略评估，并应用0.1%的佣金。

## 如何操作……

执行以下步骤以实现并回测基于RSI的策略：

1.  导入库：

    ```py
    from datetime import datetime
    import backtrader as bt
    from backtrader_strategies.strategy_utils import * 
    ```

1.  基于`bt.SignalStrategy`定义信号策略：

    ```py
    class  RsiSignalStrategy(bt.SignalStrategy):
        params = dict(rsi_periods=14, rsi_upper=70,
                      rsi_lower=30, rsi_mid=50)
        def  __init__(self):       
            # add RSI indicator
            rsi = bt.indicators.RSI(period=self.p.rsi_periods,
                                    upperband=self.p.rsi_upper,
                                    lowerband=self.p.rsi_lower)
            # add RSI from TA-lib just for reference
            bt.talib.RSI(self.data, plotname="TA_RSI")

            # long condition (with exit)
            rsi_signal_long = bt.ind.CrossUp(
                rsi, self.p.rsi_lower, plot=False
            )
            self.signal_add(bt.SIGNAL_LONG, rsi_signal_long)
            self.signal_add(
                bt.SIGNAL_LONGEXIT, -(rsi > self.p.rsi_mid)
            )

            # short condition (with exit)
            rsi_signal_short = -bt.ind.CrossDown(
                rsi, self.p.rsi_upper, plot=False
            )
            self.signal_add(bt.SIGNAL_SHORT, rsi_signal_short)
            self.signal_add(
                bt.SIGNAL_SHORTEXIT, rsi < self.p.rsi_mid
            ) 
    ```

1.  下载数据：

    ```py
    data = bt.feeds.YahooFinanceData(dataname="META",
                                     fromdate=datetime(2021, 1, 1),
                                     todate=datetime(2021, 12, 31)) 
    ```

1.  设置并运行回测：

    ```py
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(RsiSignalStrategy)
    cerebro.adddata(data)
    cerebro.addsizer(bt.sizers.SizerFix, stake=1)
    cerebro.broker.setcash(1000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addobserver(MyBuySell)
    cerebro.addobserver(bt.observers.Value)
    print(
        f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}"
    )
    cerebro.run()
    print(
        f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}"
    ) 
    ```

    运行代码片段后，我们看到以下输出：

    ```py
    Starting Portfolio Value: 1000.00
    Final Portfolio Value: 1042.56 
    ```

1.  绘制结果：

    ```py
    cerebro.plot(iplot=True, volume=False) 
    ```

    运行代码片段生成如下图表：

![](../Images/B18112_12_05.png)

图12.5：我们策略在回测期间的行为/表现总结

我们观察成对的三角形。每对中的第一个三角形表示开仓（如果三角形是绿色且朝上，则开多仓；如果三角形是红色且朝下，则开空仓）。接下来的相反方向的三角形表示平仓。我们可以将开仓和平仓与图表下方的RSI对照。有时，同色的三角形会连续出现。这是因为RSI在开仓线附近波动，多次穿越该线。但实际的开仓仅发生在信号首次出现时（默认情况下，所有回测不进行累积）。

## 它是如何工作的……

在这个示例中，我们展示了在`backtrader`中定义策略的第二种方法，即使用信号。信号表现为一个数字，例如当前数据点与某个技术分析指标的差值。如果信号为正，表示开多仓（买入）。如果信号为负，表示开空仓（卖出）。信号值为0表示没有信号。

导入库和辅助函数后，我们使用`bt.SignalStrategy`定义了交易策略。由于这是一个涉及多个信号（各种进出场条件）的策略，我们不得不使用`bt.SignalStrategy`而不是简单的`bt.Signal`。首先，我们定义了指标（RSI），并选择了相应的参数。我们还添加了第二个RSI指标实例，仅仅是为了展示`backtrader`提供了一个简便的方式来使用流行的TA-Lib库中的指标（必须安装该库才能使代码正常工作）。该交易策略并不依赖于第二个指标——它仅用于参考绘图。通常来说，我们可以添加任意数量的指标。

即使仅添加指标作为参考，它们的存在也会影响“预热期”。例如，如果我们额外包括了一个200日SMA指标，则在SMA指标至少有一个值之前，任何交易都不会执行。

下一步是定义信号。为此，我们使用了`bt.CrossUp`/`bt.CrossDown`指标，当第一个系列（价格）从下方/上方穿越第二个系列（RSI的上限或下限）时，分别返回1。为了进入空仓，我们通过在`bt.CrossDown`指标前加上负号来使信号变为负值。

我们可以通过在函数调用中添加`plot=False`来禁用任何指标的打印。

以下是可用信号类型的描述：

+   `LONGSHORT`: 此类型同时考虑了来自信号的多仓和空仓指示。

+   `LONG`: 正向信号表示开多仓；负向信号用于平多仓。

+   `SHORT`: 负向信号表示开空仓；正向信号用于平空仓。

+   `LONGEXIT`: 负向信号用于平多仓。

+   `SHORTEXIT`: 正向信号用于平空仓。

平仓可能更为复杂，这反过来允许用户构建更复杂的策略。我们在下面描述了其逻辑：

+   `LONG`: 如果出现`LONGEXIT`信号，则用于平多仓，而不是上面提到的行为。如果出现`SHORT`信号且没有`LONGEXIT`信号，则使用`SHORT`信号先平多仓，然后再开空仓。

+   `SHORT`: 如果出现`SHORTEXIT`信号，则用于平空仓，而不是上面提到的行为。如果出现`LONG`信号且没有`SHORTEXIT`信号，则使用`LONG`信号先平空仓，然后再开多仓。

正如你可能已经意识到的，信号会在每个时间点计算（如图表底部所示），这实际上会创建一个连续的开盘/平仓信号流（信号值为0的情况不太可能发生）。因此，`backtrader`默认禁用累积（即使已有仓位，也不断开新仓）和并发（在没有收到经纪商反馈之前生成新订单）。

在定义策略的最后一步，我们通过使用`signal_add`方法跟踪所有信号。对于平仓，我们使用的条件（RSI值高于/低于50）会产生一个布尔值，当退出多头仓位时，我们必须将其取反：在Python中，`-True`与`-1`的意义相同。

在*步骤3*中，我们下载了2021年Meta的股票价格。

然后，我们设置了回测。大部分步骤应该已经很熟悉了，因此我们只关注新的部分：

+   使用`addsizer`方法添加一个Sizer——在这一点上我们不必这么做，因为`backtrader`默认使用1的头寸，也就是说，每次交易会买卖1单位资产。然而，我们希望展示在使用信号法创建交易策略时，在哪个时刻可以修改订单大小。

+   使用`broker`的`setcommission`方法将佣金设置为0.1%。

+   我们还在回测运行前后访问并打印了投资组合的当前价值。为此，我们使用了`broker`的`getvalue`方法。

在最后一步，我们绘制了回测结果。

## 还有更多……

在这个示例中，我们向回测框架引入了几个新概念——Sizer和佣金。使用这两个组件，我们可以进行更多有趣的实验。

### 全部押注

之前，我们的简单策略仅仅是以单个单位的资产进行多头或空头操作。然而，我们可以轻松修改这一行为，利用所有可用的现金。我们只需通过`addsizer`方法添加`AllInSizer` Sizer：

```py
cerebro = bt.Cerebro(stdstats=False)
cerebro.addstrategy(RsiSignalStrategy)
cerebro.adddata(data)
cerebro.addsizer(bt.sizers.AllInSizer)
cerebro.broker.setcash(1000.0)
cerebro.broker.setcommission(commission=0.001)
cerebro.addobserver(bt.observers.Value)
print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
cerebro.run()
print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}") 
```

运行回测生成了以下结果：

```py
Starting Portfolio Value: 1000.00
Final Portfolio Value: 1183.95 
```

结果显然比我们每次只使用单个单位时的表现要好。

### 每股固定佣金

在我们对基于RSI的策略进行初步回测时，我们使用了0.1%的佣金费用。然而，一些经纪商可能有不同的佣金方案，例如，每股固定佣金。

为了融入这些信息，我们需要定义一个自定义类来存储佣金方案。我们可以从`bt.CommInfoBase`继承并添加所需的信息：

```py
class  FixedCommissionShare(bt.CommInfoBase):
    """
 Scheme with fixed commission per share
 """
    params = (
        ("commission", 0.03),
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_FIXED),
    )
    def  _getcommission(self, size, price, pseudoexec):
        return abs(size) * self.p.commission 
```

定义中的最重要的方面是每股固定佣金为0.03美元以及在`_getcommission`方法中计算佣金的方式。我们取大小的绝对值，并将其乘以固定佣金。

然后，我们可以轻松地将这些信息输入回测。在前面的“全部投入”策略示例的基础上，代码如下所示：

```py
cerebro = bt.Cerebro(stdstats=False)
cerebro.addstrategy(RsiSignalStrategy)
cerebro.adddata(data)
cerebro.addsizer(bt.sizers.AllInSizer)
cerebro.broker.setcash(1000.0)
cerebro.broker.addcommissioninfo(FixedCommissionShare())
cerebro.addobserver(bt.observers.Value)
print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
cerebro.run()
print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}") 
```

结果如下：

```py
Starting Portfolio Value: 1000.00
Final Portfolio Value: 1189.94 
```

这些数字得出结论：0.01%的佣金实际上比每股3美分还要高。

### 每单固定佣金

其他经纪商可能会提供每单固定的佣金。在以下代码片段中，我们定义了一个自定义佣金方案，每单支付2.5美元，不管订单大小。

我们更改了`commission`参数的值以及在`_getcommission`方法中佣金的计算方式。这一次，该方法始终返回我们之前指定的2.5美元：

```py
class  FixedCommissionOrder(bt.CommInfoBase):
    """
 Scheme with fixed commission per order
 """
    params = (
        ("commission", 2.5),
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_FIXED),
    )
    def  _getcommission(self, size, price, pseudoexec):
        return self.p.commission 
```

我们不包括回测设置，因为它几乎与之前的相同。我们只需要通过`addcommissioninfo`方法传递一个不同的类。回测结果是：

```py
Starting Portfolio Value: 1000.00
Final Portfolio Value: 1174.70 
```

## 另见

以下是一些有用的`backtrader`文档参考：

+   要了解更多关于资金分配器的信息：[https://www.backtrader.com/docu/sizers-reference/](https://www.backtrader.com/docu/sizers-reference/)

+   要了解更多关于佣金方案和可用参数的信息：[https://www.backtrader.com/docu/commission-schemes/commission-schemes/](https://www.backtrader.com/docu/commission-schemes/commission-schemes/)

# 基于布林带的买卖策略回测

布林带是一种统计方法，用于推导某一资产在一段时间内的价格和波动性信息。为了得到布林带，我们需要计算时间序列（价格）的移动平均和标准差，使用指定的窗口（通常为20天）。然后，我们将上/下布林带设置为移动标准差的K倍（通常为2），分别位于移动平均值的上下方。

布林带的解释非常简单：带宽随着波动性的增加而变宽，随着波动性的减少而收窄。

在这个示例中，我们构建了一个简单的交易策略，利用布林带识别超买和超卖的水平，然后基于这些区域进行交易。策略规则如下：

+   当价格向上突破下布林带时，进行买入。

+   当价格向下突破上布林带时，卖出（仅当持有股票时）。

+   全部投入策略——在创建买入订单时，尽可能购买尽量多的股票。

+   不允许卖空。

我们评估了2021年微软股票的策略。此外，我们将佣金设置为0.1%。

## 如何实现……

执行以下步骤来实现并回测一个基于布林带的策略：

1.  导入库：

    ```py
    import backtrader as bt
    import datetime
    import pandas as pd
    from backtrader_strategies.strategy_utils import * 
    ```

    为了让代码更具可读性，我们首先展示定义交易策略的类的大致框架，然后在接下来的子步骤中介绍各个方法。

1.  基于布林带定义策略：

    ```py
    class  BollingerBandStrategy(bt.Strategy):
        params = (("period", 20),
                  ("devfactor", 2.0),)
        def  __init__(self):
            # some code
        def  log(self, txt):
            # some code
        def  notify_order(self, order):
            # some code
        def  notify_trade(self, trade):
            # some code
        def  next_open(self):
            # some code
        def  start(self):
            print(f"Initial Portfolio Value: {self.broker.get_value():.2f}")
        def  stop(self):
            print(f"Final Portfolio Value: {self.broker.get_value():.2f}") 
    ```

    使用策略方法定义策略时，有相当多的样板代码。因此，在以下子步骤中，我们只提及与之前解释的不同的方法。你也可以在本书的GitHub仓库中找到策略的完整代码：

    1.  `__init__`方法定义如下：

        ```py
        def  __init__(self):
            # keep track of prices
            self.data_close = self.datas[0].close
            self.data_open = self.datas[0].open

            # keep track of pending orders
            self.order = None

            # add Bollinger Bands indicator and track buy/sell
         # signals
            self.b_band = bt.ind.BollingerBands(
                self.datas[0], 
                period=self.p.period, 
                devfactor=self.p.devfactor
            )
            self.buy_signal = bt.ind.CrossOver(
                self.datas[0], 
                self.b_band.lines.bot,
                plotname="buy_signal"
            )
            self.sell_signal = bt.ind.CrossOver(
                self.datas[0], 
                self.b_band.lines.top,
                plotname="sell_signal"
            ) 
        ```

    1.  `next_open`方法定义如下：

        ```py
        def  next_open(self):
            if not self.position:
                if self.buy_signal > 0:
                    # calculate the max number of shares ("all-in")
                    size = int(
                        self.broker.getcash() / self.datas[0].open
                    )
                    # buy order
                    log_str = get_action_log_string(
                        "b", "c", 
                        price=self.data_close[0], 
                        size=size,
                        cash=self.broker.getcash(),
                        open=self.data_open[0],
                        close=self.data_close[0]
                    )
                    self.log(log_str)
                    self.order = self.buy(size=size)
            else:
                if self.sell_signal < 0:
                    # sell order
                    log_str = get_action_log_string(
                        "s", "c", self.data_close[0], 
                        self.position.size
                    )
                    self.log(log_str)
                    self.order = self.sell(size=self.position.size) 
        ```

1.  下载数据：

    ```py
    data = bt.feeds.YahooFinanceData(
        dataname="MSFT",
        fromdate=datetime.datetime(2021, 1, 1),
        todate=datetime.datetime(2021, 12, 31)
    ) 
    ```

1.  设置回测：

    ```py
    cerebro = bt.Cerebro(stdstats=False, cheat_on_open=True)
    cerebro.addstrategy(BollingerBandStrategy)
    cerebro.adddata(data)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addobserver(MyBuySell)
    cerebro.addobserver(bt.observers.Value)
    cerebro.addanalyzer(
        bt.analyzers.Returns, _name="returns"
    )
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn, _name="time_return"
    ) 
    ```

1.  运行回测：

    ```py
    backtest_result = cerebro.run() 
    ```

    运行回测生成以下（简化版）日志：

    ```py
    Initial Portfolio Value: 10000.00
    2021-03-01: BUY CREATED - Price: 235.03, Size: 42.00, Cash: 10000.00, Open: 233.99, Close: 235.03
    2021-03-01: BUY EXECUTED - Price: 233.99, Size: 42.00, Cost: 9827.58, Commission: 9.83
    2021-04-13: SELL CREATED - Price: 256.40, Size: 42.00
    2021-04-13: SELL EXECUTED - Price: 255.18, Size: -42.00, Cost: 
    9827.58, Commission: 10.72
    2021-04-13: OPERATION RESULT - Gross: 889.98, Net: 869.43
    …
    2021-12-07: BUY CREATED - Price: 334.23, Size: 37.00, Cash: 12397.10, Open: 330.96, Close: 334.23
    2021-12-07: BUY EXECUTED - Price: 330.96, Size: 37.00, Cost: 12245.52, Commission: 12.25
    Final Portfolio Value: 12668.27 
    ```

1.  绘制结果：

    ```py
    cerebro.plot(iplot=True, volume=False) 
    ```

    运行代码片段会生成以下图表：

    ![](../Images/B18112_12_06.png)

    图12.6：我们策略在回测期间的行为/表现总结

    我们可以看到，即使考虑到佣金成本，策略也能赚钱。投资组合价值中的平坦期代表我们没有持仓的时期。

1.  调查不同的回报度量：

    ```py
    backtest_result[0].analyzers.returns.get_analysis() 
    ```

    运行代码生成以下输出：

    ```py
    OrderedDict([('rtot', 0.2365156915893157),
                 ('ravg', 0.0009422935919893056),
                 ('rnorm', 0.2680217199688534),
                 ('rnorm100', 26.80217199688534)]) 
    ```

1.  提取每日投资组合回报并绘制：

    ```py
    returns_dict = (
        backtest_result[0].analyzers.time_return.get_analysis()
    )
    returns_df = (
        pd.DataFrame(list(returns_dict.items()), 
                     columns = ["date", "return"])
        .set_index("date")
    )
    returns_df.plot(title="Strategy's daily returns") 
    ```

![](../Images/B18112_12_07.png)

图12.7：基于布林带的策略的每日投资组合回报

我们可以看到，投资组合回报中的平坦期（见*图12.7*）与我们没有持仓的时期相对应，正如在*图12.6*中所示。

## 它是如何工作的...

创建基于布林带的策略所用的代码与之前配方中的代码有很多相似之处。这就是为什么我们只讨论新颖之处，并将更多细节参考给*基于事件驱动的回测（使用backtrader）*配方的原因。

由于我们在这个策略中进行了全力投资，因此我们必须使用一种名为`cheat_on_open`的方法。这意味着我们使用*第t*天的收盘价计算信号，但根据*第t+1*天的开盘价计算我们希望购买的股份数量。为此，在实例化`Cerebro`对象时，我们需要设置`cheat_on_open=True`。

因此，我们还在`Strategy`类中定义了一个`next_open`方法，而不是使用`next`。这明确地向`Cerebro`表明我们在开盘时进行了作弊。在创建潜在的买单之前，我们手动计算了使用*第t+1天*的开盘价可以购买的最大股份数量。

在基于布林带计算买入/卖出信号时，我们使用了`CrossOver`指标。它返回了以下内容：

+   如果第一组数据（价格）上穿第二组数据（指标），则返回1

+   如果第一组数据（价格）下穿第二组数据（指标），则返回-1

我们还可以使用`CrossUp`和`CrossDown`函数，当我们只希望考虑单向穿越时。买入信号如下：`self.buy_signal = bt.ind.CrossUp(self.datas[0], self.b_band.lines.bot)`。

最后的补充内容包括使用分析器——`backtrader`对象，帮助评估投资组合的表现。在此配方中，我们使用了两个分析器：

+   `Returns`：一组不同的对数收益率，计算覆盖整个时间范围：总复合收益率、整个期间的平均收益率和年化收益率。

+   `TimeReturn`：一组随时间变化的收益率（使用提供的时间范围，在本例中为每日数据）。

我们可以通过添加一个同名观察器来获得与`TimeReturn`分析器相同的结果：`cerebro.addobserver(bt.observers.TimeReturn)`。唯一的区别是观察器会显示在主要结果图表上，这并非总是我们所希望的。

## 还有更多内容……

我们已经看到如何从回测中提取每日收益率。这为将这些信息与`quantstats`库的功能结合提供了一个绝佳机会。使用以下代码片段，我们可以计算多种指标，详细评估我们的投资组合表现。此外，我们还会将策略表现与简单的买入并持有策略进行对比（为了简化，买入并持有策略没有包括交易成本）：

```py
import quantstats as qs
qs.reports.metrics(returns_df,
                   benchmark="MSFT",
                   mode="basic") 
```

运行该代码片段将生成以下报告：

```py
 Strategy    Benchmark
------------------  ----------  -----------
Start Period        2021-01-04  2021-01-04
End Period          2021-12-30  2021-12-30
Risk-Free Rate      0.0%        0.0%
Time in Market      42.0%       100.0%
Cumulative Return   26.68%      57.18%
CAGR﹪              27.1%       58.17%
Sharpe              1.65        2.27
Sortino             2.68        3.63
Sortino/√2          1.9         2.57
Omega               1.52        1.52 
```

为简洁起见，我们只展示报告中可用的几条主要信息。

在*第11章*，*资产配置*中，我们提到过，`quantstats`的一个替代库是`pyfolio`。后者的潜在缺点是它已经不再积极维护。然而，`pyfolio`与`backtrader`的集成非常好。我们可以轻松添加一个专用分析器（`bt.analyzers.PyFolio`）。有关实现的示例，请参见本书的GitHub仓库。

# 使用加密数据回测移动平均交叉策略

到目前为止，我们已经创建并回测了几种股票策略。在本篇中，我们讨论了另一类流行的资产——加密货币。处理加密数据时有一些关键的区别：

+   加密货币可以进行24/7交易。

+   加密货币可以使用部分单位进行交易。

由于我们希望回测尽可能接近真实交易，因此我们应当在回测中考虑这些加密货币特有的特点。幸运的是，`backtrader`框架非常灵活，我们可以稍微调整已有的方法来处理这种新资产类别。

一些经纪商也允许购买股票的部分股份。

在本篇中，我们回测了一种移动平均交叉策略，规则如下：

+   我们只关心比特币，并使用来自2021年的每日数据。

+   我们使用两种不同的移动平均线，窗口期分别为20天（快速）和50天（慢速）。

+   如果快速移动平均线向上穿越慢速移动平均线，我们将把70%的可用现金分配用于购买比特币。

+   如果快速移动平均线下穿慢速移动平均线，我们会卖出所有持有的比特币。

+   不允许进行卖空交易。

## 如何实现……

执行以下步骤来实现并回测基于移动平均交叉的策略：

1.  导入所需的库：

    ```py
    import backtrader as bt
    import datetime
    import pandas as pd
    from backtrader_strategies.strategy_utils import * 
    ```

1.  定义允许进行部分交易的佣金方案：

    ```py
    class  FractionalTradesCommission(bt.CommissionInfo):
        def  getsize(self, price, cash):
            """Returns the fractional size"""
            return self.p.leverage * (cash / price) 
    ```

    为了提高代码的可读性，我们首先展示定义交易策略的类的大纲，然后在以下的子步骤中介绍各个独立的方法。

1.  定义SMA交叉策略：

    ```py
    class  SMACrossoverStrategy(bt.Strategy):
        params = (
            ("ma_fast", 20),
            ("ma_slow", 50),
            ("target_perc", 0.7)
        )
        def  __init__(self):
            # some code

        def  log(self, txt):
            # some code
        def  notify_order(self, order):
           # some code
        def  notify_trade(self, trade):
            # some code
        def  next(self):
            # some code
        def  start(self):
            print(f"Initial Portfolio Value: {self.broker.get_value():.2f}")
        def  stop(self):
            print(f"Final Portfolio Value: {self.broker.get_value():.2f}") 
    ```

    1.  `__init__`方法定义如下：

        ```py
        def  __init__(self):
            # keep track of close price in the series
            self.data_close = self.datas[0].close

            # keep track of pending orders
            self.order = None

            # calculate the SMAs and get the crossover signal 
            self.fast_ma = bt.indicators.MovingAverageSimple(
                self.datas[0], 
                period=self.params.ma_fast
            )
            self.slow_ma = bt.indicators.MovingAverageSimple(
                self.datas[0], 
                period=self.params.ma_slow
            )
            self.ma_crossover = bt.indicators.CrossOver(self.fast_ma, 
                                                        self.slow_ma) 
        ```

    1.  `next`方法定义如下：

        ```py
        def  next(self):

            if self.order:
                # pending order execution. Waiting in orderbook
                return  
            if not self.position:
                if self.ma_crossover > 0:
                    self.order = self.order_target_percent(
                        target=self.params.target_perc
                    )
                    log_str = get_action_log_string(
                        "b", "c", 
                        price=self.data_close[0], 
                        size=self.order.size,
                        cash=self.broker.getcash(),
                        open=self.data_open[0],
                        close=self.data_close[0]
                    )
                    self.log(log_str)
            else:
                if self.ma_crossover < 0:
                    # sell order
                    log_str = get_action_log_string(
                        "s", "c", self.data_close[0], 
                        self.position.size
                    )
                    self.log(log_str)
                    self.order = (
                        self.order_target_percent(target=0)
                    ) 
        ```

1.  下载`BTC-USD`数据：

    ```py
    data = bt.feeds.YahooFinanceData(
        dataname="BTC-USD",
        fromdate=datetime.datetime(2020, 1, 1),
        todate=datetime.datetime(2021, 12, 31)
    ) 
    ```

1.  设置回测：

    ```py
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(SMACrossoverStrategy)
    cerebro.adddata(data)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.addcommissioninfo(
        FractionalTradesCommission(commission=0.001)
    )
    cerebro.addobserver(MyBuySell)
    cerebro.addobserver(bt.observers.Value)
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn, _name="time_return"
    ) 
    ```

1.  运行回测：

    ```py
    backtest_result = cerebro.run() 
    ```

    运行代码片段会生成以下（简化的）日志：

    ```py
    Initial Portfolio Value: 10000.00
    2020-04-19: BUY CREATED - Price: 7189.42, Size: 0.97, Cash: 10000.00, Open: 7260.92, Close: 7189.42
    2020-04-20: BUY EXECUTED - Price: 7186.87, Size: 0.97, Cost: 6997.52, Commission: 7.00
    2020-06-29: SELL CREATED - Price: 9190.85, Size: 0.97
    2020-06-30: SELL EXECUTED - Price: 9185.58, Size: -0.97, Cost: 6997.52, Commission: 8.94
    2020-06-30: OPERATION RESULT - Gross: 1946.05, Net: 1930.11
    …
    Final Portfolio Value: 43547.99 
    ```

    在完整日志的摘录中，我们可以看到现在我们正使用分数仓位进行操作。此外，策略已经产生了相当可观的回报——我们大约将初始投资组合的价值翻了四倍。

1.  绘制结果：

    ```py
    cerebro.plot(iplot=True, volume=False) 
    ```

    运行代码片段会生成以下图表：

![](../Images/B18112_12_08.png)

图12.8：我们的策略在回测期间的行为/表现总结

我们已经证明，使用我们的策略，我们获得了超过300%的回报。然而，我们也可以在*图12.8*中看到，出色的表现可能仅仅是因为在考虑的期间内，比特币（BTC）价格的大幅上涨。

使用与之前示例中相同的代码，我们可以将我们的策略与简单的买入并持有策略进行比较。通过这种方式，我们可以验证我们的主动策略与静态基准的表现差异。下面展示的是简化的表现对比，代码可以在书中的GitHub仓库找到。

```py
 Strategy    Benchmark
------------------  ----------  -----------
Start Period        2020-01-01  2020-01-01
End Period          2021-12-30  2021-12-30
Risk-Free Rate      0.0%        0.0%
Time in Market      57.0%       100.0%
Cumulative Return   335.48%     555.24%
CAGR﹪              108.89%     156.31%
Sharpe              1.6         1.35
Sortino             2.63        1.97
Sortino/√2          1.86        1.4
Omega               1.46        1.46 
```

不幸的是，我们的策略在分析的时间框架内并没有超过基准。这证实了我们最初的怀疑，即优异的表现与在考虑的期间内比特币价格的上涨有关。

## 它是如何工作的…

在导入库之后，我们定义了一个自定义的佣金方案，以允许分数股份。在之前创建自定义佣金方案时，我们是从`bt.CommInfoBase`继承，并修改了`_getcommission`方法。这一次，我们从`bt.CommissionInfo`继承，并修改了`getsize`方法，以根据可用现金和资产价格返回分数值。

在*步骤3*（及其子步骤）中，我们定义了移动平均交叉策略。通过这个示例，大部分代码应该已经非常熟悉。我们在这里应用的一个新概念是不同类型的订单，也就是`order_target_percent`。使用这种类型的订单表示我们希望给定资产在我们的投资组合中占有X%的比例。

这是一种非常方便的方法，因为我们将精确的订单大小计算交给了`backtrader`。如果在发出订单时，我们低于指定的目标百分比，我们将购买更多的资产；如果我们超过了该比例，我们将卖出一部分资产。

为了退出仓位，我们表示希望比特币（BTC）在我们的投资组合中占0%，这相当于卖出我们所有持有的比特币。通过使用目标为零的`order_target_percent`，我们无需跟踪/访问当前持有的单位数量。

在*步骤 4*中，我们下载了2021年每日的BTC价格（以美元计）。在接下来的步骤中，我们设置了回测，运行了回测，并绘制了结果。唯一值得提到的是，我们需要使用`addcommissioninfo`方法添加自定义佣金方案（包含部分股份逻辑）。  

## 还有更多内容…  

在本示例中，我们介绍了目标订单。`backtrader` 提供了三种类型的目标订单：  

+   `order_target_percent`：表示我们希望在给定资产中拥有的当前投资组合价值的百分比。

+   `order_target_size`：表示我们希望在投资组合中拥有的给定资产的目标单位数。  

+   `order_target_value`：表示我们希望在投资组合中拥有的资产目标金额（以货币单位表示）。  

目标订单在我们知道给定资产的目标百分比/价值/数量时非常有用，但不想花额外的时间计算是否应该购买更多单位或卖出它们以达到目标。  

还有一件关于部分股份的重要事情需要提到。在这个示例中，我们定义了一个自定义的佣金方案，考虑了部分股份的情况，然后使用目标订单来买入/卖出资产。这样，当引擎计算出为了达到目标需要交易的单位数量时，它知道可以使用部分值。  

然而，还有一种不需要定义自定义佣金方案的方式来使用部分股份。我们只需手动计算我们想要买入/卖出的股份数量，并创建一个给定份额的订单。在前一个示例中，我们做了类似的操作，但那时我们将潜在的部分值四舍五入为整数。有关手动部分订单大小计算的SMA交叉策略实现，请参考本书的GitHub仓库。  

# 对均值方差投资组合优化的回测  

在前一章中，我们讨论了资产配置和均值方差优化。将均值方差优化与回测结合起来将是一个有趣的练习，特别是因为它涉及同时处理多个资产。  

在这个示例中，我们回测了以下配置策略：  

+   我们考虑FAANG股票。  

+   每周五市场收盘后，我们找到切线投资组合（最大化Sharpe比率）。然后，在市场周一开盘时，我们创建目标订单来匹配计算出的最佳权重。  

+   我们假设需要至少252个数据点来计算预期收益和协方差矩阵（使用Ledoit-Wolf方法）。  

对于这个练习，我们下载了2020到2021年的FAANG股票价格。由于我们为计算权重设置的预热期，实际的交易只发生在2021年。  

## 准备工作  

由于在这个示例中我们将处理部分股份，我们需要使用在前一个示例中定义的自定义佣金方案（`FractionalTradesCommission`）。  

## 如何操作…  

执行以下步骤来实现并回测基于均值-方差投资组合优化的策略：

1.  导入库：

    ```py
    from datetime import datetime
    import backtrader as bt
    import pandas as pd
    from pypfopt.expected_returns import mean_historical_return
    from pypfopt.risk_models import CovarianceShrinkage
    from pypfopt.efficient_frontier import EfficientFrontier
    from backtrader_strategies.strategy_utils import * 
    ```

    为了提高代码的可读性，我们首先展示定义交易策略的类的一般框架，然后在以下子步骤中引入各个方法。

1.  定义策略：

    ```py
    class  MeanVariancePortfStrategy(bt.Strategy):
        params = (("n_periods", 252), )
        def  __init__(self):  
            # track number of days
            self.day_counter = 0

        def  log(self, txt):
            dt = self.datas[0].datetime.date(0).isoformat()
            print(f"{dt}: {txt}")
        def  notify_order(self, order):
            # some code
        def  notify_trade(self, trade):
            # some code
        def  next(self):
            # some code
        def  start(self):
            print(f"Initial Portfolio Value: {self.broker.get_value():.2f}")
        def  stop(self):
            print(f"Final Portfolio Value: {self.broker.get_value():.2f}") 
    ```

    1.  `next`方法定义如下：

        ```py
        def  next(self):
            # check if we have enough data points
            self.day_counter += 1
            if self.day_counter < self.p.n_periods:
                return

            # check if the date is a Friday
            today = self.datas[0].datetime.date()
            if today.weekday() != 4: 
                return

            # find and print the current allocation
            current_portf = {}
            for data in self.datas:
                current_portf[data._name] = (
                    self.positions[data].size * data.close[0]
                )

            portf_df = pd.DataFrame(current_portf, index=[0])
            print(f"Current allocation as of {today}")
            print(portf_df / portf_df.sum(axis=1).squeeze())

            # extract the past price data for each asset
            price_dict = {}
            for data in self.datas:
                price_dict[data._name] = (
                    data.close.get(0, self.p.n_periods+1)
                )
            prices_df = pd.DataFrame(price_dict)

            # find the optimal portfolio weights
            mu = mean_historical_return(prices_df)
            S = CovarianceShrinkage(prices_df).ledoit_wolf()
            ef = EfficientFrontier(mu, S)
            weights = ef.max_sharpe(risk_free_rate=0)
            print(f"Optimal allocation identified on {today}")
            print(pd.DataFrame(ef.clean_weights(), index=[0]))

            # create orders
            for allocation in list(ef.clean_weights().items()):
                self.order_target_percent(data=allocation[0],
                                          target=allocation[1]) 
        ```

1.  下载FAANG股票的价格并将数据源存储在列表中：

    ```py
    TICKERS = ["META", "AMZN", "AAPL", "NFLX", "GOOG"]
    data_list = []
    for ticker in TICKERS:
        data = bt.feeds.YahooFinanceData(
            dataname=ticker,
            fromdate=datetime(2020, 1, 1),
            todate=datetime(2021, 12, 31)
        )
        data_list.append(data) 
    ```

1.  设置回测：

    ```py
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(MeanVariancePortfStrategy)
    for ind, ticker in enumerate(TICKERS):
        cerebro.adddata(data_list[ind], name=ticker)
    cerebro.broker.setcash(1000.0)
    cerebro.broker.addcommissioninfo(
        FractionalTradesCommission(commission=0)
    )
    cerebro.addobserver(MyBuySell)
    cerebro.addobserver(bt.observers.Value) 
    ```

1.  运行回测：

    ```py
    backtest_result = cerebro.run() 
    ```

运行回测后会生成如下日志：

```py
Initial Portfolio Value: 1000.00
Current allocation as of 2021-01-08
  META  AMZN  AAPL  NFLX  GOOG
0 NaN   NaN   NaN   NaN   NaN
Optimal allocation identified on 2021-01-08
  META     AMZN     AAPL  NFLX  GOOG
0  0.0  0.69394  0.30606   0.0   0.0
2021-01-11: Order Failed: AAPL
2021-01-11: BUY EXECUTED - Price: 157.40, Size: 4.36, Asset: AMZN, Cost: 686.40, Commission: 0.00
Current allocation as of 2021-01-15
  META  AMZN  AAPL  NFLX  GOOG
0  0.0   1.0   0.0   0.0   0.0
Optimal allocation identified on 2021-01-15
  META     AMZN     AAPL  NFLX  GOOG
0  0.0  0.81862  0.18138   0.0   0.0
2021-01-19: BUY EXECUTED - Price: 155.35, Size: 0.86, Asset: AMZN, Cost: 134.08, Commission: 0.00
2021-01-19: Order Failed: AAPL
Current allocation as of 2021-01-22
  META  AMZN  AAPL  NFLX  GOOG
0  0.0   1.0   0.0   0.0   0.0
Optimal allocation identified on 2021-01-22
  META     AMZN     AAPL  NFLX  GOOG
0  0.0  0.75501  0.24499   0.0   0.0
2021-01-25: SELL EXECUTED - Price: 166.43, Size: -0.46, Asset: AMZN, Cost: 71.68, Commission: 0.00
2021-01-25: Order Failed: AAPL
...
0  0.0   0.0  0.00943   0.0  0.99057
2021-12-20: Order Failed: GOOG
2021-12-20: SELL EXECUTED - Price: 167.82, Size: -0.68, Asset: AAPL, Cost: 110.92, Commission: 0.00
Final Portfolio Value: 1287.22 
```

我们不会花时间评估策略，因为这与我们在前一个示例中做的非常相似。因此，我们将其作为潜在的练习留给读者。测试该策略的表现是否优于基准*1/n*投资组合也是一个有趣的思路。

值得一提的是，一些订单没有成功执行。我们将在下一节中描述原因。

## 它是如何工作的……

在导入库后，我们使用均值-方差优化定义了策略。在`__init__`方法中，我们定义了一个计数器，用来判断是否有足够的数据点来执行优化过程。选择252天是随意的，你可以尝试不同的值。

在`next`方法中，有多个新的组件：

+   我们首先将天数计数器加1，并检查是否有足够的观察数据。如果没有，我们就简单地跳到下一个交易日。

+   我们从价格数据中提取当前日期并检查是否为星期五。如果不是，我们就继续到下一个交易日。

+   我们通过访问每个资产的头寸大小并将其乘以给定日期的收盘价来计算当前的配置。最后，我们将每个资产的价值除以总投资组合的价值，并打印权重。

+   我们需要提取每只股票的最后252个数据点来进行优化过程。`self.datas`对象是一个可迭代的集合，包含我们在设置回测时传递给`Cerebro`的所有数据源。我们创建一个字典，并用包含252个数据点的数组填充它。然后，我们使用`get`方法提取这些数据。接着，我们从字典中创建一个包含价格的`pandas`数据框。

+   我们使用`pypfopt`库找到了最大化夏普比率的权重。更多细节请参考前一章节。我们还打印了新的权重。

+   对于每个资产，我们使用`order_target_percent`方法下达目标订单，目标是最优投资组合权重。由于这次我们使用多个资产，因此需要指明为哪个资产下单。我们通过指定`data`参数来实现这一点。

在背后，`backtrader`使用`array`模块来存储类似矩阵的对象。

在*第3步*中，我们创建了一个包含所有数据源的列表。我们简单地遍历了FAANG股票的代码，下载了每只股票的数据，并将该对象添加到列表中。

在*第4步*中，我们设置了回测。许多步骤现在已经非常熟悉，包括设置分数股的佣金方案。新的部分是添加数据，我们通过已经涵盖过的`adddata`方法，逐步添加每个下载的数据源。在这一过程中，我们还需要使用`name`参数提供数据源的名称。

在最后一步，我们运行了回测。正如我们之前提到的，新的情况是订单失败。这是因为我们在周五使用收盘价计算投资组合权重，并在同一天准备订单。然而，在周一的市场开盘时，价格发生了变化，导致并非所有订单都能执行。我们尝试使用分数股和将佣金设置为0来解决这一问题，但价格差异可能仍然过大，使得这种简单方法无法正常工作。一种可能的解决方案是始终保留一些现金，以应对潜在的价格差异。

为此，我们可以假设用我们投资组合的约90%的价值购买股票，而将剩余部分保持为现金。为了实现这一点，我们可以使用`order_target_value`方法。我们可以使用投资组合的权重和投资组合价值的90%来计算每个资产的目标价值。或者，我们也可以使用`pypfopt`中的`DiscreteAllocation`方法，正如我们在前一章提到的那样。

# 摘要

在本章中，我们深入探讨了回测的话题。我们从较简单的方式——矢量化回测开始。尽管它不像事件驱动方法那样严格和稳健，但由于其矢量化特性，通常实现和执行速度更快。之后，我们将事件驱动回测框架的探索与前几章获得的知识结合起来，例如计算各种技术指标和寻找最优的投资组合权重。

我们花了最多时间使用`backtrader`库，因为它在实现各种场景时具有广泛的应用和灵活性。然而，市场上有许多其他的回测库。你可能还想研究以下内容：

+   `vectorbt` ([https://github.com/polakowo/vectorbt](https://github.com/polakowo/vectorbt)): 一个基于`pandas`的库，用于大规模高效回测交易策略。该库的作者还提供了一个专业版（收费），具有更多功能和更高性能。

+   `bt` ([https://github.com/pmorissette/bt](https://github.com/pmorissette/bt)): 一个提供基于可复用和灵活模块的框架的库，模块包含策略的逻辑。它支持多种工具，并输出详细的统计数据和图表。

+   `backtesting.py` ([https://github.com/kernc/backtesting.py](https://github.com/kernc/backtesting.py)): 一个建立在`backtrader`之上的回测框架。

+   `fastquant` ([https://github.com/enzoampil/fastquant](https://github.com/enzoampil/fastquant)): 一个围绕`backtrader`的包装库，旨在减少为流行交易策略（例如移动平均交叉）运行回测时需要编写的样板代码量。

+   `zipline` ([https://github.com/quantopian/zipline](https://github.com/quantopian/zipline) / [https://github.com/stefan-jansen/zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded)): 该库曾是最受欢迎的回测库（基于GitHub星标），也可能是最复杂的开源回测库之一。然而，正如我们已经提到的，Quantopian已经关闭，该库也不再维护。你可以使用由Stefan Jansen维护的分支（`zipline-reloaded`）。

回测是一个非常有趣的领域，值得深入学习。以下是一些非常有趣的参考资料，介绍了更稳健的回测方法：

+   Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2016). “回测过拟合的概率。” *计算金融杂志，待刊*。

+   Bailey, D. H., & De Prado, M. L. (2014). “调整夏普比率：修正选择偏差、回测过拟合和非正态性。” *投资组合管理杂志*, *40* (5), 94-107。

+   Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2014). “伪数学与金融江湖术士：回测过拟合对样本外表现的影响。” *美国数学会通报*, *61* (5), 458-471。

+   De Prado, M. L. (2018). *金融机器学习的进展*. 约翰·威利与子公司。
