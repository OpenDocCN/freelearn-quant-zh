# *第七章*：Python 中的金融市场数据访问

本章概述了几个关键的市场数据源，从免费到付费的数据源都有涵盖。可从[`github.com/wilsonfreitas/awesome-quant#data-sources`](https://github.com/wilsonfreitas/awesome-quant#data-sources)获得更完整的可用资源列表。

算法交易模型信号的质量基本取决于正在分析的市场数据的质量。市场数据是否已清理出错误记录，并且是否有质量保证流程来在发生错误时更正任何错误？如果市场数据源有问题，那么数据可以多快被纠正？

下述描述的免费数据源适用于学习目的，但不适用于专业交易目的 - 每天的 API 调用次数可能非常有限，API 可能较慢，并且如果数据不正确，则没有支持和更正。此外，在使用任何这些数据提供者时，请注意其使用条款。

在本章中，我们将涵盖以下主要内容：

+   探索 yahoofinancials Python 库

+   探索 pandas_datareader Python 库

+   探索 Quandl 数据源

+   探索 IEX Cloud 数据源

+   探索 MarketStack 数据源

# 技术要求

本章中使用的 Python 代码可在书籍代码存储库的`Chapter07/marketdata.ipynb`笔记本中找到。

# 探索 yahoofinancials Python 库

yahoofinancials Python 库提供了对雅虎财经市场数据的免费访问，其提供商是 ICE Data Services。库存储库位于[`github.com/JECSand/yahoofinancials`](https://github.com/JECSand/yahoofinancials)。

它提供以下资产的历史和大多数资产的实时定价数据访问：

+   货币

+   索引

+   股票

+   商品

+   ETF

+   共同基金

+   美国国债

+   加密货币

要找到正确的股票代码，请使用[`finance.yahoo.com/`](https://finance.yahoo.com/)上的查找功能。

每个 IP 地址每小时的调用次数有严格的限制（每小时每个 IP 地址约为 1,000-2,000 次请求），一旦达到限制，您的 IP 地址将被阻止一段时间。此外，提供的功能不断变化。

库的安装是标准的：

```py
pip install yahoofinancials
```

访问数据非常简单，如下所示：

```py
from yahoofinancials import YahooFinancials
```

该库支持单一股票检索和多个股票检索。

## 单一股票检索

单一股票检索的步骤如下：

1.  首先，我们定义`AAPL`的股票对象：

    ```py
    aapl = yf.Ticker("AAPL")
    ```

1.  然后，还有历史数据检索的问题。让我们打印出 2020 年的所有历史每日价格数据：

    ```py
    hist = aapl.get_historical_price_data('2020-01-01', 
                                          '2020-12-31', 
                                          'daily')
    print(hist)
    ```

    输出以以下内容开始：

    ```py
    {'AAPL': {'eventsData': {'dividends': {'2020-02-07': {'amount': 0.1925, 'date': 1581085800, 'formatted_date': '2020-02-07'}, '2020-05-08': {'amount': 0.205, 'date': 1588944600, 'formatted_date': '2020-05-08'}, '2020-08-07': {'amount': 0.205, 'date': 1596807000, 'formatted_date': '2020-08-07'}, '2020-11-06': {'amount': 0.205, 'date': 1604673000, 'formatted_date': '2020-11-06'}}, 'splits': {'2020-08-31': {'date': 1598880600, 'numerator': 4, 'denominator': 1, 'splitRatio': '4:1', 'formatted_date': '2020-08-31'}}}, 'firstTradeDate': {'formatted_date': '1980-12-12', 'date': 345479400}, 'currency': 'USD', 'instrumentType': 'EQUITY', 'timeZone': {'gmtOffset': -18000}, 'prices': [{'date': 1577975400, 'high': 75.1500015258789, 'low': 73.79750061035156, 'open': 74.05999755859375, 'close': 75.0875015258789, 'volume': 135480400, 'adjclose': 74.4446029663086, 'formatted_date': '2020-01-02'}, {'date': 1578061800, 'high': 75.1449966430664, 'low': 74.125, 'open': 74.2874984741211, 'close': 74.35749816894531, 'volume': 146322800, 'adjclose': 73.72084045410156, 'formatted_date': '2020-01-03'}, {'date': 1578321000, 'high': 74.98999786376953, 'low': 73.1875, 'open': 73.44750213623047, 'close': 74.94999694824219, 'volume': 118387200, 'adjclose': 74.30826568603516, 'formatted_date': '2020-01-06'}, {'date': 1578407400, 'high': 75.2249984741211, 'low': 74.37000274658203, 'open': 74.95999908447266, 'close': 74.59750366210938, 'volume': 108872000, 'adjclose': 73.95879364013672, 'formatted_date': '2020-01-07'}, {'date': 1578493800, 'high': 76.11000061035156, 'low': 74.29000091552734, 'open': 74.29000091552734, 'close': 75.79750061035156, 'volume': 132079200, 'adjclose': 75.14852142333984, 'formatted_date': '2020-01-08'}, {'date': 1578580200, 'high': 77.60749816894531, 'low': 76.55000305175781, 'open': 76.80999755859375, 'close': 77.40750122070312, 'volume': 170108400, 'adjclose': 76.7447280883789, 'formatted_date': '2020-01-09'}, {'date': 1578666600, 'high': 78.1675033569336, 'low': 77.0625, 'open': 77.6500015258789, 'close': 77.5824966430664, 'volume': 140644800, 'adjclose': 76.91822052001953, 'formatted_date': '2020-01-10'}, {'date': 1578925800, 'high': 79.26750183105469, 'low': 77.7874984741211, 'open': 77.91000366210938, 'close': 79.23999786376953, 'volume': 121532000, 'adjclose': 78.56153106689453, 'formatted_date': '2020-01-13'}, {'date': 1579012200, 'high': 79.39250183105469, 'low': 78.0425033569336, 'open': 79.17500305175781, 'close': 78.16999816894531, 'volume': 161954400, 'adjclose': 77.50070190429688, 'formatted_date': '2020-01-14'}, {'date': 1579098600, 'high': 78.875, 'low': 77.38749694824219, 'open': 77.9625015258789, 'close': 77.83499908447266, 'volume': 121923600, 'adjclose': 77.16856384277344, 'formatted_date': '2020-01-15'}, {'date': 1579185000, 'high': 78.92500305175781, 'low': 78.02249908447266, 'open': 78.39749908447266, 'close': 78.80999755859375, 'volume': 108829200, 'adjclose': 78.13522338867188, 'formatted_date': '2020-01-16'}, {'date': 1579271400, 'high': 79.68499755859375, 'low': 78.75, 'open': 79.06749725341797, 'close': 79.68250274658203, 'volume': 137816400, 'adjclose': 79.000244140625, 'formatted_date': '2020-01-17'}, {'date': 1579617000, 'high': 79.75499725341797, 'low': 79.0, 'open': 79.29750061035156, 'close': 79.14250183105469, 'volume': 110843200, 'adjclose': 78.46488189697266, 'formatted_date': '2020-01-21'}, {'date': 1579703400, 'high': 79.99749755859375, 'low': 79.32749938964844, 'open': 79.6449966430664, 'close': 79.42500305175781, 'volume': 101832400, 'adjclose': 78.74495697021484, 'formatted_date': '2020-01-22'}, ... 
    ```

    注意

    您可以将频率从`'daily'`更改为`'weekly'`或`'monthly'`。

1.  现在，让我们查看每周数据结果：

    ```py
    hist = aapl.get_historical_price_data('2020-01-01', 
                                          '2020-12-31', 
                                          'weekly')
    print(hist)
    ```

    输出如下：

    ```py
    {'AAPL': {'eventsData': {'dividends': {'2020-02-05': {'amount': 0.1925, 'date': 1581085800, 'formatted_date': '2020-02-07'}, '2020-05-06': {'amount': 0.205, 'date': 1588944600, 'formatted_date': '2020-05-08'}, '2020-08-05': {'amount': 0.205, 'date': 1596807000, 'formatted_date': '2020-08-07'}, '2020-11-04': {'amount': 0.205, 'date': 1604673000, 'formatted_date': '2020-11-06'}}, 'splits': {'2020-08-26': {'date': 1598880600, 'numerator': 4, 'denominator': 1, 'splitRatio': '4:1', 'formatted_date': '2020-08-31'}}}, 'firstTradeDate': {'formatted_date': '1980-12-12', 'date': 345479400}, 'currency': 'USD', 'instrumentType': 'EQUITY', 'timeZone': {'gmtOffset': -18000}, 'prices': [{'date': 1577854800, 'high': 75.2249984741211, 'low': 73.1875, 'open': 74.05999755859375, 'close': 74.59750366210938, 'volume': 509062400, 'adjclose': 73.95879364013672, 'formatted_date': '2020-01-01'}, {'date': 1578459600, 'high': 79.39250183105469, 'low': 74.29000091552734, 'open': 74.29000091552734, 'close': 78.16999816894531, 'volume': 726318800, 'adjclose': 77.50070190429688, 'formatted_date': '2020-01-08'}, {'date': 1579064400, 'high': 79.75499725341797, 'low': 77.38749694824219, 'open': 77.9625015258789, 'close': 79.14250183105469, 'volume': 479412400, 'adjclose': 78.46488189697266, 'formatted_date': '2020-01-15'}, {'date': 1579669200, 'high': 80.8324966430664, 'low': 76.22000122070312, 'open': 79.6449966430664, 'close': 79.42250061035156, 'volume': 677016000, 'adjclose': 78.74247741699219, 'formatted_date': '2020-01-22'}, {'date': 1580274000, 'high': 81.9625015258789, 'low': 75.55500030517578, 'open': 81.11250305175781, 'close': 79.7125015258789, 'volume': 853162800, 'adjclose': 79.02999877929688, 'formatted_date': '2020-01-29'}, {'date': 1580878800, 'high': 81.30500030517578, 'low': 78.4625015258789, 'open': 80.87999725341797, 'close': 79.90249633789062, 'volume': 545608400, 'adjclose': 79.21836853027344, 'formatted_date': '2020-02-05'}, {'date': 1581483600, 'high': 81.80500030517578, 'low': 78.65249633789062, 'open': 80.36750030517578, 'close': 79.75, 'volume': 441122800, 'adjclose': 79.25482177734375, 'formatted_date': '2020-02-12'}, {'date': 1582088400, 'high': 81.1624984741211, 'low': 71.53250122070312, 'open': 80.0, 'close': 72.0199966430664, 'volume': 776972800, 'adjclose': 71.57282257080078, 'formatted_date': '2020-02-19'}, {'date': 1582693200, 'high': 76.0, 'low': 64.09249877929688, 'open': 71.63249969482422, 'close': 72.33000183105469, 'volume': 1606418000, 'adjclose': 71.88089752197266, 'formatted_date': '2020-02-26'}, {'date': 1583298000, 'high': 75.8499984741211, 'low': 65.75, 'open': 74.11000061035156, 'close': 71.33499908447266, 'volume': 1204962800, 'adjclose': 70.89207458496094, 'formatted_date': '2020-03-04'}, {'date': 1583899200, 'high': 70.3050003051757 ...
    ```

1.  然后，我们检查月度数据结果：  

    ```py
    hist = aapl.get_historical_price_data('2020-01-01', 
                                          '2020-12-31', 
                                          'monthly')
    print(hist)
    ```

    输出如下：  

    ```py
    {'AAPL': {'eventsData': {'dividends': {'2020-05-01': {'amount': 0.205, 'date': 1588944600, 'formatted_date': '2020-05-08'}, '2020-08-01': {'amount': 0.205, 'date': 1596807000, 'formatted_date': '2020-08-07'}, '2020-02-01': {'amount': 0.1925, 'date': 1581085800, 'formatted_date': '2020-02-07'}, '2020-11-01': {'amount': 0.205, 'date': 1604673000, 'formatted_date': '2020-11-06'}}, 'splits': {'2020-08-01': {'date': 1598880600, 'numerator': 4, 'denominator': 1, 'splitRatio': '4:1', 'formatted_date': '2020-08-31'}}}, 'firstTradeDate': {'formatted_date': '1980-12-12', 'date': 345479400}, 'currency': 'USD', 'instrumentType': 'EQUITY', 'timeZone': {'gmtOffset': -18000}, 'prices': [{'date': 1577854800, 'high': 81.9625015258789, 'low': 73.1875, 'open': 74.05999755859375, 'close': 77.37750244140625, 'volume': 2934370400, 'adjclose': 76.7149887084961, 'formatted_date': '2020-01-01'}, {'date': 1580533200, 'high': 81.80500030517578, 'low': 64.09249877929688, 'open': 76.07499694824219, 'close': 68.33999633789062, 'volume': 3019851200, 'adjclose': 67.75486755371094, 'formatted_date': '2020-02-01'}, {'date': 1583038800, 'high': 76.0, 'low': 53.15250015258789, 'open': 70.56999969482422, 'close': 63 ...
    ```

1.  嵌套的 JSON 可轻松转换为 pandas 的 DataFrame：  

    ```py
    import pandas as pd
    hist_df = \
    pd.DataFrame(hist['AAPL']['prices']).drop('date', axis=1).set_index('formatted_date')
    print(hist_df)
    ```

    输出如下：  

![图 7.1 - 嵌套 JSON 转换为 pandas 的 DataFrame](img/Figure_7.1_B15029.jpg)  

图 7.1 - 嵌套 JSON 转换为 pandas 的 DataFrame  

注意两列 - `adjclose`和`close`。调整后的收盘价是根据股利、股票拆分和其他公司事件调整的收盘价。  

### 实时数据检索  

要获取实时股票价格数据，请使用`get_stock_price_data()`函数：  

```py
print(aapl.get_stock_price_data())
```

输出如下：  

```py
{'AAPL': {'quoteSourceName': 'Nasdaq Real Time Price', 'regularMarketOpen': 137.35, 'averageDailyVolume3Month': 107768827, 'exchange': 'NMS', 'regularMarketTime': '2021-02-06 03:00:02 UTC+0000', 'volume24Hr': None, 'regularMarketDayHigh': 137.41, 'shortName': 'Apple Inc.', 'averageDailyVolume10Day': 115373562, 'longName': 'Apple Inc.', 'regularMarketChange': -0.42500305, 'currencySymbol': '$', 'regularMarketPreviousClose': 137.185, 'postMarketTime': '2021-02-06 06:59:58 UTC+0000', 'preMarketPrice': None, 'exchangeDataDelayedBy': 0, 'toCurrency': None, 'postMarketChange': -0.0800018, 'postMarketPrice': 136.68, 'exchangeName': 'NasdaqGS', 'preMarketChange': None, 'circulatingSupply': None, 'regularMarketDayLow': 135.86, 'priceHint': 2, 'currency': 'USD', 'regularMarketPrice': 136.76, 'regularMarketVolume': 72317009, 'lastMarket': None, 'regularMarketSource': 'FREE_REALTIME', 'openInterest': None, 'marketState': 'CLOSED', 'underlyingSymbol': None, 'marketCap': 2295940513792, 'quoteType': 'EQUITY', 'volumeAllCurrencies': None, 'postMarketSource': 'FREE_REALTIME', 'strikePrice': None, 'symbol': 'AAPL', 'postMarketChangePercent': -0.00058498, 'preMarketSource': 'FREE_REALTIME', 'maxAge': 1, 'fromCurrency': None, 'regularMarketChangePercent': -0.0030980287}}
```

免费数据源的实时数据通常延迟 10 到 30 分钟。  

至于获取财务报表，让我们获取苹果股票的财务报表 - 损益表、现金流量表和资产负债表：  

```py
statements = aapl.get_financial_stmts('quarterly', 
                                      ['income', 'cash', 
                                       'balance'])
print(statements)
```

输出如下：  

```py
{'incomeStatementHistoryQuarterly': {'AAPL': [{'2020-12-26': {'researchDevelopment': 5163000000, 'effectOfAccountingCharges': None, 'incomeBeforeTax': 33579000000, 'minorityInterest': None, 'netIncome': 28755000000, 'sellingGeneralAdministrative': 5631000000, 'grossProfit': 44328000000, 'ebit': 33534000000, 'operatingIncome': 33534000000, 'otherOperatingExpenses': None, 'interestExpense': -638000000, 'extraordinaryItems': None, 'nonRecurring': None, 'otherItems': None, 'incomeTaxExpense': 4824000000, 'totalRevenue': 111439000000, 'totalOperatingExpenses': 77905000000, 'costOfRevenue': 67111000000, 'totalOtherIncomeExpenseNet': 45000000, 'discontinuedOperations': None, 'netIncomeFromContinuingOps': 28755000000, 'netIncomeApplicableToCommonShares': 28755000000}}, {'2020-09-26': {'researchDevelopment': 4978000000, 'effectOfAccountingCharges': None, 'incomeBeforeTax': 14901000000, 'minorityInterest': None, 'netIncome': 12673000000, 'sellingGeneralAdministrative': 4936000000, 'grossProfit': ...
```

金融报表数据在算法交易中有多种用途。首先，它可用于确定要交易的股票的总体情况。其次，从非价格数据创建算法交易信号会增加额外的价值。  

### 摘要数据检索  

摘要数据可通过`get_summary_data`方法获取：  

```py
print(aapl.get_summary_data())
```

输出如下：  

```py
{'AAPL': {'previousClose': 137.185, 'regularMarketOpen': 137.35, 'twoHundredDayAverage': 119.50164, 'trailingAnnualDividendYield': 0.0058825673, 'payoutRatio': 0.2177, 'volume24Hr': None, 'regularMarketDayHigh': 137.41, 'navPrice': None, 'averageDailyVolume10Day': 115373562, 'totalAssets': None, 'regularMarketPreviousClose': 137.185, 'fiftyDayAverage': 132.86455, 'trailingAnnualDividendRate': 0.807, 'open': 137.35, 'toCurrency': None, 'averageVolume10days': 115373562, 'expireDate': '-', 'yield': None, 'algorithm': None, 'dividendRate': 0.82, 'exDividendDate': '2021-02-05', 'beta': 1.267876, 'circulatingSupply': None, 'startDate': '-', 'regularMarketDayLow': 135.86, 'priceHint': 2, 'currency': 'USD', 'trailingPE': 37.092484, 'regularMarketVolume': 72317009, 'lastMarket': None, 'maxSupply': None, 'openInterest': None, 'marketCap': 2295940513792, 'volumeAllCurrencies': None, 'strikePrice': None, 'averageVolume': 107768827, 'priceToSalesTrailing12Months': 7.805737, 'dayLow': 135.86, 'ask': 136.7, 'ytdReturn': None, 'askSize': 1100, 'volume': 72317009, 'fiftyTwoWeekHigh': 145.09, 'forwardPE': 29.410751, 'maxAge': 1, 'fromCurrency': None, 'fiveYearAvgDividendYield': 1.44, 'fiftyTwoWeekLow': 53.1525, 'bid': 136.42, 'tradeable': False, 'dividendYield': 0.0061000003, 'bidSize': 2900, 'dayHigh': 137.41}}
```

使用此函数检索的摘要数据是财务报表函数和实时数据函数的摘要。  

## 多股票检索  

多股票检索，也称为**批量检索**，比单股票检索更高效快速，因为每个下载请求关联的大部分时间都用于建立和关闭网络连接。  

### 历史数据检索  

让我们获取这些外汇对的历史价格：`EURCHF`、`USDEUR`和`GBPUSD`：  

```py
currencies = YahooFinancials(['EURCHF=X', 'USDEUR=X', 
                              'GBPUSD=x'])
print(currencies.get_historical_price_data('2020-01-01', 
                                           '2020-12-31', 
                                           'weekly'))
```

输出如下：  

```py
{'EURCHF=X': {'eventsData': {}, 'firstTradeDate': {'formatted_date': '2003-01-23', 'date': 1043280000}, 'currency': 'CHF', 'instrumentType': 'CURRENCY', 'timeZone': {'gmtOffset': 0}, 'prices': [{'date': 1577836800, 'high': 1.0877000093460083, 'low': 1.0818699598312378, 'open': 1.0872000455856323, 'close': 1.084280014038086, 'volume': 0, 'adjclose': 1.084280014038086, 'formatted_date': '2020-01-01'}, {'date': 1578441600, 'high': 1.083299994468689, 'low': 1.0758999586105347, 'open': 1.080530047416687, 'close': 1.0809999704360962, 'volume': 0, 'adjclose': 1.0809999704360962, 'formatted_date': '2020-01-08'}, {'date': 1579046400, 'high': 1.0774999856948853, 'low': 1.0729299783706665, 'open': 1.076300024986267, 'close': 1.0744800567626953, 'volume': 0, 'adjclose': 1.0744800567626953, 'formatted_date': '2020-01-15'}, {'date': 1579651200, 'high': 1.0786099433898926, 'low': 1.0664700269699097, 'open': 1.0739500522613525, 'close': 1.068600058555603, 'volume': 0, 'adjclose': 1.068600058555603, 'formatted_date': '2020-01-22'}, {'date': 1580256000, 'high': 1.0736199617385864, 'low': 1.0663000345230103, 'open': 1.0723999738693237, 'close': 1.0683200359344482, 'volume': 0, 'adjclose': 1.068320035 ...
```

我们发现历史数据不包含任何财务报表数据。  

写作本书时库支持的全部方法如下：  

+   `get_200day_moving_avg()`  

+   `get_50day_moving_avg()`  

+   `get_annual_avg_div_rate()`  

+   `get_annual_avg_div_yield()`  

+   `get_beta()`  

+   `get_book_value()`  

+   `get_cost_of_revenue()`  

+   `get_currency()`  

+   `get_current_change()`  

+   `get_current_percent_change()`  

+   `get_current_price()`  

+   `get_current_volume()`  

+   `get_daily_dividend_data(start_date, end_date)`  

+   `get_daily_high()`  

+   `get_daily_low()`  

+   `get_dividend_rate()`  

+   `get_dividend_yield()`  

+   `get_earnings_per_share()`  

+   `get_ebit()`  

+   `get_exdividend_date()`  

+   `get_financial_stmts(frequency, statement_type, reformat=True)`  

+   `get_five_yr_avg_div_yield()`  

+   `get_gross_profit()`  

+   `get_historical_price_data(start_date, end_date, time_interval)`  

+   `get_income_before_tax()`  

+   `get_income_tax_expense()`  

+   `get_interest_expense()`  

+   `get_key_statistics_data()`  

+   `get_market_cap()`  

+   `get_net_income()`  

+   `get_net_income_from_continuing_ops()`  

+   `get_num_shares_outstanding(price_type='current')`  

+   `get_open_price()`  

+   `get_operating_income()`  

+   `get_payout_ratio()`  

+   `get_pe_ratio()`  

+   `get_prev_close_price()`  

+   `get_price_to_sales()`  

+   `get_research_and_development()`  

+   `get_stock_earnings_data(reformat=True)`

+   `get_stock_exchange()`

+   `get_stock_price_data(reformat=True)`

+   `get_stock_quote_type_data()`

+   `get_summary_data(reformat=True)`

+   `get_ten_day_avg_daily_volume()`

+   `get_three_month_avg_daily_volume()`

+   `get_total_operating_expense()`

+   `get_total_revenue()`

+   `get_yearly_high()`

+   `get_yearly_low()`

我们将在下一部分中探索 `pandas_datareader` 库。

# 探索 pandas_datareader Python 库

`pandas_datareader` 是用于金融数据的最先进的库之一，提供对多个数据源的访问。

支持的一些数据源如下：

+   雅虎财经

+   圣路易斯联邦储备银行的 FRED

+   IEX

+   Quandl

+   Kenneth French 的数据库

+   世界银行

+   经济合作与发展组织

+   Eurostat

+   Econdb

+   纳斯达克交易员符号定义

参考[`pandas-datareader.readthedocs.io/en/latest/remote_data.html`](https://pandas-datareader.readthedocs.io/en/latest/remote_data.html)以获取完整列表。

安装很简单：

```py
pip install pandas-datareader
```

现在，让我们设置基本的数据检索参数：

```py
from pandas_datareader import data
start_date = '2010-01-01'
end_date = '2020-12-31'
```

下载数据的一般访问方法是 `data.DataReader(ticker, data_source, start_date, end_date)`。

## 访问雅虎财经

让我们下载 Apple 过去 10 年的股票价格：

```py
aapl = data.DataReader('AAPL', 'yahoo', start_date, 
                       end_date)
aapl
       High      Low     Open    Close     Volume Adj Close
Date            
2010-01-04 7.660714 7.585000 7.622500 7.643214 493729600.0 6.593426
2010-01-05 7.699643 7.616071 7.664286 7.656428 601904800.0 6.604825
2010-01-06 7.686786 7.526786 7.656428 7.534643 552160000.0 6.499768
2010-01-07 7.571429 7.466072 7.562500 7.520714 477131200.0 6.487752
2010-01-08 7.571429 7.466429 7.510714 7.570714 447610800.0 6.530883
...  ...  ...  ...  ...  ...  ...
2020-12 -21 128.309998 123.449997 125.019997 128.229996 121251600.0 128.229996
2020-12-22 134.410004 129.649994 131.610001 131.880005 168904800.0 131.880005
2020-12-23 132.429993 130.779999 132.160004 130.960007 88223700.0 130.960007
2020-12-24 133.460007 131.100006 131.320007 131.970001 54930100.0 131.970001
2020-12-28 137.339996 133.509995 133.990005 136.690002 124182900.0 136.690002
```

输出与前一部分中的 `yahoofinancials` 库的输出几乎相同。

## 访问 EconDB

可用股票标记列表在[`www.econdb.com/main-indicators`](https://www.econdb.com/main-indicators)上可用。

让我们下载美国过去 10 年的月度石油产量时间序列：

```py
oilprodus = data.DataReader('ticker=OILPRODUS', 'econdb', 
                            start_date, end_date)
oilprodus
 Reference Area         United States of America
 Energy product                        Crude oil
 Flow breakdown                       Production
Unit of measure  Thousand Barrels per day (kb/d)
TIME_PERIOD  
2010-01-01  5390
2010-02-01  5548
2010-03-01  5506
2010-04-01  5383
2010-05-01  5391
       ...   ...
2020-04-01  11990
2020-05-01  10001
2020-06-01  10436
2020-07-01  10984
2020-08-01  10406
```

每个数据源都有不同的输出列。

## 访问圣路易斯联邦储备银行的 FRED

可以在[`fred.stlouisfed.org/`](https://fred.stlouisfed.org/)检查可用数据列表和标记。

让我们下载美国过去 10 年的实际国内生产总值：

```py
import pandas as pd
pd.set_option('display.max_rows', 2)
gdp = data.DataReader('GDP', 'fred', start_date, end_date)
gdp
```

我们将输出限制为只有两行：

```py
                  GDP
      DATE  
2010-01-01  14721.350
       ...        ...
2020-07-01  21170.252
43 rows × 1 columns
```

现在，让我们研究美国政府债券 20 年期恒久收益率的 5 年数据：

```py
gs10 = data.get_data_fred('GS20')
gs10
            GS20
      DATE  
2016-01-01  2.49
       ...   ...
2020-11-01  1.40
59 rows × 1 columns
```

圣路易斯联邦储备银行的 FRED 数据是可用的最清洁的数据源之一，提供免费支持。

## 缓存查询

该库的一个关键优势是实现了查询结果的缓存，从而节省带宽，加快代码执行速度，并防止因 API 过度使用而禁止 IP。

举例来说，让我们下载 Apple 股票的全部历史数据：

```py
import datetime
import requests_cache
session = \
requests_cache.CachedSession(cache_name='cache', 
                             backend='sqlite', 
                             expire_after = \
                             datetime.timedelta(days=7))
aapl_full_history = \
data.DataReader("AAPL",'yahoo',datetime.datetime(1980,1,1), 
                datetime.datetime(2020, 12, 31), 
                session=session)
aapl_full_history
       High      Low    Open    Close      Volume Adj Close
Date            
1980-12-12 0.128906 0.128348 0.128348 0.128348 469033600.0 0.101087
...  ...  ...  ...  ...  ...  ...
2020-12-28 137.339996 133.509995 133.990005 136.690002 124182900.0 136.690002
```

现在，让我们只访问一个数据点：

```py
aapl_full_history.loc['2013-01-07']
High         18.903572
               ...    
Adj Close    16.284145
Name: 2013-01-07 00:00:00, Length: 6, dtype: float64
```

缓存也可以为所有以前的示例启用。

# 探索 Quandl 数据源

Quandl 是互联网上最大的经济/金融数据存储库之一。其数据源可以免费访问。它还提供高级数据源，需要付费。

安装很简单：

```py
pip install quandl
```

要访问数据，您必须提供访问密钥（在[`quandl.com`](https://quandl.com)申请）：

```py
import quandl
quandl.ApiConfig.api_key = 'XXXXXXX'
```

要查找股票和数据源，请使用[`www.quandl.com/search`](https://www.quandl.com/search)。

现在让我们下载`法国大都市地区每月平均消费价格 - 苹果（1 公斤）；欧元`数据：

```py
papple = quandl.get('ODA/PAPPLE_USD')
papple
               Value
Date  
1998-01-31  1.735999
    ...          ...
2020-11-30  3.350000
275 rows × 1 columns
```

现在让我们下载苹果公司的基本数据：

```py
aapl_fundamental_data = quandl.get_table('ZACKS/FC', 
                                         ticker='AAPL')
  m_ticker  ticker  comp_name  comp_name_2  exchange  currency_code  per_end_date  per_type  per_code  per_fisc_year  ...  stock_based_compsn_qd  cash_flow_oper_activity_qd  net_change_prop_plant_equip_qd  comm_stock_div_paid_qd  pref_stock_div_paid_qd  tot_comm_pref_stock_div_qd  wavg_shares_out  wavg_shares_out_diluted  eps_basic_net  eps_diluted_net
None                                          
0  AAPL  AAPL  APPLE INC  Apple Inc.  NSDQ  USD  2018-09-30  A  None  2018  ...  NaN  NaN  NaN  NaN  None  NaN  19821.51  20000.44  3.000  2.980
...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...
4  AAPL  AAPL  APPLE INC  Apple Inc.  NSDQ  USD  2018-12-31  Q  None  2019  ...  1559.0  26690.0  -3355.0  -3568.0  None  -3568.0  18943.28  19093.01  1.055  1.045
5 rows × 249 columns
```

Yahoo 和 Quandl 数据之间的区别在于，Quandl 数据更可靠、更完整。

# 探索 IEX Cloud 数据源

IEX Cloud 是其中一个商业产品。它为个人提供每月 9 美元的计划。它还提供一个免费计划，每月限制为 50,000 次 API 调用。

Python 库的安装是标准的：

```py
pip install iexfinance
```

完整的库文档可在[`addisonlynch.github.io/iexfinance/stable/index.html`](https://addisonlynch.github.io/iexfinance/stable/index.html)上找到。

以下代码旨在检索所有符号：

```py
from iexfinance.refdata import get_symbols
get_symbols(output_format='pandas', token="XXXXXX")
symbol  exchange  exchangeSuffix  exchangeName  name  date  type  iexId  region  currency  isEnabled  figi  cik  lei
0  A  NYS  UN  NEW YORK STOCK EXCHANGE, INC.  Agilent Technologies Inc.  2020-12-29  cs  IEX_46574843354B2D52  US  USD  True  BBG000C2V3D6  0001090872  QUIX8Y7A2WP0XRMW7G29
...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...
9360  ZYXI  NAS    NASDAQ CAPITAL MARKET  Zynex Inc  2020-12-29  cs  IEX_4E464C4C4A462D52  US  USD  True  BBG000BJBXZ2  0000846475  None
9361 rows × 14 columns
```

以下代码旨在获取苹果公司的资产负债表（免费账户不可用）：

```py
from iexfinance.stocks import Stock
aapl = Stock("aapl", token="XXXXXX")
aapl.get_balance_sheet()
```

以下代码旨在获取当前价格（免费账户不可用）：

```py
aapl.get_price()
```

以下代码旨在获取部门绩效报告（免费账户不可用）：

```py
from iexfinance.stocks import get_sector_performance
get_sector_performance(output_format='pandas', 
                       token =token)
```

以下代码旨在获取苹果公司的历史市场数据：

```py
from iexfinance.stocks import get_historical_data
get_historical_data("AAPL", start="20190101", 
                    end="20200101", 
                    output_format='pandas', token=token)
close  high  low  open  symbol  volume  id  key  subkey  updated  ...  uLow  uVolume  fOpen  fClose  fHigh  fLow  fVolume  label  change  changePercent
2019-01-02  39.48  39.7125  38.5575  38.7225  AAPL  148158948  HISTORICAL_PRICES  AAPL    1606830572000  ...  154.23  37039737  37.8227  38.5626  38.7897  37.6615  148158948  Jan 2, 19  0.045  0.0011
...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...
2019-12-31  73.4125  73.42  72.38  72.4825  AAPL  100990500  HISTORICAL_PRICES  AAPL    1606830572000  ...  289.52  25247625  71.8619  72.7839  72.7914  71.7603  100990500  Dec 31, 19  0.5325  0.0073
252 rows × 25 columns
```

我们可以看到每个数据源提供了略有不同的输出列。

# 探索 MarketStack 数据源

MarketStack 提供跨主要全球股票交易所的实时、盘内和历史市场数据的广泛数据库。它为每月高达 1,000 次的 API 请求提供免费访问。

虽然没有官方的 MarketStack Python 库，但 REST JSON API 在 Python 中提供了对其所有数据的舒适访问。

让我们下载苹果公司的调整后收盘数据：

```py
import requests
params = {
  'access_key': 'XXXXX'
}
api_result = \
requests.get('http://api.marketstack.com/v1/tickers/aapl/eod', params)
api_response = api_result.json()
print(f"Symbol = {api_response['data']['symbol']}")
for eod in api_response['data']['eod']:
    print(f"{eod['date']}: {eod['adj_close']}")
Symbol = AAPL
2020-12-28T00:00:00+0000: 136.69
2020-12-24T00:00:00+0000: 131.97
2020-12-23T00:00:00+0000: 130.96
2020-12-22T00:00:00+0000: 131.88
2020-12-21T00:00:00+0000: 128.23
2020-12-18T00:00:00+0000: 126.655
2020-12-17T00:00:00+0000: 128.7
2020-12-16T00:00:00+0000: 127.81
2020-12-15T00:00:00+0000: 127.88
2020-12-14T00:00:00+0000: 121.78
2020-12-11T00:00:00+0000: 122.41
2020-12-10T00:00:00+0000: 123.24
2020-12-09T00:00:00+0000: 121.78
2020-12-08T00:00:00+0000: 124.38
2020-12-07T00:00:00+0000: 123.75
2020-12-04T00:00:00+0000: 122.25
```

现在让我们下载纳斯达克证券交易所的所有股票代码：

```py
api_result = \
requests.get('http://api.marketstack.com/v1/exchanges/XNAS/tickers', params)
api_response = api_result.json()
print(f"Exchange Name = {api_response['data']['name']}")
for ticker in api_response['data']['tickers']:
    print(f"{ticker['name']}: {ticker['symbol']}")
Exchange Name = NASDAQ Stock Exchange
Microsoft Corp: MSFT
Apple Inc: AAPL
Amazoncom Inc: AMZN
Alphabet Inc Class C: GOOG
Alphabet Inc Class A: GOOGL
Facebook Inc: FB
Vodafone Group Public Limited Company: VOD
Intel Corp: INTC
Comcast Corp: CMCSA
PepsiCo Inc: PEP
Adobe Systems Inc: ADBE
Cisco Systems Inc: CSCO
NVIDIA Corp: NVDA
Netflix Inc: NFLX
```

MarketStack 的票务宇宙检索功能是最有价值的功能之一。所有回测的第一步之一是确定股票交易的宇宙（即完整列表）。然后，您可以通过仅交易具有某些趋势或某些交易量的股票等方式将自己限制在该列表的子集中。

# 概要

在本章中，我们概述了在 Python 中获取金融和经济数据的不同方法。在实践中，您通常同时使用多个数据源。我们探索了`yahoofinancials` Python 库，并看到了单个和多个股票检索。然后，我们探索了`pandas_datareader` Python 库，以访问 Yahoo Finance、EconDB 和 Fed 的 Fred 数据，并缓存查询。然后我们探索了 Quandl、IEX Cloud 和 MarketStack 数据源。

在下一章中，我们将介绍回测库 Zipline，以及交易组合绩效和风险分析库 PyFolio。
