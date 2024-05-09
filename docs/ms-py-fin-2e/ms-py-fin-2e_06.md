# 第四章：期权定价的数值方法

衍生品是一种合同，其回报取决于某些基础资产的价值。在封闭形式衍生品定价可能复杂甚至不可能的情况下，数值程序表现出色。数值程序是使用迭代计算方法试图收敛到解的方法。其中一个基本实现是二项树。在二项树中，一个节点代表某一时间点的资产状态，与价格相关。每个节点在下一个时间步骤导致另外两个节点。同样，在三项树中，每个节点在下一个时间步骤导致另外三个节点。然而，随着树的节点数量或时间步骤的增加，消耗的计算资源也会增加。栅格定价试图通过仅在每个时间步骤存储新信息，同时在可能的情况下重复使用价值来解决这个问题。

在有限差分定价中，树的节点也可以表示为网格。网格上的终端值包括终端条件，而网格的边缘代表资产定价中的边界条件。我们将讨论有限差分方案的显式方法、隐式方法和 Crank-Nicolson 方法，以确定资产的价格。

尽管香草期权和某些奇异期权，如欧式障碍期权和回望期权，可能具有封闭形式解，但其他奇异产品，如亚洲期权，没有封闭形式解。在这些情况下，可以使用数值程序来定价期权。

在本章中，我们将涵盖以下主题：

+   使用二项树定价欧式和美式期权

+   使用 Cox-Ross-Rubinstein 二项树

+   使用 Leisen-Reimer 树定价期权

+   使用三项树定价期权

+   从树中推导希腊字母

+   使用二项和三项栅格定价期权

+   使用显式、隐式和 Crank-Nicolson 方法的有限差分

+   使用 LR 树和二分法的隐含波动率建模

# 期权介绍

**期权**是资产的衍生品，它赋予所有者在特定日期以特定价格交易基础资产的权利，称为到期日和行权价格。

**认购期权**赋予买方在特定日期以特定价格购买资产的权利。认购期权的卖方或写方有义务在约定日期以约定价格向买方出售基础证券，如果买方行使其权利。

**认沽期权**赋予买方在特定日期以特定价格出售基础资产的权利。认沽期权的卖方或写方有义务在约定日期以约定价格从买方购买基础证券，如果买方行使其权利。

最常见的期权包括欧式期权和美式期权。其他奇异期权包括百慕大期权和亚洲期权。本章主要涉及欧式期权和美式期权。欧式期权只能在到期日行使。而美式期权则可以在期权的整个生命周期内的任何时间行使。

# 期权定价中的二项树

在二项期权定价模型中，假设在一个时间段内，代表具有给定价格的节点的基础证券在下一个时间步骤中遍历到另外两个节点，代表上升状态和下降状态。由于期权是基础资产的衍生品，二项定价模型以离散时间为基础跟踪基础条件。二项期权定价可用于估值欧式期权、美式期权以及百慕大期权。

根节点的初始值是基础证券的现货价格*S[0]*，具有风险中性概率的上涨*q*和下跌的风险中性概率*1-q*，在下一个时间步骤。基于这些概率，为每个价格上涨或下跌的状态计算了证券的预期值。终端节点代表了每个预期证券价格的值，对应于上涨状态和下跌状态的每种组合。然后我们可以计算每个节点的期权价值，通过风险中性期望遍历树状图，并在从远期利率贴现后，我们可以推导出看涨期权或看跌期权的价值。

# 定价欧式期权

考虑一个两步二叉树。不支付股息的股票价格从 50 美元开始，在两个时间步骤中，股票可能上涨 20%，也可能下跌 20%。假设无风险利率为每年 5%，到期时间*T*为两年。我们想要找到行权价*K*为 52 美元的欧式看跌期权的价值。以下图表显示了使用二叉树定价股票和终端节点的回报：

![](img/ebfbaecf-a7a2-4763-bb3d-cf9673e98c4d.png)

这里，节点的计算如下：

![](img/56cb6902-2c50-4589-8d03-71981ff9cd15.png)

![](img/6c899404-b04e-4695-be22-8f1f4a46fac0.png)

![](img/70e15ac6-f601-410d-9371-32cfb0dbe24e.png)

![](img/6eda97f5-8781-4acf-9a3a-a88c41375fd5.png)

![](img/a959a1cf-4af3-4fba-904a-3e316030ad56.png)

![](img/98ee5ef1-3fa8-47fd-afd2-5fecdbe22756.png)

![](img/c7579cd4-bc65-439b-9657-25691097682c.png)

![](img/82886492-9cb6-4ebe-b4fc-40d5b97d7da3.png)

在终端节点，行使欧式看涨期权的回报如下：

![](img/401d00a1-d6b5-4980-9cad-d589da05a7cf.png)

在欧式看跌期权的情况下，回报如下：

![](img/91b879ad-9ac4-4540-b9a2-a9a9dbf3b42d.png)

欧式看涨期权和看跌期权通常用小写字母*c*和*p*表示，而美式看涨期权和看跌期权通常用大写字母*C*和*P*表示。

从期权回报值中，我们可以向后遍历二叉树到当前时间，并在从无风险利率贴现后，我们将获得期权的现值。向后遍历树状图考虑了期权上涨状态和下跌状态的风险中性概率。

我们可以假设投资者对风险不感兴趣，并且所有资产的预期回报相等。在通过风险中性概率投资股票的情况下，持有股票的回报并考虑上涨和下跌状态的可能性将等于在下一个时间步骤中预期的连续复利无风险利率，如下所示：

![](img/59bc58a4-3aa6-47d1-b491-e305786a3925.png)

投资股票的风险中性概率*q*可以重写如下：

![](img/8996391c-6994-421f-99a1-116ac1e83498.png)

这些公式与股票相关吗？期货呢？

与投资股票不同，投资者无需提前付款来持有期货合约。在风险中性意义上，持有期货合约的预期增长率为零，投资期货的风险中性概率*q*可以重写如下：

![](img/83f95aed-08c9-447c-aa1d-e5f5a372d01b.png)

让我们计算前面示例中给出的股票的风险中性概率*q*：

```py
In [ ]:
    import math

    r = 0.05
    T = 2
    t = T/2
    u = 1.2
    d = 0.8

    q = (math.exp(r*t)-d)/(u-d)
In [ ]:
    print('q is', q)
Out[ ]:   
    q is 0.6281777409400603
```

在终端节点行使欧式看跌期权的回报分别为 0 美元、4 美元和 20 美元。看跌期权的现值可以定价如下：

![](img/748f9fd6-e9de-4b9b-946a-1719ace53d65.png)

这给出了看跌期权价格为 4.19 美元。使用二叉树估算每个节点的欧式看跌期权的价值如下图所示：

![](img/66ef05f5-a024-498b-9228-720be29e4ef9.png)

# 编写 StockOption 基类

在进一步实现我们将要讨论的各种定价模型之前，让我们创建一个`StockOption`类，用于存储和计算股票期权的共同属性，这些属性将在本章中被重复使用：

```py
In [ ]:
    import math

    """ 
    Stores common attributes of a stock option 
    """
    class StockOption(object):
        def __init__(
            self, S0, K, r=0.05, T=1, N=2, pu=0, pd=0, 
            div=0, sigma=0, is_put=False, is_am=False):
            """
            Initialize the stock option base class.
            Defaults to European call unless specified.

            :param S0: initial stock price
            :param K: strike price
            :param r: risk-free interest rate
            :param T: time to maturity
            :param N: number of time steps
            :param pu: probability at up state
            :param pd: probability at down state
            :param div: Dividend yield
            :param is_put: True for a put option,
                    False for a call option
            :param is_am: True for an American option,
                    False for a European option
            """
            self.S0 = S0
            self.K = K
            self.r = r
            self.T = T
            self.N = max(1, N)
            self.STs = [] # Declare the stock prices tree

            """ Optional parameters used by derived classes """
            self.pu, self.pd = pu, pd
            self.div = div
            self.sigma = sigma
            self.is_call = not is_put
            self.is_european = not is_am

        @property
        def dt(self):
            """ Single time step, in years """
            return self.T/float(self.N)

        @property
        def df(self):
            """ The discount factor """
            return math.exp(-(self.r-self.div)*self.dt)  
```

当前的标的价格、行权价格、无风险利率、到期时间和时间步数是定价期权的强制共同属性。时间步长`dt`和折现因子`df`的增量作为类的属性计算，并且如果需要，可以被实现类覆盖。

# 使用二项式树的欧式期权类

欧式期权的 Python 实现是`BinomialEuropeanOption`类，它继承自`StockOption`类的共同属性。该类中方法的实现如下：

1.  `BinomialEuropeanOption`类的`price()`方法是该类所有实例的入口

1.  它调用`setup_parameters()`方法来设置所需的模型参数，然后调用`init_stock_price_tree()`方法来模拟期间内股票价格的预期值直到*T*

1.  最后，调用`begin_tree_traversal()`方法来初始化支付数组并存储折现支付值，因为它遍历二项式树回到现在的时间

1.  支付树节点作为 NumPy 数组对象返回，其中欧式期权的现值在初始节点处找到

`BinomialEuropeanOption`的类实现如下 Python 代码：

```py
In [ ]:
    import math
    import numpy as np
    from decimal import Decimal

    """ 
    Price a European option by the binomial tree model 
    """
    class BinomialEuropeanOption(StockOption):

        def setup_parameters(self):
            # Required calculations for the model
            self.M = self.N+1  # Number of terminal nodes of tree
            self.u = 1+self.pu  # Expected value in the up state
            self.d = 1-self.pd  # Expected value in the down state
            self.qu = (math.exp(
                (self.r-self.div)*self.dt)-self.d)/(self.u-self.d)
            self.qd = 1-self.qu

        def init_stock_price_tree(self):
            # Initialize terminal price nodes to zeros
            self.STs = np.zeros(self.M)

            # Calculate expected stock prices for each node
            for i in range(self.M):
                self.STs[i] = self.S0 * \
                    (self.u**(self.N-i)) * (self.d**i)

        def init_payoffs_tree(self):
            """
            Returns the payoffs when the option 
            expires at terminal nodes
            """ 
            if self.is_call:
                return np.maximum(0, self.STs-self.K)
            else:
                return np.maximum(0, self.K-self.STs)

        def traverse_tree(self, payoffs):
            """
            Starting from the time the option expires, traverse
            backwards and calculate discounted payoffs at each node
            """
            for i in range(self.N):
                payoffs = (payoffs[:-1]*self.qu + 
                           payoffs[1:]*self.qd)*self.df

            return payoffs

        def begin_tree_traversal(self):
            payoffs = self.init_payoffs_tree()
            return self.traverse_tree(payoffs)

        def price(self):
            """ Entry point of the pricing implementation """
            self.setup_parameters()
            self.init_stock_price_tree()
            payoffs = self.begin_tree_traversal()

            # Option value converges to first node
            return payoffs[0]
```

让我们使用我们之前讨论的两步二项式树示例中的值来定价欧式看跌期权：

```py
In [ ]:
    eu_option = BinomialEuropeanOption(
        50, 52, r=0.05, T=2, N=2, pu=0.2, pd=0.2, is_put=True)
In [ ]:
    print('European put option price is:', eu_option.price())
Out[ ]:    
    European put option price is: 4.1926542806038585
```

使用二项式期权定价模型，我们得到了欧式看跌期权的现值为 4.19 美元。

# 使用二项式树的美式期权类

与只能在到期时行使的欧式期权不同，美式期权可以在其寿命内的任何时候行使。

为了在 Python 中实现美式期权的定价，我们可以像`BinomialEuropeanOption`类一样创建一个名为`BinomialTreeOption`的类，该类继承自`Stockoption`类。`setup_parameters()`方法中使用的参数保持不变，只是删除了一个未使用的`M`参数。

美式期权中使用的方法如下：

+   `init_stock_price_tree`：使用二维 NumPy 数组存储所有时间步的股票价格的预期回报。这些信息用于计算在每个期间行使期权时的支付值。该方法编写如下：

```py
def init_stock_price_tree(self):
    # Initialize a 2D tree at T=0
    self.STs = [np.array([self.S0])]

    # Simulate the possible stock prices path
    for i in range(self.N):
        prev_branches = self.STs[-1]
        st = np.concatenate(
            (prev_branches*self.u, 
             [prev_branches[-1]*self.d]))
        self.STs.append(st) # Add nodes at each time step
```

+   `init_payoffs_tree`：创建支付树作为二维 NumPy 数组，从期满时期的期权内在价值开始。该方法编写如下：

```py
def init_payoffs_tree(self):
    if self.is_call:
        return np.maximum(0, self.STs[self.N]-self.K)
    else:
        return np.maximum(0, self.K-self.STs[self.N])
```

+   `check_early_exercise`：返回在提前行使美式期权和根本不行使期权之间的最大支付值。该方法编写如下：

```py
def check_early_exercise(self, payoffs, node):
    if self.is_call:
        return np.maximum(payoffs, self.STs[node] - self.K)
    else:
        return np.maximum(payoffs, self.K - self.STs[node])
```

+   `traverse_tree`：这还包括调用`check_early_exercise()`方法，以检查是否在每个时间步提前行使美式期权是最优的。该方法编写如下：

```py
def traverse_tree(self, payoffs):
    for i in reversed(range(self.N)):
        # The payoffs from NOT exercising the option
        payoffs = (payoffs[:-1]*self.qu + 
                   payoffs[1:]*self.qd)*self.df

        # Payoffs from exercising, for American options
        if not self.is_european:
            payoffs = self.check_early_exercise(payoffs,i)

    return payoffs
```

`begin_tree_traversal()`和`price()`方法的实现保持不变。

当在类的实例化期间将`is_put`关键字参数设置为`False`或`True`时，`BinomialTreeOption`类可以定价欧式和美式期权。

以下代码是用于定价美式期权的：

```py
In [ ]:
    am_option = BinomialTreeOption(50, 52, 
        r=0.05, T=2, N=2, pu=0.2, pd=0.2, is_put=True, is_am=True)
In [ ]:
    print('American put option price is:', am_option.price())
Out[ ]:    
    American put option price is: 5.089632474198373
```

美式看跌期权的价格为 5.0896 美元。由于美式期权可以在任何时候行使，而欧式期权只能在到期时行使，因此美式期权的这种灵活性在某些情况下增加了其价值。

对于不支付股息的基础资产的美式看涨期权，可能没有超过其欧式看涨期权对应的额外价值。由于时间价值的原因，今天在行权价上行使美式看涨期权的成本比在未来以相同行权价行使更高。对于实值的美式看涨期权，提前行使期权会失去对抗行权价以下不利价格波动的保护，以及其内在时间价值。没有股息支付的权利，没有动机提前行使美式看涨期权。

# Cox–Ross–Rubinstein 模型

在前面的例子中，我们假设基础股价在相应的*u*上升状态和*d*下降状态分别增加 20％和减少 20％。**Cox-Ross-Rubinstein**（**CRR**）模型提出，在风险中性世界的短时间内，二项式模型与基础股票的均值和方差相匹配。基础股票的波动性，或者股票回报的标准差，如下所示：

![](img/8a91adde-554a-429b-96bb-180e991f1ce2.png)

![](img/55fa99b2-ac5a-40bf-8adb-d09102cf749c.png)

# CRR 二项式树期权定价模型的类

二项式 CRR 模型的实现与我们之前讨论的二项式树相同，唯一的区别在于`u`和`d`模型参数。

在 Python 中，让我们创建一个名为`BinomialCRROption`的类，并简单地继承`BinomialTreeOption`类。然后，我们只需要覆盖`setup_parameters()`方法，使用 CRR 模型中的值。

`BinomialCRROption`对象的实例将调用`price()`方法，该方法调用父类`BinomialTreeOption`的所有其他方法，除了被覆盖的`setup_parameters()`方法：

```py
In [ ]:
    import math

    """ 
    Price an option by the binomial CRR model 
    """
    class BinomialCRROption(BinomialTreeOption):
        def setup_parameters(self):
            self.u = math.exp(self.sigma * math.sqrt(self.dt))
            self.d = 1./self.u
            self.qu = (math.exp((self.r-self.div)*self.dt) - 
                       self.d)/(self.u-self.d)
            self.qd = 1-self.qu
```

再次考虑两步二项式树。不支付股息的股票当前价格为 50 美元，波动率为 30％。假设无风险利率为年利率 5％，到期时间*T*为两年。我们想要找到 CRR 模型下的行权价为 52 美元的欧式看跌期权的价值：

```py
In [ ]:
    eu_option = BinomialCRROption(
        50, 52, r=0.05, T=2, N=2, sigma=0.3, is_put=True)
In [ ]:
    print('European put:', eu_option.price())
Out[ ]:
    European put: 6.245708445206436
In [ ]:
    am_option = BinomialCRROption(50, 52, 
        r=0.05, T=2, N=2, sigma=0.3, is_put=True, is_am=True)
In [ ]:
    print('American put option price is:', am_option.price())
Out[ ]:
    American put option price is: 7.428401902704834
```

通过使用 CRR 两步二项式树模型，欧式看跌期权和美式看跌期权的价格分别为 6.2457 美元和 7.4284 美元。

# 使用 Leisen-Reimer 树

在我们之前讨论的二项式模型中，我们对上升和下降状态的概率以及由此产生的风险中性概率做出了几个假设。除了我们讨论的具有 CRR 参数的二项式模型之外，在数学金融中广泛讨论的其他形式的参数化包括 Jarrow-Rudd 参数化、Tian 参数化和 Leisen-Reimer 参数化。让我们详细看看 Leisen-Reimer 模型。

Dietmar Leisen 博士和 Matthias Reimer 提出了一个二项式树模型，旨在在步数增加时逼近 Black-Scholes 解。它被称为**Leisen-Reimer**（**LR**）树，节点不会在每个交替步骤重新组合。它使用反演公式在树遍历期间实现更好的准确性。

有关该公式的详细解释可在 1995 年 3 月的论文*Binomial Models For Option Valuation - Examining And Improving Convergence*中找到，网址为[`papers.ssrn.com/sol3/papers.cfm?abstract_id=5976`](http://papers.ssrn.com/sol3/papers.cfm?abstract_id=5976)。我们将使用 Peizer 和 Pratt 反演函数*f*的第二种方法，具有以下特征参数：

![](img/9de89a37-6682-434b-b36a-c2397fa9ed49.png)

![](img/ec38b5a2-90e5-4563-b006-bb1b848c4512.png)

![](img/b8910449-3784-4f4e-a9d1-3ae2d5cc8834.png)

![](img/b38b80b9-8653-42e5-8463-ff6f3922b8eb.png)

![](img/07c34467-3bfb-468c-aab2-edc83990fdba.png)

![](img/648c1914-ac35-48f2-aec2-14d91b432121.png)

![](img/7e22357f-3cfd-4a3b-a57c-5bfad34f8b60.png)

![](img/2cf66a09-cded-4bac-88dc-c66e4a75b10e.png)

*S[0]*参数是当前股票价格，*K*是期权的行权价格，σ是基础股票的年化波动率，*T*是期权的到期时间，*r*是年化无风险利率，*y*是股息收益，*Δt*是每个树步之间的时间间隔。

# LR 二项树期权定价模型的一个类

LR 树的 Python 实现在以下`BinomialLROption`类中给出。与`BinomialCRROption`类类似，我们只需继承`BinomialTreeOption`类，并用 LR 树模型的变量覆盖`setup_parameters`方法中的变量：

```py
In [ ]:
    import math

    """ 
    Price an option by the Leisen-Reimer tree
    """
    class BinomialLROption(BinomialTreeOption):

        def setup_parameters(self):
            odd_N = self.N if (self.N%2 == 0) else (self.N+1)
            d1 = (math.log(self.S0/self.K) +
                  ((self.r-self.div) +
                   (self.sigma**2)/2.)*self.T)/\
                (self.sigma*math.sqrt(self.T))
            d2 = (math.log(self.S0/self.K) +
                  ((self.r-self.div) -
                   (self.sigma**2)/2.)*self.T)/\
                (self.sigma * math.sqrt(self.T))

            pbar = self.pp_2_inversion(d1, odd_N)
            self.p = self.pp_2_inversion(d2, odd_N)
            self.u = 1/self.df * pbar/self.p
            self.d = (1/self.df-self.p*self.u)/(1-self.p)
            self.qu = self.p
            self.qd = 1-self.p

        def pp_2_inversion(self, z, n):
            return .5 + math.copysign(1, z)*\
                math.sqrt(.25 - .25*
                    math.exp(
                        -((z/(n+1./3.+.1/(n+1)))**2.)*(n+1./6.)
                    )
                )
```

使用我们之前使用的相同示例，我们可以使用 LR 树定价期权：

```py
In [ ]:
    eu_option = BinomialLROption(
        50, 52, r=0.05, T=2, N=4, sigma=0.3, is_put=True)
In [ ]:
    print('European put:', eu_option.price())
Out[ ]:      
    European put: 5.878650106601964
In [ ]:
    am_option = BinomialLROption(50, 52, 
        r=0.05, T=2, N=4, sigma=0.3, is_put=True, is_am=True)
In [ ]:
    print('American put:', am_option.price())
Out[ ]:
    American put: 6.763641952939979
```

通过使用具有四个时间步长的 LR 二项树模型，欧式看跌期权的价格和美式看跌期权的价格分别为$5.87865 和$6.7636。

# 希腊字母免费

在我们迄今为止涵盖的二项树定价模型中，我们在每个时间点上上下遍历树来确定节点值。根据每个节点的信息，我们可以轻松地重用这些计算出的值。其中一种用途是计算希腊字母。

希腊字母衡量衍生品价格对基础资产参数变化的敏感性，例如期权，通常用希腊字母表示。在数学金融中，与希腊字母相关的常见名称包括 alpha、beta、delta、gamma、vega、theta 和 rho。

期权的两个特别有用的希腊字母是 delta 和 gamma。Delta 衡量期权价格对基础资产价格的敏感性。Gamma 衡量 delta 相对于基础价格的变化率。

如下图所示，在我们原始的两步树周围添加了额外的节点层，使其成为一个四步树，向时间向后延伸了两步。即使有额外的期末支付节点，所有节点将包含与我们原始的两步树相同的信息。我们感兴趣的期权价值现在位于树的中间，即**t=0**：

![](img/f93d5207-b1b0-4c63-b4be-ecc0bd5e3324.png)

注意，在**t=0**处存在两个额外节点的信息，我们可以使用它们来计算 delta 公式，如下所示：

![](img/da8a6f89-51c5-4052-8f68-cc0eb611150a.png)

三角洲公式规定，期权价格在上涨和下跌状态之间的差异表示为时间**t=0**时各自股票价格之间的差异的单位。

相反，gamma 公式可以计算如下：

![](img/6838f01f-47b1-4485-b121-1a222efddc4a.png)

伽玛公式规定，上节点和下节点中期权价格的 delta 之间的差异与初始节点值相对于各自状态下股票价格的差异的单位进行计算。

# LR 二项树的希腊字母类

为了说明在 LR 树中计算希腊字母的过程，让我们创建一个名为`BinomialLRWithGreeks`的新类，该类继承了`BinomialLROption`类，并使用我们自己的`price`方法的实现。

在`price`方法中，我们将首先调用父类的`setup_parameters()`方法来初始化 LR 树所需的所有变量。然而，这一次，我们还将调用`new_stock_price_tree()`方法，这是一个用于在原始树周围创建额外节点层的新方法。

调用`begin_tree_traversal()`方法执行父类中的通常 LR 树实现。返回的 NumPy 数组对象现在包含**t=0**处三个节点的信息，其中中间节点是期权价格。在数组的第一个和最后一个索引处是**t=0**处上升和下降状态的支付。

有了这些信息，`price()`方法计算并返回期权价格、delta 和 gamma 值：

```py
In [ ]:
    import numpy as np

    """ 
    Compute option price, delta and gamma by the LR tree 
    """
    class BinomialLRWithGreeks(BinomialLROption):

        def new_stock_price_tree(self):
            """
            Creates an additional layer of nodes to our
            original stock price tree
            """
            self.STs = [np.array([self.S0*self.u/self.d,
                                  self.S0,
                                  self.S0*self.d/self.u])]

            for i in range(self.N):
                prev_branches = self.STs[-1]
                st = np.concatenate((prev_branches*self.u,
                                     [prev_branches[-1]*self.d]))
                self.STs.append(st)

        def price(self):
            self.setup_parameters()
            self.new_stock_price_tree()
            payoffs = self.begin_tree_traversal()

            # Option value is now in the middle node at t=0
            option_value = payoffs[len(payoffs)//2]

            payoff_up = payoffs[0]
            payoff_down = payoffs[-1]
            S_up = self.STs[0][0]
            S_down = self.STs[0][-1]
            dS_up = S_up - self.S0
            dS_down = self.S0 - S_down

            # Calculate delta value
            dS = S_up - S_down
            dV = payoff_up - payoff_down
            delta = dV/dS

            # calculate gamma value
            gamma = ((payoff_up-option_value)/dS_up - 
                     (option_value-payoff_down)/dS_down) / \
                ((self.S0+S_up)/2\. - (self.S0+S_down)/2.)

            return option_value, delta, gamma

```

使用 LR 树的相同示例，我们可以计算具有 300 个时间步的欧式看涨期权和看跌期权的期权价值和希腊值：

```py
In [ ]:
    eu_call = BinomialLRWithGreeks(50, 52, r=0.05, T=2, N=300, sigma=0.3)
    results = eu_call.price()
In [ ]:
    print('European call values')
    print('Price: %s\nDelta: %s\nGamma: %s' % results)
Out[ ]:
    European call values
    Price: 9.69546807138366
    Delta: 0.6392477816643529
    Gamma: 0.01764795890533088

In [ ]:
    eu_put = BinomialLRWithGreeks(
        50, 52, r=0.05, T=2, N=300, sigma=0.3, is_put=True)
    results = eu_put.price()
In [ ]:
    print('European put values')
    print('Price: %s\nDelta: %s\nGamma: %s' % results)
Out[ ]:   
    European put values
    Price: 6.747013809252746
    Delta: -0.3607522183356649
    Gamma: 0.0176479589053312
```

从`price()`方法和结果中可以看出，我们成功地从修改后的二项树中获得了希腊附加信息，而没有增加计算复杂性。

# 期权定价中的三项树

在二项树中，每个节点导致下一个时间步中的两个其他节点。同样，在三项树中，每个节点导致下一个时间步中的三个其他节点。除了具有上升和下降状态外，三项树的中间节点表示状态不变。当扩展到两个以上的时间步时，三项树可以被视为重新组合树，其中中间节点始终保留与上一个时间步相同的值。

让我们考虑 Boyle 三项树，其中树被校准，使得上升、下降和平稳移动的概率*u*、*d*和*m*与风险中性概率*q[u]*、*q[d]*和*q[m]*如下：

![](img/6bccf093-15d1-474a-aa19-2bbb3b5b4669.png)

![](img/548f26bc-063d-49c1-b5a5-c5b88cf07150.png)

![](img/b6f6844b-4f98-41ca-bc2d-277b5d155c6b.png)

![](img/5f592c17-82bd-4348-be54-2fe893501558.png)

![](img/3331afe8-85f2-4160-9e74-aadfe9e7253f.png)

![](img/d6514bfb-2bfa-457a-8a95-eac1f6ec2e1a.png)

我们可以看到 ![](img/368560b2-b79d-4f65-adb1-50cf118f29dc.png) 重新组合为 *m =1*。通过校准，无状态移动 *m* 以 1 的固定利率增长，而不是以无风险利率增长。变量 *v* 是年化股息收益，*σ* 是基础股票的年化波动率。

一般来说，处理更多节点时，三项树在建模较少时间步时比二项树具有更好的精度，可以节省计算速度和资源。下图说明了具有两个时间步的三项树的股价变动：

![](img/ebfc055a-39ea-418c-a865-425c697b4255.png)

# 三项树期权定价模型的类

让我们创建一个`TrinomialTreeOption`类，继承自`BinomialTreeOption`类。

`TrinomialTreeOption`的方法如下所示：

+   `setup_parameters()`方法实现了三项树的模型参数。该方法编写如下：

```py
def setup_parameters(self):
    """ Required calculations for the model """
    self.u = math.exp(self.sigma*math.sqrt(2.*self.dt))
    self.d = 1/self.u
    self.m = 1
    self.qu = ((math.exp((self.r-self.div) *
                         self.dt/2.) -
                math.exp(-self.sigma *
                         math.sqrt(self.dt/2.))) /
               (math.exp(self.sigma *
                         math.sqrt(self.dt/2.)) -
                math.exp(-self.sigma *
                         math.sqrt(self.dt/2.))))**2
    self.qd = ((math.exp(self.sigma *
                         math.sqrt(self.dt/2.)) -
                math.exp((self.r-self.div) *
                         self.dt/2.)) /
               (math.exp(self.sigma *
                         math.sqrt(self.dt/2.)) -
                math.exp(-self.sigma *
                         math.sqrt(self.dt/2.))))**2.

    self.qm = 1 - self.qu - self.qd
```

+   `init_stock_price_tree()`方法设置了三项树，包括股价的平稳移动。该方法编写如下：

```py
def init_stock_price_tree(self):
    # Initialize a 2D tree at t=0
    self.STs = [np.array([self.S0])]

    for i in range(self.N):
        prev_nodes = self.STs[-1]
        self.ST = np.concatenate(
            (prev_nodes*self.u, [prev_nodes[-1]*self.m,
                                 prev_nodes[-1]*self.d]))
        self.STs.append(self.ST)
```

+   `traverse_tree()`方法在打折后考虑中间节点的收益：

```py
def traverse_tree(self, payoffs):
    # Traverse the tree backwards 
    for i in reversed(range(self.N)):
        payoffs = (payoffs[:-2] * self.qu +
                   payoffs[1:-1] * self.qm +
                   payoffs[2:] * self.qd) * self.df

        if not self.is_european:
            payoffs = self.check_early_exercise(payoffs,i)

    return payoffs
```

+   使用二项树的相同示例，我们得到以下结果：

```py
In [ ]:
   eu_put = TrinomialTreeOption(
        50, 52, r=0.05, T=2, N=2, sigma=0.3, is_put=True)
In [ ]:
   print('European put:', eu_put.price())
Out[ ]:
   European put: 6.573565269142496
In [ ]:
   am_option = TrinomialTreeOption(50, 52, 
        r=0.05, T=2, N=2, sigma=0.3, is_put=True, is_am=True)
In [ ]:
   print('American put:', am_option.price())
Out[ ]:
   American put: 7.161349217272585
```

通过三项树模型，我们得到了欧式看跌期权和美式看跌期权的价格分别为$6.57 和$7.16。

# 期权定价中的栅格

在二项树中，每个节点在每个交替节点处重新组合。在三项树中，每个节点在每个其他节点处重新组合。重新组合树的这种属性也可以表示为栅格，以节省内存而无需重新计算和存储重新组合的节点。

# 使用二项栅格

我们将从二项 CRR 树创建一个二项栅格，因为在每个交替的上升和下降节点处，价格重新组合为相同的*ud=1*概率。在下图中，**S[u]**和**S[d]**与**S[du]** = **S[ud]** = **S*[0]***重新组合。现在可以将树表示为单个列表：

![](img/1d131a2a-96bb-4663-9199-e84c4a8cbe84.png)

对于*N*步二项式树，需要一个大小为*2N +1*的列表来包含关于基础股票价格的信息。对于欧式期权定价，列表的奇数节点代表到期时的期权价值。树向后遍历以获得期权价值。对于美式期权定价，随着树向后遍历，列表的两端收缩，奇数节点代表任何时间步的相关股票价格。然后可以考虑早期行权的回报。

# CRR 二项式栅格期权定价模型的类

让我们通过 CRR 将二项式树定价转换为栅格。我们可以继承`BinomialCRROption`类（该类又继承自`BinomialTreeOption`类），并创建一个名为`BinomialCRRLattice`的新类，如下所示：

```py
In [ ]:
    import numpy as np

    class BinomialCRRLattice(BinomialCRROption):

        def setup_parameters(self):
            super(BinomialCRRLattice, self).setup_parameters()
            self.M = 2*self.N + 1

        def init_stock_price_tree(self):
            self.STs = np.zeros(self.M)
            self.STs[0] = self.S0 * self.u**self.N

            for i in range(self.M)[1:]:
                self.STs[i] = self.STs[i-1]*self.d

        def init_payoffs_tree(self):
            odd_nodes = self.STs[::2]  # Take odd nodes only
            if self.is_call:
                return np.maximum(0, odd_nodes-self.K)
            else:
                return np.maximum(0, self.K-odd_nodes)

        def check_early_exercise(self, payoffs, node):
            self.STs = self.STs[1:-1]  # Shorten ends of the list
            odd_STs = self.STs[::2]  # Take odd nodes only
            if self.is_call:
                return np.maximum(payoffs, odd_STs-self.K)
            else:
                return np.maximum(payoffs, self.K-odd_STs)
```

以下方法被覆盖，同时保留所有其他定价函数的行为：

+   `setup_parameters`：覆盖父类方法以初始化父类的 CRR 参数，并声明新变量`M`为列表大小

+   `init_stock_price_tree`：覆盖父类方法，设置一个一维 NumPy 数组作为具有`M`大小的栅格

+   `init_payoffs_tree`和`check_early_exercise`：覆盖父类方法，只考虑奇数节点的回报

使用我们二项式 CRR 模型示例中的相同股票信息，我们可以使用二项式栅格定价来定价欧式和美式看跌期权：

```py
In [ ]:
    eu_option = BinomialCRRLattice(
        50, 52, r=0.05, T=2, N=2, sigma=0.3, is_put=True)
In [ ] :
    print('European put:', eu_option.price())
Out[ ]:  European put: 6.245708445206432
In [ ]:
    am_option = BinomialCRRLattice(50, 52, 
        r=0.05, T=2, N=2, sigma=0.3, is_put=True, is_am=True)
In [ ] :
    print("American put:", am_option.price())
Out[ ]:   
    American put: 7.428401902704828
```

通过使用 CRR 二项式树格定价模型，我们得到了欧式和美式看跌期权的价格分别为$6.2457 和$7.428。

# 使用三项式栅格

三项式栅格与二项式栅格的工作方式基本相同。由于每个节点在每个其他节点重新组合，而不是交替节点，因此不需要从列表中提取奇数节点。由于列表的大小与二项式栅格中的大小相同，在三项式栅格定价中没有额外的存储要求，如下图所示：

![](img/ac536179-2cd2-4ee1-ba53-e3debaa9e4e6.png)

# 三项式栅格期权定价模型的类

在 Python 中，让我们创建一个名为`TrinomialLattice`的类，用于继承`TrinomialTreeOption`类的三项式栅格实现。

就像我们为`BinomialCRRLattice`类所做的那样，覆盖了`setup_parameters`、`init_stock_price_tree`、`init_payoffs_tree`和`check_early_exercise`方法，而不必考虑奇数节点的回报：

```py
In [ ]:
    import numpy as np

    """ 
    Price an option by the trinomial lattice 
    """
    class TrinomialLattice(TrinomialTreeOption):

        def setup_parameters(self):
            super(TrinomialLattice, self).setup_parameters()
            self.M = 2*self.N + 1

        def init_stock_price_tree(self):
            self.STs = np.zeros(self.M)
            self.STs[0] = self.S0 * self.u**self.N

            for i in range(self.M)[1:]:
                self.STs[i] = self.STs[i-1]*self.d

        def init_payoffs_tree(self):
            if self.is_call:
                return np.maximum(0, self.STs-self.K)
            else:
                return np.maximum(0, self.K-self.STs)

        def check_early_exercise(self, payoffs, node):
            self.STs = self.STs[1:-1]  # Shorten ends of the list
            if self.is_call:
                return np.maximum(payoffs, self.STs-self.K)
            else:
                return np.maximum(payoffs, self.K-self.STs)
```

使用与之前相同的示例，我们可以使用三项式栅格模型定价欧式和美式期权：

```py
In [ ]:
    eu_option = TrinomialLattice(
        50, 52, r=0.05, T=2, N=2, sigma=0.3, is_put=True)
    print('European put:', eu_option.price())
Out[ ]:
    European put: 6.573565269142496
In [ ]:
    am_option = TrinomialLattice(50, 52, 
        r=0.05, T=2, N=2, sigma=0.3, is_put=True, is_am=True)
    print('American put:', am_option.price())
Out[ ]:
    American put: 7.161349217272585
```

输出与从三项式树期权定价模型获得的结果一致。

# 期权定价中的有限差分

有限差分方案与三项式树期权定价非常相似，其中每个节点依赖于另外三个节点，即上升、下降和平移。有限差分的动机是应用 Black-Scholes**偏微分方程**（**PDE**）框架（涉及函数及其偏导数），其中价格*S(t)*是*f(S,t)*的函数，*r*是无风险利率，*t*是到期时间，*σ*是基础证券的波动率：

![](img/95125917-f109-4f59-82dd-44ff66ff99dc.png)

有限差分技术往往比栅格更快地收敛，并且很好地逼近复杂的异国期权。

通过有限差分向后工作来解决 PDE，建立大小为*M*乘以*N*的离散时间网格，以反映一段时间内的资产价格，使得*S*和*t*在网格上的每个点上取以下值：

由网格符号表示，*f[i,j]=f( idS, j dt)*。*S[max]*是一个适当大的资产价格，无法在到期时间*T*到达。因此*dS*和*dt*是网格中每个节点之间的间隔，分别由价格和时间递增。到期时间*T*的终端条件对于每个*S*的值是一个具有行权价*K*的看涨期权的*max(S − K, 0)*和一个具有行权价*K*的看跌期权的*max(K − S, 0)*。网格从终端条件向后遍历，遵守 PDE，同时遵守网格的边界条件，例如早期行权的支付。

边界条件是节点的两端的定义值，其中*i=0*和*i=N*对于每个时间*t*。边界处的值用于使用 PDE 迭代计算所有其他格点的值。

网格的可视化表示如下图所示。当*i*和*j*从网格的左上角增加时，价格*S*趋向于网格的右下角的*S[max]*（可能的最高价格）：

![](img/d8d7bd1b-8af0-4d25-bcdc-5ea644964aa5.png)

近似 PDE 的一些方法如下：

+   前向差分：

![](img/967661cb-b539-45f5-8a7f-999297c8d67d.png)

+   后向差分：

![](img/1208b133-921f-4777-992a-5dbfa9349a1a.png)

+   中心或对称差分：

![](img/63ff2ce0-6167-479e-a73b-92aadde910b7.png)

+   二阶导数：

![](img/5886183c-bf2c-4320-8bbe-0ffbcc29da77.png)

一旦我们设置好边界条件，现在可以使用显式、隐式或 Crank-Nicolson 方法进行迭代处理。 

# 显式方法

用于近似*f[i,j]*的显式方法由以下方程给出：

![](img/cd4d03e8-dab2-4445-bfbe-4f353e2b5985.png)

在这里，我们可以看到第一个差分是关于*t*的后向差分，第二个差分是关于*S*的中心差分，第三个差分是关于*S*的二阶差分。当我们重新排列项时，我们得到以下方程：

![](img/1c511137-07bc-420c-a905-b4084a60e3e4.png)

其中：

![](img/2f7a5ad4-5e48-42af-9366-c1d7818baf68.png)

![](img/c5f4f25e-9d7a-484f-b8e5-336f01d49559.png)

然后：

![](img/0db97616-b3a6-44c4-92c8-9f97fb0af73b.png)

![](img/496c44aa-5985-4a78-8cec-0e4111dbfcf2.png)

![](img/e3caf1b2-b486-4785-a6d9-40e6531a77ed.png)

显式方法的迭代方法可以通过以下图表进行可视化表示：

![](img/2b4772e0-cba3-4545-b93f-c264c4f564a0.png)

# 编写有限差分基类

由于我们将在 Python 中编写有限差分的显式、隐式和 Crank-Nicolson 方法，让我们编写一个基类，该基类继承了所有三种方法的共同属性和函数。

我们将创建一个名为`FiniteDifferences`的类，该类在`__init__`构造方法中接受并分配所有必需的参数。`price()`方法是调用特定有限差分方案实现的入口点，并将按以下顺序调用这些方法：`setup_boundary_conditions()`、`setup_coefficients()`、`traverse_grid()`和`interpolate()`。这些方法的解释如下：

+   `setup_boundary_conditions`：设置网格结构的边界条件为 NumPy 二维数组

+   `setup_coefficients`：设置用于遍历网格结构的必要系数

+   `traverse_grid`：向后迭代网格结构，将计算值存储到网格的第一列

+   `interpolate`：使用网格第一列上的最终计算值，这种方法将插值这些值以找到最接近初始股价`S0`的期权价格

所有这些方法都是可以由派生类实现的抽象方法。如果我们忘记实现这些方法，将抛出`NotImplementedError`异常类型。

基类应该包含以下强制方法：

```py
In [ ]:
    from abc import ABC, abstractmethod
    import numpy as np

    """ 
    Base class for sharing 
    attributes and functions of FD 
    """
    class FiniteDifferences(object):

        def __init__(
            self, S0, K, r=0.05, T=1, 
            sigma=0, Smax=1, M=1, N=1, is_put=False
        ):
            self.S0 = S0
            self.K = K
            self.r = r
            self.T = T
            self.sigma = sigma
            self.Smax = Smax
            self.M, self.N = M, N
            self.is_call = not is_put

            self.i_values = np.arange(self.M)
            self.j_values = np.arange(self.N)
            self.grid = np.zeros(shape=(self.M+1, self.N+1))
            self.boundary_conds = np.linspace(0, Smax, self.M+1)

        @abstractmethod
        def setup_boundary_conditions(self):
            raise NotImplementedError('Implementation required!')

        @abstractmethod
        def setup_coefficients(self):
            raise NotImplementedError('Implementation required!')

        @abstractmethod
        def traverse_grid(self):
            """  Iterate the grid backwards in time"""
            raise NotImplementedError('Implementation required!')

        @abstractmethod
        def interpolate(self):
            """ Use piecewise linear interpolation on the initial
            grid column to get the closest price at S0.
            """
            return np.interp(
                self.S0, self.boundary_conds, self.grid[:,0])
```

**抽象基类**（**ABCs**）提供了定义类接口的方法。`@abstractmethod()`装饰器声明了子类应该实现的抽象方法。与 Java 的抽象方法不同，这些方法可能有一个实现，并且可以通过`super()`机制从覆盖它的类中调用。

除了这些方法，我们还需要定义`dS`和`dt`，即每单位时间内`S`的变化和每次迭代中`T`的变化。我们可以将这些定义为类属性：

```py
@property
def dS(self):
    return self.Smax/float(self.M)

@property
def dt(self):
    return self.T/float(self.N)
```

最后，将`price()`方法添加为入口点，显示调用我们讨论的抽象方法的步骤：

```py
def price(self):
    self.setup_boundary_conditions()
    self.setup_coefficients()
    self.traverse_grid()
    return self.interpolate()
```

# 使用有限差分的显式方法对欧式期权进行定价的类

使用显式方法的有限差分的 Python 实现如下`FDExplicitEu`类，它继承自`FiniteDifferences`类并覆盖了所需的实现方法：

```py
In [ ]:
    import numpy as np

    """ 
    Explicit method of Finite Differences 
    """
    class FDExplicitEu(FiniteDifferences):

        def setup_boundary_conditions(self):
            if self.is_call:
                self.grid[:,-1] = np.maximum(
                    0, self.boundary_conds - self.K)
                self.grid[-1,:-1] = (self.Smax-self.K) * \
                    np.exp(-self.r*self.dt*(self.N-self.j_values))
            else:
                self.grid[:,-1] = np.maximum(
                    0, self.K-self.boundary_conds)
                self.grid[0,:-1] = (self.K-self.Smax) * \
                    np.exp(-self.r*self.dt*(self.N-self.j_values))

        def setup_coefficients(self):
            self.a = 0.5*self.dt*((self.sigma**2) *
                                  (self.i_values**2) -
                                  self.r*self.i_values)
            self.b = 1 - self.dt*((self.sigma**2) *
                                  (self.i_values**2) +
                                  self.r)
            self.c = 0.5*self.dt*((self.sigma**2) *
                                  (self.i_values**2) +
                                  self.r*self.i_values)

        def traverse_grid(self):
            for j in reversed(self.j_values):
                for i in range(self.M)[2:]:
                    self.grid[i,j] = \
                        self.a[i]*self.grid[i-1,j+1] +\
                        self.b[i]*self.grid[i,j+1] + \
                        self.c[i]*self.grid[i+1,j+1]
```

完成网格结构的遍历后，第一列包含**t=0**时刻的初始资产价格的现值。NumPy 的`interp`函数用于执行线性插值以近似期权价值。

除了线性插值作为插值方法的最常见选择外，还可以使用其他方法，如样条或三次插值来近似期权价值。

考虑一个欧式看跌期权的例子。标的股票价格为 50 美元，波动率为 40%。看跌期权的行权价为 50 美元，到期时间为五个月。无风险利率为 10%。

我们可以使用显式方法对该期权进行定价，`Smax`值为`100`，`M`值为`100`，`N`值为`1000`：

```py
In [ ]:
    option = FDExplicitEu(50, 50, r=0.1, T=5./12., 
        sigma=0.4, Smax=100, M=100, N=1000, is_put=True)
    print(option.price())
Out[ ]:
    4.072882278148043
```

当选择其他不合适的`M`和`N`值时会发生什么？

```py
In [ ]:
    option = FDExplicitEu(50, 50, r=0.1, T=5./12., 
        sigma=0.4, Smax=100, M=80, N=100, is_put=True)
    print(option.price())
Out[ ]:   
    -8.109445694129245e+35
```

显然，有限差分方案的显式方法存在不稳定性问题。

# 隐式方法

显式方法的不稳定问题可以通过对时间的前向差分来克服。用于近似*f[i,j]*的隐式方法由以下方程给出：

![](img/2b6e9a19-c56e-4d89-9da4-4fec358a90b3.png)

在这里，可以看到隐式和显式近似方案之间唯一的区别在于第一个差分，隐式方案中使用了对*t*的前向差分。当我们重新排列项时，我们得到以下表达式：

![](img/2d290bf9-5f31-463a-819c-8efaed57c26b.png)

其中：

![](img/1cd1639f-1740-4616-b9f7-9e45fb885cda.png)

![](img/7d83d409-627f-4267-b857-f714a158fad8.png)

在这里：

![](img/193c6822-3155-41e6-b4b8-110a15e4354d.png)

![](img/3a7cd64e-4488-47fb-a3eb-588da2b9c77f.png)

![](img/42dc58ea-64de-49be-947b-5b397c2ada52.png)

隐式方案的迭代方法可以用以下图表进行可视化表示：

![](img/b760a196-4c5e-48a6-a6d3-67799d0a657b.png)

从前面的图表中，我们可以注意到需要在下一次迭代步骤中计算出*j+1*的值，因为网格是向后遍历的。在隐式方案中，网格可以被认为在每次迭代中代表一个线性方程组，如下所示：

![](img/8978c0d4-f8ef-421c-89e0-713504b1dcd4.png)

通过重新排列项，我们得到以下方程：

![](img/f23e2337-d4f1-458d-8409-1f55a40b9b69.png)

线性方程组可以表示为*Ax = B*的形式，我们希望在每次迭代中解出*x*的值。由于矩阵*A*是三对角的，我们可以使用 LU 分解，其中*A=LU*，以加快计算速度。请记住，我们在第二章中使用 LU 分解解出了线性方程组，该章节名为《金融中的线性关系的重要性》。

# 使用有限差分的隐式方法对欧式期权进行定价的类

隐式方案的 Python 实现在以下`FDImplicitEu`类中给出。我们可以从之前讨论的`FDExplicitEu`类中继承显式方法的实现，并覆盖感兴趣的必要方法，即`setup_coefficients`和`traverse_grid`方法：

```py
In [ ]:
    import numpy as np
    import scipy.linalg as linalg

    """ 
    Explicit method of Finite Differences 
    """
    class FDImplicitEu(FDExplicitEu):

        def setup_coefficients(self):
            self.a = 0.5*(self.r*self.dt*self.i_values -
                          (self.sigma**2)*self.dt*\
                              (self.i_values**2))
            self.b = 1 + \
                     (self.sigma**2)*self.dt*\
                        (self.i_values**2) + \
                    self.r*self.dt
            self.c = -0.5*(self.r*self.dt*self.i_values +
                           (self.sigma**2)*self.dt*\
                               (self.i_values**2))
            self.coeffs = np.diag(self.a[2:self.M],-1) + \
                          np.diag(self.b[1:self.M]) + \
                          np.diag(self.c[1:self.M-1],1)

        def traverse_grid(self):
            """ Solve using linear systems of equations """
            P, L, U = linalg.lu(self.coeffs)
            aux = np.zeros(self.M-1)

            for j in reversed(range(self.N)):
                aux[0] = np.dot(-self.a[1], self.grid[0, j])
                x1 = linalg.solve(L, self.grid[1:self.M, j+1]+aux)
                x2 = linalg.solve(U, x1)
                self.grid[1:self.M, j] = x2
```

使用与显式方案相同的示例，我们可以使用隐式方案定价欧式看跌期权：

```py
In [ ]:
    option = FDImplicitEu(50, 50, r=0.1, T=5./12., 
        sigma=0.4, Smax=100, M=100, N=1000, is_put=True)
    print(option.price())
Out[ ]:
    4.071594188049893
In [ ]:
    option = FDImplicitEu(50, 50, r=0.1, T=5./12., 
        sigma=0.4, Smax=100, M=80, N=100, is_put=True)
    print(option.price())
Out[ ]:
    4.063684691731647
```

鉴于当前参数和输入数据，我们可以看到隐式方案没有稳定性问题。

# Crank-Nicolson 方法

另一种避免稳定性问题的方法，如显式方法中所见，是使用 Crank-Nicolson 方法。Crank-Nicolson 方法通过使用显式和隐式方法的组合更快地收敛，取两者的平均值。这导致以下方程：

![](img/a8840a9a-b4a3-408f-a0ba-a82b85b3767e.png)

这个方程也可以重写如下：

![](img/f847380f-ae9d-46f6-bfa1-2b17617b7e49.png)

其中：

![](img/58d5e72d-41d6-4dad-bd00-d0cee25e8b34.png)

隐式方案的迭代方法可以用以下图表形式表示：

![](img/25895799-5f87-4158-970c-988563cc595c.png)

我们可以将方程视为矩阵形式的线性方程组：

![](img/08f3b87b-e8dc-4d42-8a72-87b0e990a26f.png)

其中：

![](img/0955b807-c75c-47ac-8abd-2256d570b329.png)

![](img/d26eccb2-880e-41b4-9ddd-302affc103b1.png)

![](img/a92e8ca5-5361-478b-9cbf-85edad31fa45.png)

我们可以在每个迭代过程中解出矩阵*M*。

# 使用有限差分的 Crank-Nicolson 方法定价欧式期权的类

Crank-Nicolson 方法的 Python 实现在以下`FDCnEu`类中给出，该类继承自`FDExplicitEu`类，并仅覆盖`setup_coefficients`和`traverse_grid`方法：

```py
In [ ]:
    import numpy as np
    import scipy.linalg as linalg

    """ 
    Crank-Nicolson method of Finite Differences 
    """
    class FDCnEu(FDExplicitEu):

        def setup_coefficients(self):
            self.alpha = 0.25*self.dt*(
                (self.sigma**2)*(self.i_values**2) - \
                self.r*self.i_values)
            self.beta = -self.dt*0.5*(
                (self.sigma**2)*(self.i_values**2) + self.r)
            self.gamma = 0.25*self.dt*(
                (self.sigma**2)*(self.i_values**2) +
                self.r*self.i_values)
            self.M1 = -np.diag(self.alpha[2:self.M], -1) + \
                      np.diag(1-self.beta[1:self.M]) - \
                      np.diag(self.gamma[1:self.M-1], 1)
            self.M2 = np.diag(self.alpha[2:self.M], -1) + \
                      np.diag(1+self.beta[1:self.M]) + \
                      np.diag(self.gamma[1:self.M-1], 1)

        def traverse_grid(self):
            """ Solve using linear systems of equations """
            P, L, U = linalg.lu(self.M1)

            for j in reversed(range(self.N)):
                x1 = linalg.solve(
                    L, np.dot(self.M2, self.grid[1:self.M, j+1]))
                x2 = linalg.solve(U, x1)
                self.grid[1:self.M, j] = x2
```

使用与显式和隐式方法相同的示例，我们可以使用 Crank-Nicolson 方法为不同的时间点间隔定价欧式看跌期权：

```py
In [ ]:
    option = FDCnEu(50, 50, r=0.1, T=5./12.,
        sigma=0.4, Smax=100, M=100, N=1000, is_put=True)
    print(option.price())
Out[ ]:   
    4.072238354486825
In [ ]:
    option = FDCnEu(50, 50, r=0.1, T=5./12., 
        sigma=0.4, Smax=100, M=80, N=100, is_put=True)
    print(option.price())
Out[ ]: 
    4.070145703042843
```

从观察到的值来看，Crank-Nicolson 方法不仅避免了我们在显式方案中看到的不稳定性问题，而且比显式和隐式方法都更快地收敛。隐式方法需要更多的迭代，或者更大的*N*值，才能产生接近 Crank-Nicolson 方法的值。

# 定价异国情调的障碍期权

有限差分在定价异国情调期权方面特别有用。期权的性质将决定边界条件的规格。

在本节中，我们将看一个使用有限差分的 Crank-Nicolson 方法定价低迷障碍期权的例子。由于其相对复杂性，通常会使用其他分析方法，如蒙特卡罗方法，而不是有限差分方案。

# 一种低迷的选择

让我们看一个低迷期权的例子。在期权的任何生命周期中，如果标的资产价格低于*S[barrier]*障碍价格，则认为该期权毫无价值。由于在网格中，有限差分方案代表所有可能的价格点，我们只需要考虑以下价格范围的节点：

![](img/7aa99c36-a86a-45d1-b1d4-0b6f30bd9371.png)

然后我们可以设置边界条件如下：

![](img/843cd7f2-7cec-4c82-b4bd-dc3debce8798.png)

# 使用有限差分的 Crank-Nicolson 方法定价低迷期权的类

让我们创建一个名为`FDCnDo`的类，它继承自之前讨论的`FDCnEu`类。我们将在构造方法中考虑障碍价格，而将`FDCnEu`类中的 Crank-Nicolson 实现的其余部分保持不变：

```py
In [ ]:
    import numpy as np

    """
    Price a down-and-out option by the Crank-Nicolson
    method of finite differences.
    """
    class FDCnDo(FDCnEu):

        def __init__(
            self, S0, K, r=0.05, T=1, sigma=0, 
            Sbarrier=0, Smax=1, M=1, N=1, is_put=False
        ):
            super(FDCnDo, self).__init__(
                S0, K, r=r, T=T, sigma=sigma,
                Smax=Smax, M=M, N=N, is_put=is_put
            )
            self.barrier = Sbarrier
            self.boundary_conds = np.linspace(Sbarrier, Smax, M+1)
            self.i_values = self.boundary_conds/self.dS

        @property
        def dS(self):
            return (self.Smax-self.barrier)/float(self.M)
```

让我们考虑一个敲出期权的例子。标的股票价格为 50 美元，波动率为 40%。期权的行权价为 50 美元，到期时间为五个月。无风险利率为 10%。敲出价格为 40 美元。

我们可以使用 `Smax` 为 `100`，`M` 为 `120`，`N` 为 `500` 来定价看涨期权和敲出看跌期权：

```py
In [ ]:
    option = FDCnDo(50, 50, r=0.1, T=5./12., 
        sigma=0.4, Sbarrier=40, Smax=100, M=120, N=500)
    print(option.price())
Out[ ]:   
    5.491560552934787
In [ ]:
    option = FDCnDo(50, 50, r=0.1, T=5./12., sigma=0.4, 
        Sbarrier=40, Smax=100, M=120, N=500, is_put=True)
    print(option.price())
Out[ ]:
   0.5413635028954452
```

敲出看涨期权和敲出看跌期权的价格分别为 5.4916 美元和 0.5414 美元。

# 使用有限差分定价美式期权

到目前为止，我们已经定价了欧式期权和奇异期权。由于美式期权中存在提前行权的可能性，因此定价此类期权并不那么直接。在隐式 Crank-Nicolson 方法中需要迭代过程，当前期内的提前行权收益要考虑先前期内的提前行权收益。在 Crank-Nicolson 方法中，建议使用高斯-西德尔迭代方法定价美式期权。

回顾一下，在第二章中，我们讨论了在金融中线性性的重要性，我们介绍了解决线性方程组的高斯-西德尔方法，形式为 *Ax=B*。在这里，矩阵 *A* 被分解为 *A=L+U*，其中 *L* 是下三角矩阵，*U* 是上三角矩阵。让我们看一个 4 x 4 矩阵 *A* 的例子：

![](img/6437c715-0e2b-426c-b17f-cbefd8a42c2b.png)

然后通过迭代方式获得解决方案，如下所示：

![](img/c141f535-da24-4178-a6a0-8500d7d7bd5a.png)

我们可以将高斯-西德尔方法调整到我们的 Crank-Nicolson 实现中，如下所示：

![](img/11cd3627-b98b-407a-b244-cc620283ca3c.png)

这个方程满足提前行权特权方程：

![](img/596908ed-7de4-4a12-ba73-f3b28424570a.png)

# 使用有限差分的 Crank-Nicolson 方法定价美式期权的类

让我们创建一个名为 `FDCnAm` 的类，该类继承自 `FDCnEu` 类，后者是定价欧式期权的 Crank-Nicolson 方法的对应物。`setup_coefficients` 方法可以被重用，同时覆盖所有其他方法，以包括先前行权的收益，如果有的话。

`__init__()` 构造函数和 `setup_boundary_conditions()` 方法在 `FDCnAm` 类中给出：

```py
In [ ]:
    import numpy as np
    import sys

    """ 
    Price an American option by the Crank-Nicolson method 
    """
    class FDCnAm(FDCnEu):

        def __init__(self, S0, K, r=0.05, T=1, 
                Smax=1, M=1, N=1, omega=1, tol=0, is_put=False):
            super(FDCnAm, self).__init__(S0, K, r=r, T=T, 
                sigma=sigma, Smax=Smax, M=M, N=N, is_put=is_put)
            self.omega = omega
            self.tol = tol
            self.i_values = np.arange(self.M+1)
            self.j_values = np.arange(self.N+1)

        def setup_boundary_conditions(self):
            if self.is_call:
                self.payoffs = np.maximum(0, 
                    self.boundary_conds[1:self.M]-self.K)
            else:
                self.payoffs = np.maximum(0, 
                    self.K-self.boundary_conds[1:self.M])

            self.past_values = self.payoffs
            self.boundary_values = self.K * np.exp(
                    -self.r*self.dt*(self.N-self.j_values))
```

接下来，在同一类中实现 `traverse_grid()` 方法：

```py
def traverse_grid(self):
    """ Solve using linear systems of equations """
    aux = np.zeros(self.M-1)
    new_values = np.zeros(self.M-1)

    for j in reversed(range(self.N)):
        aux[0] = self.alpha[1]*(self.boundary_values[j] +
                                self.boundary_values[j+1])
        rhs = np.dot(self.M2, self.past_values) + aux
        old_values = np.copy(self.past_values)
        error = sys.float_info.max

        while self.tol < error:
            new_values[0] = \
                self.calculate_payoff_start_boundary(
                    rhs, old_values)    

            for k in range(self.M-2)[1:]:
                new_values[k] = \
                    self.calculate_payoff(
                        k, rhs, old_values, new_values)                  

            new_values[-1] = \
                self.calculate_payoff_end_boundary(
                    rhs, old_values, new_values)

            error = np.linalg.norm(new_values-old_values)
            old_values = np.copy(new_values)

        self.past_values = np.copy(new_values)

    self.values = np.concatenate(
        ([self.boundary_values[0]], new_values, [0]))
```

在 `while` 循环的每个迭代过程中，计算收益时要考虑开始和结束边界。此外，`new_values` 不断根据现有和先前的值进行新的收益计算替换。

在开始边界处，索引为 0 时，通过省略 alpha 值来计算收益。在类内实现 `calculate_payoff_start_boundary()` 方法：

```py
 def calculate_payoff_start_boundary(self, rhs, old_values):
    payoff = old_values[0] + \
        self.omega/(1-self.beta[1]) * \
            (rhs[0] - \
             (1-self.beta[1])*old_values[0] + \
             self.gamma[1]*old_values[1])

    return max(self.payoffs[0], payoff)       
```

在结束边界处，最后一个索引时，通过省略 gamma 值来计算收益。在类内实现 `calculate_payoff_end_boundary()` 方法：

```py
 def calculate_payoff_end_boundary(self, rhs, old_values, new_values):
    payoff = old_values[-1] + \
        self.omega/(1-self.beta[-2]) * \
            (rhs[-1] + \
             self.alpha[-2]*new_values[-2] - \
             (1-self.beta[-2])*old_values[-1])

    return max(self.payoffs[-1], payoff)
```

对于不在边界的收益，通过考虑 alpha 和 gamma 值来计算收益。在类内实现 `calculate_payoff()` 方法：

```py
def calculate_payoff(self, k, rhs, old_values, new_values):
    payoff = old_values[k] + \
        self.omega/(1-self.beta[k+1]) * \
            (rhs[k] + \
             self.alpha[k+1]*new_values[k-1] - \
             (1-self.beta[k+1])*old_values[k] + \
             self.gamma[k+1]*old_values[k+1])

    return max(self.payoffs[k], payoff)
```

由于新变量 `values` 包含我们的终端收益值作为一维数组，因此重写父类的 `interpolate` 方法以考虑这一变化，使用以下代码：

```py
def interpolate(self):
    # Use linear interpolation on final values as 1D array
    return np.interp(self.S0, self.boundary_conds, self.values)
```

容差参数用于高斯-西德尔方法的收敛标准。`omega` 变量是过度松弛参数。更高的 `omega` 值提供更快的收敛速度，但这也伴随着算法不收敛的可能性更高。

让我们定价一个标的资产价格为 50，波动率为 40%，行权价为 50，无风险利率为 10%，到期日为五个月的美式看涨期权和看跌期权。我们选择 `Smax` 值为 `100`，`M` 为 `100`，`N` 为 `42`，`omega` 参数值为 `1.2`，容差值为 `0.001`：

```py
In [ ]:
    option = FDCnAm(50, 50, r=0.1, T=5./12., 
        sigma=0.4, Smax=100, M=100, N=42, omega=1.2, tol=0.001)
    print(option.price())
Out[ ]:
    6.108682815392217
In [ ]:
    option = FDCnAm(50, 50, r=0.1, T=5./12., sigma=0.4, Smax=100, 
        M=100, N=42, omega=1.2, tol=0.001, is_put=True)
    print(option.price())
Out[ ]:   
    4.277764229383736
```

使用 Crank-Nicolson 方法计算美式看涨和看跌期权的价格分别为 6.109 美元和 4.2778 美元。

# 将所有内容整合在一起-隐含波动率建模

到目前为止，我们学到的期权定价方法中，有一些参数被假定为常数：利率、行权价、股息和波动率。在这里，感兴趣的参数是波动率。在定量研究中，波动率比率被用来预测价格趋势。

要得出隐含波动率，我们需要参考第三章*金融中的非线性*，在那里我们讨论了非线性函数的根查找方法。在我们的下一个示例中，我们将使用数值程序的二分法来创建一个隐含波动率曲线。

# AAPL 美式看跌期权的隐含波动率

让我们考虑股票**苹果**（**AAPL**）的期权数据，这些数据是在 2014 年 10 月 3 日收集的。以下表格提供了这些细节。期权到期日为 2014 年 12 月 20 日。所列价格为买入价和卖出价的中间价：

| **行权价** | **看涨期权价格** | **看跌期权价格** |
| --- | --- | --- |
| 75 | 30 | 0.16 |
| 80 | 24.55 | 0.32 |
| 85 | 20.1 | 0.6 |
| 90 | 15.37 | 1.22 |
| 92.5 | 10.7 | 1.77 |
| 95 | 8.9 | 2.54 |
| 97.5 | 6.95 | 3.55 |
| 100 | 5.4 | 4.8 |
| 105 | 4.1 | 7.75 |
| 110 | 2.18 | 11.8 |
| 115 | 1.05 | 15.96 |
| 120 | 0.5 | 20.75 |
| 125 | 0.26 | 25.8 |

AAPL 的最后交易价格为 99.62 美元，利率为 2.48%，股息率为 1.82%。美式期权在 78 天后到期。

利用这些信息，让我们创建一个名为`ImpliedVolatilityModel`的新类，它在构造函数中接受股票期权的参数。如果需要，导入我们在本章前面部分介绍的用于 LR 二项树的`BinomialLROption`类。还需要导入我们在第三章*金融中的非线性*中介绍的`bisection`函数。

`option_valuation()`方法接受`K`行权价和`sigma`波动率值，计算期权的价值。在这个例子中，我们使用`BinomialLROption`定价方法。

`get_implied_volatilities()`方法接受一个行权价和期权价格的列表，通过`bisection`方法计算每个价格的隐含波动率。因此，这两个列表的长度必须相同。

`ImpliedVolatilityModel`类的 Python 代码如下所示：

```py
In [ ]:
    """
    Get implied volatilities from a Leisen-Reimer binomial
    tree using the bisection method as the numerical procedure.
    """
    class ImpliedVolatilityModel(object):

        def __init__(self, S0, r=0.05, T=1, div=0, 
                     N=1, is_put=False):
            self.S0 = S0
            self.r = r
            self.T = T
            self.div = div
            self.N = N
            self.is_put = is_put

        def option_valuation(self, K, sigma):
            """ Use the binomial Leisen-Reimer tree """
            lr_option = BinomialLROption(
                self.S0, K, r=self.r, T=self.T, N=self.N, 
                sigma=sigma, div=self.div, is_put=self.is_put
            )
            return lr_option.price()

        def get_implied_volatilities(self, Ks, opt_prices):
            impvols = []
            for i in range(len(strikes)):
                # Bind f(sigma) for use by the bisection method
                f = lambda sigma: \
                    self.option_valuation(Ks[i], sigma)-\
                    opt_prices[i]
                impv = bisection(f, 0.01, 0.99, 0.0001, 100)[0]
                impvols.append(impv)

            return impvols

```

导入我们在上一章讨论过的`bisection`函数：

```py
In [ ]:
    def bisection(f, a, b, tol=0.1, maxiter=10):
        """
        :param f: The function to solve
        :param a: The x-axis value where f(a)<0
        :param b: The x-axis value where f(b)>0
        :param tol: The precision of the solution
        :param maxiter: Maximum number of iterations
        :return: The x-axis value of the root,
                    number of iterations used
        """
        c = (a+b)*0.5  # Declare c as the midpoint ab
        n = 1  # Start with 1 iteration
        while n <= maxiter:
            c = (a+b)*0.5
            if f(c) == 0 or abs(a-b)*0.5 < tol:
                # Root is found or is very close
                return c, n

            n += 1
            if f(c) < 0:
                a = c
            else:
                b = c

        return c, n
```

利用这个模型，让我们使用这组特定数据找出美式看跌期权的隐含波动率：

```py
In [ ]:
    strikes = [75, 80, 85, 90, 92.5, 95, 97.5, 
               100, 105, 110, 115, 120, 125]
    put_prices = [0.16, 0.32, 0.6, 1.22, 1.77, 2.54, 3.55, 
                  4.8, 7.75, 11.8, 15.96, 20.75, 25.81]
In [ ]:
    model = ImpliedVolatilityModel(
        99.62, r=0.0248, T=78/365., div=0.0182, N=77, is_put=True)
    impvols_put = model.get_implied_volatilities(strikes, put_prices)
```

隐含波动率值现在以`list`对象的形式存储在`impvols_put`变量中。让我们将这些值绘制成隐含波动率曲线：

```py
In [ ]:
    %matplotlib inline
    import matplotlib.pyplot as plt

    plt.plot(strikes, impvols_put)
    plt.xlabel('Strike Prices')
    plt.ylabel('Implied Volatilities')
    plt.title('AAPL Put Implied Volatilities expiring in 78 days')
    plt.show()
```

这将给我们提供波动率微笑，如下图所示。在这里，我们建立了一个包含 77 个步骤的 LR 树，每一步代表一天：

![](img/f5e906bf-b5cc-4c74-bc7a-1881acfd58e2.png)

当然，每天定价一个期权可能并不理想，因为市场变化是以毫秒为单位的。我们使用了二分法来解决隐含波动率，这是由二项树隐含的，而不是直接从市场价格观察到的实现波动率值。

我们是否应该将这条曲线与多项式曲线拟合，以确定潜在的套利机会？或者推断曲线，以从远离实值和虚值期权的隐含波动率中获得更多见解？好吧，这些问题是供像你这样的期权交易员去发现的！

# 总结

在本章中，我们研究了衍生品定价中的一些数值程序，最常见的是期权。其中一种程序是使用树，二叉树是最简单的结构来建模资产信息，其中一个节点在每个时间步长延伸到另外两个节点，分别代表上升状态和下降状态。在三叉树中，每个节点在每个时间步长延伸到另外三个节点，分别代表上升状态、下降状态和无移动状态。随着树向上遍历，基础资产在每个节点处被计算和表示。然后期权采用这棵树的结构，并从期末回溯并向根部遍历，收敛到当前折现期权价格。除了二叉树和三叉树，树还可以采用 CRR、Jarrow-Rudd、Tian 或 LR 参数的形式。

通过在我们的树周围添加另一层节点，我们引入了额外的信息，从中我们可以推导出希腊字母，如 delta 和 gamma，而不会产生额外的计算成本。

晶格被引入是为了节省存储成本，相比二叉树和三叉树。在晶格定价中，只保存具有新信息的节点一次，并在以后需要不改变信息的节点上重复使用。

我们还讨论了期权定价中的有限差分方案，包括期末和边界条件。从期末条件开始，网格使用显式方法、隐式方法和 Crank-Nicolson 方法向后遍历时间。除了定价欧式和美式期权，有限差分定价方案还可以用于定价异国期权，我们看了一个定价下触及障碍期权的例子。

通过引入在第三章中学到的二分根查找方法，以及本章中的二叉 LR 树模型，我们使用美式期权的市场价格来创建隐含波动率曲线以进行进一步研究。

在下一章中，我们将研究利率和衍生品建模。
