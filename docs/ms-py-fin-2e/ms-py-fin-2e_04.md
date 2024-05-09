# 第二章：金融中线性的重要性

非线性动力学在我们的世界中起着至关重要的作用。由于更容易研究和更容易建模的能力，线性模型经常在经济学中使用。在金融领域，线性模型被广泛用于帮助定价证券和执行最优投资组合分配，以及其他有用的事情。金融建模中线性的一个重要方面是它保证问题在全局最优解处终止。

为了进行预测和预测，回归分析在统计学领域被广泛使用，以估计变量之间的关系。由于 Python 具有丰富的数学库是其最大的优势之一，因此 Python 经常被用作科学脚本语言来帮助解决这些问题。像 SciPy 和 NumPy 这样的模块包含各种线性回归函数，供数据科学家使用。

在传统的投资组合管理中，资产配置遵循线性模式，投资者有个人的投资风格。我们可以将投资组合分配问题描述为一个线性方程组，包含等式或不等式。然后，这些线性系统可以以矩阵形式表示为*Ax=B*，其中*A*是已知的系数值，*B*是观察到的结果，*x*是我们想要找出的值的向量。往往，*x*包含最大化效用的最优证券权重。使用矩阵代数，我们可以使用直接或间接方法高效地解出*x*。

在本章中，我们将涵盖以下主题：

+   检查资本资产定价模型和证券市场线

+   使用回归解决证券市场线问题

+   检查 APT 模型并执行多元线性回归

+   理解投资组合中的线性优化

+   使用 Pulp 软件包执行线性优化

+   理解线性规划的结果

+   整数规划简介

+   使用二进制条件实现线性整数规划模型

+   使用矩阵线性代数解等式的线性方程组

+   使用 LU、Cholesky 和 QR 分解直接解线性方程组

+   使用 Jacobi 和 Gauss-Seidel 方法间接解线性方程组

# 资本资产定价模型和证券市场线

很多金融文献都专门讨论了**资本资产定价模型**（**CAPM**）。在本节中，我们将探讨突出金融中线性的重要性的关键概念。

在著名的 CAPM 中，描述了证券的风险和回报率之间的关系如下：

![](img/90b9951f-98de-427b-9d67-6b9bc56c0e71.png)

对于证券*i*，其回报被定义为*R[i]*，其 beta 被定义为*β[i]*。CAPM 将证券的回报定义为无风险利率*R[f]*和其 beta 与风险溢价的乘积之和。风险溢价可以被视为市场投资组合的超额回报，不包括无风险利率。以下是 CAPM 的可视化表示：

![](img/a8ce88b0-498f-4c1e-be55-8212d1ecabbb.png)

Beta 是股票系统风险的度量 - 无法分散的风险。实质上，它描述了股票回报与市场波动的敏感性。例如，beta 为零的股票无论市场走向如何都不会产生超额回报。它只能以无风险利率增长。beta 为 1 的股票表示该股票与市场完全同步。

beta 是通过将股票与市场回报的协方差除以市场回报的方差来数学推导的。

CAPM 模型衡量了投资组合篮子中每支股票的风险和回报之间的关系。通过概述这种关系的总和，我们可以得到在每个投资组合回报水平下产生最低投资组合风险的风险证券的组合或权重。希望获得特定回报的投资者将拥有一个最佳投资组合的组合，以提供可能的最低风险。最佳投资组合的组合位于一条称为**有效边界**的线上。

在有效边界上，存在一个切线点，表示最佳的可用最优投资组合，并以可能的最低风险换取最高的回报率。切线点处的最佳投资组合被称为**市场投资组合**。

从市场投资组合到无风险利率之间存在一条直线。这条线被称为**资本市场线**（**CML**）。CML 可以被认为是所有最优投资组合中最高夏普比率的夏普比率。**夏普比率**是一个风险调整后的绩效指标，定义为投资组合超额回报与标准差风险单位的比率。投资者特别感兴趣持有沿着 CML 线的资产组合。以下图表说明了有效边界、市场投资组合和 CML：

![](img/c344df85-44e8-4b9a-bab2-317311a22093.png)

CAPM 研究中另一个有趣的概念是**证券市场线**（**SML**）。SML 绘制了资产的预期回报与其贝塔值的关系。对于贝塔值为 1 的证券，其回报完全匹配市场回报。任何定价高于 SML 的证券被认为是被低估的，因为投资者期望在相同风险下获得更高的回报。相反，任何定价低于 SML 的证券被认为是被高估的。

![](img/1905e49e-f52c-4fd5-a639-e33fd1115c03.png)

假设我们有兴趣找到证券的贝塔值*β[i]*。我们可以对公司的股票回报*R[i]*与市场回报*R[M]*进行回归分析，同时加上一个截距*α*，形成*R[i]=α+βR[M]*的方程。

考虑以下一组在五个时间段内测得的股票回报和市场回报数据：

| **时间段** | **股票回报** | **市场回报** |
| --- | --- | --- |
| 1 | 0.065 | 0.055 |
| 2 | 0.0265 | -0.09 |
| 3 | -0.0593 | -0.041 |
| 4 | -0.001 | 0.045 |
| 5 | 0.0346 | 0.022 |

使用 SciPy 的`stats`模块，我们将对 CAPM 模型进行最小二乘回归，并通过在 Python 中运行以下代码来得出α和*β[i]*的值：

```py
In [ ]:
    """ 
    Linear regression with SciPy 
    """
    from scipy import stats

    stock_returns = [0.065, 0.0265, -0.0593, -0.001, 0.0346]
    mkt_returns = [0.055, -0.09, -0.041, 0.045, 0.022]
    beta, alpha, r_value, p_value, std_err = \
        stats.linregress(stock_returns, mkt_returns)
```

`scipty.stats.linregress`函数返回五个值：回归线的斜率、回归线的截距、相关系数、零斜率假设的假设检验的 p 值，以及估计的标准误差。我们有兴趣通过打印`beta`和`alpha`的值来找到线的斜率和截距，分别为：

```py
In [ ]:
    print(beta, alpha)
Out[ ]:
 0.5077431878770808 -0.008481900352462384 
```

股票的贝塔值为 0.5077，α几乎为零。

描述 SML 的方程可以写成：

![](img/a5b59b68-c605-4528-87d7-5ca98588efb1.png)

术语*E[R[M]]−R[f]*是市场风险溢价，*E[R[M]]*是市场投资组合的预期回报。*R[f]*是无风险利率的回报，*E[R[i]]*是资产*i*的预期回报，*β[i]*是资产的贝塔值。

假设无风险利率为 5%，市场风险溢价为 8.5%。股票的预期回报率是多少？根据 CAPM，贝塔值为 0.5077 的股票将有 0.5077×8.5%的风险溢价，即 4.3%。无风险利率为 5%，因此股票的预期回报率为 9.3%。

如果在同一时间段内观察到证券的回报率高于预期的股票回报（例如，10.5%），则可以说该证券被低估，因为投资者可以期望在承担相同风险的情况下获得更高的回报。

相反，如果观察到证券的回报率低于 SML 所暗示的预期回报率（例如，7%），则可以说该证券被高估。投资者在承担相同风险的情况下获得了降低的回报。

# 套利定价理论模型

CAPM 存在一些局限性，比如使用均值-方差框架和事实上回报被一个风险因素（市场风险因素）捕获。在一个分散投资组合中，各种股票的非系统风险相互抵消，基本上被消除了。

**套利定价理论**（**APT**）模型被提出来解决这些缺点，并提供了一种除了均值和方差之外确定资产价格的一般方法。

APT 模型假设证券回报是根据多因素模型生成的，这些模型由几个系统风险因素的线性组合组成。这些因素可能是通货膨胀率、GDP 增长率、实际利率或股利。

根据 APT 模型的均衡资产定价方程如下：

![](img/90170a01-87f1-4d7d-8966-6afcd2921d6c.png)

在这里，*E[R[i]]*是第*i*个证券的预期回报率，*α[i]*是如果所有因素都可以忽略时第*i*个股票的预期回报，*β[i,j]*是第*i*个资产对第*j*个因素的敏感性，*F[j]*是影响第*i*个证券回报的第*j*个因素的值。

由于我们的目标是找到*α[i]*和*β*的所有值，我们将在 APT 模型上执行**多元线性回归**。

# 因子模型的多元线性回归

许多 Python 包（如 SciPy）都带有几种回归函数的变体。特别是，`statsmodels`包是 SciPy 的补充，具有描述性统计信息和统计模型的估计。Statsmodels 的官方页面是[`www.statsmodels.org`](https://www.statsmodels.org)。

如果您的 Python 环境中尚未安装 Statsmodels，请运行以下命令进行安装：

```py
$ pip install -U statsmodels
```

如果您已经安装了一个现有的包，`-U`开关告诉`pip`将选定的包升级到最新可用版本。

在这个例子中，我们将使用`statsmodels`模块的`ols`函数执行普通最小二乘回归，并查看其摘要。

假设您已经实现了一个包含七个因素的 APT 模型，返回*Y*的值。考虑在九个时间段*t[1]*到*t[9]*内收集的以下数据集。*X*[1]到*X[7]*是在每个时间段观察到的自变量。因此，回归问题的结构如下：

![](img/028272c7-3f1d-46e8-8845-32c433ca2f6b.png)

可以使用以下代码对*X*和*Y*的值进行简单的普通最小二乘回归：

```py
In [ ]:
    """ 
    Least squares regression with statsmodels 
    """
    import numpy as np
    import statsmodels.api as sm

    # Generate some sample data
    num_periods = 9
    all_values = np.array([np.random.random(8) \
                           for i in range(num_periods)])

    # Filter the data
    y_values = all_values[:, 0] # First column values as Y
    x_values = all_values[:, 1:] # All other values as X
    x_values = sm.add_constant(x_values) # Include the intercept
    results = sm.OLS(y_values, x_values).fit() # Regress and fit the model
```

让我们查看回归的详细统计信息：

```py
In [ ]:
    print(results.summary())
```

OLS 回归结果将输出一个相当长的统计信息表。然而，我们感兴趣的是一个特定部分，它给出了我们 APT 模型的系数：

```py
===================================================================
                 coef    std err          t      P>|t|      [0.025      
-------------------------------------------------------------------
const          0.7229      0.330      2.191      0.273      -3.469
x1             0.4195      0.238      1.766      0.328      -2.599
x2             0.4930      0.176      2.807      0.218      -1.739
x3             0.1495      0.102      1.473      0.380      -1.140
x4            -0.1622      0.191     -0.847      0.552      -2.594
x5            -0.6123      0.172     -3.561      0.174      -2.797
x6            -0.2414      0.161     -1.499      0.375      -2.288
x7            -0.5079      0.200     -2.534      0.239      -3.054
```

`coef`列给出了我们回归的系数值，包括*c*常数和*X[1]*到*X[7]*。同样，我们可以使用`params`属性来显示这些感兴趣的系数：

```py
In [ ]:    
    print(results.params)
Out[ ]:
    [ 0.72286627  0.41950411  0.49300959  0.14951292 -0.16218313 -0.61228465 -0.24143028 -0.50786377]
```

两个函数调用以相同的顺序产生了 APT 模型的相同系数值。

# 线性优化

在 CAPM 和 APT 定价理论中，我们假设模型是线性的，并使用 Python 中的回归来解决预期的证券价格。

随着我们投资组合中证券数量的增加，也会引入一定的限制。投资组合经理在追求投资者规定的某些目标时会受到这些规则的约束。

线性优化有助于克服投资组合分配的问题。优化侧重于最小化或最大化目标函数的值。一些例子包括最大化回报和最小化波动性。这些目标通常受到某些规定的约束，例如不允许空头交易规则，或者对要投资的证券数量的限制。

不幸的是，在 Python 中，没有一个官方的包支持这个解决方案。但是，有第三方包可用，其中包含线性规划的单纯形算法的实现。为了演示目的，我们将使用 Pulp，一个开源线性规划建模器，来帮助我们解决这个特定的线性规划问题。

# 获取 Pulp

您可以从[`github.com/coin-or/pulp`](https://github.com/coin-or/pulp)获取 Pulp。该项目页面包含了一份全面的文档列表，以帮助您开始优化过程。

您还可以使用`pip`包管理器获取 Pulp 包：

```py
$ pip install pulp
```

# 线性规划的最大化示例

假设我们有兴趣投资两种证券*X*和*Y*。我们想要找出每三单位*X*和两单位*Y*的实际投资单位数，使得总投资单位数最大化，如果可能的话。然而，我们的投资策略有一定的限制：

+   对于每 2 单位*X*和 1 单位*Y*的投资，总量不得超过 100

+   对于每单位*X*和*Y*的投资，总量不得超过 80

+   允许投资*X*的总量不得超过 40

+   不允许对证券进行空头交易

最大化问题可以用数学表示如下：

![](img/02a2f49c-93e7-40f6-a947-208df527a4e2.png)

受限于：

![](img/b97a3c75-ad70-4c84-a5de-f1b995473e4c.png)

![](img/752f3101-6e8c-4def-92ea-8edc58ffa41d.png)

![](img/12d846a0-8f99-4a3a-b8ea-54b0c0d05e39.png)

![](img/71b1620c-2dbb-4d04-924e-3fc006dfc30b.png)

通过在*x*和*y*图上绘制约束条件，可以看到一组可行解，由阴影区域给出：

![](img/178c4e94-bb47-4d90-97d3-e72dcee6b704.png)

该问题可以用`pulp`包在 Python 中进行翻译，如下所示：

```py
In [ ]:
    """ 
    A simple linear optimization problem with 2 variables 
    """
    import pulp

    x = pulp.LpVariable('x', lowBound=0)
    y = pulp.LpVariable('y', lowBound=0)

    problem = pulp.LpProblem(
        'A simple maximization objective', 
        pulp.LpMaximize)
    problem += 3*x + 2*y, 'The objective function'
    problem += 2*x + y <= 100, '1st constraint'
    problem += x + y <= 80, '2nd constraint'
    problem += x <= 40, '3rd constraint'
    problem.solve()
```

`LpVariable`函数声明要解决的变量。`LpProblem`函数用问题的文本描述和优化类型初始化问题，本例中是最大化方法。`+=`操作允许添加任意数量的约束，以及文本描述。最后，调用`.solve()`方法开始执行线性优化。要显示优化器解决的值，使用`.variables()`方法循环遍历每个变量并打印出其`varValue`。

当代码运行时生成以下输出：

```py
In [ ]:
    print("Maximization Results:")
    for variable in problem.variables():
        print(variable.name, '=', variable.varValue)
Out[ ]:
    Maximization Results:
    x = 20.0
    y = 60.0
```

结果显示，在满足给定的一组约束条件的情况下，当*x*的值为 20，*y*的值为 60 时，可以获得最大值 180。

# 线性规划的结果

线性优化有三种结果，如下：

+   线性规划的局部最优解是一个可行解，其目标函数值比其附近的所有其他可行解更接近。它可能是也可能不是**全局最优解**，即优于每个可行解的解。

+   如果找不到解决方案，线性规划是**不可行**的。

+   如果最优解是无界的或无限的，线性规划是**无界**的。

# 整数规划

在我们之前调查的简单优化问题中，*线性规划的最大化示例*，变量被允许是连续的或分数的。如果使用分数值或结果不现实怎么办？这个问题被称为**线性整数规划**问题，其中所有变量都受限于整数。整数变量的一个特殊情况是二进制变量，可以是 0 或 1。在给定一组选择时，二进制变量在模型决策时特别有用。

整数规划模型经常用于运筹学中来模拟现实工作问题。通常情况下，将非线性问题陈述为线性或甚至二进制的问题需要更多的艺术而不是科学。

# 整数规划的最小化示例

假设我们必须从三家经销商那里购买 150 份某种场外奇特证券。经销商*X*报价每份合同 500 美元加上 4000 美元的手续费，无论卖出的合同数量如何。经销商*Y*每份合同收费 450 美元，加上 2000 美元的交易费。经销商*Z*每份合同收费 450 美元，加上 6000 美元的费用。经销商*X*最多销售 100 份合同，经销商*Y*最多销售 90 份，经销商*Z*最多销售 70 份。从任何经销商那里交易的最低交易量是 30 份合同。我们应该如何最小化购买 150 份合同的成本？

使用`pulp`包，让我们设置所需的变量：

```py
In [ ]:
    """ 
    An example of implementing an integer 
    programming model with binary conditions 
    """
    import pulp

    dealers = ['X', 'Y', 'Z']
    variable_costs = {'X': 500, 'Y': 350, 'Z': 450}
    fixed_costs = {'X': 4000, 'Y': 2000, 'Z': 6000}

    # Define PuLP variables to solve
    quantities = pulp.LpVariable.dicts('quantity', 
                                       dealers, 
                                       lowBound=0,
                                       cat=pulp.LpInteger)
    is_orders = pulp.LpVariable.dicts('orders', 
                                      dealers,
                                      cat=pulp.LpBinary)
```

`dealers`变量只是包含用于稍后引用列表和字典的字典标识符的字典。`variable_costs`和`fixed_costs`变量是包含每个经销商收取的相应合同成本和费用的字典对象。Pulp 求解器解决了`quantities`和`is_orders`的值，这些值由`LpVariable`函数定义。`dicts()`方法告诉 Pulp 将分配的变量视为字典对象，使用`dealers`变量进行引用。请注意，`quantities`变量具有一个下限（0），防止我们在任何证券中进入空头头寸。`is_orders`值被视为二进制对象，指示我们是否应该与任何经销商进行交易。

对建模这个整数规划问题的最佳方法是什么？乍一看，通过应用这个方程似乎相当简单：

![](img/938c9cac-ca49-43ab-bfd7-b3447d480a2c.png)

其中以下内容为真：

![](img/982dfb3a-3cc6-4858-8a10-336823094b9b.png)

![](img/1c0a8992-5668-4ac8-9504-0e7e4ad4a537.png)

![](img/2ab9139e-96e8-4a5b-8111-2a5fe54c6242.png)

![](img/ee23bb15-cf12-4f08-8675-dcfe7de1a8b7.png)

![](img/76571b12-5400-47c7-bf0d-ed4f8c7f1976.png)

该方程简单地陈述了我们希望最小化总成本，并使用二进制变量*isOrder[i]*来确定是否考虑从特定经销商购买的成本。

让我们在 Python 中实现这个模型：

```py
In [ ]:
    """
    This is an example of implementing an integer programming model
    with binary variables the wrong way.
    """
    # Initialize the model with constraints
    model = pulp.LpProblem('A cost minimization problem',
                           pulp.LpMinimize)
    model += sum([(variable_costs[i] * \
                   quantities[i] + \
                   fixed_costs[i])*is_orders[i] \
                  for i in dealers]), 'Minimize portfolio cost'
    model += sum([quantities[i] for i in dealers]) == 150\
        , 'Total contracts required'
    model += 30 <= quantities['X'] <= 100\
        , 'Boundary of total volume of X'
    model += 30 <= quantities['Y'] <= 90\
        , 'Boundary of total volume of Y'
    model += 30 <= quantities['Z'] <= 70\
        , 'Boundary of total volume of Z'
    model.solve() # You will get an error running this code!
```

当我们运行求解器时会发生什么？看一下：

```py
Out[ ]:
    TypeError: Non-constant expressions cannot be multiplied
```

事实证明，我们试图对两个未知变量`quantities`和`is_order`进行乘法运算，无意中导致我们执行了非线性规划。这就是在执行整数规划时遇到的陷阱。

我们应该如何解决这个问题？我们可以考虑使用**二进制变量**，如下一节所示。

# 具有二进制条件的整数规划

制定最小化目标的另一种方法是将所有未知变量线性排列，使它们是可加的：

![](img/59f4559c-4f21-4482-980f-92ca0b30d1ba.png)

与先前的目标方程相比，我们将获得相同的固定成本值。但是，未知变量*quantity[i]*仍然在方程的第一项中。因此，需要将*quantity[i]*变量作为*isOrder[i]*的函数来求解，约束如下所述：

![](img/6e531ad8-a8c9-4945-ad24-45c1f84efa0d.png)

![](img/11c3eecb-48e5-43ec-adc9-f49f534d4932.png)

![](img/2fbd74dc-0c7b-4ff5-a0aa-57a03a8788c7.png)

让我们在 Python 中应用这些公式：

```py
In [ ]:
    """
    This is an example of implementing an 
    IP model with binary variables the correct way.
    """
    # Initialize the model with constraints
    model = pulp.LpProblem('A cost minimization problem',
                           pulp.LpMinimize)
    model += sum(
        [variable_costs[i]*quantities[i] + \
             fixed_costs[i]*is_orders[i] for i in dealers])\
        , 'Minimize portfolio cost'
    model += sum([quantities[i] for i in dealers]) == 150\
        ,  'Total contracts required'
    model += is_orders['X']*30 <= quantities['X'] <= \
        is_orders['X']*100, 'Boundary of total volume of X'
    model += is_orders['Y']*30 <= quantities['Y'] <= \
        is_orders['Y']*90, 'Boundary of total volume of Y'
    model += is_orders['Z']*30 <= quantities['Z'] <= \
        is_orders['Z']*70, 'Boundary of total volume of Z'
    model.solve()
```

当我们尝试运行求解器时会发生什么？让我们看看：

```py
In [ ]:
    print('Minimization Results:')
    for variable in model.variables():
        print(variable, '=', variable.varValue)

    print('Total cost:',  pulp.value(model.objective))
Out[ ]:
    Minimization Results:
    orders_X = 0.0
    orders_Y = 1.0
    orders_Z = 1.0
    quantity_X = 0.0
    quantity_Y = 90.0
    quantity_Z = 60.0
    Total cost: 66500.0
```

输出告诉我们，从经销商*Y*购买 90 份合同和从经销商*Z*购买 60 份合同可以以最低成本 66,500 美元满足所有其他约束。

正如我们所看到的，需要在整数规划模型的设计中进行仔细规划，以便得出准确的解决方案，使其在决策中有用。

# 使用矩阵解决线性方程

在前一节中，我们看到了如何解决带有不等式约束的线性方程组。如果一组系统线性方程有确定性约束，我们可以将问题表示为矩阵，并应用矩阵代数。矩阵方法以紧凑的方式表示多个线性方程，同时使用现有的矩阵库函数。

假设我们想要建立一个包含三种证券*a*、*b*和*c*的投资组合。投资组合的分配必须满足一定的约束条件：必须持有证券*a*的多头头寸 6 单位。对于每两单位的证券*a*，必须投资一单位的证券*b*和一单位的证券*c*，净头寸必须是多头四单位。对于每一单位的证券*a*，必须投资三单位的证券*b*和两单位的证券*c*，净头寸必须是多头五单位。

要找出要投资的证券数量，我们可以用数学方式表述问题，如下：

![](img/305091a5-a752-4963-8a03-8c4274cd3c76.png)

![](img/310d1491-85a1-4798-9e64-4ce19f7632c1.png)

![](img/72498221-ef19-42a8-95d0-cfc4d9d0744c.png)

在所有系数可见的情况下，方程如下：

![](img/2357e60f-d857-4038-99ca-e1d6889e49d0.png)

![](img/c0c87d82-37a1-4d52-8ca1-1f0d6340d24e.png)

![](img/b45e50af-e151-40bd-ba08-776f7971b4b4.png)

让我们把方程的系数表示成矩阵形式：

![](img/534ad0ee-7490-4bb7-aadd-da72c240ceda.png)

线性方程现在可以陈述如下：

![](img/802ae3dc-26ee-47d7-a18e-7d64d42faea7.png)

要解出包含要投资的证券数量的*x*向量，需要取矩阵*A*的逆，方程写为：

![](img/b0bb9923-d74f-4cbf-a228-a8956441ebf8.png)

使用 NumPy 数组，*A*和*B*矩阵分配如下：

```py
In [ ]:
    """ 
    Linear algebra with NumPy matrices 
    """
    import numpy as np

    A = np.array([[2, 1, 1],[1, 3, 2],[1, 0, 0]])
    B = np.array([4, 5, 6])
```

我们可以使用 NumPy 的`linalg.solve`函数来解决一组线性标量方程：

```py
In [ ]:
    print(np.linalg.solve(A, B))
Out[ ]:
   [  6\.  15\. -23.]
```

投资组合需要持有 6 单位的证券*a*的多头头寸，15 单位的证券*b*，和 23 单位的证券*c*的空头头寸。

在投资组合管理中，我们可以使用矩阵方程系统来解决给定一组约束条件的证券的最佳权重分配。随着投资组合中证券数量的增加，*A*矩阵的大小增加，计算*A*的矩阵求逆变得计算成本高昂。因此，人们可以考虑使用 Cholesky 分解、LU 分解、QR 分解、雅各比方法或高斯-赛德尔方法等方法，将*A*矩阵分解为更简单的矩阵进行因式分解。

# LU 分解

**LU 分解**，又称**下三角-上三角分解**，是解决方阵线性方程组的方法之一。顾名思义，LU 分解将矩阵*A*分解为两个矩阵的乘积：一个下三角矩阵*L*和一个上三角矩阵*U*。分解可以表示如下：

![](img/71af5750-4c95-4836-b774-46acd01886f2.png)

![](img/0243079e-6788-49c3-9ac0-1be05d2f85fa.png)

在这里，我们可以看到*a=l[11]u[11]*，*b=l[11]u[12]*，依此类推。下三角矩阵是一个矩阵，它在其下三角中包含值，其余的上三角中填充了零。上三角矩阵的情况相反。

LU 分解方法相对于 Cholesky 分解方法的明显优势在于它适用于任何方阵。后者只适用于对称和正定矩阵。

回想一下前面的例子，*使用矩阵解线性方程*，一个 3 x 3 的*A*矩阵。这次，我们将使用 SciPy 模块的`linalg`包来执行 LU 分解，使用以下代码：

```py
In  [ ]:
    """ 
    LU decomposition with SciPy 
    """
    import numpy as np
    import scipy.linalg as linalg

    # Define A and B
    A = np.array([
        [2., 1., 1.],
        [1., 3., 2.],
        [1., 0., 0.]])
    B = np.array([4., 5., 6.])

    # Perform LU decomposition
    LU = linalg.lu_factor(A)
    x = linalg.lu_solve(LU, B)
```

要查看`x`的值，请执行以下命令：

```py
In  [ ]:
   print(x)
Out[ ]:
   [  6\.  15\. -23.]
```

我们得到了*a*、*b*和*c*的值分别为`6`、`15`和`-23`。

请注意，我们在这里使用了`scipy.linalg`的`lu_factor()`方法，它给出了*A*矩阵的置换 LU 分解作为`LU`变量。我们使用了`lu_solve()`方法，它接受置换的 LU 分解和`B`向量来解方程组。

我们可以使用`scipy.linalg`的`lu()`方法显示*A*矩阵的 LU 分解。`lu()`方法返回三个变量——置换矩阵*P*，下三角矩阵*L*和上三角矩阵*U*——分别返回：

```py
In [ ]:
    import scipy

    P, L, U = scipy.linalg.lu(A)

    print('P=\n', P)
    print('L=\n', L)
    print('U=\n', U)
```

当我们打印出这些变量时，我们可以得出 LU 分解和*A*矩阵之间的关系如下：

![](img/226067c6-bb1e-447a-98b1-fa67ab8d1ac2.png)

LU 分解可以看作是在两个更简单的矩阵上执行的高斯消元的矩阵形式：上三角矩阵和下三角矩阵。

# Cholesky 分解

Cholesky 分解是解线性方程组的另一种方法。它可以比 LU 分解快得多，并且使用的内存要少得多，因为它利用了对称矩阵的性质。然而，被分解的矩阵必须是 Hermitian（或者是实对称的并且是方阵）和正定的。这意味着*A*矩阵被分解为*A=LL^T*，其中*L*是一个下三角矩阵，对角线上有实数和正数，*L^T*是*L*的共轭转置。

让我们考虑另一个线性方程组的例子，其中*A*矩阵既是 Hermitian 又是正定的。同样，方程的形式是*Ax=B*，其中*A*和*B*取以下值：

![](img/8ff30e92-f62d-46ec-ab09-a60a2fa81f49.png)

让我们将这些矩阵表示为 NumPy 数组：

```py
In  [ ]:
    """ 
    Cholesky decomposition with NumPy 
    """
    import numpy as np

    A = np.array([
        [10., -1., 2., 0.],
        [-1., 11., -1., 3.],
        [2., -1., 10., -1.],
        [0., 3., -1., 8.]])
    B = np.array([6., 25., -11., 15.])

    L = np.linalg.cholesky(A)
```

`numpy.linalg`的`cholesky()`函数将计算*A*矩阵的下三角因子。让我们查看下三角矩阵：

```py
In  [ ]:
    print(L)
Out[ ]:
   [[ 3.16227766  0\.          0\.          0\.        ]
    [-0.31622777  3.3015148   0\.          0\.        ]
    [ 0.63245553 -0.24231301  3.08889696  0\.        ]
    [ 0\.          0.9086738  -0.25245792  2.6665665 ]]
```

为了验证 Cholesky 分解的结果是否正确，我们可以使用 Cholesky 分解的定义，将*L*乘以它的共轭转置，这将使我们回到*A*矩阵的值：

```py
In  [ ]:
    print(np.dot(L, L.T.conj())) # A=L.L*
Out [ ]:
    [[10\. -1\.  2\.  0.]
     [-1\. 11\. -1\.  3.]
     [ 2\. -1\. 10\. -1.]
     [ 0\.  3\. -1\.  8.]]
```

在解出*x*之前，我们需要将*L^Tx*解为*y*。让我们使用`numpy.linalg`的`solve()`方法：

```py
In  [ ]:
    y = np.linalg.solve(L, B)  # L.L*.x=B; When L*.x=y, then L.y=B
```

要解出*x*，我们需要再次使用*L*的共轭转置和*y*来解：

```py
In  [ ]:
    x = np.linalg.solve(L.T.conj(), y)  # x=L*'.y
```

让我们打印出*x*的结果：

```py
In  [ ]:
    print(x)
Out[ ]:
   [ 1\.  2\. -1\.  1.]
```

输出给出了我们的*a*、*b*、*c*和*d*的*x*的值。

为了证明 Cholesky 分解给出了正确的值，我们可以通过将*A*矩阵乘以*x*的转置来验证答案，从而得到*B*的值：

```py
In [ ] :
    print(np.mat(A) * np.mat(x).T)  # B=Ax
Out[ ]:
    [[  6.]
     [ 25.]
     [-11.]
     [ 15.]]
```

这表明通过 Cholesky 分解得到的*x*的值将与*B*给出的相同。

# QR 分解

**QR 分解**，也称为**QR 分解**，是使用矩阵解线性方程的另一种方法，非常类似于 LU 分解。要解的方程是*Ax*=*B*的形式，其中矩阵*A*=*QR*。然而，在这种情况下，*A*是正交矩阵*Q*和上三角矩阵*R*的乘积。QR 算法通常用于解线性最小二乘问题。

正交矩阵具有以下特性：

+   它是一个方阵。

+   将正交矩阵乘以其转置返回单位矩阵：

![](img/41efd8bc-3c81-4151-8192-8b66a4b75e01.png)

+   正交矩阵的逆等于其转置：

![](img/f7719a52-7912-4827-b5d0-077d5a4f5f2a.png)

单位矩阵也是一个方阵，其主对角线包含 1，其他位置包含 0。

现在问题*Ax=B*可以重新表述如下：

![](img/76ec1aa1-ad1a-4e44-a99b-6e2463dc3660.png)

![](img/97535969-d66b-4e3b-bbca-e2436beab295.png)

使用 LU 分解示例中的相同变量，我们将使用`scipy.linalg`的`qr()`方法来计算我们的*Q*和*R*的值，并让*y*变量代表我们的*BQ^T*的值，代码如下：

```py
In  [ ]:
    """ 
    QR decomposition with scipy 
    """
    import numpy as np
    import scipy.linalg as linalg

    A = np.array([
        [2., 1., 1.],
        [1., 3., 2.],
        [1., 0., 0]])
    B = np.array([4., 5., 6.])

    Q, R = scipy.linalg.qr(A)  # QR decomposition
    y = np.dot(Q.T, B)  # Let y=Q'.B
    x = scipy.linalg.solve(R, y)  # Solve Rx=y
```

注意`Q.T`只是`Q`的转置，也就是*Q*的逆：

```py
In [ ]:
    print(x)
Out[ ]:
    [  6\.  15\. -23.]
```

我们得到了与 LU 分解示例中相同的答案。

# 使用其他矩阵代数方法求解

到目前为止，我们已经看过了使用矩阵求逆、LU 分解、Cholesky 分解和 QR 分解来解线性方程组。如果*A*矩阵中的财务数据规模很大，可以通过多种方案进行分解，以便使用矩阵代数更快地收敛。量化投资组合分析师应该熟悉这些方法。

在某些情况下，我们寻找的解可能不会收敛。因此，您可能需要考虑使用迭代方法。解决线性方程组的常见迭代方法包括雅各比方法、高斯-赛德尔方法和 SOR 方法。我们将简要介绍实现雅各比和高斯-赛德尔方法的示例。

# 雅各比方法

雅各比方法沿着其对角线元素迭代地解决线性方程组。当解收敛时，迭代过程终止。同样，要解决的方程式是*Ax=B*的形式，其中矩阵*A*可以分解为两个相同大小的矩阵，使得*A=D+R*。矩阵 D 仅包含 A 的对角分量，另一个矩阵 R 包含其余分量。让我们看一个 4 x 4 的*A*矩阵的例子：

![](img/7db794c0-7bda-4251-becc-1c94352bac6f.png)

然后迭代地获得解如下：

![](img/406fc895-12fa-49cd-9169-65d220203f1c.png)

![](img/3a031bd8-7c92-4c0a-9fe5-c59071007046.png)

![](img/1adeeed4-2adf-41b1-9e41-b180888d6393.png)

![](img/f2e91e47-124e-4b15-9f34-1fd282f11b31.png)

与高斯-赛德尔方法相反，在雅各比方法中，需要在每次迭代中使用*x[n]*的值来计算*x[n+1]*，并且不能被覆盖。这将占用两倍的存储空间。然而，每个元素的计算可以并行进行，这对于更快的计算是有用的。

如果*A*矩阵是严格不可约对角占优的，这种方法保证收敛。严格不可约对角占优矩阵是指每一行的绝对对角元素大于其他项的绝对值之和。

在某些情况下，即使不满足这些条件，雅各比方法也可以收敛。Python 代码如下：

```py
In [ ]:
    """
    Solve Ax=B with the Jacobi method 
    """
    import numpy as np

    def jacobi(A, B, n, tol=1e-10):
        # Initializes x with zeroes with same shape and type as B
        x = np.zeros_like(B)

        for iter_count in range(n):
            x_new = np.zeros_like(x)
            for i in range(A.shape[0]):
                s1 = np.dot(A[i, :i], x[:i])
                s2 = np.dot(A[i, i + 1:], x[i + 1:])
                x_new[i] = (B[i] - s1 - s2) / A[i, i]

            if np.allclose(x, x_new, tol):
                break

            x = x_new

        return x
```

考虑 Cholesky 分解示例中的相同矩阵值。我们将在我们的`jacobi`函数中使用 25 次迭代来找到*x*的值：

```py
In [ ] :
    A = np.array([
        [10., -1., 2., 0.], 
        [-1., 11., -1., 3.], 
        [2., -1., 10., -1.], 
        [0.0, 3., -1., 8.]])
    B = np.array([6., 25., -11., 15.])
    n = 25
```

初始化值后，我们现在可以调用函数并求解*x*：

```py
In [ ]:
    x = jacobi(A, B, n)
    print('x', '=', x)
Out[ ]:
    x = [ 1\.  2\. -1\.  1.]
```

我们求解了*x*的值，这与 Cholesky 分解的答案类似。

# 高斯-赛德尔方法

高斯-赛德尔方法与雅各比方法非常相似。这是使用迭代过程以*Ax**=**B*形式的方程解决线性方程组的另一种方法。在这里，*A*矩阵被分解为*A**=**L+U*，其中*A*矩阵是下三角矩阵*L*和上三角矩阵*U*的和。让我们看一个 4 x 4 *A*矩阵的例子：

![](img/28f828fb-7354-4f6e-98b6-8db9994b2d99.png)

然后通过迭代获得解决方案，如下所示：

![](img/8f6c1a8f-b4a1-4edc-8a95-767c34a75cf2.png)

![](img/e0a8bd69-a01b-4f19-a603-b631e2cff6d8.png)

![](img/7a7694e0-2b36-4079-bdd6-d06dd8a942f1.png)

![](img/68f67be8-9055-459b-8819-0d824806c9e3.png)

使用下三角矩阵*L*，其中零填充上三角，可以在每次迭代中覆盖*x[n]*的元素，以计算*x[n+1]*。这样做的好处是使用雅各比方法时所需的存储空间减少了一半。

高斯-赛德尔方法的收敛速度主要取决于*A*矩阵的性质，特别是如果需要严格对角占优或对称正定的*A*矩阵。即使不满足这些条件，高斯-赛德尔方法也可能收敛。

高斯-赛德尔方法的 Python 实现如下：

```py
In  [ ]:
    """ 
    Solve Ax=B with the Gauss-Seidel method 
    """
    import numpy as np

    def gauss(A, B, n, tol=1e-10):
        L = np.tril(A)  # returns the lower triangular matrix of A
        U = A-L  # decompose A = L + U
        L_inv = np.linalg.inv(L)
        x = np.zeros_like(B)

        for i in range(n):
            Ux = np.dot(U, x)
            x_new = np.dot(L_inv, B - Ux)

            if np.allclose(x, x_new, tol):
                break

            x = x_new

        return x
```

在这里，NumPy 的`tril()`方法返回下三角*A*矩阵，从中我们可以找到下三角*U*矩阵。将剩余的值迭代地插入*x*，将会得到以下解，其中由`tol`定义了一些容差。

让我们考虑雅各比方法和乔列斯基分解示例中的相同矩阵值。我们将在我们的`gauss()`函数中使用最多 100 次迭代来找到*x*的值，如下所示：

```py
In  [ ]:
    A = np.array([
        [10., -1., 2., 0.], 
        [-1., 11., -1., 3.], 
        [2., -1., 10., -1.], 
        [0.0, 3., -1., 8.]])
    B = np.array([6., 25., -11., 15.])
    n = 100
    x = gauss(A, B, n)
```

让我们看看我们的*x*值是否与雅各比方法和乔列斯基分解中的值匹配：

```py
In [ ]:
    print('x', '=', x)
Out[ ]:   
    x = [ 1\.  2\. -1\.  1.]
```

我们解出了*x*的值，这些值与雅各比方法和乔列斯基分解的答案类似。

# 总结

在本章中，我们简要介绍了 CAPM 模型和 APT 模型在金融中的应用。在 CAPM 模型中，我们访问了 CML 的有效边界，以确定最佳投资组合和市场投资组合。然后，我们使用回归解决了 SML，这有助于我们确定资产是被低估还是被高估。在 APT 模型中，我们探讨了除使用均值方差框架之外，各种因素如何影响证券回报。我们进行了多元线性回归，以帮助我们确定导致证券价格估值的因素的系数。

在投资组合配置中，投资组合经理通常被投资者授权实现一组目标，同时遵循某些约束。我们可以使用线性规划来建模这个问题。使用 Pulp Python 包，我们可以定义一个最小化或最大化的目标函数，并为我们的问题添加不等式约束以解决未知变量。线性优化的三种结果可以是无界解、仅有一个解或根本没有解。

线性优化的另一种形式是整数规划，其中所有变量都受限于整数，而不是分数值。整数变量的特殊情况是二进制变量，它可以是 0 或 1，特别适用于在给定一组选择时建模决策。我们研究了一个包含二进制条件的简单整数规划模型，并看到了遇到陷阱有多容易。需要仔细规划整数规划模型的设计，以便它们在决策中有用。

投资组合分配问题也可以表示为一个具有相等性的线性方程组，可以使用矩阵形式的*Ax=B*来求解。为了找到*x*的值，我们使用了各种类型的*A*矩阵分解来求解*A^(−1)B*。矩阵分解方法有两种类型，直接和间接方法。直接方法在固定次数的迭代中执行矩阵代数运算，包括 LU 分解、Cholesky 分解和 QR 分解方法。间接或迭代方法通过迭代计算*x*的下一个值，直到达到一定的精度容差。这种方法特别适用于计算大型矩阵，但也面临着解不收敛的风险。我们使用的间接方法有雅各比方法和高斯-赛德尔方法。

在下一章中，我们将讨论金融中的非线性建模。
