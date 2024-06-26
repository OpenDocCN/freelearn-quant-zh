# 附录 A：期权定价的 C++数值库

在 C++中实现金融衍生品可能是一个复杂的任务。正如我们在本书中所展示的，它不仅需要对数学模型和数值方法的知识，以便在 C++代码的形式中实现它们，还需要使用可靠的数学和金融库的支持。例如，当您需要从标准正态分布中获取随机样本，或者当您需要求逆矩阵时。在这些情况下，我们可以使用已经存在的数值库，而不是从头开始实现这些算法。这些库包含多年来使用的算法，因此在许多用户之前已经得到验证。使用这些库将显著加速我们对高级定价模型的实现。这些库的一些示例将在接下来的章节中提到。

# 数值配方

许可证：商业。

网站：[`www.nr.com`](http://www.nr.com)。

在书籍*“Numerical Recipes: The Art of Scientific Computing, 3rd Edition”*中可以找到一套广泛使用和可靠的 C++数值例程。这套例程被世界各地顶尖大学和研究机构视为“黄金标准”。该书包含了这些例程的理论背景描述，并提供了 C++代码的访问。书中包含了 400 多个 C++数值例程，涵盖了线性代数方程的解、矩阵代数、插值和外推、积分和随机数等主题。

# 金融数值配方

许可证：免费/GNU。

网站：[`finance.bi.no/~bernt/gcc_prog/`](http://finance.bi.no/~bernt/gcc_prog/)。

这个网站包含了由 Bernt Arne Odegaard 开发的大量非常有用的 C++数值和金融程序。它们遵循 ANSI C++标准，并有一个名为*Circa*（250 页）的大型附带手册，其中包含使用的公式和相关参考资料。这个库可以在[`finance.bi.no/~bernt/gcc_prog/`](http://finance.bi.no/~bernt/gcc_prog/)找到。

# QuantLib 项目

许可证：免费/GNU。

网站：[`quantlib.org/`](http://quantlib.org/)。

QuantLib 项目是一个为量化金融提供软件的大型项目。它已被用于金融领域的建模、交易和风险管理。该软件是用 C++编写的，并已导出到各种语言，如 C＃、Objective Caml、Java、Perl、Python、GNU R、Ruby 和 Scheme。QuantLib 有许多有用的工具，包括收益曲线模型、求解器、PDEs、蒙特卡洛（低差异性）、异国期权、VAR 等。

# Boost 库

许可证：免费/GNU。

网站：[www.boost.org](http://www.boost.org)。

Boost 项目提供了经过同行评审的便携式 C++源代码库，可以在 GNU GPL 下免费使用。这些库旨在使它们在广泛的应用程序中有用和可用。十个 Boost 库包括在 C++标准委员会的库技术报告（TR1）和新的 C++11 标准中。示例包括累加器、数组、时间、文件系统、几何、数学、数学/统计分布和 MPI。

# GSL 库

许可证：免费/GNU。

网站：[www.gnu.org/s/gsl/](http://www.gnu.org/s/gsl/)。

GNU 科学图书馆（GSL）是用于 C 和 C ++的数值库。该库提供了各种数学数值例程，包括随机数生成器、特殊函数和最小二乘拟合。总共有 1000 多个函数。该库涵盖的主题领域的示例包括复数、多项式的根、特殊函数、向量和矩阵、排列、线性代数、特征系统、快速傅立叶变换、积分、随机数、准随机序列、统计、直方图和蒙特卡洛积分。
