# 第五章：使用 pandas 进行数据操作和分析

在本章中，您将学习基于 NumPy 构建的 Python `pandas` 库，该库为结构化数据框提供了数据操作和分析方法。根据维基百科对 pandas 的页面，**pandas** 这个名字是从 **panel** **data** 派生而来，它是一个描述多维结构化数据集的计量经济学术语。

`pandas`库包含两种基本数据结构来表示和操作带有各种索引选项的结构化矩形数据集：Series 和 DataFrames。两者都使用索引数据结构。

Python 中处理金融数据的大多数操作都是基于 DataFrames 的。DataFrame 就像一个 Excel 工作表 - 一个可能包含多个时间序列的二维表格，存储在列中。因此，我们建议您在您的环境中执行本章中的所有示例，以熟悉语法并更好地了解可能的操作。

在本章中，我们将涵盖以下主题：

+   介绍 pandas Series、pandas DataFrames 和 pandas Indexes

+   学习 pandas DataFrames 上的基本操作

+   使用 pandas DataFrames 探索文件操作

# 技术要求

本章中使用的 Python 代码在书籍代码存储库中的`Chapter04/pandas.ipynb`笔记本中可用。

# 介绍 pandas Series、pandas DataFrames 和 pandas Indexes

pandas Series、pandas DataFrames 和 pandas Indexes 是 pandas 的基本数据结构。

## pandas.Series

`pandas.Series`数据结构表示同质值（整数值、字符串值、双精度值等）的一维系列。 Series 是一种列表类型，只能包含带索引的单个列表。另一方面，Data Frame 是一个包含一个或多个 series 的集合。

让我们创建一个`pandas.Series`数据结构：

```py
import pandas as pd
ser1 = pd.Series(range(1, 6)); 
ser1
```

该系列包含在第一列中的索引，第二列中的索引对应的值：

```py
0    1
1    2
2    3
3    4
4    5
dtype: int64
```

我们可以通过指定`index`参数来指定自定义索引名称：

```py
ser2 = pd.Series(range(1, 6), 
                 index=['a', 'b', 'c', 'd', 'e']); 
ser2
```

输出将如下所示：

```py
a    1
b    2
c    3
d    4
e    5
dtype: int64
```

我们还可以通过字典指定`index -> value`映射来创建一个系列：

```py
ser3 = pd.Series({ 'a': 1.0, 'b': 2.0, 'c': 3.0, 
                   'd': 4.0, 'e': 5.0 }); 
ser3
```

输出如下所示：

```py
a    1.0
b    2.0
c    3.0
d    4.0
e    5.0
dtype: float64
```

`pandas.Series.index`属性允许我们访问索引：

```py
ser3.index
```

索引的类型是`pandas.Index`：

```py
Index(['a', 'b', 'c', 'd', 'e'], dtype='object')
```

可以使用`pandas.Series.values`属性访问系列的值：

```py
ser3.values
```

值如下：

```py
array([ 1.,  2.,  3.,  4.,  5.])
```

我们可以通过修改`pandas.Series.name`属性为系列指定一个名称：

```py
ser3.name = 'Alphanumeric'; ser3
```

输出如下所示：

```py
a    1.0
b    2.0
c    3.0
d    4.0
e    5.0
Name: Alphanumeric, dtype: float64
```

上述示例演示了构建 pandas Series 的多种方式。让我们了解一下 DataFrame，这是一种可能包含多个 Series 的数据结构。

## pandas.DataFrame

`pandas.DataFrame`数据结构是多个可能不同类型的`pandas.Series`对象的集合，由相同的公共 Index 对象索引。

所有统计时间序列操作的大部分都是在数据框上执行的，`pandas.DataFrame`针对数据框的并行超快处理进行了优化，比在单独系列上进行处理快得多。

我们可以从字典创建一个数据框，其中键是列名，该键的值包含相应系列/列的数据：

```py
df1 = pd.DataFrame({'A': range(1,5,1), 
                    'B': range(10,50,10), 
                    'C': range(100, 500, 100)}); 
df1
```

输出如下所示：

```py
     A    B     C
0    1    10    100
1    2    20    200
2    3    30    300
3    4    40    400
```

我们也可以在这里传递`index=`参数来标记索引：

```py
df2 = pd.DataFrame({'A': range(1,5,1), 
                    'B': range(10,50,10), 
                    'C': range(100, 500, 100)}, 
                    index=['a', 'b', 'c', 'd']); 
df2
```

这构建了以下数据框：

```py
     A    B     C
a    1    10    100
b    2    20    200
c    3    30    300
d    4    40    400
```

`pandas.DataFrame.columns`属性返回不同列的名称：

```py
df2.columns
```

结果是一个`Index`对象：

```py
Index(['A', 'B', 'C'], dtype='object')
```

索引可以从`pandas.DataFrame.index`属性中访问：

```py
df2.index
```

这给了我们这个：

```py
Index(['a', 'b', 'c', 'd'], dtype='object')
```

数据框还包含`pandas.DataFrame.values`属性，该属性返回列中包含的值：

```py
df2.values
```

结果是以下 2D 数组：

```py
array([[  1,  10, 100],
       [  2,  20, 200],
       [  3,  30, 300],
       [  4,  40, 400]])
```

我们可以通过以下方式向数据框添加具有指定值和相同索引的新列：

```py
df2['D'] = range(1000,5000,1000); 
df2
```

更新后的数据框如下：

```py
     A    B     C      D
a    1    10    100    1000
b    2    20    200    2000
c    3    30    300    3000
d    4    40    400    4000
```

我们可以为数据框的索引和列指定名称。

我们可以通过修改`pandas.DataFrame.index.name`属性来命名索引：

```py
df2.index.name = 'lowercase'; df2
```

这导致以下更新后的数据框：

```py
         A    B     C      D
lowercase                
a        1    10    100    1000
b        2    20    200    2000
c        3    30    300    3000
d        4    40    400    4000
```

可以使用`pandas.DataFrame.columns.name`属性重命名列：

```py
df2.columns.name = 'uppercase'; df2
```

新数据框如下所示：

```py
uppercase  A    B     C      D
lowercase                
a          1    10    100    1000
b          2    20    200    2000
c          3    30    300    3000
d          4    40    400    4000
```

前面的例子演示了如何构造数据框。

## pandas.Index

`pandas.Series`和`pandas.DataFrame`数据结构都利用`pandas.Index`数据结构。

有许多特殊类型的`Index`对象：

+   `Int64Index`：`Int64Index`包含整数索引值。

+   `MultiIndex`：`MultiIndex`包含用于分层索引的元组索引，我们将在本章中探讨。

+   `DatetimeIndex`：`DatetimeIndex`，我们之前已经见过，包含时间序列数据集的日期时间索引值。

我们可以通过以下方式创建一个`pandas.Index`对象：

```py
ind2 = pd.Index(list(range(5))); ind2
```

结果是这样的：

```py
Int64Index([0, 1, 2, 3, 4], dtype='int64')
```

注意

`Index`对象是不可变的，因此无法就地修改。

让我们看看如果我们尝试修改`Index`对象中的元素会发生什么：

```py
ind2[0] = -1
```

我们得到以下输出：

```py
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-34-20c233f961b2> in <module>()
----> 1 ind2[0] = -1
...
TypeError: Index does not support mutable operations
```

Python 警告我们无法手动修改索引对象。

我们现在已经学会了如何构建系列和数据框。让我们探索对数据框进行的基本操作。

# 学习重要的 pandas.DataFrame 操作

本节描述了对数据框进行的基本操作。了解它们的存在以及如何使用它们将为您节省大量时间。

## 数据框的索引、选择和过滤

pandas 数据结构通过特殊的`Index`对象进行索引（而`numpy.ndarrays`和 Python 列表对象仅可通过整数索引）。本课程的步骤如下：

1.  让我们检查在本章前面创建的`df2`数据框的内容：

    ```py
    df2
    ```

    输出如下所示：

    ```py
    uppercase A    B     C      D
    lowercase                
    a         1    10    100    1000
    b         2    20    200    2000
    c         3    30    300    3000
    d         4    40    400    4000
    ```

1.  我们可以通过执行以下操作选择列`B`中的值序列：

    ```py
    df2['B']
    ```

    这产生了以下序列：

    ```py
    lowercase
    a    10
    b    20
    c    30
    d    40
    Name: B, dtype: int64
    ```

1.  我们可以通过传递列名列表来选择多个列（与我们在 `numpy.ndarrays` 中看到的有些相似）：

    ```py
    df2[['A', 'C']]
    ```

    这产生了以下具有两列的 DataFrame：

    ```py
    uppercase  A    C
    lowercase        
    a          1    100
    b          2    200
    c          3    300
    d          4    400
    ```

1.  我们可以通过以下方式使用 DataFrame 进行布尔选择：

    ```py
    df2[(df2['D'] > 1000) & (df2['D'] <= 3000)]
    ```

    这选择了满足提供条件的以下行：

    ```py
    uppercase    A    B     C      D
    lowercase                
    b            2    20    200    2000
    c            3    30    300    3000
    ```

1.  `pandas.DataFrame.loc[...]` 属性允许我们索引行而不是列。以下选择了两行 `c` 和 `d`：

    ```py
    df2.loc[['c', 'd']]
    ```

    这产生了以下子集 DataFrame：

    ```py
    uppercase    A    B     C      D
    lowercase                
    c            3    30    300    3000
    d            4    40    400    4000
    ```

1.  pandas DataFrame 仍然支持通过 `pandas.DataFrame.iloc[...]` 属性进行标准整数索引。我们可以通过这样做来选择第一行：

    ```py
    df2.iloc[[0]]
    ```

    这选择了以下单行 DataFrame：

    ```py
    uppercase    A    B     C      D
    lowercase                
    a            1    10    100    1000
    ```

    我们可以通过类似这样的操作修改 DataFrame：

    ```py
    df2[df2['D'] == 2000] = 0; df2
    ```

    这将 DataFrame 更新为这个新 DataFrame：

    ```py
    uppercase    A    B     C      D
    lowercase                
    a            1    10    100    1000
    b            0    0     0      0
    c            3    30    300    3000
    d            4    40    400    4000
    ```

在本节中，我们学习了如何索引、选择和过滤 DataFrame。在下一节中，我们将学习如何删除行和列。

## 从 DataFrame 中删除行和列

从 DataFrame 中删除行和列是一个关键操作——它不仅有助于节省计算机的内存，还确保 DataFrame 只包含逻辑上需要的信息。步骤如下：

1.  让我们显示当前 DataFrame：

    ```py
    df2
    ```

    此 DataFrame 包含以下内容：

    ```py
    uppercase    A    B     C      D
    lowercase                
    a            1    10    100    1000
    b            0    0     0      0
    c            3    30    300    3000
    d            4    40    400    4000
    ```

1.  要删除索引为 `b` 的行，我们使用 `pandas.DataFrame.drop(...)` 方法：

    ```py
    df2.drop('b')
    ```

    这产生了一个没有索引为 `b` 的行的新 DataFrame：

    ```py
    uppercase    A    B     C      D
    lowercase                
    a            1    10    100    1000
    c            3    30    300    3000
    d            4    40    400    4000
    ```

    让我们检查原始 DataFrame 是否已更改：

    ```py
    df2
    ```

    输出显示没有，也就是说，默认情况下 `pandas.DataFrame.drop(...)` 不是原位的：

    ```py
    uppercase    A    B     C      D
    lowercase                
    a            1    10    100    1000
    b            0    0     0      0
    c            3    30    300    3000
    d            4    40    400    4000
    ```

1.  要修改原始 DataFrame，我们使用 `inplace=` 参数：

    ```py
    df2.drop('b', inplace=True); 
    df2
    ```

    新的原地修改的 DataFrame 如下所示：

    ```py
    uppercase    A    B     C      D
    lowercase                
    a            1    10    100    1000
    c            3    30    300    3000
    d            4    40    400    4000
    ```

1.  我们也可以删除多个行：

    ```py
    df2.drop(['a', 'd'])
    ```

    这返回了以下新 DataFrame：

    ```py
    uppercase    A    B     C      D
    lowercase                
    c            3    30    300    3000
    ```

1.  要删除列而不是行，我们指定额外的 `axis=` 参数：

    ```py
    df2.drop(['A', 'B'], axis=1)
    ```

    这给了我们具有两个删除列的新 DataFrame：

    ```py
    uppercase   C      D
    lowercase        
    a           100    1000
    c           300    3000
    d           400    4000
    ```

我们在本节中学习了如何删除行和列。在下一节中，我们将学习如何对值进行排序和 rand。

## 对 DataFrame 进行排序值和排列值顺序

首先，让我们创建一个具有整数行索引、整数列名和随机值的 DataFrame：

```py
import numpy as np
df = pd.DataFrame(np.random.randn(5,5),
                  index=np.random.randint(0, 100, size=5), 
                  columns=np.random.randint(0,100,size=5)); 
df
```

DataFrame 包含以下数据：

```py
87        79        74        3        61
7     0.355482  -0.246812  -1.147618  -0.293973  -0.560168
52    1.748274   0.304760  -1.346894  -0.548461   0.457927
80   -0.043787  -0.680384   1.918261   1.080733   1.346146
29    0.237049   0.020492   1.212589  -0.462218   1.284134
0    -0.153209   0.995779   0.100585  -0.350576   0.776116
```

`pandas.DataFrame.sort_index(...)` 按索引值对 DataFrame 进行排序：

```py
df.sort_index()
```

结果如下：

```py
87        79        74        3        61
0    -0.153209   0.995779   0.100585  -0.350576   0.776116
7     0.355482  -0.246812  -1.147618  -0.293973  -0.560168
29    0.237049   0.020492   1.212589  -0.462218   1.284134
52    1.748274   0.304760  -1.346894  -0.548461   0.457927
80   -0.043787  -0.680384   1.918261   1.080733   1.346146
```

我们也可以通过指定 `axis` 参数按列名值进行排序：

```py
df.sort_index(axis=1)
```

这产生了以下按顺序排列的 DataFrame：

```py
     3         61         74         79         87
7    -0.293973  -0.560168  -1.147618  -0.246812   0.355482
52   -0.548461   0.457927  -1.346894   0.304760   1.748274
80    1.080733   1.346146   1.918261  -0.680384  -0.043787
29   -0.462218   1.284134   1.212589   0.020492   0.237049
0    -0.350576   0.776116   0.100585   0.995779  -0.153209
```

要对 DataFrame 中的值进行排序，我们使用 `pandas.DataFrame.sort_values(...)` 方法，该方法采用 `by=` 参数指定要按其排序的列：

```py
df.sort_values(by=df.columns[0])
```

这产生了以下按第一列值排序的 DataFrame：

```py
    87         79         74         3         61
0     -0.153209   0.995779   0.100585  -0.350576   0.776116
80    -0.043787  -0.680384   1.918261   1.080733   1.346146
29     0.237049   0.020492   1.212589  -0.462218   1.284134
7      0.355482  -0.246812  -1.147618  -0.293973  -0.560168
52     1.748274   0.304760  -1.346894  -0.548461   0.457927
```

`pandas.DataFrame.rank(...)` 方法产生一个包含每列值的排名/顺序的 DataFrame：

```py
df.rank()
```

输出包含值的排名（按升序）：

```py
     87     79     74     3      61
7    4.0    2.0    2.0    4.0    1.0
52   5.0    4.0    1.0    1.0    2.0
80   2.0    1.0    5.0    5.0    5.0
29   3.0    3.0    4.0    2.0    4.0
0    1.0    5.0    3.0    3.0    3.0
```

本课程完成后，在下一节中，我们将对 DataFrame 执行算术运算。

## DataFrame 上的算术操作

首先，让我们为我们的示例创建两个 DataFrames：

```py
df1 = pd.DataFrame(np.random.randn(3,2), 
                   index=['A', 'C', 'E'], 
                   columns=['colA', 'colB']); 
df1
```

`df1` DataFrame 包含以下内容：

```py
     colA         colB
A     0.519105    -0.127284
C    -0.840984    -0.495306
E    -0.137020     0.987424
```

现在我们创建`df2` DataFrame：

```py
df2 = pd.DataFrame(np.random.randn(4,3), 
                   index=['A', 'B', 'C', 'D'], 
                   columns=['colA', 'colB', 'colC']); 
df2
```

这包含以下内容：

```py
     colA          colB         colC
A    -0.718550     1.938035     0.220391
B    -0.475095     0.238654     0.405642
C     0.299659     0.691165    -1.905837
D     0.282044    -2.287640    -0.551474
```

我们可以将两个 DataFrame 相加。请注意它们具有不同的索引值以及不同的列：

```py
df1 + df2
```

输出是元素的总和，如果索引和列存在于两个 DataFrame 中，则为 NaN：

```py
     colA         colB        colC
A    -0.199445    1.810751    NaN
B     NaN         NaN         NaN
C    -0.541325    0.195859    NaN
D     NaN         NaN         NaN
E     NaN         NaN         NaN
```

我们可以使用`pandas.DataFrame.add(...)`方法并带有`fill_value=`参数指定一个值来替代`NaN`（在这种情况下是`0`）：

```py
df1.add(df2, fill_value=0)
```

输出如下所示：

```py
     colA         colB         colC
A    -0.199445    1.810751     0.220391
B    -0.475095    0.238654     0.405642
C    -0.541325    0.195859    -1.905837
D     0.282044   -2.287640    -0.551474
E    -0.137020    0.987424     NaN
```

我们还可以在 DataFrame 和 Series 之间执行算术操作：

```py
df1 - df2[['colB']]
```

这个操作的输出如下（因为右侧只有`colB`）：

```py
     colA    colB
A    NaN     -2.065319
B    NaN     NaN
C    NaN     -1.186471
D    NaN     NaN
E    NaN     NaN
```

现在让我们学习如何将多个 DataFrame 合并和组合成一个单独的 DataFrame。

## 将多个 DataFrame 合并和组合成一个 DataFrame

让我们首先创建两个 DataFrame，`df1` 和 `df2`：

```py
df1.index.name = 'Index'; df1.columns.name = 'Columns'; df1
```

`df1` DataFrame 包含以下数据：

```py
Columns    colA          colB
Index        
A           0.519105    -0.127284
C          -0.840984    -0.495306
E          -0.137020     0.987424
```

现在我们创建`df2`：

```py
df2.index.name = 'Index'; df2.columns.name = 'Columns'; df2
```

`df2` DataFrame 包含以下数据：

```py
Columns    colA         colB         colC
Index            
A          -0.718550     1.938035     0.220391
B          -0.475095     0.238654     0.405642
C           0.299659     0.691165    -1.905837
D           0.282044    -2.287640    -0.551474
```

`pandas.merge(...)`方法连接/合并两个 DataFrames。`left_index=`和`right_index=`参数指示合并应该在两个 DataFrames 的索引值上执行：

```py
pd.merge(df1, df2, left_index=True, right_index=True)
```

这产生了以下合并后的 DataFrame。`_x` 和 `_y` 后缀用于区分左右两个 DataFrame 中具有相同名称的列：

```py
Columns colA_x    colB_x     colA_y     colB_y     colC
Index                    
A       0.519105  -0.127284  -0.718550  1.938035   0.220391
C      -0.840984  -0.495306   0.299659  0.691165  -1.905837
```

我们可以使用`suffixes=`参数指定自定义后缀：

```py
pd.merge(df1, df2, left_index=True, right_index=True, 
         suffixes=('_1', '_2'))
```

结果是带有我们提供的后缀的以下 DataFrame：

```py
Columns  colA_1    colB_1     colA_2     colB_2    colC
Index                    
A        0.519105  -0.127284  -0.718550  1.938035  0.220391
C       -0.840984  -0.495306   0.299659  0.691165 -1.905837
```

我们可以使用`how=`参数指定连接的行为（外部、内部、左连接或右连接）：

```py
pd.merge(df1, df2, left_index=True, right_index=True, 
         suffixes=('_1', '_2'), how='outer')
```

这会产生以下带有`NaNs`的 DataFrame，用于缺失值：

```py
Columns  colA_1    colB_1    colA_2    colB_2    colC
Index                    
A        0.519105  -0.127284  -0.718550  1.938035  0.220391
B        NaN        NaN       -0.475095  0.238654  0.405642
C       -0.840984  -0.495306   0.299659  0.691165 -1.905837
D        NaN        NaN        0.282044 -2.287640 -0.551474
E       -0.137020   0.987424   NaN       NaN       NaN
```

pandas DataFrame 本身具有`pandas.DataFrame.merge(...)`方法，其行为方式相同：

```py
df1.merge(df2, left_index=True, right_index=True, 
          suffixes=('_1', '_2'), how='outer')
```

这会产生以下结果：

```py
Columns  colA_1     colB_1     colA_2    colB_2    colC
Index                    
A        0.519105  -0.127284  -0.718550  1.938035  0.220391
B        NaN        NaN       -0.475095  0.238654  0.405642
C       -0.840984  -0.495306   0.299659  0.691165 -1.905837
D        NaN        NaN        0.282044 -2.287640 -0.551474
E       -0.137020   0.987424   NaN       NaN        NaN
```

另一种选择是`pandas.DataFrame.join(...)`方法：

```py
df1.join(df2, lsuffix='_1', rsuffix='_2')
```

并且连接的输出（默认为左连接）如下所示：

```py
Columns  colA_1    colB_1    colA_2    colB_2    colC
Index                    
A        0.519105  -0.127284  -0.718550  1.938035  0.220391
C       -0.840984  -0.495306   0.299659  0.691165 -1.905837
E       -0.137020  0.987424    NaN       NaN       NaN
```

`pandas.concat(...)`方法通过将行连接在一起来组合 DataFrame：

```py
pd.concat([df1, df2])
```

这会产生以下带有`NaNs`的连接 DataFrame：

```py
      colA         colB        colC
Index            
A     0.519105    -0.127284    NaN
C    -0.840984    -0.495306    NaN
E    -0.137020     0.987424    NaN
A    -0.718550     1.938035     0.220391
B    -0.475095     0.238654     0.405642
C     0.299659     0.691165    -1.905837
D     0.282044    -2.287640    -0.551474
```

我们可以通过指定`axis=`参数在列之间进行连接：

```py
pd.concat([df1, df2], axis=1)
```

这会产生以下带有来自`df2`的额外列的 DataFrame：

```py
Columns  colA       colB       colA      colB      colC
A        0.519105  -0.127284  -0.718550  1.938035  0.220391
B        NaN        NaN       -0.475095  0.238654  0.405642
C       -0.840984  -0.495306   0.299659  0.691165 -1.905837
D        NaN        NaN        0.282044 -2.287640 -0.551474
E       -0.137020  0.987424    NaN       NaN       NaN
```

现在我们将学习分层索引。

## 分层索引

到目前为止，我们一直在处理的索引对象都是一个简单的单个值。分层索引使用`MultiIndex`对象，它是每个索引的多个值的元组。这使我们能够在单个 DataFrame 内创建子 DataFrame。

让我们创建一个`MultiIndex` DataFrame：

```py
df = pd.DataFrame(np.random.randn(10, 2),
                  index=[list('aaabbbccdd'), 
                  [1, 2, 3, 1, 2, 3, 1, 2, 1, 2]], 
                  columns=['A', 'B']); 
df
```

这是使用分层索引的`MultiIndex` DataFrame 的布局：

```py
                 A             B
a    1     0.289379    -0.157919
     2    -0.409463    -1.103412
     3     0.812444    -1.950786
b    1    -1.549981     0.947575
     2     0.344725    -0.709320
     3     1.384979    -0.716733
c    1    -0.319983     0.887631
     2    -1.763973     1.601361
d    1     0.171177    -1.285323
     2    -0.143279     0.020981
```

我们可以使用`pandas.MultiIndex.names`属性为`MultiIndex`对象分配名称 - 它需要一个名称列表，其维度与`MultiIndex` DataFrame 的维度相同（在本例中为两个元素）：

```py
df.index.names = ['alpha', 'numeric']; df
```

这会得到以下结果：

```py
                    A            B
alpha    numeric        
a       1           0.289379    -0.157919
        2          -0.409463    -1.103412
        3           0.812444    -1.950786
...
```

`pandas.DataFrame.reset_index(...)`方法默认情况下从`MultiIndex`DataFrame 中移除所有索引级别，但可以用于移除一个或多个级别：

```py
df.reset_index()
```

这导致以下整数索引 DataFrame 以及`MultiIndex`值被添加为此 DataFrame 的列：

```py
       alpha    numeric      A            B
0    a        1              0.289379    -0.157919
1    a        2             -0.409463    -1.103412
2    a        3              0.812444    -1.950786
...
```

`pandas.DataFrame.unstack(...)`方法的行为类似，并将内部索引的级别旋转并将其转换为列：

```py
df.unstack()
```

让我们检查新的 DataFrame，其中最内层的索引级别`[1, 2, 3]`变为列：

```py
             A                                      B
numeric        1             2             3              1             2             3
alpha                        
a            0.289379    -0.409463    0.812444     -0.157919    -1.103412    -1.950786
b            -1.549981    0.344725    1.384979     0.947575    -0.709320    -0.716733
c            -0.319983    -1.763973    NaN             0.887631    1.601361        NaN
d            0.171177    -0.143279    NaN             -1.285323    0.020981        NaN
```

`pandas.DataFrame.stack(...)`方法的作用与`unstack(...)`相反：

```py
df.stack()
```

输出 DataFrame 是具有分层索引的原始 DataFrame：

```py
alpha  numeric   
a      1        A    0.289379
                B   -0.157919
       2        A   -0.409463
                B   -1.103412
       3        A    0.812444
                B   -1.950786
...
dtype: float64
```

让我们检查`MultiIndex`DataFrame 的结构。请注意，我们首先调用`pandas.DataFrame.stack(...)`将列`[A, B]`转换为`MultiIndex`DataFrame 中的第三个索引级别：

```py
df.stack().index
```

这给我们一个具有三个索引级别的`MultiIndex`对象：

```py
MultiIndex(levels=[['a', 'b', 'c', 'd'], 
                   [1, 2, 3], ['A', 'B']],
           labels=[[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]],
           names=['alpha', 'numeric', None])
```

现在我们将学习如何在 DataFrames 中进行分组操作。

## DataFrames 中的分组操作

pandas 中的分组操作通常遵循操作的分割-应用-组合过程：

1.  首先，数据根据一个或多个键分成组。

1.  然后，我们对这些组应用必要的函数来计算所需的结果。

1.  最后，我们将它们组合起来构建转换后的数据集。

因此，对单索引 DataFrame 进行分组会构建一个分层 DataFrame。步骤如下：

1.  让我们使用`pandas.DataFrame.reset_index(…)`方法从先前的`df`DataFrame 中移除所有分层索引：

    ```py
    df = df.reset_index(); df
    ```

    这返回了以下带有整数索引的 DataFrame：

    ```py
        alpha    numeric        A           B
    0    a        1            -0.807285    0.170242
    1    a        2             0.704596    1.568901
    2    a        3            -1.417366    0.573896
    3    b        1             1.110121    0.366712
    ...
    ```

1.  让我们使用`pandas.DataFrame.groupby(...)`方法来按`alpha`列对`A`和`B`列进行分组：

    ```py
    grouped = df[['A','B']].groupby(df['alpha']); grouped
    ```

    这产生了以下的`DataFrameGroupBy`对象，随后我们可以对其进行操作：

    ```py
    <pandas.core.groupby.DataFrameGroupBy object at 0x7fd21f24cc18>
    ```

1.  我们可以使用`DataFrameGroupBy.describe(...)`方法来收集摘要描述性统计信息：

    ```py
    grouped.describe()
    ```

    这产生了以下输出，其中生成了`A`和`B`的统计信息，但是按`alpha`列分组：

    ```py
            A        B
    alpha            
    a        count    3.000000    3.000000
    mean   -0.506685    0.771013
    std     1.092452    0.719863
    min    -1.417366    0.170242
    25%    -1.112325    0.372069
    50%    -0.807285    0.573896
    75%    -0.051344    1.071398
    max     0.704596    1.568901
    ...
    ```

1.  我们可以使用`DataFrameGroupBy.apply(...)`方法应用`pandas.DataFrame.unstack(...)`方法，该方法接受不同的函数并将它们应用于`grouped`对象的每个组：

    ```py
    grouped.apply(pd.DataFrame.unstack)
    ```

    这产生了以下分层 DataFrame：

    ```py
    alpha      
    a      A  0   -0.807285
              1    0.704596
              2   -1.417366
           B  0    0.170242
              1    1.568901
              2    0.573896
    ...
    dtype: float64
    ```

1.  还存在`DataFrameGroupBy.agg(...)`方法，它接受函数并使用该方法为每个组的每个列聚合该方法。下一个示例使用`mean`方法进行聚合：

    ```py
    grouped[['A', 'B']].agg('mean')
    ```

    输出包含了按`alpha`值分组的`A`和`B`列的均值：

    ```py
                    A            B
    alpha        
        a    -0.506685     0.771013
        b     0.670435     0.868550
        c     0.455688    -0.497468
        d    -0.786246     0.107246
    ```

1.  类似的方法是`DataFrameGroupBy.transform(...)`方法，唯一的区别在于 transform 一次只对一列起作用，并返回与系列长度相同的值序列，而 apply 可以返回任何类型的结果：

    ```py
    from scipy import stats
    grouped[['A', 'B']].transform(stats.zscore)
    ```

    这会为列`A`和`B`生成 Z 得分，我们在*第二章*中解释了这个*探索性数据分析*：

    ```py
                A             B
    0    -0.337002    -1.022126
    1     1.357964     1.357493
    2    -1.020962    -0.335367
    3     0.610613    -0.567813
    4    -1.410007     1.405598
    5     0.799394    -0.837785
    6    -1.000000     1.000000
    7     1.000000    -1.000000
    8    -1.000000    -1.000000
    9     1.000000     1.000000
    ```

我们现在将学习如何转换 DataFrame 轴索引中的值。

## 转换 DataFrame 轴索引中的值

让我们首先重新检查我们将在这些示例中使用的 `df2` DataFrame：

```py
df2
```

这包含以下数据：

```py
Columns   colA         colB         colC
Index            
A        -2.071652     0.742857     0.632307
B         0.113046    -0.384360     0.414585
C         0.690674     1.511816     2.220732
D         0.184174    -1.069291    -0.994885
```

我们可以使用 `pandas.DataFrame.index` 属性重命名索引标签，就像我们之前看到的那样：

```py
df2.index = ['Alpha', 'Beta', 'Gamma', 'Delta']; 
df2
```

这会生成以下转换后的 DataFrame：

```py
Columns       colA             colB             colC
Alpha        -2.071652         0.742857         0.632307
Beta          0.113046        -0.384360         0.414585
Gamma         0.690674         1.511816         2.220732
Delta         0.184174        -1.069291        -0.994885
```

`pandas.Index.map(...)` 方法应用于转换索引的函数。

在以下示例中，`map` 函数取名称的前三个字符并将其设置为新名称：

```py
df2.index = df2.index.map(lambda x : x[:3]); df2
```

输出如下：

```py
Columns     colA         colB         colC
Alp        -2.071652     0.742857     0.632307
Bet         0.113046    -0.384360     0.414585
Gam         0.690674     1.511816     2.220732
Del         0.184174    -1.069291    -0.994885
```

`pandas.DataFrame.rename(...)` 方法允许我们转换索引名称和列名称，并接受从旧名称到新名称的字典映射：

```py
df2.rename(index={'Alp': 0, 'Bet': 1, 'Gam': 2, 'Del': 3}, 
           columns={'colA': 'A', 'colB': 'B', 'colC': 'C'})
```

结果 DataFrame 在两个轴上都有新标签：

```py
Columns      A            B            C
0           -2.071652     0.742857     0.632307
1            0.113046    -0.384360     0.414585
2            0.690674     1.511816     2.220732
3            0.184174    -1.069291    -0.994885
```

通过学习这个课程，我们将学习如何处理 DataFrame 中的缺失数据。

## 处理 DataFrame 中的缺失数据

缺失数据是数据科学中常见的现象，可能由多种原因导致 - 例如，技术错误，人为错误，市场假期。

### 过滤掉缺失数据

在处理缺失数据时，第一个选择是删除具有任何缺失数据的所有观察。

此代码块使用 `pandas.DataFrame.at[...]` 属性修改了 `df2` DataFrame，并将一些值设置为 `NaN`：

```py
for row, col in [('Bet', 'colA'), ('Bet', 'colB'), 
  ('Bet', 'colC'), ('Del', 'colB'), ('Gam', 'colC')]:
    df2.at[row, col] = np.NaN
df2
```

修改后的 DataFrame 如下：

```py
Columns      colA         colB          colC
Alp         -1.721523    -0.425150      1.425227
Bet          NaN          NaN           NaN
Gam         -0.408566    -1.121813      NaN
Del          0.361053     NaN           0.580435
```

`pandas.DataFrame.isnull(...)` 方法在 DataFrame 中查找缺失值：

```py
df2.isnull()
```

结果是一个 DataFrame，其中缺失值为 `True`，否则为 `False`：

```py
Columns     colA     colB     colC
Alp         False    False    False
Bet         True     True     True
Gam         False    False    True
Del         False    True     False
```

`pandas.DataFrame.notnull(...)` 方法执行相反操作（检测到非缺失值）：

```py
df2.notnull()
```

输出是以下 DataFrame：

```py
Columns    colA    colB    colC
Alp        True    True    True
Bet        False   False   False
Gam        True    True    False
Del        True    False   True
```

`pandas.DataFrame.dropna(...)` 方法允许我们删除具有缺失值的行。 附加的 `how=` 参数控制哪些行被删除。 要删除所有字段都为 `NaN` 的行，我们执行以下操作：

```py
df2.dropna(how='all')
```

结果是以下修改后的 DataFrame，其中 `Bet` 行被移除，因为那是唯一一个所有值都为 `NaN` 的行：

```py
Columns    colA         colB         colC
Alp       -1.721523    -0.425150     1.425227
Gam       -0.408566    -1.121813     NaN
Del        0.361053     NaN          0.580435
```

将 `how=` 设置为 `any` 会删除具有任何 NaN 值的行：

```py
df2.dropna(how='any')
```

这给我们以下包含所有非 NaN 值的 DataFrame：

```py
Columns     colA         colB       colC
Alp        -1.721523    -0.42515    1.425227
```

现在我们将看看如何填充缺失数据。

### 填充缺失数据

处理缺失数据的第二个选择是使用我们选择的值或使用同一列中的其他有效值来填充缺失值以复制/推断缺失值。

让我们首先重新检查一下 `df2` DataFrame：

```py
df2
```

这产生以下带有一些缺失值的 DataFrame：

```py
Columns     colA         colB        colC
Alp        -1.721523    -0.425150    1.425227
Bet         NaN          NaN         NaN
Gam        -0.408566    -1.121813    NaN
Del         0.361053     NaN         0.580435
```

现在，让我们使用 `pandas.DataFrame.fillna(...)` 方法，使用 `method='backfill'` 和 `inplace=True` 参数来使用 `backfill` 方法从其他值向后填充缺失值并就地更改 DataFrame：

```py
df2.fillna(method='backfill', inplace=True); 
df2
```

新的 DataFrame 包含以下内容：

```py
Columns     colA         colB        colC
Alp        -1.721523    -0.425150    1.425227
Bet        -0.408566    -1.121813    0.580435
Gam        -0.408566    -1.121813    0.580435
Del         0.361053     NaN         0.580435
```

`(Del,colB)` 处的 `NaN` 值是因为该行后没有观察到值，因此无法执行向后填充。 这可以使用向前填充来修复。

## 使用函数和映射来转换 DataFrame

pandas DataFrame 的值也可以通过传递函数和字典映射来修改，这些函数和映射作用于一个或多个数据值，并生成新的转换值。

让我们通过添加一个新列 `Category` 来修改 `df2` DataFrame，其中包含离散文本数据：

```py
df2['Category'] = ['HIGH', 'LOW', 'LOW', 'HIGH']; df2
```

新的 DataFrame 包含以下内容：

```py
Columns     colA         colB        colC        Category
Alp         1.017961     1.450681   -0.328989    HIGH
Bet        -0.079838    -0.519025    1.460911    LOW
Gam        -0.079838    -0.519025    1.460911    LOW
Del         0.359516     NaN         1.460911    HIGH
```

`pandas.Series.map(...)` 方法接受包含从旧值到新值的映射的字典，并对值进行转换。以下代码片段将 `Category` 中的文本值更改为单个字符：

```py
df2['Category'] = df2['Category'].map({'HIGH': 'H', 
                                       'LOW': 'L'}); 
df2
```

更新后的 DataFrame 如下所示：

```py
Columns     colA         colB        colC        Category
Alp         1.017961     1.450681   -0.328989    H
Bet        -0.079838    -0.519025    1.460911    L
Gam        -0.079838    -0.519025    1.460911    L
Del         0.359516     NaN         1.460911    H
```

`pandas.DataFrame.applymap(...)` 方法允许我们在 DataFrame 中对数据值应用函数。

以下代码应用了 `numpy.exp(...)` 方法，计算指数：

```py
df2.drop('Category', axis=1).applymap(np.exp)
```

结果是一个包含原始 DataFrame 值的指数值的 DataFrame（除了 `NaN` 值）：

```py
Columns    colA        colB        colC
Alp        2.767545    4.266020    0.719651
Bet        0.923266    0.595101    4.309883
Gam        0.923266    0.595101    4.309883
Del        1.432636    NaN         4.309883
```

现在我们已经学会了如何转换 DataFrame，我们将看到如何对 DataFrame 中的值进行离散化和分桶。

## DataFrame 值的离散化/分桶

实现离散化的最简单方法是创建数值范围，并为落入某个区间的所有值分配一个单独的离散标签。

首先，让我们为我们的使用生成一个随机值 ndarray：

```py
arr = np.random.randn(10); 
arr
```

这包括以下内容：

```py
array([  1.88087339e-01,  7.94570445e-01,  -5.97384701e-01,
        -3.01897668e+00, -5.42185315e-01,   1.10094663e+00,
         1.16002554e+00,  1.51491444e-03,  -2.21981570e+00,
         1.11903929e+00])
```

`pandas.cut(...)` 方法可用于离散化这些数值。以下代码使用 `bins=` 和 `labels=[...]` 参数将值分为五个离散值，并提供标签：

```py
cat = pd.cut(arr, bins=5, labels=['Very Low', 'Low', 'Med', 
                                  'High', 'Very High']); 
cat
```

在转换后，我们得到了离散值：

```py
 [High, Very High, Med, Very Low, Med, Very High, Very High, High, Very Low, Very High]
Categories (5, object): [Very Low < Low < Med < High < Very High]
```

`pandas.qcut(...)` 方法类似，但使用四分位数将连续值划分为离散值，以便每个类别具有相同数量的观测值。

以下使用 `q=` 参数构建了五个离散区间：

```py
qcat = pd.qcut(arr, q=5, labels=['Very Low', 'Low', 'Med', 
                                 'High', 'Very High']); 
qcat
```

四分位数离散化产生以下类别：

```py
[Med, High, Low, Very Low, Low, High, Very High, Med, Very Low, Very High]
Categories (5, object): [Very Low < Low < Med < High < Very High]
```

以下代码块构建了一个包含原始连续值以及由 `cut` 和 `qcut` 生成的类别的 pandas DataFrame：

```py
pd.DataFrame({'Value': arr, 'Category': cat, 
              'Quartile Category': qcat})
```

此 DataFrame 允许并列比较：

```py
Category    Quartile     Category    Value
0           High         Med         0.188087
1           Very High    High        0.794570
2           Med          Low        -0.597385
3           Very Low     Very Low   -3.018977
4           Med          Low        -0.542185
5           Very High    High        1.100947
6           Very High    Very High   1.160026
7           High         Med         0.001515
8           Very Low     Very Low   -2.219816
9           Very High    Very High   1.119039
```

`pandas.Categorical.categories` 属性为我们提供了区间范围：

```py
pd.cut(arr, bins=5).categories
```

在这种情况下，区间/数值范围如下：

```py
Index(['(-3.0232, -2.183]', '(-2.183, -1.347]', 
       '(-1.347, -0.512]', '(-0.512, 0.324]', 
       '(0.324, 1.16]'],
      dtype='object')
```

我们也可以检查 `qcut` 的区间：

```py
pd.qcut(arr, q=5).categories
```

它们与先前的区间略有不同，并显示如下：

```py
Index(['[-3.019, -0.922]', '(-0.922, -0.216]', 
       '(-0.216, 0.431]', '(0.431, 1.105]', 
       '(1.105, 1.16]'],
      dtype='object')
```

现在我们将看到如何对 DataFrame 值进行排列和抽样以生成新的 DataFrame。

## 对 DataFrame 值进行排列和抽样以生成新的 DataFrame

对可用数据集进行排列以生成新数据集，以及对数据集进行抽样以进行子抽样（减少观测数量）或超抽样（增加观测数量）是统计分析中常见的操作。

首先，让我们生成一个随机值 DataFrame 进行操作：

```py
df = pd.DataFrame(np.random.randn(10,5), 
                  index=np.sort(np.random.randint(0, 100, 
                                                 size=10)), 
                  columns=list('ABCDE')); 
df
```

结果如下：

```py
            A          B          C          D          E
 0  -0.564568  -0.188190  -1.678637  -0.128102  -1.880633
 0  -0.465880   0.266342   0.950357  -0.867568   1.504719
29   0.589315  -0.968324  -0.432725   0.856653  -0.683398
...
```

当应用于 DataFrame 时，`numpy.random.permutation(...)` 方法会沿着索引轴随机洗牌，并且可以用于对数据集的行进行置换：

```py
df.loc[np.random.permutation(df.index)]
```

这产生了以下随机打乱行的 DataFrame：

```py
            A         B          C           D         E
42   0.214554   1.108811   1.352568   0.238083  -1.090455
 0  -0.564568  -0.188190  -1.678637  -0.128102  -1.880633
 0  -0.465880   0.266342   0.950357  -0.867568   1.504719
62  -0.266102   0.831051  -0.164629   0.349047   1.874955
...
```

我们可以使用 `numpy.random.randint(...)` 方法在一定范围内生成随机整数，然后使用 `pandas.DataFrame.iloc[...]` 属性从我们的 DataFrame 中进行随机替换采样（同一观察结果可能会被多次选择

以下代码块随机选择了五行，并进行了替换采样：

```py
df.iloc[np.random.randint(0, len(df), size=5)]
```

这导致了以下随机子采样的 DataFrame：

```py
           A          B           C         D          E
54   0.692757  -0.584690  -0.176656   0.728395  -0.434987
98  -0.517141   0.109758  -0.132029   0.614610  -0.235801
29   0.589315  -0.968324  -0.432725   0.856653  -0.683398
35   0.520140   0.143652   0.973510   0.440253   1.307126
62  -0.266102   0.831051  -0.164629   0.349047   1.874955
```

在接下来的章节中，我们将探索使用 `pandas.DataFrames` 进行文件操作。

# 使用 pandas.DataFrames 探索文件操作

pandas 支持将 DataFrames 持久化到纯文本和二进制格式中。常见的文本格式是 CSV 和 JSON 文件，最常用的二进制格式是 Excel XLSX、HDF5 和 pickle。

在本书中，我们专注于纯文本持久化。

## CSV 文件

**CSV** 文件（**逗号分隔值** 文件）是数据交换标准文件。

### 写入 CSV 文件

使用 `pandas.DataFrame.to_csv(...)` 方法可以轻松将 pandas DataFrame 写入 CSV 文件。`header=` 参数控制是否将标题写入文件顶部，而 `index=` 参数控制是否将索引轴值写入文件：

```py
df.to_csv('df.csv', sep=',', header=True, index=True)
```

我们可以使用以下 Linux 命令检查写入磁盘的文件。`!` 字符指示笔记本运行一个 shell 命令：

```py
!head -n 4 df.csv
```

文件包含以下行：

```py
,A,B,C,D,E
4,-0.6329164608486778,0.3733235944037599,0.8225354680198685,-0.5171618315489593,0.5492241692404063
17,0.7664860447792711,0.8427366352142621,0.9621402130525599,-0.41134468872009666,-0.9704305306626816
24,-0.22976016405853183,0.38081314413811984,-1.526376189972014,0.07229102135441286,-0.3297356221604555
```

### 读取 CSV 文件

使用 `pandas.read_csv(...)` 方法可以读取 CSV 文件并构建一个 pandas DataFrame。在这里，我们将指定字符（虽然这是 `read_csv` 的默认值），`index_col=` 参数来指定哪一列作为 DataFrame 的索引，以及 `nrows=` 参数来指定要读取的行数：

```py
pd.read_csv('df.csv', sep=',', index_col=0, nrows=5)
```

这构建了以下 DataFrame，该 DataFrame 与写入磁盘的相同：

```py
           A          B          C           D         E
 4  -0.632916   0.373324   0.822535  -0.517162   0.549224
17   0.766486   0.842737   0.962140  -0.411345  -0.970431
24  -0.229760   0.380813  -1.526376   0.072291  -0.329736
33   0.662259  -1.457732  -2.268573   0.332456   0.496143
33   0.335710   0.452842  -0.977736   0.677470   1.164602
```

我们还可以指定 `chunksize=` 参数，该参数一次读取指定数量的行，这在探索非常大的文件中包含的非常大的数据集时会有所帮助：

```py
pd.read_csv('df.csv', sep=',', index_col=0, chunksize=2)
```

这将返回一个 pandas `TextFileReader` 生成器，我们可以根据需要迭代它，而不是一次加载整个文件：

```py
<pandas.io.parsers.TextFileReader at 0x7fb4e9933a90>
```

我们可以通过将生成器包装在列表中来强制生成器完成评估，并观察按两行一组加载的整个 DataFrame：

```py
list(pd.read_csv('df.csv', sep=',', index_col=0, 
                  chunksize=2))
```

这给我们带来了以下两行块的列表：

```py
[           A         B         C         D         E
 4  -0.632916  0.373324  0.822535 -0.517162  0.549224
 17  0.766486  0.842737  0.962140 -0.411345 -0.970431,
            A         B         C         D         E
 24 -0.229760  0.380813 -1.526376  0.072291 -0.329736
 33  0.662259 -1.457732 -2.268573  0.332456  0.496143,
...
```

现在我们将看看如何探索 JSON 文件中的文件操作。

## JSON 文件

JSON 文件基于与 Python 字典相同的数据结构。这使得 JSON 文件非常方便，可用于许多目的，包括表示 DataFrames 和表示配置文件。

`pandas.DataFrame.to_json(...)` 方法方便地将 DataFrame 写入磁盘上的 JSON 文件。在这里，我们只写入了前四行：

```py
df.iloc[:4].to_json('df.json')
```

让我们来看看写入磁盘的 JSON 文件：

```py
!cat df.json
```

这样我们就得到了以下写入磁盘的字典样式 JSON 文件：

```py
{"A":{"4":-0.6329164608,"17":0.7664860448,"24":-0.2297601641,"33":0.6622594878},"B":{"4":0.3733235944,"17":0.8427366352,"24":0.3808131441,"33":-1.4577321521},"C":{"4":0.822535468,"17":0.9621402131,"24":-1.52637619,"33":-2.2685732447},"D":{"4":-0.5171618315,"17":-0.4113446887 ,"24":0.0722910214,"33":0.3324557226},"E":{"4":0.5492241692 ,"17":-0.9704305307,"24":-0.3297356222,"33":0.4961425281}}
```

使用`pandas.read_json(...)`方法将 JSON 文件读回到 Pandas DataFrames 中同样很容易：

```py
pd.read_json('df.json')
```

这样我们就能得到原始的写入磁盘的四行 DataFrame：

```py
            A         B           C         D           E
 4  -0.632916   0.373324   0.822535  -0.517162   0.549224
17   0.766486   0.842737   0.962140  -0.411345  -0.970431
24  -0.229760   0.380813  -1.526376   0.072291  -0.329736
33   0.662259  -1.457732  -2.268573   0.332456   0.496143
```

恭喜成功完成本课程！

# 总结

本章介绍了 pandas 库，几乎所有 Python 中的时间序列操作都是基于它完成的。我们已经学会了如何创建 DataFrame，如何修改它以及如何持久化它。

Pandas DataFrames 主要用于高性能的大规模数据操作、选择和重塑数据。它们是 Python 版本的 Excel 工作表。

在下一章中，我们将使用 Matplotlib 在 Python 中进行可视化探索。
