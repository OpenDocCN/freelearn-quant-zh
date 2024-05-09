处理和操纵日期、时间和时间序列数据

当涉及到算法交易时，时间序列数据是无处不在的。因此，处理、管理和操纵时间序列数据对于成功执行算法交易至关重要。本章包含了各种食谱，演示了如何使用 Python 标准库和`pandas`来进行算法交易，`pandas`是一个 Python 数据分析库。

对于我们的上下文，时间序列数据是一系列数据，由等间隔的时间戳和描述特定时间段内交易数据的多个数据点组成。

处理时间序列数据时，您首先应该了解的是如何读取、修改和创建理解日期和时间的 Python 对象。Python 标准库包括了`datetime`模块，它提供了`datetime`和`timedelta`对象，可以处理关于日期和时间的所有内容。本章的前七个食谱讨论了这个模块。本章的剩余部分讨论了如何使用`pandas`库处理时间序列数据，`pandas`是一个非常高效的数据分析库。我们的食谱将使用`pandas.DataFrame`类。

以下是本章的食谱列表：

+   创建日期时间对象

+   创建时间差对象

+   对日期时间对象进行操作

+   修改日期时间对象

+   将日期时间转换为字符串

+   从字符串创建日期时间对象

+   日期时间对象和时区

+   创建一个 pandas.DataFrame 对象

+   DataFrame 操作——重命名、重新排列、反转和切片

+   DataFrame 操作——应用、排序、迭代和连接

+   将 DataFrame 转换为其他格式

+   从其他格式创建 DataFrame

# 技术要求

您将需要以下内容才能成功执行本章的食谱：

+   Python 3.7+

+   Python 包：

+   `pandas` (`$ pip install pandas`)

对于本章中的所有食谱，您将需要本章的 Jupyter 笔记本，位于[`github.com/PacktPublishing/Python-Algorithmic-Trading-Cookbook/tree/master/Chapter01`](https://github.com/PacktPublishing/Python-Algorithmic-Trading-Cookbook/tree/master/Chapter01)。

您还可以打开一个新的 Jupyter 笔记本，并直接尝试食谱中显示的实践练习。请注意，对于其中一些食谱，您的输出可能会有所不同，因为它们取决于提供的日期、时间和时区信息。

# 创建日期时间对象

`datetime`模块提供了一个`datetime`类，它可以用于准确捕获与时间戳、日期、时间和时区相关的信息。在本食谱中，您将以多种方式创建`datetime`对象，并检查其属性。

## 如何做...

按照以下步骤执行本食谱：

1.  从 Python 标准库中导入必要的模块：

```py
>>> from datetime import datetime
```

1.  使用`now()`方法创建一个持有当前时间戳的`datetime`对象并打印它：

```py
>>> dt1 = datetime.now()
>>> print(f'Approach #1: {dt1}')
```

我们得到以下输出。您的输出将有所不同：

```py
Approach #1: 2020-08-12 20:55:39.680195
```

1.  打印与`dt1`相关的日期和时间属性：

```py
>>> print(f'Year: {dt1.year}')
>>> print(f'Month: {dt1.month}')
>>> print(f'Day: {dt1.day}')
>>> print(f'Hours: {dt1.hour}')
>>> print(f'Minutes: {dt1.minute}')
>>> print(f'Seconds: {dt1.second}')
>>> print(f'Microseconds: {dt1.microsecond}')
>>> print(f'Timezone: {dt1.tzinfo}')
```

我们得到以下输出。您的输出可能会有所不同：

```py
Year: 2020
Month: 8
Day: 12
Hours: 20
Minutes: 55
Seconds: 39
Microseconds: 680195
Timezone: None
```

1.  创建一个持有 2021 年 1 月 1 日的时间戳的`datetime`对象：

```py
>>> dt2 = datetime(year=2021, month=1, day=1)
>>> print(f'Approach #2: {dt2}')
```

您将得到以下输出：

```py
Approach #2: 2021-01-01 00:00:00
```

1.  打印与`dt2`相关的各种日期和时间属性：

```py
>>> print(f'Year: {dt.year}')
>>> print(f'Month: {dt.month}')
>>> print(f'Day: {dt.day}')
>>> print(f'Hours: {dt.hour}')
>>> print(f'Minutes: {dt.minute}')
>>> print(f'Seconds: {dt.second}')
>>> print(f'Microseconds: {dt.microsecond}')
>>> print(f'Timezone: {dt2.tzinfo}')
```

您将得到以下输出：

```py
Year: 2021
Month: 1
Day: 1
Hours: 0
Minutes: 0
Seconds: 0
Microseconds: 0
Timezone: None
```

## 工作原理...

在*步骤 1*中，您从`datetime`模块中导入`datetime`类。在*步骤 2*中，您使用`now()`方法创建并打印一个`datetime`对象，并将其分配给`dt1`。该对象保存当前的时间戳信息。

一个`datetime`对象具有以下与日期、时间和时区信息相关的属性：

| 1 | `year` | 一个介于 0 和 23 之间的整数，包括 0 和 23 |
| --- | --- | --- |
| 2 | `month` | 一个介于 1 和 12 之间的整数，包括 1 和 12 |
| 3 | `day` | 一个介于 1 和 31 之间的整数，包括 1 和 31 |
| 4 | `hour` | 一个介于 0 和 23 之间的整数，包括 0 和 23 |
| 5 | `minute` | 一个介于 0 和 59 之间的整数，包括 0 和 59 |
| 6 | `second` | 一个介于 0 和 59 之间的整数，包括 0 和 59 |
| 7 | `microsecond` | 一个介于 0 和 999999 之间的整数，包括 0 和 999999 |
| 8 | `tzinfo` | 一个`timezone`类的对象。（有关时区的更多信息，请参阅*日期时间对象和时区*示例。） |

在*步骤 3*中，这些属性被打印为`dt1`。您可以看到它们保存了当前时间戳信息。

在*步骤 4*中，您创建并打印另一个`datetime`对象。这次您创建了一个特定的时间戳，即 2021 年 1 月 1 日，午夜。您将构造函数本身与参数一起调用——`year`为`2021`，`month`为`1`，`day`为`1`。其他与时间相关的属性默认为`0`，时区默认为`None`。在*步骤 5*中，您打印了`dt2`的属性。您可以看到它们与您在*步骤 4*中传递给构造函数的值完全相同。

## 还有更多

您可以使用`datetime`对象的`date()`和`time()`方法提取日期和时间信息，分别作为`datetime.date`和`datetime.time`类的实例：

1.  使用`date()`方法从`dt1`中提取日期。注意返回值的类型。

```py
>>> print(f"Date: {dt1.date()}")
>>> print(f"Type: {type(dt1.date())}")
```

您将得到以下输出。您的输出可能会有所不同：

```py
Date: 2020-08-12
Type: <class 'datetime.date'>
```

1.  使用`time()`方法从`dt1`中提取日期。注意返回值的类型。

```py
>>> print(f"Time: {dt1.time()}")
>>> print(f"Type: {type(dt1.time())}")
```

我们得到以下输出。您的输出可能会有所不同：

```py
Time: 20:55:39.680195
Type: <class 'datetime.time'>
```

1.  使用`date()`方法从`dt2`中提取日期。注意返回值的类型。

```py
>>> print(f"Date: {dt2.date()}")
>>> print(f"Type: {type(dt2.date())}")
```

我们得到以下输出：

```py
Date: 2021-01-01
Type: <class 'datetime.date'>
```

1.  使用`time()`方法从`dt2`中提取日期。注意返回值的类型。

```py
>>> print(f"Time: {dt2.time()}")
>>> print(f"Type: {type(dt2.time())}")
```

我们得到以下输出：

```py
Time: 00:00:00
Type: <class 'datetime.time'>
```

# 创建 timedelta 对象

`datetime`模块提供了一个`timedelta`类，可用于表示与日期和时间差异相关的信息。在本示例中，您将创建`timedelta`对象并对其执行操作。

## 如何做…

按照以下步骤执行此示例：

1.  从 Python 标准库中导入必要的模块：

```py
>>> from datetime import timedelta
```

1.  创建一个持续时间为 5 天的`timedelta`对象。将其分配给`td1`并打印它：

```py
>>> td1 = timedelta(days=5)
>>> print(f'Time difference: {td1}')
```

我们得到以下输出：

```py
Time difference: 5 days, 0:00:00
```

1.  创建一个持续 4 天的`timedelta`对象。将其赋值给`td2`并打印出来：

```py
>>> td2 = timedelta(days=4)
>>> print(f'Time difference: {td2}')
```

我们得到以下输出：

```py
Time difference: 4 days, 0:00:00
```

1.  将`td1`和`td2`相加并打印输出：

```py
>>> print(f'Addition: {td1} + {td2} = {td1 + td2}')
```

我们得到以下输出：

```py
Addition: 5 days, 0:00:00 + 4 days, 0:00:00 = 9 days, 0:00:00
```

1.  将`td2`从`td1`中减去并打印输出：

```py
>>> print(f'Subtraction: {td1} - {td2} = {td1 - td2}')
```

我们将得到以下输出：

```py
Subtraction: 5 days, 0:00:00 - 4 days, 0:00:00 = 1 day, 0:00:00
```

1.  将`td1`乘以一个数字（一个`浮点数`）：

```py
>>> print(f'Multiplication: {td1} * 2.5 = {td1 * 2.5}')
```

我们得到以下输出：

```py
Multiplication: 5 days, 0:00:00 * 2.5 = 12 days, 12:00:00
```

## 工作原理...

在*步骤 1*中，您从`datetime`模块中导入`timedelta`类。在*步骤 2*中，您创建一个持有`5 天`时间差值的`timedelta`对象，并将其赋值给`td1`。您调用构造函数来创建具有单个属性`days`的对象。您在此处传递值为`5`。类似地，在*步骤 3*中，您创建另一个`timedelta`对象，其中包含`4 天`的时间差值，并将其赋值给`td2`。

在接下来的步骤中，您对`timedelta`对象执行操作。在*步骤 4*中，您将`td1`和`td2`相加。这将返回另一个`timedelta`对象，其中包含`9 天`的时间差值，这是由`td1`和`td2`持有的时间差值的总和。在*步骤 5*中，您将`td2`从`td1`中减去。这将返回另一个`timedelta`对象，其中包含`1 天`的时间差值，这是由`td1`和`td2`持有的时间差值之间的差异。在*步骤 6*中，您将`td1`乘以`2.5`，一个`浮点数`。这再次返回一个`timedelta`对象，其中包含十二天半的时间差值。

## 还有更多内容

可以使用一个或多个可选参数创建`timedelta`对象：

| 1 | `weeks` | 一个整数，默认值为 0。 |
| --- | --- | --- |
| 2 | `days` | 一个整数，默认值为 0。 |
| 3 | `hours` | 一个整数，默认值为 0。 |
| 4 | `minutes` | 一个整数，默认值为 0。 |
| 5 | `seconds` | 一个整数，默认值为 0。 |
| 6 | `milliseconds` | 一个整数，默认值为 0。 |
| 7 | `microseconds` | 一个整数，默认值为 0。 |

在*步骤 2*和*步骤 3*中，我们仅使用了`days`参数。您也可以使用其他参数。此外，这些属性在创建时被标准化。对`timedelta`对象的这种标准化是为了确保每个时间差值都有一个唯一的表示形式。以下代码演示了这一点：

1.  创建一个小时为`23`，分钟为`59`，秒数为`60`的`timedelta`对象。将其赋值给`td3`并打印出来。它将被标准化为一个`timedelta`对象，其中`days`为`1`（其他日期和时间相关属性为`0`）：

```py
>>> td3 = timedelta(hours=23, minutes=59, seconds=60)
>>> print(f'Time difference: {td3}')
```

我们得到以下输出：

```py
Time difference: 1 day, 0:00:00
```

`timedelta`对象有一个方便的方法，`total_seconds()`。该方法返回一个`浮点数`，表示`timedelta`对象持续的总秒数。

1.  在`td3`上调用`total_seconds()`方法。您将得到`86400.0`作为输出：

```py
>>> print(f'Total seconds in 1 day: {td3.total_seconds()}')
```

我们得到以下输出：

```py
Total seconds in 1 day: 86400.0
```

# 时间对象上的操作

`datetime`和`timedelta`类支持各种数学操作，以获取未来或过去的日期。使用这些操作返回另一个`datetime`对象。在这个示例中，您将创建`datetime`、`date`、`time`和`timedelta`对象，并对它们执行数学运算。

## 如何做...

按照这些步骤执行此操作：

1.  从 Python 标准库中导入必要的模块：

```py
>>> from datetime import datetime, timedelta
```

1.  获取今天的日期。将其赋值给`date_today`并打印出来：

```py
>>> date_today = date.today()              
>>> print(f"Today's Date: {date_today}")
```

我们得到以下输出。您的输出可能有所不同：

```py
Today's Date: 2020-08-12
```

1.  使用`timedelta`对象将 5 天添加到今天的日期。将其赋值给`date_5days_later`并打印出来：

```py
>>> date_5days_later = date_today + timedelta(days=5)
>>> print(f"Date 5 days later: {date_5days_later}")
```

我们得到以下输出。您的输出可能有所不同：

```py
Date 5 days later: 2020-08-17
```

1.  使用`timedelta`对象从今天的日期减去 5 天。将其赋值给`date_5days_ago`并打印出来：

```py
>>> date_5days_ago = date_today - timedelta(days=5)
>>> print(f"Date 5 days ago: {date_5days_ago}")
```

我们得到以下输出。您的输出可能有所不同：

```py
Date 5 days ago: 2020-08-07
```

1.  使用`>`操作符将`date_5days_later`与`date_5days_ago`进行比较：

```py
>>> date_5days_later > date_5days_ago
```

我们得到以下输出：

```py
True
```

1.  使用`<`操作符将`date_5days_later`与`date_5days_ago`进行比较：

```py
>>> date_5days_later < date_5days_ago
```

我们得到以下输出：

```py
False
```

1.  使用`>`操作符将`date_5days_later`、`date_today`和`date_5days_ago`一起进行比较：

```py
>>> date_5days_later > date_today > date_5days_ago
```

我们得到以下输出：

```py
True
```

1.  获取当前时间戳。将其赋值给`current_timestamp`：

```py
>>> current_timestamp = datetime.now()
```

1.  获取当前时间。将其赋值给`time_now`并打印出来：

```py
>>> time_now = current_timestamp.time()
>>> print(f"Time now: {time_now}")
```

我们得到以下输出。您的输出可能有所不同：

```py
Time now: 20:55:45.239177
```

1.  使用`timedelta`对象将 5 分钟添加到当前时间。将其赋值给`time_5minutes_later`并打印出来：

```py
>>> time_5minutes_later = (current_timestamp + 
                                timedelta(minutes=5)).time()
>>> print(f"Time 5 minutes later: {time_5minutes_later}")
```

我们得到以下输出。您的输出可能有所不同：

```py
Time 5 minutes later: 21:00:45.239177
```

1.  使用`timedelta`对象从当前时间减去 5 分钟。将其赋值给`time_5minutes_ago`并打印出来：

```py
>>> time_5minutes_ago = (current_timestamp - 
                            timedelta(minutes=5)).time()
>>> print(f"Time 5 minutes ago: {time_5minutes_ago}")
```

我们得到以下输出。您的输出可能有所不同：

```py
Time 5 minutes ago: 20:50:45.239177
```

1.  使用`<`操作符将`time_5minutes_later`与`time_5minutes_ago`进行比较：

```py
>>> time_5minutes_later < time_5minutes_ago
```

我们得到以下输出。您的输出可能有所不同：

```py
False
```

1.  使用`>`操作符将`time_5minutes_later`与`time_5minutes_ago`进行比较：

```py
>>> time_5minutes_later > time_5minutes_ago
```

我们得到以下输出。您的输出可能有所不同：

```py
True
```

1.  使用`>`操作符将`time_5minutes_later`、`time_now`和`time_5minutes_ago`一起进行比较：

```py
>> time_5minutes_later > time_now > time_5minutes_ago
```

我们得到以下输出。您的输出可能有所不同：

```py
True
```

## 工作原理...

在*步骤 1*中，您从`datetime`模块导入`date`、`datetime`和`timedelta`类。在*步骤 2*中，您使用类`date`提供的`today()` `classmethod`获取今天的日期，并将其赋值给一个新属性`date_today`。（`classmethod`允许您直接在类上调用方法而不创建实例。）返回的对象类型为`datetime.date`。在*步骤 3*中，您通过将持续时间为 5 天的`timedelta`对象添加到`date_today`来创建一个比今天晚 5 天的日期。您将此赋值给一个新属性`date_5days_later`。同样，在*步骤 4*中，您创建一个 5 天前的日期并将其赋值给一个新属性`date_5days_ago`。

在*步骤 5* 和 *步骤 6* 中，你使用 `>` 和 `<` 操作符分别比较 `date_5days_later` 和 `date_5days_ago`。如果第一个操作数保存的日期在第二个操作数之后，则 `>` 操作符返回 `True`。类似地，如果第二个操作数保存的日期在第一个操作数之后，则 `<` 操作符返回 `True`。在 *步骤 7* 中，你比较到目前为止创建的所有三个日期对象。注意输出。

*步骤 8* 到 *步骤 14* 执行与 *步骤 2* 到 *步骤 7* 相同的操作，但这次是在`datetime.time`对象上——获取当前时间、获取当前时间之后的 5 分钟、获取当前时间之前的 5 分钟，并比较所有创建的`datetime.time`对象。无法直接将`timedelta`对象添加到`datetime.time`对象中以获取过去或未来的时间。为了克服这一点，你可以将`timedelta`对象添加到`datetime`对象中，然后使用`time()`方法从中提取时间。你在 *步骤 10* 和 *步骤 11* 中执行此操作。

## 还有更多

本示例展示了对`date`和`time`对象的操作，这些操作可以类似地在`datetime`对象上执行。除了`+`、`-`、`<`和`>`之外，你还可以在`datetime`、`date`和`time`对象上使用以下操作符：

| `>=` | 仅在第一个操作数保持的`datetime`/`date`/`time`晚于或等于第二个操作数时返回`True` |
| --- | --- |
| `<=` | 仅在第一个操作数保持的`datetime`/`date`/`time`早于或等于第二个操作数时返回`True` |
| `==` | 仅在第一个操作数保持的`datetime`/`date`/`time`等于第二个操作数时返回`True` |

这不是允许的操作符的详尽列表。有关更多信息，请参阅`datetime`模块的官方文档：[`docs.python.org/3.8/library/datetime.html`](https://docs.python.org/3.8/library/datetime.html)。

# 修改 datetime 对象

通常，你可能希望修改现有的`datetime`对象以表示不同的日期和时间。本示例包括演示此操作的代码。

## 如何做…

按照以下步骤执行此示例：

1.  从 Python 标准库中导入必要的模块：

```py
>>> from datetime import datetime
```

1.  获取当前时间戳。将其分配给`dt1`并打印：

```py
>>> dt1 = datetime.now()
>>> print(dt1)
```

我们得到以下输出。你的输出会有所不同：

```py
2020-08-12 20:55:46.753899
```

1.  通过替换`dt1`的`year`、`month`和`day`属性来创建一个新的`datetime`对象。将其分配给`dt2`并打印：

```py
>>> dt2 = dt1.replace(year=2021, month=1, day=1)
>>> print(f'A timestamp from 1st January 2021: {dt2}')
```

我们得到以下输出。你的输出会有所不同：

```py
A timestamp from 1st January 2021: 2021-01-01 20:55:46.753899
```

1.  通过直接指定所有属性来创建一个新的`datetime`对象。将其分配给`dt3`并打印它：

```py
>>> dt3 = datetime(year=2021, 
                   month=1, 
                   day=1,
                   hour=dt1.hour,
                   minute=dt1.minute, 
                   second=dt1.second, 
                   microsecond=dt1.microsecond, 
                   tzinfo=dt1.tzinfo)
print(f'A timestamp from 1st January 2021: {dt3}')
```

我们得到以下输出。你的输出会有所不同：

```py
A timestamp from 1st January 2021: 2021-01-01 20:55:46.753899
```

1.  比较`dt2`和`dt3`：

```py
>>> dt2 == dt3
```

我们得到以下输出。

```py
True
```

## 工作原理...

在*步骤 1*中，您从`datetime`模块中导入`datetime`类。在*步骤 2*中，您使用`datetime`的`now()`方法获取当前时间戳并将其赋值给新属性`dt1`。要从现有的`datetime`对象获取修改后的时间戳，可以使用`replace()`方法。在*步骤 3*中，您通过调用`replace()`方法从`dt1`创建一个新的`datetime`对象`dt2`。您指定要修改的属性，即`year`、`month`和`day`。其余属性保持不变，即`hour`、`minute`、`second`、`microsecond`和`timezone`。您可以通过比较*步骤 2*和*步骤 3*的输出来确认这一点。在*步骤 4*中，您创建另一个`datetime`对象`dt3`。这次，您直接调用`datetime`构造函数。您将所有属性传递给构造函数，使创建的时间戳与`dt2`相同。在*步骤 5*中，您使用`==`运算符确认`dt2`和`dt3`持有完全相同的时间戳，该运算符返回`True`。

# 将`datetime`对象转换为字符串

本配方演示了将`datetime`对象转换为字符串的过程，该过程在打印和日志记录中应用。此外，在通过 web API 发送时间戳时也很有帮助。

## 如何做…

执行此配方的以下步骤：

1.  从 Python 标准库中导入必要的模块：

```py
>>> from datetime import datetime
```

1.  获取带有时区信息的当前时间戳。将其分配给`now`并打印出来：

```py
>>> now = datetime.now().astimezone()
```

1.  将`now`强制转换为字符串并打印出来：

```py
>>> print(str(now))
```

我们得到以下输出。您的输出可能会有所不同：

```py
2020-08-12 20:55:48.366130+05:30
```

1.  使用`strftime()`将`now`转换为具有特定日期时间格式的字符串并打印出来：

```py
>>> print(now.strftime("%d-%m-%Y %H:%M:%S %Z"))
```

我们得到以下输出。您的输出可能会有所不同：

```py
12-08-2020 20:55:48 +0530
```

## 如何运作...

在*步骤 1*中，您从`datetime`模块中导入`datetime`类。在*步骤 2*中，您使用带有时区的当前时间戳并将其赋值给新属性`now`。`datetime`的`now()`方法获取当前时间戳，但没有时区信息。这样的对象称为时区本地的`datetime`对象。`astimezone()`方法从此时区无关对象上添加系统本地时区的时区信息，从而将其转换为时区感知对象。（有关更多信息，请参阅*datetime 对象和时区*配方）。在*步骤 3*中，您将`now`转换为字符串对象并将其打印出来。请注意，输出的日期格式是固定的，可能不是您的选择。`datetime`模块有一个`strftime()`方法，它可以按需要将对象转换为特定格式的字符串。在*步骤 4*中，您将`now`转换为格式为`DD-MM-YYYY HH:MM:SS +Z`的字符串。*步骤 4*中使用的指令描述如下：

| **指令** | **意义** |
| --- | --- |
| `%d` | 以零填充的十进制数表示的月份中的一天 |
| `%m` | 以零填充的十进制月份 |
| `%Y` | 十进制数世纪年份 |
| `%H` | 小时（24 小时制）以零填充的十进制数 |
| `%M` | 分钟，以零填充的十进制数 |
| `%S` | 秒，以零填充的十进制数 |
| `%Z` | 时区名称（如果对象是无时区的，则为空字符串） |

可以在[`docs.python.org/3.7/library/datetime.html#strftime-and-strptime-behavior`](https://docs.python.org/3.7/library/datetime.html#strftime-and-strptime-behavior)找到可以提供给`.strptime()`的指令的完整列表。

# 从字符串创建 datetime 对象

此配方演示了将格式良好的字符串转换为`datetime`对象。这在从文件中读取时间戳时很有用。此外，在通过 Web API 接收时间戳作为 JSON 数据时也很有帮助。

## 如何做...

执行此配方的以下步骤：

1.  从 Python 标准库中导入必要的模块：

```py
>>> from datetime import datetime
```

1.  创建一个包含日期、时间和时区的时间戳的字符串表示形式。将其赋值给`now_str`：

```py
>>> now_str = '13-1-2021 15:53:39 +05:30'
```

1.  将`now_str`转换为`now`，一个`datetime.datetime`对象。打印出来：

```py
>>> now = datetime.strptime(now_str, "%d-%m-%Y %H:%M:%S %z")
>>> print(now)
```

我们得到以下输出：

```py
2021-01-13 15:53:39+05:30
```

1.  确认 now 是`datetime`类型：

```py
>>> print(type(now))
```

我们得到以下输出：

```py
<class 'datetime.datetime'>
```

## 如何工作...

在*步骤 1*中，你从`datetime`模块中导入`datetime`类。在*步骤 2*中，你创建一个包含有效时间戳的字符串，并将其赋值给一个新属性`now_str`。`datetime`模块有一个`strptime()`方法，可以将一个特定格式的字符串转换为`datetime`对象。在*步骤 3*中，你将`now_str`，一个格式为`DD-MM-YYYY HH:MM:SS +Z`的字符串，转换为`now`。在*步骤 4*中，你确认`now`确实是`datetime`类型的对象。在*步骤 3*中使用的指令与*将 datetime 对象转换为字符串*配方中描述的相同。

## 还有更多

当将字符串读入`datetime`对象时，应使用适当的指令消耗整个字符串。部分消耗字符串将引发异常，如下面的代码片段所示。错误消息显示了未转换的数据，并可用于修复提供给`strptime()`方法的指令。

尝试使用`strptime()`方法将`now_str`转换为`datetime`对象。只传递包含字符串日期部分指令的字符串。注意错误：

```py
>>> now = datetime.strptime(now_str, "%d-%m-%Y")
```

输出如下：

```py
# Note: It's expected to have an error below
---------------------------------------------------------------------------
ValueError Traceback (most recent call last)
<ipython-input-96-dc92a0358ed8> in <module>
----> 1 now = datetime.strptime(now_str, "%d-%m-%Y")
      2 # Note: It's expected to get an error below

/usr/lib/python3.8/_strptime.py in _strptime_datetime(cls, data_string, format)
    566 """Return a class cls instance based on the input string and the
    567 format string."""
--> 568 tt, fraction, gmtoff_fraction = _strptime(data_string, format)
    569 tzname, gmtoff = tt[-2:]
    570 args = tt[:6] + (fraction,)

/usr/lib/python3.8/_strptime.py in _strptime(data_string, format)
    350 (data_string, format))
    351 if len(data_string) != found.end():
--> 352 raise ValueError("unconverted data remains: %s" %
    353 data_string[found.end():])
    354 

ValueError: unconverted data remains: 15:53:39 +05:30
```

# `datetime`对象和时区

有两种类型的`datetime`对象——时区无关和时区感知。时区无关对象不包含时区信息，而时区感知对象包含时区信息。这个配方演示了在`datetime`对象上执行多个与时区相关的操作：创建时区无关和时区感知对象，向时区感知对象添加时区信息，从时区无关对象中删除时区信息，以及比较时区感知和时区无关对象。

## 如何做...

执行此配方的以下步骤：

1.  从 Python 标准库中导入必要的模块：

```py
>>> from datetime import datetime
```

1.  创建一个时区无关的 `datetime` 对象。将其赋给 `now_tz_naive` 并打印它：

```py
>>> now_tz_unaware = datetime.now()
>>> print(now_tz_unaware)
```

我们得到了以下输出。您的输出可能会有所不同：

```py
2020-08-12 20:55:50.598800
```

1.  打印 `now_tz_naive` 的时区信息。注意输出：

```py
>>> print(now_tz_unaware.tzinfo)
```

我们得到了以下输出：

```py
None
```

1.  创建一个时区感知的 `datetime` 对象。将其赋给 `now_tz_aware` 并打印它：

```py
>>> now_tz_aware = datetime.now().astimezone()
>>> print(now_tz_aware)
```

我们得到了以下输出。您的输出可能会有所不同：

```py
2020-08-12 20:55:51.004671+05:30
```

1.  打印 `now_tz_aware` 的时区信息。注意输出：

```py
>>> print(now_tz_aware.tzinfo)
```

我们得到了以下输出。您的输出可能会有所不同：

```py
IST
```

1.  通过从 `now_tz_aware` 中添加时区信息创建一个新时间戳。将其赋给 `new_tz_aware` 并打印它：

```py
>>> new_tz_aware = now_tz_naive.replace(tzinfo=now_tz_aware.tzinfo)
>>> print(new_tz_aware)
```

输出如下。您的输出可能会有所不同：

```py
2020-08-12 20:55:50.598800+05:30
```

1.  使用 `tzinfo` 属性打印 `new_tz_aware` 的时区信息。注意输出：

```py
>>> print(new_tz_aware.tzinfo)
```

输出如下。您的输出可能会有所不同：

```py
IST
```

1.  通过从 `new_tz_aware` 中移除时区信息创建一个新的时间戳。将其赋给 `new_tz_naive` 并打印它：

```py
>>> new_tz_naive = new_tz_aware.replace(tzinfo=None)
>>> print(new_tz_naive)
```

输出如下。您的输出可能会有所不同：

```py
2020-08-12 20:55:50.598800
```

1.  使用 `tzinfo` 属性打印 `new_tz_naive` 的时区信息。注意输出：

```py
>>> print(new_tz_naive.tzinfo)
```

输出如下：

```py
None
```

## 工作原理如下...

在 *步骤 1* 中，从 `datetime` 模块中导入 `datetime` 类。在 *步骤 2* 中，使用 `now()` 方法创建一个时区无关的 `datetime` 对象，并将其赋给一个新属性 `now_tz_naive`。在 *步骤 3* 中，使用 `tzinfo` 属性打印 `now_tz_naive` 所持有的时区信息。观察到输出为 `None`，因为这是一个时区无关的对象。

*步骤 4* 中，使用 `now()` 和 `astimezone()` 方法创建了一个时区感知的 `datetime` 对象，并将其赋给一个新属性 `now_tz_aware`。*步骤 5* 中，使用 `tzinfo` 属性打印了 `now_tz_aware` 所持有的时区信息。注意输出为 `IST` 而不是 `None`，因为这是一个时区感知对象。

在 *步骤 6* 中，通过向 `now_tz_naive` 添加时区信息来创建一个新的 `datetime` 对象。时区信息来自 `now_tz_aware`。你可以使用 `replace()` 方法实现这一点（有关更多信息，请参阅 *修改 datetime 对象* 配方）。将其赋给一个新变量 `new_tz_aware`。在 *步骤 7* 中，打印 `new_tz_aware` 所持有的时区信息。观察到它与 *步骤 5* 中的输出相同，因为你从 `now_tz_aware` 中取了时区信息。同样，在 *步骤 8* 和 *步骤 9* 中，你创建了一个新的 `datetime` 对象 `new_tz_naive`，但这次你移除了时区信息。

## 还有更多

您只能在时区无关或时区感知的 `datetime` 对象之间使用比较运算符。你不能比较一个时区无关的 `datetime` 对象和一个时区感知的 `datetime` 对象。这样做会引发异常。这在以下步骤中得到了证明：

1.  比较两个时区无关对象，`new_tz_naive` 和 `now_tz_naive`。注意输出：

```py
>>> new_tz_naive <= now_tz_naive
```

1.  比较两个时区感知对象，`new_tz_aware` 和 `now_tz_aware`。注意输出：

```py
>>> new_tz_aware <= now_tz_aware
```

我们得到了以下输出：

```py
True
```

1.  比较一个时区感知对象和一个时区不感知对象，`new_tz_aware`和`now_tz_naive`。注意错误：

```py
>>> new_tz_aware > now_tz_naive
```

我们得到以下输出：

```py
-------------------------------------------------------------------
            TypeError Traceback (most recent call last)
<ipython-input-167-a9433bb51293> in <module>
----> 1 new_tz_aware > now_tz_naive
      2 # Note: It's expected to get an error below

TypeError: can't compare offset-naive and offset-aware datetimes
```

# 创建一个 pandas.DataFrame 对象

现在我们已经完成了日期和时间的处理，让我们转向处理时间序列数据。`pandas`库有一个`pandas.DataFrame`类，对于处理和操作这样的数据很有用。这个示例从创建这些对象开始。

## 如何做...

对于这个示例，执行以下步骤：

1.  从 Python 标准库中导入必要的模块：

```py
>>> from datetime import datetime
>>> import pandas
```

1.  创建一个时间序列数据的示例，作为一个字典对象列表。将其分配给`time_series`数据：

```py
>>> time_series_data = \
[{'date': datetime.datetime(2019, 11, 13, 9, 0),   
  'open': 71.8075, 'high': 71.845,  'low': 71.7775, 
  'close': 71.7925, 'volume': 219512},
{'date': datetime.datetime(2019, 11, 13, 9, 15),  
 'open': 71.7925, 'high': 71.8,    'low': 71.78,   
 'close': 71.7925, 'volume': 59252},
{'date': datetime.datetime(2019, 11, 13, 9, 30),  
 'open': 71.7925, 'high': 71.8125, 'low': 71.76,
 'close': 71.7625, 'volume': 57187},
{'date': datetime.datetime(2019, 11, 13, 9, 45),  
 'open': 71.76,   'high': 71.765,  'low': 71.735,  
 'close': 71.7425, 'volume': 43048}, 
{'date': datetime.datetime(2019, 11, 13, 10, 0),  
 'open': 71.7425, 'high': 71.78,   'low': 71.7425, 
 'close': 71.7775, 'volume': 45863},
{'date': datetime.datetime(2019, 11, 13, 10, 15), 
 'open': 71.775,  'high': 71.8225, 'low': 71.77,   
 'close': 71.815,  'volume': 42460},
{'date': datetime.datetime(2019, 11, 13, 10, 30), 
 'open': 71.815,  'high': 71.83,   'low': 71.7775, 
 'close': 71.78,   'volume': 62403},
{'date': datetime.datetime(2019, 11, 13, 10, 45), 
 'open': 71.775,  'high': 71.7875, 'low': 71.7475,
 'close': 71.7525, 'volume': 34090},
{'date': datetime.datetime(2019, 11, 13, 11, 0),  
 'open': 71.7525, 'high': 71.7825, 'low': 71.7475,
 'close': 71.7625, 'volume': 39320},
{'date': datetime.datetime(2019, 11, 13, 11, 15), 
 'open': 71.7625, 'high': 71.7925, 'low': 71.76,
 'close': 71.7875, 'volume': 20190}]
```

1.  从`time_series_data`创建一个新的`DataFrame`。将其分配给`df`并打印它：

```py
>>> df = pandas.DataFrame(time_series_data)
>>> df
```

我们得到以下输出：

```py
                 date    open    high     low   close volume
0 2019-11-13 09:00:00 71.8075 71.8450 71.7775 71.7925 219512
1 2019-11-13 09:15:00 71.7925 71.8000 71.7800 71.7925  59252
2 2019-11-13 09:30:00 71.7925 71.8125 71.7600 71.7625  57187
3 2019-11-13 09:45:00 71.7600 71.7650 71.7350 71.7425  43048
4 2019-11-13 10:00:00 71.7425 71.7800 71.7425 71.7775  45863
5 2019-11-13 10:15:00 71.7750 71.8225 71.7700 71.8150  42460
6 2019-11-13 10:30:00 71.8150 71.8300 71.7775 71.7800  62403
7 2019-11-13 10:45:00 71.7750 71.7875 71.7475 71.7525  34090
8 2019-11-13 11:00:00 71.7525 71.7825 71.7475 71.7625  39320
9 2019-11-13 11:15:00 71.7625 71.7925 71.7600 71.7875  20190
```

1.  获取`df`中的列列表：

```py
>>> df.columns.tolist()
```

我们得到以下输出：

```py
['date', 'open', 'high', 'low', 'close', 'volume']
```

1.  再次使用`time_series_data`创建一个`DataFrame`对象。这次，按照你想要的顺序指定列：

```py
>>> pandas.DataFrame(time_series_data, 
         columns=['close','date', 'open', 'high', 'low', 'volume'])
```

我们得到以下输出：

```py
    close                date    open    high     low volume
0 71.7925 2019-11-13 09:00:00 71.8075 71.8450 71.7775 219512
1 71.7925 2019-11-13 09:15:00 71.7925 71.8000 71.7800  59252
2 71.7625 2019-11-13 09:30:00 71.7925 71.8125 71.7600  57187
3 71.7425 2019-11-13 09:45:00 71.7600 71.7650 71.7350  43048
4 71.7775 2019-11-13 10:00:00 71.7425 71.7800 71.7425  45863
5 71.8150 2019-11-13 10:15:00 71.7750 71.8225 71.7700  42460
6 71.7800 2019-11-13 10:30:00 71.8150 71.8300 71.7775  62403
7 71.7525 2019-11-13 10:45:00 71.7750 71.7875 71.7475  34090
8 71.7625 2019-11-13 11:00:00 71.7525 71.7825 71.7475  39320
9 71.7875 2019-11-13 11:15:00 71.7625 71.7925 71.7600  20190
```

## 它是如何工作的...

在*步骤 1*中，从`datetime`模块导入`datetime`类和`pandas`包。在*步骤 2*中，创建一个时间序列数据，这通常由第三方 API 返回历史数据。这个数据是一个字典列表，每个字典有相同的键集——`date`、`open`、`high`、`low`、`close`和`volume`。注意`date`键的值是一个`datetime`对象，其他键的值是`float`对象。

在*步骤 3*中，通过直接调用构造函数并将`time_series_data`作为参数来创建一个 pandas `DataFrame`对象，并将返回数据分配给`df`。字典的键成为`df`的列名，值成为数据。在*步骤 4*中，使用`columns`属性和`tolist()`方法将`df`的列作为列表提取出来。您可以验证`time_series_data`中字典的键与列名相同。

在*步骤 5*中，通过向构造函数传递`columns`参数以特定顺序的列来创建一个`DataFrame`，该参数是一个字符串列表。

## 还有更多

当创建一个`DataFrame`对象时，会自动分配一个索引，这是所有行的地址。前面示例中最左边的列是索引列。默认情况下，索引从`0`开始。可以通过向`DataFrame`构造函数传递一个`index`参数以迭代器的形式设置自定义索引。如下所示：

1.  从`time_series_data`创建一个新的 DataFrame 对象，带有自定义索引：

```py
>>> pandas.DataFrame(time_series_data, index=range(10, 20)) 
```

我们得到以下输出：

```py
                  date    open    high     low   close volume
10 2019-11-13 09:00:00 71.8075 71.8450 71.7775 71.7925 219512
11 2019-11-13 09:15:00 71.7925 71.8000 71.7800 71.7925  59252
12 2019-11-13 09:30:00 71.7925 71.8125 71.7600 71.7625  57187
13 2019-11-13 09:45:00 71.7600 71.7650 71.7350 71.7425  43048
14 2019-11-13 10:00:00 71.7425 71.7800 71.7425 71.7775  45863
15 2019-11-13 10:15:00 71.7750 71.8225 71.7700 71.8150  42460
16 2019-11-13 10:30:00 71.8150 71.8300 71.7775 71.7800  62403
17 2019-11-13 10:45:00 71.7750 71.7875 71.7475 71.7525  34090
18 2019-11-13 11:00:00 71.7525 71.7825 71.7475 71.7625  39320
19 2019-11-13 11:15:00 71.7625 71.7925 71.7600 71.7875  20190
```

注意输出中的索引从`10`开始到`19`结束。默认索引值应该是从`0`到`9`。

# DataFrame 操作—重命名、重新排列、反转和切片

创建`DataFrame`对象后，你可以对其执行各种操作。本示例涵盖了对`DataFrame`对象进行以下操作。重命名列、重新排列列、反转`DataFrame`，以及对`DataFrame`进行切片以提取行、列和数据子集。

## 准备工作完成

确保`df`对象在你的 Python 命名空间中可用。请参考本章的*创建 pandas.DataFrame 对象*示例来设置该对象。

## 如何执行…

对这个示例执行以下步骤：

1.  将`df`的`date`列重命名为`timestamp`。打印它：

```py
>>> df.rename(columns={'date':'timestamp'}, inplace=True)
>>> df
```

我们得到以下输出：

```py
            timestamp    open    high     low   close volume
0 2019-11-13 09:00:00 71.8075 71.8450 71.7775 71.7925 219512
1 2019-11-13 09:15:00 71.7925 71.8000 71.7800 71.7925  59252
2 2019-11-13 09:30:00 71.7925 71.8125 71.7600 71.7625  57187
3 2019-11-13 09:45:00 71.7600 71.7650 71.7350 71.7425  43048
4 2019-11-13 10:00:00 71.7425 71.7800 71.7425 71.7775  45863
5 2019-11-13 10:15:00 71.7750 71.8225 71.7700 71.8150  42460
6 2019-11-13 10:30:00 71.8150 71.8300 71.7775 71.7800  62403
7 2019-11-13 10:45:00 71.7750 71.7875 71.7475 71.7525  34090
8 2019-11-13 11:00:00 71.7525 71.7825 71.7475 71.7625  39320
9 2019-11-13 11:15:00 71.7625 71.7925 71.7600 71.7875  20190
```

1.  通过重新排列`df`中的列创建一个新的`DataFrame`对象：

```py
>>> df.reindex(columns=[
               'volume', 
               'close', 
               'timestamp', 
               'high', 
               'open', 
               'low'
            ])
```

我们得到以下输出：

```py
  volume   close           timestamp    high    open     low
0 219512 71.7925 2019-11-13 09:00:00 71.8450 71.8075 71.7775
1  59252 71.7925 2019-11-13 09:15:00 71.8000 71.7925 71.7800
2  57187 71.7625 2019-11-13 09:30:00 71.8125 71.7925 71.7600
3  43048 71.7425 2019-11-13 09:45:00 71.7650 71.7600 71.7350
4  45863 71.7775 2019-11-13 10:00:00 71.7800 71.7425 71.7425
5  42460 71.8150 2019-11-13 10:15:00 71.8225 71.7750 71.7700
6  62403 71.7800 2019-11-13 10:30:00 71.8300 71.8150 71.7775
7  34090 71.7525 2019-11-13 10:45:00 71.7875 71.7750 71.7475
8  39320 71.7625 2019-11-13 11:00:00 71.7825 71.7525 71.7475
9  20190 71.7875 2019-11-13 11:15:00 71.7925 71.7625 71.7600
```

1.  通过反转`df`中的行创建一个新的`DataFrame`对象：

```py
>>> df[::-1]
```

我们得到以下输出：

```py
            timestamp    open    high     low   close volume
9 2019-11-13 11:15:00 71.7625 71.7925 71.7600 71.7875  20190
8 2019-11-13 11:00:00 71.7525 71.7825 71.7475 71.7625  39320
7 2019-11-13 10:45:00 71.7750 71.7875 71.7475 71.7525  34090
6 2019-11-13 10:30:00 71.8150 71.8300 71.7775 71.7800  62403
5 2019-11-13 10:15:00 71.7750 71.8225 71.7700 71.8150  42460
4 2019-11-13 10:00:00 71.7425 71.7800 71.7425 71.7775  45863
3 2019-11-13 09:45:00 71.7600 71.7650 71.7350 71.7425  43048
2 2019-11-13 09:30:00 71.7925 71.8125 71.7600 71.7625  57187
1 2019-11-13 09:15:00 71.7925 71.8000 71.7800 71.7925  59252
0 2019-11-13 09:00:00 71.8075 71.8450 71.7775 71.7925 219512
```

1.  从`df`中提取`close`列：

```py
>>> df['close']
```

我们得到以下输出：

```py
0    71.7925
1    71.7925
2    71.7625
3    71.7425
4    71.7775
5    71.8150
6    71.7800
7    71.7525
8    71.7625
9    71.7875
Name: close, dtype: float64
```

1.  从`df`中提取第一行：

```py
>>> df.iloc[0]
```

我们得到以下输出：

```py
timestamp    2019-11-13 09:00:00
open                     71.8075
high                      71.845
low                      71.7775
close                    71.7925
volume                    219512
Name: 10, dtype: object
```

1.  提取一个*2 × 2*矩阵，只包括前两行和前两列：

```py
>>> df.iloc[:2, :2]
```

我们得到以下输出：

```py
            timestamp    open
0 2019-11-13 09:00:00 71.8075
1 2019-11-13 09:15:00 71.7925
```

## 它的工作原理...

**重命名**：在*步骤 1* 中，你使用 pandas 的 DataFrame 的`rename()`方法将`date`列重命名为`timestamp`。你通过将`columns`参数作为一个字典传递，其中要替换的现有名称作为键，其新名称作为相应的值。你还将`inplace`参数传递为`True`，以便直接修改`df`。如果不传递，其默认值为`False`，意味着将创建一个新的`DataFrame`而不是修改`df`。

**重新排列**：在*步骤 2* 中，你使用`reindex()`方法从`df`创建一个新的`DataFrame`，重新排列其列。你通过传递`columns`参数以字符串列表的形式传递所需的顺序的列名。

**反转**：在*步骤 3* 中，你通过以一种特殊的方式使用索引运算符`[::-1]`从`df`创建一个新的`DataFrame`，其中的行被反转。这类似于我们反转常规的 Python 列表的方式。

**切片**：在*步骤 4* 中，你使用`df`上的索引运算符提取列`close`。你在这里传递列名`close`作为索引。返回的数据是一个`pandas.Series`对象。你可以在 DataFrame 对象上使用`iloc`属性来提取行、列或子集 DataFrame 对象。在*步骤 5* 中，你使用`iloc`提取第一行，并使用`0`作为索引。返回的数据是一个`pandas.Series`对象。在*步骤 6* 中，你使用`iloc`提取从`df`中的`(:2, :2)`开始的 2x2 子集。这意味着提取直到索引 2（即 0 和 1）的所有行和直到索引 2（再次是 0 和 1）的所有列的数据。返回的数据是一个`pandas.DataFrame`对象。

在此示例中显示的所有操作中，返回一个新的`DataFrame`对象的地方，原始的`DataFrame`对象保持不变。

## 还有更多

`.iloc()`属性也可以用于从`DataFrame`中提取列。以下代码展示了这一点。

从`df`中提取第四列。观察输出：

```py
>>> df.iloc[:, 4]
```

我们得到以下输出：

```py
0    71.7925
1    71.7925
2    71.7625
3    71.7425
4    71.7775
5    71.8150
6    71.7800
7    71.7525
8    71.7625
9    71.7875
Name: close, dtype: float64
```

注意，此输出和 *步骤 4* 的输出相同。

# DataFrame 操作 — 应用、排序、迭代和连接

在上一个食谱的基础上，本食谱演示了可以对 `DataFrame` 对象执行的更多操作：对列中的所有元素应用函数、基于列进行排序、迭代行以及垂直和水平连接多个 `DataFrame` 对象。

## 准备工作

在尝试此食谱之前，请确保您已经完成了上一个食谱。确保您的 Python 命名空间中有来自上一个食谱的 `df`。

## 如何做...

为此食谱执行以下步骤：

1.  导入必要的模块

```py
>>> import random
>>> import pandas
```

1.  使用不同的日期和时间格式 `DD-MM-YYYY HH:MM:SS` 修改 `df` 的时间戳列中的值：

```py
>>> df['timestamp'] = df['timestamp'].apply(
                        lambda x: x.strftime("%d-%m-%Y %H:%M:%S"))
>>> df
```

我们得到以下输出：

```py
            timestamp    open    high     low   close volume
0 13-11-2019 09:00:00 71.8075 71.8450 71.7775 71.7925 219512
1 13-11-2019 09:15:00 71.7925 71.8000 71.7800 71.7925  59252
2 13-11-2019 09:30:00 71.7925 71.8125 71.7600 71.7625  57187
3 13-11-2019 09:45:00 71.7600 71.7650 71.7350 71.7425  43048
4 13-11-2019 10:00:00 71.7425 71.7800 71.7425 71.7775  45863
5 13-11-2019 10:15:00 71.7750 71.8225 71.7700 71.8150  42460
6 13-11-2019 10:30:00 71.8150 71.8300 71.7775 71.7800  62403
7 13-11-2019 10:45:00 71.7750 71.7875 71.7475 71.7525  34090
8 13-11-2019 11:00:00 71.7525 71.7825 71.7475 71.7625  39320
9 13-11-2019 11:15:00 71.7625 71.7925 71.7600 71.7875  20190
```

1.  创建一个按照 `close` 列升序排列的新的 `DataFrame` 对象：

```py
>>> df.sort_values(by='close', ascending=True)
```

我们得到以下输出：

```py
            timestamp    open    high     low   close volume
3 13-11-2019 09:45:00 71.7600 71.7650 71.7350 71.7425  43048
7 13-11-2019 10:45:00 71.7750 71.7875 71.7475 71.7525  34090
2 13-11-2019 09:30:00 71.7925 71.8125 71.7600 71.7625  57187
8 13-11-2019 11:00:00 71.7525 71.7825 71.7475 71.7625  39320
4 13-11-2019 10:00:00 71.7425 71.7800 71.7425 71.7775  45863
6 13-11-2019 10:30:00 71.8150 71.8300 71.7775 71.7800  62403
9 13-11-2019 11:15:00 71.7625 71.7925 71.7600 71.7875  20190
0 13-11-2019 09:00:00 71.8075 71.8450 71.7775 71.7925 219512
1 13-11-2019 09:15:00 71.7925 71.8000 71.7800 71.7925  59252
5 13-11-2019 10:15:00 71.7750 71.8225 71.7700 71.8150  42460
```

1.  创建一个按照 `open` 列降序排列的新的 `DataFrame` 对象：

```py
>>> df.sort_values(by='open', ascending=False)
```

我们得到以下输出：

```py
            timestamp    open    high     low   close volume
6 13-11-2019 10:30:00 71.8150 71.8300 71.7775 71.7800  62403
0 13-11-2019 09:00:00 71.8075 71.8450 71.7775 71.7925 219512
2 13-11-2019 09:30:00 71.7925 71.8125 71.7600 71.7625  57187
1 13-11-2019 09:15:00 71.7925 71.8000 71.7800 71.7925  59252
7 13-11-2019 10:45:00 71.7750 71.7875 71.7475 71.7525  34090
5 13-11-2019 10:15:00 71.7750 71.8225 71.7700 71.8150  42460
9 13-11-2019 11:15:00 71.7625 71.7925 71.7600 71.7875  20190
3 13-11-2019 09:45:00 71.7600 71.7650 71.7350 71.7425  43048
8 13-11-2019 11:00:00 71.7525 71.7825 71.7475 71.7625  39320
4 13-11-2019 10:00:00 71.7425 71.7800 71.7425 71.7775  45863
```

1.  遍历 `df` 以找到每行的 `open`、`close`、`high` 和 `low` 的平均值：

```py
>>> for _, row in df.iterrows():
       avg = (row['open'] + row['close'] + row['high'] + 
              row['low'])/4
       print(f"Index: {_} | Average: {avg}")
```

我们得到以下输出：

```py
Index: 0 | Average: 71.805625
Index: 1 | Average: 71.79124999999999
Index: 2 | Average: 71.781875
Index: 3 | Average: 71.750625
Index: 4 | Average: 71.760625
Index: 5 | Average: 71.795625
Index: 6 | Average: 71.800625
Index: 7 | Average: 71.765625
Index: 8 | Average: 71.76124999999999
Index: 9 | Average: 71.775625
```

1.  逐列迭代 `df` 的第一行的所有值：

```py
>>> for value in df.iloc[0]:
        print(value)
```

我们得到以下输出：

```py
13-11-2019 09:00:00
71.8075
71.845
71.7775
71.7925
219512
```

1.  创建一个样本时间序列数据作为字典对象的列表。将其分配给 `df_new`：

```py
>>> df_new = pandas. DataFrame([
    {'timestamp': datetime.datetime(2019, 11, 13, 11, 30),
     'open': 71.7875,
     'high': 71.8075,
     'low': 71.77,
     'close': 71.7925,
     'volume': 18655},
    {'timestamp': datetime.datetime(2019, 11, 13, 11, 45),
     'open': 71.7925,
     'high': 71.805,
     'low': 71.7625,
     'close': 71.7625,
     'volume': 25648},
    {'timestamp': datetime.datetime(2019, 11, 13, 12, 0),
     'open': 71.7625,
     'high': 71.805,
     'low': 71.75,
     'close': 71.785,
     'volume': 37300},
    {'timestamp': datetime.datetime(2019, 11, 13, 12, 15),
     'open': 71.785,
     'high': 71.7925,
     'low': 71.7575,
     'close': 71.7775,
     'volume': 15431},
    {'timestamp': datetime.datetime(2019, 11, 13, 12, 30),
     'open': 71.7775,
     'high': 71.795,
     'low': 71.7725,
     'close': 71.79,
     'volume': 5178}])
>>> df_new
```

我们得到以下输出：

```py
            timestamp    open    high     low   close volume
0 2019-11-13 11:30:00 71.7875 71.8075 71.7700 71.7925  18655
1 2019-11-13 11:45:00 71.7925 71.8050 71.7625 71.7625  25648
2 2019-11-13 12:00:00 71.7625 71.8050 71.7500 71.7850  37300
3 2019-11-13 12:15:00 71.7850 71.7925 71.7575 71.7775  15431
4 2019-11-13 12:30:00 71.7775 71.7950 71.7725 71.7900   5178
```

1.  通过垂直连接 `df` 和 `df_new` 创建一个新的 DataFrame：

```py
>>> pandas.concat([df, df_new]).reset_index(drop=True)
```

我们得到以下输出：

```py
             timestamp    open    high     low   close volume
0  13-11-2019 09:00:00 71.8075 71.8450 71.7775 71.7925 219512
1  13-11-2019 09:15:00 71.7925 71.8000 71.7800 71.7925  59252
2  13-11-2019 09:30:00 71.7925 71.8125 71.7600 71.7625  57187
3  13-11-2019 09:45:00 71.7600 71.7650 71.7350 71.7425  43048
4  13-11-2019 10:00:00 71.7425 71.7800 71.7425 71.7775  45863
5  13-11-2019 10:15:00 71.7750 71.8225 71.7700 71.8150  42460
6  13-11-2019 10:30:00 71.8150 71.8300 71.7775 71.7800  62403
7  13-11-2019 10:45:00 71.7750 71.7875 71.7475 71.7525  34090
8  13-11-2019 11:00:00 71.7525 71.7825 71.7475 71.7625  39320
9  13-11-2019 11:15:00 71.7625 71.7925 71.7600 71.7875  20190
10 2019-11-13 11:30:00 71.7875 71.8075 71.7700 71.7925  18655
11 2019-11-13 11:45:00 71.7925 71.8050 71.7625 71.7625  25648
12 2019-11-13 12:00:00 71.7625 71.8050 71.7500 71.7850  37300
13 2019-11-13 12:15:00 71.7850 71.7925 71.7575 71.7775  15431
14 2019-11-13 12:30:00 71.7775 71.7950 71.7725 71.7900   5178
```

## 它是如何工作的...

在 *步骤 1* 中，您导入 `pandas` 包。

**应用**：在 *步骤 2* 中，您通过使用 `apply` 方法修改 `df` 的 `timestamp` 列中的所有值。此方法接受要应用的函数作为输入。您在此处传递一个期望一个 `datetime` 对象作为单个输入的 lambda 函数，并使用 `strftime()` 将其转换为所需格式的字符串。（有关 `strftime()` 的更多详细信息，请参阅 *将 datetime 对象转换为字符串* 食谱）。`apply` 方法调用在 `df` 的 `timestamp` 列上，这是一个 `pandas.Series` 对象。lambda 函数应用于列中的每个值。此调用返回一个新的 `pandas.Series` 对象，您将其重新分配给 `df` 的 `timestamp` 列。注意，之后，`df` 的 `timestamp` 列保存的是字符串对象，而不是之前的 `datetime` 对象。

**排序**：在 *步骤 3* 中，您通过按照 `df` 的 `close` 列升序排列来创建一个新的 `DataFrame` 对象。您使用 `sort_values()` 方法来执行排序。类似地，在 *步骤 4* 中，您通过按照 `df` 的 `open` 列降序排列来创建一个新的 `DataFrame` 对象。

**迭代**：在*步骤 5*中，您使用`iterrows()`方法迭代`df`以找到并打印出每行的`open`、`close`、`high`和`low`的平均值。`iterrows()`方法将每行作为一个（`index, pandas.Series`）对进行迭代。在*步骤 6*中，您使用`df.iloc[0]`迭代`df`的第一行的所有值。您将第一行的`timestamp`、`open`、`high`、`low`、`close`和`volume`列值作为输出。

**连接**：在*步骤 6*中，您创建了一个新的`DataFrame`，类似于*创建 pandas.DataFrame 对象*配方中创建的那个，并将其赋值给`df_new`。您使用`pandas.concat()`函数通过垂直连接`dt`和`df_new`来创建一个新的`DataFrame`。这意味着将创建一个新的`DataFrame`，其中`df_new`的行附加在`df`的行下面。您将包含`df`和`df_new`的列表作为参数传递给`pandas.concat()`函数。另外，为了创建一个从`0`开始的新索引，您使用了`reset_index()`方法，并将参数 drop 传递为`True`。如果您不使用`reset_index()`，则连接的`DataFrame`的索引会看起来像这样—`0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4`。（有关`DataFrame`索引的更多信息，请参阅*创建 pandas.DataFrame 对象*配方。）

## 还有更多

您也可以使用`pandas.concat()`函数将两个`DataFrame`对象水平连接在一起，即列方向上，通过将`axis`参数传递给`pandas.concat()`方法一个值为`1`。这在以下步骤中显示：

1.  从 Python 标准库导入`random`模块：

```py
>>> import random
```

1.  使用一个单列`open`和随机值创建一个`DataFrame`对象。将其赋值给`df1`并打印出来：

```py
>>> df1 = pandas.DataFrame([random.randint(1,100) for i in 
                            range(10)], columns=['open'])
>>> df1
```

我们得到以下输出。您的输出可能会有所不同：

```py
   open
0    99
1    73
2    16
3    53
4    47
5    74
6    21
7    22
8     2
9    30
```

1.  使用一个单列`close`和随机值创建另一个`DataFrame`对象。将其赋值给`df2`并打印出来：

```py
>>> df2 = pandas.DataFrame([random.randint(1,100) for i in 
                            range(10)], columns=['close'])
>>> df2
```

我们得到以下输出：

```py
   close
0     63
1     84
2     44
3     56
4     25
5      1
6     41
7     55
8     93
9     82
```

1.  通过水平连接`df1`和`df2`创建一个新的`DataFrame`

```py
>>> pandas.concat([df1, df2], axis=1)
```

我们得到以下输出。您的输出可能会有所不同：

```py
    open  close
0     99     93
1     73     42
2     16     57
3     53     56
4     47     25
5     74      1
6     21     41
7     22     55
8      2     93
9     30     82
```

# 将 DataFrame 转换为其他格式

本配方演示了将`DataFrame`对象转换为其他格式，如`.csv`文件、`json`对象和`pickle`对象。将其转换为`.csv`文件可以使进一步使用电子表格应用程序处理数据变得更加容易。`json`格式对于通过网络 API 传输`DataFrame`对象非常有用。`pickle`格式对于通过套接字将一个 Python 会话中创建的`DataFrame`对象传输到另一个 Python 会话中而无需重新创建它们非常有用。

## 准备工作

确保在您的 Python 命名空间中可用对象`df`。请参阅本章的*创建 pandas.DataFrame 对象*配方来设置此对象。

## 如何做...

执行此配方的以下步骤：

1.  将`df`转换并保存为 CSV 文件：

```py
>>> df.to_csv('dataframe.csv', index=False)
```

1.  将`df`转换为 JSON 字符串：

```py
>>> df.to_json()
```

我们得到以下输出：

```py
'{
    "timestamp":{
        "0":"13-11-2019 09:00:00","1":"13-11-2019 09:15:00",
        "2":"13-11-2019 09:30:00","3":"13-11-2019 09:45:00",
        "4":"13-11-2019 10:00:00","5":"13-11-2019 10:15:00",
        "6":"13-11-2019 10:30:00","7":"13-11-2019 10:45:00",
        "8":"13-11-2019 11:00:00","9":"13-11-2019 11:15:00"},
    "open":{
        "0":71.8075,"1":71.7925,"2":71.7925, "3":71.76,         
        "4":71.7425,"5":71.775,"6":71.815, "7":71.775,
        "8":71.7525,"9":71.7625},
    "high"{
        "0":71.845,"1":71.8,"2":71.8125,"3":71.765,
        "4":71.78,"5":71.8225,"6":71.83,"7":71.7875,
        "8":71.7825,"9":71.7925},
    "low":{
        "0":71.7775,"1":71.78,"2":71.76,"3":71.735,
        "4":71.7425,"5":71.77,"6":71.7775,"7":71.7475,
        "8":71.7475,"9":71.76},
    "close":{
        "0":71.7925,"1":71.7925,"2":71.7625,"3":71.7425,
        "4":71.7775,"5":71.815,"6":71.78,"7":71.7525,
        "8":71.7625,"9":71.7875},
    "volume":{
        "0":219512,"1":59252,"2":57187,"3":43048,
        "4":45863,"5":42460,"6":62403,"7":34090,
        "8":39320,"9":20190}}'
```

1.  将`df`保存为一个文件：

```py
>>> df.to_pickle('df.pickle')
```

## 工作原理...

在 *步骤 1* 中，你使用 `to_csv()` 方法将 `df` 保存为 `.csv` 文件。你将 `dataframe.csv`，一个生成 `.csv` 文件的文件路径，作为第一个参数传递，将索引设置为 `False` 作为第二个参数。将索引设置为 `False` 可以防止索引被转储到 `.csv` 文件中。如果你想将 `DataFrame` 与其索引一起保存，可以将索引设置为 `True` 传递给 `to_csv()` 方法。

在 *步骤 2* 中，你使用 `to_json()` 方法将 `df` 转换为 JSON 字符串。你没有向 `to_json()` 方法传递任何额外的参数。

在 *步骤 3* 中，你使用 `to_pickle()` 方法对对象进行 pickle（序列化）。同样，你没有向 `to_pickle()` 方法传递任何额外的参数。

方法 `to_csv()`、`to_json()` 和 `to_pickle()` 可以接受比本示例中显示的更多可选参数。有关这些方法的完整信息，请参阅官方文档：

+   `to_csv()`: [`pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html)

+   `to_json()`: [`pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html)

+   `to_pickle()`: [`pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_pickle.html`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_pickle.html)

# 从其他格式创建 DataFrame

在这个示例中，你将从其他格式（如 `.csv` 文件、`.json` 字符串和 `pickle` 文件）创建 `DataFrame` 对象。使用电子表格应用程序创建的 `.csv` 文件、通过 web API 接收的有效 JSON 数据或通过套接字接收的有效 pickle 对象都可以通过将它们转换为 `DataFrame` 对象来进一步处理。

从不受信任的来源加载 pickled 数据可能是不安全的。请谨慎使用 `read_pickle()`。你可以在这里找到更多详细信息：[`docs.python.org/3/library/pickle.html`](https://docs.python.org/3/library/pickle.html)。如果你在之前的示例中使用此函数的 pickle 文件，那么使用 `read_pickle()` 是完全安全的。

## 准备工作

在开始此示例之前，请确保你已经按照上一个示例的步骤进行了操作。

## 如何操作…

执行以下步骤来完成这个示例：

1.  通过读取 CSV 文件创建一个 DataFrame 对象：

```py
>>> pandas.read_csv('dataframe.csv')
```

我们得到以下输出：

```py
            timestamp    open    high     low   close volume
0 2019-11-13 09:00:00 71.8075 71.8450 71.7775 71.7925 219512
1 2019-11-13 09:15:00 71.7925 71.8000 71.7800 71.7925  59252
2 2019-11-13 09:30:00 71.7925 71.8125 71.7600 71.7625  57187
3 2019-11-13 09:45:00 71.7600 71.7650 71.7350 71.7425  43048
4 2019-11-13 10:00:00 71.7425 71.7800 71.7425 71.7775  45863
5 2019-11-13 10:15:00 71.7750 71.8225 71.7700 71.8150  42460
6 2019-11-13 10:30:00 71.8150 71.8300 71.7775 71.7800  62403
7 2019-11-13 10:45:00 71.7750 71.7875 71.7475 71.7525  34090
8 2019-11-13 11:00:00 71.7525 71.7825 71.7475 71.7625  39320
9 2019-11-13 11:15:00 71.7625 71.7925 71.7600 71.7875  20190
```

1.  通过读取 JSON 字符串创建一个 DataFrame 对象：

```py
>>> pandas.read_json("""{
        "timestamp": {
            "0":"13-11-2019 09:00:00", "1":"13-11-2019 09:15:00", 
            "2":"13-11-2019 09:30:00","3":"13-11-2019 09:45:00", 
            "4":"13-11-2019 10:00:00","5":"13-11-2019 10:15:00",
            "6":"13-11-2019 10:30:00","7":"13-11-2019 10:45:00",
            "8":"13-11-2019 11:00:00","9":"13-11-2019 11:15:00"},

        "open":{
            "0":71.8075,"1":71.7925,"2":71.7925,"3":71.76,
            "4":71.7425,"5":71.775,"6":71.815,"7":71.775,
            "8":71.7525,"9":71.7625},

        "high":{
            "0":71.845,"1":71.8,"2":71.8125,"3":71.765,"4":71.78,
            "5":71.8225,"6":71.83,"7":71.7875,"8":71.7825,
            "9":71.7925},

        "low":{
            "0":71.7775,"1":71.78,"2":71.76,"3":71.735,"4":71.7425,
            "5":71.77,"6":71.7775,"7":71.7475,"8":71.7475,
            "9":71.76},

        "close":{
            "0":71.7925,"1":71.7925,"2":71.7625,"3":71.7425,
            "4":71.7775,"5":71.815,"6":71.78,"7":71.7525,
            "8":71.7625,"9":71.7875},

        "volume":{
            "0":219512,"1":59252,"2":57187,"3":43048,"4":45863,
            "5":42460,"6":62403,"7":34090,"8":39320,"9":20190}}
            """)
```

我们得到以下输出：

```py
            timestamp    open    high     low   close volume
0 2019-11-13 09:00:00 71.8075 71.8450 71.7775 71.7925 219512
1 2019-11-13 09:15:00 71.7925 71.8000 71.7800 71.7925  59252
2 2019-11-13 09:30:00 71.7925 71.8125 71.7600 71.7625  57187
3 2019-11-13 09:45:00 71.7600 71.7650 71.7350 71.7425  43048
4 2019-11-13 10:00:00 71.7425 71.7800 71.7425 71.7775  45863
5 2019-11-13 10:15:00 71.7750 71.8225 71.7700 71.8150  42460
6 2019-11-13 10:30:00 71.8150 71.8300 71.7775 71.7800  62403
7 2019-11-13 10:45:00 71.7750 71.7875 71.7475 71.7525  34090
8 2019-11-13 11:00:00 71.7525 71.7825 71.7475 71.7625  39320
9 2019-11-13 11:15:00 71.7625 71.7925 71.7600 71.7875  20190
```

1.  通过取消反序列化 `df.pickle` 文件来创建一个 `DataFrame` 对象：

```py
>>> pandas.read_pickle('df.pickle')
```

我们得到以下输出：

```py
            timestamp    open    high     low   close volume
0 2019-11-13 09:00:00 71.8075 71.8450 71.7775 71.7925 219512
1 2019-11-13 09:15:00 71.7925 71.8000 71.7800 71.7925  59252
2 2019-11-13 09:30:00 71.7925 71.8125 71.7600 71.7625  57187
3 2019-11-13 09:45:00 71.7600 71.7650 71.7350 71.7425  43048
4 2019-11-13 10:00:00 71.7425 71.7800 71.7425 71.7775  45863
5 2019-11-13 10:15:00 71.7750 71.8225 71.7700 71.8150  42460
6 2019-11-13 10:30:00 71.8150 71.8300 71.7775 71.7800  62403
7 2019-11-13 10:45:00 71.7750 71.7875 71.7475 71.7525  34090
8 2019-11-13 11:00:00 71.7525 71.7825 71.7475 71.7625  39320
9 2019-11-13 11:15:00 71.7625 71.7925 71.7600 71.7875  20190
```

## 工作原理如下...

在 *步骤 1* 中，你使用 `pandas.read_csv()` 函数从 `.csv` 文件创建一个 DataFrame 对象。你将 `dataframe.csv`，即 `.csv` 文件应该读取的文件路径，作为参数传递。回想一下，在前一个示例的 *步骤 1* 中创建了 `dataframe.csv`。

在*第 2 步*中，你使用`pandas.read_json()`函数从有效的 JSON 字符串创建一个`DataFrame`对象。你将前一个示例中*第 2 步*的输出的 JSON 字符串作为此函数的参数传递。

在*第 3 步*中，你使用`pandas.read_pickle()`方法从`pickle`文件创建一个`DataFrame`对象。你将`df.pickle`，即 pickle 文件应该读取的文件路径，作为此函数的参数传递。回忆一下，你在前一个示例的*第 3 步*中创建了`df.pickle`。

如果你遵循了前一个示例，那么所有三个步骤的输出都将是相同的`DataFrame`对象。这与前一个示例中的`df`完全相同。

方法`read_csv()`、`read_json()`和`read_pickle()`可以接受比本示例中显示的更多的可选参数。请参考官方文档以获取有关这些方法的完整信息。

+   `read_csv()`: [`pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv)

+   `read_json()`: [`pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html#pandas.read_json`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html#pandas.read_json)

+   `read_pickle()`: [`pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_pickle.html#pandas.read_pickle`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_pickle.html#pandas.read_pickle)
