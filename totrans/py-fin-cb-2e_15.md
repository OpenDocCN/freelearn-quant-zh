# 15

# 金融中的深度学习

近年来，我们看到深度学习技术在许多领域取得了惊人的成功。深度神经网络成功应用于传统机器学习算法无法成功的任务——大规模图像分类、自动驾驶、以及在围棋或经典视频游戏（从超级马里奥到星际争霸II）中超越人类的表现。几乎每年，我们都能看到一种新型网络的推出，它在某些方面打破了性能记录并取得了最先进的（SOTA）成果。

随着**图形处理单元**（**GPU**）的持续改进，涉及CPU/GPU的免费计算资源（如Google Colab、Kaggle等）的出现，以及各种框架的快速发展，深度学习在研究人员和从业者中越来越受到关注，他们希望将这些技术应用到自己的业务案例中。

本章将展示深度学习在金融领域的两个可能应用场景——预测信用卡违约（分类任务）和时间序列预测。深度学习在处理语音、音频和视频等序列数据时表现出色。这也是它自然适用于处理时间序列数据（包括单变量和多变量）的原因。金融时间序列通常表现得非常不稳定且复杂，这也是建模它们的挑战所在。深度学习方法特别适合这一任务，因为它们不对底层数据的分布做任何假设，并且能够对噪声具有较强的鲁棒性。

在本书的第一版中，我们重点介绍了用于时间序列预测的传统神经网络架构（CNN、RNN、LSTM和GRU）及其在PyTorch中的实现。在本书中，我们将使用更复杂的架构，并借助专用的Python库来实现。得益于这些库，我们不必重新创建神经网络的逻辑，可以专注于预测挑战。

在本章中，我们介绍以下几种方法：

+   探索fastai的Tabular Learner

+   探索Google的TabNet

+   使用亚马逊DeepAR进行时间序列预测

+   使用NeuralProphet进行时间序列预测

# 探索fastai的Tabular Learner

深度学习通常不与表格或结构化数据联系在一起，因为这类数据涉及一些可能的问题：

+   我们应该如何以神经网络能够理解的方式表示特征？在表格数据中，我们通常处理数值型和类别型特征，因此我们需要正确表示这两种类型的输入。

+   我们如何使用特征交互，包括特征之间以及与目标之间的交互？

+   我们如何有效地对数据进行采样？表格数据集通常比用于计算机视觉或NLP问题的典型数据集要小。没有简单的方法可以应用数据增强，例如图像的随机裁剪或旋转。此外，也没有通用的大型数据集具备一些普适的属性，我们可以基于这些属性轻松地应用迁移学习。

+   我们如何解释神经网络的预测结果？

这就是为什么实践者倾向于使用传统的机器学习方法（通常基于某种梯度提升树）来处理涉及结构化数据的任务。然而，使用深度学习处理结构化数据的一个潜在优势是，它需要的特征工程和领域知识要少得多。

在这个食谱中，我们展示了如何成功地使用深度学习处理表格数据。为此，我们使用了流行的`fastai`库，该库建立在PyTorch之上。

使用`fastai`库的一些优点包括：

+   它提供了一些API，可以大大简化与**人工神经网络**（**ANNs**）的工作——从加载和批处理数据到训练模型。

+   它结合了经实验证明有效的最佳方法，用于深度学习处理各种任务，如图像分类、自然语言处理（NLP）和表格数据（包括分类和回归问题）。

+   它自动处理数据预处理——我们只需定义要应用的操作。

`fastai`的一个亮点是使用**实体嵌入**（或嵌入层）处理分类数据。通过使用它，模型可以学习分类特征观测值之间潜在的有意义的关系。你可以把嵌入看作是潜在特征。对于每个分类列，都会有一个可训练的嵌入矩阵，每个唯一值都被映射到一个指定的向量。幸运的是，`fastai`为我们完成了所有这些工作。

使用实体嵌入有很多优点。首先，它减少了内存使用，并且比使用独热编码加速了神经网络的训练。其次，它将相似的值映射到嵌入空间中彼此接近的位置，这揭示了分类变量的内在特性。第三，这种技术对于具有许多高基数特征的数据集尤其有用，而其他方法往往会导致过拟合。

在这个食谱中，我们将深度学习应用于一个基于信用卡违约数据集的分类问题。我们已经在*第13章*，*应用机器学习：识别信用卡违约*中使用过这个数据集。

## 如何做…

执行以下步骤来训练一个神经网络，用于分类违约客户：

1.  导入库：

    ```py
    from fastai.tabular.all import *
    from sklearn.model_selection import train_test_split
    from chapter_15_utils import performance_evaluation_report_fastai
    import pandas as pd 
    ```

1.  从CSV文件加载数据集：

    ```py
    df = pd.read_csv("../Datasets/credit_card_default.csv",
                     na_values="") 
    ```

1.  定义目标、分类/数值特征列表和预处理步骤：

    ```py
    TARGET = "default_payment_next_month"
    cat_features = list(df.select_dtypes("object").columns)
    num_features = list(df.select_dtypes("number").columns)
    num_features.remove(TARGET)
    preprocessing = [FillMissing, Categorify, Normalize] 
    ```

1.  定义用于创建训练集和验证集的分割器：

    ```py
    splits = RandomSplitter(valid_pct=0.2, seed=42)(range_of(df))
    splits 
    ```

    执行这个代码片段会生成以下数据集预览：

    ```py
    ((#24000) [27362,16258,19716,9066,1258,23042,18939,24443,4328,4976...],
     (#6000) [7542,10109,19114,5209,9270,15555,12970,10207,13694,1745...]) 
    ```

1.  创建`TabularPandas`数据集：

    ```py
    tabular_df = TabularPandas(
        df,
        procs=preprocessing,
        cat_names=cat_features,
        cont_names=num_features,
        y_names=TARGET,
        y_block=CategoryBlock(),
        splits=splits
    )
    PREVIEW_COLS = ["sex", "education", "marriage",
                    "payment_status_sep", "age_na", "limit_bal",
                    "age", "bill_statement_sep"]
    tabular_df.xs.iloc[:5][PREVIEW_COLS] 
    ```

    执行代码片段会生成以下数据集的预览：

    ![](../Images/B18112_15_01.png)

    图15.1：编码数据集的预览

    我们仅打印了少量列，以保持数据框的可读性。我们可以观察到以下内容：

    +   类别列使用标签编码器进行编码

    +   连续列已经被标准化

    +   含有缺失值的连续列（如`age`）有一个额外的列，表示该特定值在插补前是否缺失。

1.  从`TabularPandas`数据集中定义`DataLoaders`对象：

    ```py
    data_loader = tabular_df.dataloaders(bs=64, drop_last=True)
    data_loader.show_batch() 
    ```

    执行代码片段会生成以下批次的预览：

    ![](../Images/B18112_15_02.png)

    图15.2：`DataLoaders`对象中一个批次的预览

    正如我们在*图15.2*中看到的，这里的特征处于其原始表示形式。

1.  定义所选择的指标和表格学习器：

    ```py
    recall = Recall()
    precision = Precision()
    learn = tabular_learner(
        data_loader,
        [500, 200],
        metrics=[accuracy, recall, precision]
    )
    learn.model 
    ```

    执行代码片段会打印出模型的架构：

    ```py
    TabularModel(
      (embeds): ModuleList(
        (0): Embedding(3, 3)
        (1): Embedding(5, 4)
        (2): Embedding(4, 3)
        (3): Embedding(11, 6)
        (4): Embedding(11, 6)
        (5): Embedding(11, 6)
        (6): Embedding(11, 6)
        (7): Embedding(10, 6)
        (8): Embedding(10, 6)
        (9): Embedding(3, 3)
      )
      (emb_drop): Dropout(p=0.0, inplace=False)
      (bn_cont): BatchNorm1d(14, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (layers): Sequential(
        (0): LinBnDrop(
          (0): Linear(in_features=63, out_features=500, bias=False)
          (1): ReLU(inplace=True)
          (2): BatchNorm1d(500, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): LinBnDrop(
          (0): Linear(in_features=500, out_features=200, bias=False)
          (1): ReLU(inplace=True)
          (2): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): LinBnDrop(
          (0): Linear(in_features=200, out_features=2, bias=True)
        )
      )
    ) 
    ```

    为了提供对嵌入的解释，`Embedding(11`, `6)`表示创建了一个类别嵌入，输入值为11个，输出潜在特征为6个。

1.  查找建议的学习率：

    ```py
    learn.lr_find() 
    ```

    执行代码片段会生成以下图表：

    ![](../Images/B18112_15_03.png)

    图15.3：我们模型的建议学习率

    它还会打印出以下输出，显示建议学习率的精确值：

    ```py
    SuggestedLRs(valley=0.0010000000474974513) 
    ```

1.  训练表格学习器：

    ```py
    learn.fit(n_epoch=25, lr=1e-3, wd=0.2) 
    ```

    在模型训练过程中，我们可以观察到每个训练周期后性能的更新。以下是代码片段：

    ![](../Images/B18112_15_04.png)

    图15.4：表格学习器训练的前10个周期

    在前10个训练周期中，损失值仍然有些不稳定，随着时间的推移有时上升/下降。评估指标也是如此。

1.  绘制损失值图表：

    ```py
    learn.recorder.plot_loss() 
    ```

    执行代码片段会生成以下图表：

    ![](../Images/B18112_15_05.png)

    图15.5：训练和验证损失随训练时间（批次）的变化

    我们可以观察到验证损失有所平稳，偶尔有些波动。这可能意味着模型对于我们的数据来说有些过于复杂，我们可能需要减少隐藏层的大小。

1.  定义验证集的`DataLoaders`：

    ```py
    valid_data_loader = learn.dls.test_dl(df.loc[list(splits[1])]) 
    ```

1.  在验证集上评估性能：

    ```py
    learn.validate(dl=valid_data_loader) 
    ```

    执行代码片段会生成以下输出：

    ```py
    (#4)[0.424113571643829,0.8248333334922,0.36228482003129,0.66237482117310] 
    ```

    这些是验证集的指标：损失，准确率，召回率和精度。

1.  获取验证集的预测结果：

    ```py
    preds, y_true = learn.get_preds(dl=valid_data_loader) 
    ```

    `y_true`包含验证集中的实际标签。`preds`对象是一个包含预测概率的张量。其内容如下：

    ```py
    tensor([[0.8092, 0.1908],
            [0.9339, 0.0661],
            [0.8631, 0.1369],
            ...,
            [0.9249, 0.0751],
            [0.8556, 0.1444],
            [0.8670, 0.1330]]) 
    ```

    为了获取预测的类别，我们可以使用以下命令：

    ```py
    preds.argmax(dim=-1) 
    ```

1.  检查性能评估指标：

    ```py
    perf = performance_evaluation_report_fastai(
        learn, valid_data_loader, show_plot=True
    ) 
    ```

    执行代码片段会生成以下图表：

![](../Images/B18112_15_06.png)

图15.6：表格学习器在验证集上的性能评估

`perf` 对象是一个字典，包含各种评估指标。由于篇幅原因，我们在这里没有展示它，但我们也可以看到，准确率、精确度和召回率的值与我们在*第 12 步*中看到的相同。

## 它是如何工作的……

在*第 2 步*中，我们使用 `read_csv` 函数将数据集加载到 Python 中。在此过程中，我们指明了哪些符号表示缺失值。

在*第 3 步*中，我们识别了因变量（目标）以及数值特征和类别特征。为此，我们使用了 `select_dtypes` 方法并指定了要提取的数据类型。我们将特征存储在列表中。我们还需要将因变量从包含数值特征的列表中移除。最后，我们创建了一个包含所有要应用于数据的转换的列表。我们选择了以下内容：

+   `FillMissing`：缺失值将根据数据类型进行填充。在类别变量的情况下，缺失值将成为一个独立的类别。在连续特征的情况下，缺失值将使用该特征值的中位数（默认方法）、众数或常量值进行填充。此外，还会添加一个额外的列，标记该值是否缺失。

+   `Categorify`：将类别特征映射为它们的整数表示。

+   `Normalize`：特征值被转换，使其均值为零，方差为单位。这使得训练神经网络变得更容易。

需要注意的是，相同的转换将同时应用于训练集和验证集。为了防止数据泄漏，转换仅基于训练集进行。

在*第 4 步*中，我们定义了用于创建训练集和验证集的分割。我们使用了 `RandomSplitter` 类，它在后台进行分层分割。我们指明了希望按照 80-20 的比例进行数据分割。此外，在实例化分割器之后，我们还需要使用 `range_of` 函数，该函数返回一个包含 DataFrame 所有索引的列表。

在*第 5 步*中，我们创建了一个 `TabularPandas` 数据集。它是对 `pandas` DataFrame 的封装，添加了一些方便的实用工具——它处理所有的预处理和分割。在实例化 `TabularPandas` 类时，我们提供了原始 DataFrame、包含所有预处理步骤的列表、目标和类别/连续特征的名称，以及我们在*第 4 步*中定义的分割器对象。我们还指定了 `y_block=CategoryBlock()`。当我们处理分类问题且目标已被编码为二进制表示（零和一的列）时，必须这样做。否则，它可能会与回归问题混淆。

我们可以轻松地将一个`TabularPandas`对象转换为常规的`pandas` DataFrame。我们可以使用`xs`方法提取特征，使用`ys`方法提取目标。此外，我们可以使用`cats`和`conts`方法分别提取类别特征和连续特征。如果我们直接在`TabularPandas`对象上使用这四个方法中的任何一个，将会提取整个数据集。或者，我们可以使用`train`和`valid`访问器仅提取其中一个数据集。例如，要从名为`tabular_df`的`TabularPandas`对象中提取验证集特征，我们可以使用以下代码：

```py
tabular_df.valid.xs 
```

在*步骤6*中，我们将`TabularPandas`对象转换为`DataLoaders`对象。为此，我们使用了`TabularPandas`数据集的`dataloaders`方法。此外，我们指定了批量大小为64，并要求丢弃最后一个不完整的批量。我们使用`show_batch`方法显示了一个示例批量。

我们本来也可以直接从CSV文件创建一个`DataLoaders`对象，而不是转换一个`pandas` DataFrame。为此，我们可以使用`TabularDataLoaders.from_csv`功能。

在*步骤7*中，我们使用`tabular_learner`定义了学习器。首先，我们实例化了额外的度量标准：精确度和召回率。在使用`fastai`时，度量标准以类的形式表示（类名为大写），我们需要先实例化它们，然后再将其传递给学习器。

然后，我们实例化了学习器。这是我们定义网络架构的地方。我们决定使用一个有两个隐藏层的网络，分别有500个和200个神经元。选择网络架构通常被认为是一门艺术而非科学，可能需要大量的试验和错误。另一种常见的方法是使用之前有人使用过且有效的架构，例如基于学术论文、Kaggle竞赛、博客文章等。至于度量标准，我们希望考虑准确率，以及前面提到的精确度和召回率。

与机器学习一样，防止神经网络过拟合至关重要。我们希望网络能够推广到新的数据。解决过拟合的一些常见技术包括以下几种：

+   **权重衰减（Weight decay）**：每次更新权重时，它们都会乘以一个小于1的因子（通常的经验法则是使用0.01到0.1之间的值）。

+   **丢弃法（Dropout）**：在训练神经网络时，对于每个小批量，某些激活值会被随机丢弃。丢弃法也可以用于类别特征的嵌入向量的连接。

+   **批量归一化（Batch normalization）**：该技术通过确保少数异常输入不会对训练后的网络产生过大的影响，从而减少过拟合。

然后，我们检查了模型的架构。在输出中，我们首先看到了分类嵌入层及其对应的dropout，或者在本例中，缺少dropout。接下来，在`(layers)`部分，我们看到了输入层（63个输入特征和500个输出特征），接着是**ReLU**（**修正线性单元**）激活函数和批量归一化。潜在的dropout在`LinBnDrop`层中控制。对于第二个隐藏层，重复了相同的步骤，最后一个线性层产生了类别概率。

`fastai`使用一个规则来确定嵌入层的大小。这个规则是通过经验选择的，它选择600和1.6乘以某个变量的基数的0.56次方中的较小值。要手动计算嵌入层的大小，可以使用`get_emb_sz`函数。如果没有手动指定大小，`tabular_learner`会在后台自动处理。

在*第8步*中，我们尝试确定“合适的”学习率。`fastai`提供了一个辅助方法`lr_find`，帮助简化这一过程。该方法开始训练网络并逐步提高学习率——从非常低的学习率开始，逐渐增加到非常高的学习率。然后，它会绘制学习率与损失值的关系图，并显示建议的学习率值。我们应该选择一个值，它位于最小值之前，但损失仍然在改善（减少）时的点。

在*第9步*中，我们使用学习器的`fit`方法训练了神经网络。我们将简要描述训练算法。整个训练集被划分为**批次**。对于每个批次，网络用来进行预测，并将预测结果与目标值进行比较，从而计算误差。然后，误差被用来更新网络中的权重。**一个epoch**指的是完整地遍历所有批次，换句话说，就是用整个数据集进行训练。在我们的案例中，我们训练了25个epoch。此外，我们还指定了学习率和权重衰减。在*第10步*中，我们绘制了批次中的训练和验证损失。

不深入细节，默认情况下，`fastai`使用（展平后的）**交叉熵损失函数**（用于分类任务）和**Adam（自适应矩估计）**作为优化器。报告的训练和验证损失来自于损失函数，评估指标（如召回率）在训练过程中并未使用。

在*第11步*中，我们定义了一个验证数据加载器。为了确定验证集的索引，我们从分割器中提取了它们。在下一步中，我们使用学习器对象的`validate`方法评估神经网络在验证集上的表现。作为方法的输入，我们传递了验证数据加载器。

在*第13步*中，我们使用了`get_preds`方法来获取验证集的预测结果。为了从`preds`对象中获取预测，我们需要使用`argmax`方法。

最后，我们使用了稍作修改的辅助函数（在前几章中使用过）来恢复评估指标，如精确度和召回率。

## 还有更多……

`fastai`在表格数据集上的一些显著特点包括：

+   在训练神经网络时使用回调。回调用于在训练循环中的不同时间插入自定义代码/逻辑，例如，在每个epoch开始时或在拟合过程开始时。

+   `fastai`提供了一个辅助函数`add_datepart`，用于从包含日期的列（例如购买日期）中提取各种特征。提取的特征可能包括星期几、月份几号，以及一个表示月/季/年开始或结束的布尔值。

+   我们可以使用拟合的表格学习器的`predict`方法，直接预测源数据框中单行的类别。

+   我们可以使用`fit_one_cycle`方法，代替`fit`方法。该方法采用超收敛策略，其基本思想是通过变化的学习率来训练网络。它从较低值开始，逐渐增加到指定的最大值，再回到较低值。这种方法被认为比选择单一学习率效果更好。

+   由于我们处理的是一个相对较小的数据集和简单的模型，因此可以轻松地在CPU上训练神经网络。`fastai`自然支持使用GPU。有关如何使用GPU的更多信息，请参见`fastai`的文档。

+   使用自定义索引进行训练和验证集的划分。当我们处理类别不平衡等问题时，这个功能特别有用，可以确保训练集和验证集包含相似的类别比例。我们可以将`IndexSplitter`与`scikit-learn`的`StratifiedKFold`结合使用。以下代码片段展示了实现示例：

    ```py
    from sklearn.model_selection import StratifiedKFold
    X = df.copy()
    y = X.pop(TARGET)
    strat_split = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )
    train_ind, test_ind = next(strat_split.split(X, y))
    ind_splits = IndexSplitter(valid_idx=list(test_ind))(range_of(df))
    tabular_df = TabularPandas(
        df,
        procs=preprocessing,
        cat_names=cat_features,
        cont_names=num_features,
        y_names=TARGET,
        y_block=CategoryBlock(),
        splits=ind_splits
    ) 
    ```

## 另见

关于`fastai`的更多信息，我们推荐以下资源：

+   `fastai`课程网站：[https://course.fast.ai/](https://course.fast.ai/)。

+   Howard, J., & Gugger, S. 2020\. *使用fastai和PyTorch进行深度学习编程*。O'Reilly Media。 [https://github.com/fastai/fastbook](https://github.com/fastai/fastbook)。

额外的资源可以在这里找到：

+   Guo, C., & Berkhahn, F. 2016\. *类别变量的实体嵌入*。arXiv预印本arXiv:1604.06737。

+   Ioffe, S., & Szegedy, C. 2015\. *批量归一化：通过减少内部协变量偏移加速深度网络训练*。arXiv预印本arXiv:1502.03167。

+   Krogh, A., & Hertz, J. A. 1991\. “简单的权重衰减可以改善泛化能力。” 见于*神经信息处理系统的进展*：9950-957。

+   Ryan, M. 2020\. *结构化数据的深度学习*。Simon和Schuster。

+   Shwartz-Ziv, R., & Armon, A. 2022\. “表格数据：深度学习并不是你所需要的一切”，*信息融合*，81：84-90。

+   Smith, L. N. 2018\. *一种有纪律的方法来调整神经网络的超参数：第一部分 - 学习率、批量大小、动量和权重衰减*。arXiv预印本arXiv:1803.09820。

+   Smith, L. N., & Topin, N. 2019年5月。超收敛：使用大学习率快速训练神经网络。在*人工智能与机器学习在多领域操作应用中的应用*（1100612）中。国际光学与光子学学会。

+   Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. 2014\. “Dropout: 防止神经网络过拟合的一种简单方法”，*机器学习研究期刊*，15（1）：1929-1958。

# 探索谷歌的TabNet

另一种使用神经网络建模表格数据的方法是谷歌的**TabNet**。由于TabNet是一个复杂的模型，我们不会深入描述它的架构。关于这一点，请参阅原始论文（见*另见*部分）。相反，我们提供一个TabNet主要特性的高层次概述：

+   TabNet使用原始的表格数据，不需要任何预处理。

+   TabNet中使用的优化过程基于梯度下降。

+   TabNet结合了神经网络拟合复杂函数的能力和基于树的算法的特征选择特性。通过使用**顺序注意力**在每个决策步骤中选择特征，TabNet能够专注于仅从最有用的特征中学习。

+   TabNet的架构包含两个关键的构建模块：特征变换器和注意力变换器。前者将特征处理为更有用的表示。后者在下一步中选择最相关的特征进行处理。

+   TabNet还具有另一个有趣的组成部分——输入特征的可学习掩码。该掩码应该是稀疏的，也就是说，它应选择一小部分特征来解决预测任务。与决策树（以及其他基于树的模型）不同，由掩码启用的特征选择允许**软决策**。实际上，这意味着决策可以在更大的值范围内做出，而不是基于单一的阈值。

+   TabNet的特征选择是按实例进行的，即可以为训练数据中的每个观测（行）选择不同的特征。

+   TabNet也非常独特，因为它使用单一的深度学习架构来同时进行特征选择和推理。

+   与绝大多数深度学习模型不同，TabNet是可解释的（在某种程度上）。所有的设计选择使TabNet能够提供局部和全局的可解释性。局部可解释性让我们能够可视化特征的重要性，并了解它们是如何为单行数据组合的。全局可解释性则提供了每个特征对训练模型的贡献的汇总度量（基于整个数据集）。

在这个例子中，我们展示了如何将TabNet（其PyTorch实现）应用于我们在前一个例子中讨论的相同信用卡违约数据集。

## 如何做到……

执行以下步骤以使用信用卡欺诈数据集训练一个TabNet分类器：

1.  导入库：

    ```py
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import recall_score
    from pytorch_tabnet.tab_model import TabNetClassifier
    from pytorch_tabnet.metrics import Metric
    import torch
    import pandas as pd
    import numpy as np 
    ```

1.  从CSV文件加载数据集：

    ```py
    df = pd.read_csv("../Datasets/credit_card_default.csv",
                     na_values="") 
    ```

1.  将目标从特征中分离，并创建包含数值/类别特征的列表：

    ```py
    X = df.copy()
    y = X.pop("default_payment_next_month")
    cat_features = list(X.select_dtypes("object").columns)
    num_features = list(X.select_dtypes("number").columns) 
    ```

1.  填充类别特征的缺失值，使用`LabelEncoder`进行编码，并存储每个特征的唯一类别数量：

    ```py
    cat_dims = {}
    for col in cat_features:
        label_encoder = LabelEncoder()
        X[col] = X[col].fillna("Missing")
        X[col] = label_encoder.fit_transform(X[col].values)
        cat_dims[col] = len(label_encoder.classes_)
    cat_dims 
    ```

    执行代码片段生成以下输出：

    ```py
    {'sex': 3,
     'education': 5,
     'marriage': 4,
     'payment_status_sep': 10,
     'payment_status_aug': 10,
     'payment_status_jul': 10,
     'payment_status_jun': 10,
     'payment_status_may': 9,
     'payment_status_apr': 9} 
    ```

    基于EDA（探索性数据分析），我们可能会认为`sex`特征只有两个唯一值。然而，由于我们使用`Missing`类别填充了缺失值，因此有三个唯一可能的值。

1.  使用70-15-15的比例创建训练/验证/测试集划分：

    ```py
    # create the initial split - training and temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )
    # create the valid and test sets
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42
    ) 
    ```

1.  填充所有数据集中的数值特征的缺失值：

    ```py
    for col in num_features:
        imp_mean = X_train[col].mean()
        X_train[col] = X_train[col].fillna(imp_mean)
        X_valid[col] = X_valid[col].fillna(imp_mean)
        X_test[col] = X_test[col].fillna(imp_mean) 
    ```

1.  准备包含类别特征索引和唯一类别数量的列表：

    ```py
    features = X.columns.to_list()
    cat_ind = [features.index(feat) for feat in cat_features]
    cat_dims = list(cat_dims.values()) 
    ```

1.  定义自定义召回率指标：

    ```py
    class Recall(Metric):
        def __init__(self):
            self._name = "recall"
            self._maximize = True
        def __call__(self, y_true, y_score):
            y_pred = np.argmax(y_score, axis=1)
            return recall_score(y_true, y_pred) 
    ```

1.  定义TabNet的参数并实例化分类器：

    ```py
    tabnet_params = {
        "cat_idxs": cat_ind,
        "cat_dims": cat_dims,
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": dict(lr=2e-2),
        "scheduler_params": {
            "step_size":20,
            "gamma":0.9
        },
        "scheduler_fn": torch.optim.lr_scheduler.StepLR,
        "mask_type": "sparsemax",
        "seed": 42,
    }
    tabnet = TabNetClassifier(**tabnet_params) 
    ```

1.  训练TabNet分类器：

    ```py
    tabnet.fit(
        X_train=X_train.values,
        y_train=y_train.values,
        eval_set=[
            (X_train.values, y_train.values),
            (X_valid.values, y_valid.values)
        ],
        eval_name=["train", "valid"],
        eval_metric=["auc", Recall],
        max_epochs=200,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128,
        weights=1,
    ) 
    ```

    下面我们可以看到训练过程中的简略日志：

    ```py
    epoch 0  | loss: 0.69867 | train_auc: 0.61461 | train_recall: 0.3789  | valid_auc: 0.62232 | valid_recall: 0.37286 |  0:00:01s
    epoch 1  | loss: 0.62342 | train_auc: 0.70538 | train_recall: 0.51539 | valid_auc: 0.69053 | valid_recall: 0.48744 |  0:00:02s
    epoch 2  | loss: 0.59902 | train_auc: 0.71777 | train_recall: 0.51625 | valid_auc: 0.71667 | valid_recall: 0.48643 |  0:00:03s
    epoch 3  | loss: 0.59629 | train_auc: 0.73428 | train_recall: 0.5268  | valid_auc: 0.72767 | valid_recall: 0.49447 |  0:00:04s
    …
    epoch 42 | loss: 0.56028 | train_auc: 0.78509 | train_recall: 0.6028  | valid_auc: 0.76955 | valid_recall: 0.58191 |  0:00:47s
    epoch 43 | loss: 0.56235 | train_auc: 0.7891  | train_recall: 0.55651 | valid_auc: 0.77126 | valid_recall: 0.5407  |  0:00:48s
    Early stopping occurred at epoch 43 with best_epoch = 23 and best_valid_recall = 0.6191
    Best weights from best epoch are automatically used! 
    ```

1.  准备历史记录DataFrame并绘制每个epoch的分数：

    ```py
    history_df = pd.DataFrame(tabnet.history.history) 
    ```

    然后，我们开始绘制每个epoch的损失值：

    ```py
    history_df["loss"].plot(title="Loss over epochs") 
    ```

    执行代码片段生成以下图表：

    ![](../Images/B18112_15_07.png)

    图15.7：训练损失随epoch变化

    然后，以类似的方式，我们生成了一个展示每个epoch召回率的图。为了简洁起见，我们没有包括生成图表的代码。

    ![](../Images/B18112_15_08.png)

    图15.8：训练和验证召回率随epoch变化

1.  为测试集创建预测并评估其性能：

    ```py
    y_pred = tabnet.predict(X_test.values)
    print(f"Best validation score: {tabnet.best_cost:.4f}")
    print(f"Test set score: {recall_score(y_test, y_pred):.4f}") 
    ```

    执行代码片段生成以下输出：

    ```py
    Best validation score: 0.6191
    Test set score: 0.6275 
    ```

    如我们所见，测试集的表现略优于使用验证集计算的召回率。

1.  提取并绘制全局特征重要性：

    ```py
    tabnet_feat_imp = pd.Series(tabnet.feature_importances_,
                                index=X_train.columns)
    (
        tabnet_feat_imp
        .nlargest(20)
        .sort_values()
        .plot(kind="barh",
              title="TabNet's feature importances")
    ) 
    ```

    执行代码片段生成以下图表：

![](../Images/B18112_15_09.png)

图15.9：从拟合的TabNet分类器中提取的全局特征重要性值

根据TabNet，预测10月违约的最重要特征是9月、7月和5月的支付状态。另一个重要特征是信用额度余额。

在此时有两点值得注意。首先，最重要的特征与我们在*第14章 高级机器学习项目概念*中的*调查特征重要性*方法中识别的特征相似。其次，特征重要性是按特征层面计算的，而非按特征和类别层面计算的，正如我们在使用类别特征的独热编码时所看到的那样。

## 它是如何工作的…

导入库后，我们从CSV文件加载数据集。然后，我们将目标从特征中分离，并提取类别特征和数值特征的名称。我们将其存储为列表。

在*步骤* *4*中，我们对分类特征进行了几项操作。首先，我们用一个新的类别——`Missing`来填补任何缺失的值。然后，我们使用`scikit-learn`的`LabelEncoder`对每一列分类特征进行编码。在此过程中，我们创建了一个字典，记录了每个分类特征的唯一类别数量（包括为缺失值新创建的类别）。

在*步骤* *5*中，我们使用`train_test_split`函数创建了训练/验证/测试数据集划分。我们决定采用70-15-15的比例进行划分。由于数据集不平衡（少数类大约在22%的观测中可见），因此我们在划分数据时使用了分层抽样。

在*步骤* *6*中，我们为数值特征填补了缺失值。我们使用训练集计算的平均值来填充缺失值。

在*步骤* *7*中，我们准备了两个列表。第一个列表包含了分类特征的数值索引，而第二个列表则包含了每个分类特征的唯一类别数量。确保这两个列表对齐至关重要，以便特征的索引与该特征的唯一类别数量一一对应。

在*步骤* *8*中，我们创建了一个自定义的召回率度量。`pytorch-tabnet`提供了一些度量（对于分类问题，包括准确率、ROC AUC和均衡准确率），但我们也可以轻松定义更多度量。为了创建自定义度量，我们进行了以下操作：

+   我们定义了一个继承自`Metric`类的类。

+   在`__init__`方法中，我们定义了度量的名称（如训练日志中所示），并指明了目标是否是最大化该度量。对于召回率而言，目标就是最大化该度量。

+   在`__call__`方法中，我们使用`scikit-learn`的`recall_score`函数计算召回率的值。但首先，我们需要将包含每个类别预测概率的数组转换为一个包含预测类别的对象。我们通过使用`np.argmax`函数来实现这一点。

在*步骤* *9*中，我们定义了一些TabNet的超参数并实例化了该模型。`pytorch-tabnet`提供了一个类似于`scikit-learn`的API，用于训练TabNet模型，无论是分类任务还是回归任务。因此，我们不需要精通PyTorch就能训练模型。首先，我们定义了一个字典，包含了模型的超参数。

通常，一些超参数是在模型级别定义的（在实例化类时传递给类），而其他超参数则是在拟合级别定义的（在使用`fit`方法时传递给模型）。此时，我们定义了模型的超参数：

+   分类特征的索引及其对应的唯一类别数量

+   选择的优化器：ADAM

+   学习率调度器

+   掩码类型

+   随机种子

在所有这些参数中，学习率调度器可能需要一些澄清。根据TabNet的文档，我们使用了逐步衰减的学习率。为此，我们指定了`torch.optim.lr_scheduler.StepLR`作为调度器函数。然后，我们提供了一些其他参数。最初，我们在`optimizer_params`中将学习率设置为`0.02`。接着，我们在`scheduler_params`中定义了逐步衰减的参数。我们指定每经过20个epoch，就应用一个衰减率`0.9`。在实践中，这意味着每经过20个epoch，学习率将变为0.9乘以0.02，等于0.018。衰减过程在每20个epoch后继续进行。

完成这些步骤后，我们通过指定的超参数实例化了`TabNetClassifier`类。默认情况下，TabNet使用交叉熵损失函数进行分类问题，使用均方误差（MSE）进行回归任务。

在*第10步*中，我们使用`fit`方法训练了`TabNetClassifier`。我们提供了相当多的参数：

+   训练数据

+   评估集——在此案例中，我们使用了训练集和验证集，这样每个epoch后，我们可以看到两个数据集的计算指标。

+   评估集的名称

+   用于评估的指标——我们使用了ROC AUC和在*第8步*中定义的自定义召回指标。

+   最大epoch数

+   **patience**参数，表示如果我们在*X*个连续的epoch中没有看到评估指标的改善，训练将停止，并且我们将使用最佳epoch的权重进行预测。

+   批量大小和虚拟批量大小（用于幽灵批量归一化；更多细节请参见*更多内容...*部分）

+   `weights`参数，仅适用于分类问题。它与采样有关，在处理类别不平衡时非常有帮助。将其设置为`0`表示没有采样。将其设置为`1`则启用基于类别发生频率的反向比例进行加权采样。最后，我们可以提供一个包含自定义类别权重的字典。

需要注意的是，TabNet的训练中，我们提供的数据集必须是`numpy`数组，而不是`pandas` DataFrame。因此，我们使用`values`方法从DataFrame中提取数组。使用`numpy`数组的需求也是我们需要定义类别特征的数值索引，而不能提供特征名称列表的原因。

与许多神经网络架构相比，TabNet使用了相对较大的批量大小。原始论文建议，我们可以使用总训练观测数的10%作为批量大小。还建议虚拟批量大小应小于批量大小，且后者可以整除前者。

在*步骤 11*中，我们从拟合的 TabNet 模型的`history`属性中提取了训练信息。它包含了训练日志中显示的相同信息，即每个 epoch 的损失、学习率和评估指标。然后，我们绘制了每个 epoch 的损失和召回率图。

在*步骤 12*中，我们使用 `predict` 方法创建了预测。与训练步骤类似，我们也需要将输入特征提供为 `numpy` 数组。和 `scikit-learn` 中一样，`predict` 方法返回预测的类别，而我们可以使用 `predict_proba` 方法获取类别概率。我们还使用 `scikit-learn` 的 `recall_score` 函数计算了测试集上的召回率。

在最后一步，我们提取了全局特征重要性值。与 `scikit-learn` 模型类似，它们存储在拟合模型的 `feature_importances_` 属性下。然后，我们绘制了最重要的 20 个特征。值得注意的是，全局特征重要性值是标准化的，它们的总和为 1。

## 还有更多……

这里有一些关于 TabNet 和其在 PyTorch 中实现的有趣点：

+   TabNet 使用**幽灵批量归一化**来训练大批量数据，并同时提供更好的泛化能力。该过程的基本思想是将输入批次分割成大小相等的子批次（由虚拟批次大小参数确定）。然后，我们对这些子批次应用相同的批量归一化层。

+   `pytorch-tabnet` 允许我们在训练过程中应用自定义的数据增强管道。目前，库提供了对分类和回归任务使用 SMOTE 方法的功能。

+   TabNet 可以作为无监督模型进行预训练，从而提高模型的表现。在预训练过程中，某些单元会被故意遮蔽，模型通过预测这些缺失（遮蔽）值，学习这些被遮蔽单元与相邻列之间的关系。然后，我们可以将这些权重用于有监督任务。通过学习特征之间的关系，无监督表示学习作为有监督学习任务的改进编码器模型。在预训练时，我们可以决定遮蔽多少比例的特征。

+   TabNet 使用**sparsemax**作为遮蔽函数。一般来说，sparsemax 是一种非线性归一化函数，其分布比流行的 softmax 函数更稀疏。这个函数使神经网络能够更有效地选择重要的特征。此外，该函数采用稀疏性正则化（其强度由超参数决定）来惩罚不够稀疏的遮蔽。`pytorch-tabnet` 库还包含了 `EntMax` 遮蔽函数。

+   在本教程中，我们介绍了如何提取全局特征重要性。要提取局部重要性，我们可以使用拟合后的TabNet模型的`explain`方法。该方法返回两个元素：一个矩阵，包含每个观察值和特征的重要性，以及模型在特征选择中使用的注意力掩码。

## 另见

+   Arik, S. Ö., & Pfister, T. 2021年5月。Tabnet：可解释的注意力表格学习。在*AAAI人工智能会议论文集*，35(8)：6679-6687。

+   描述上述论文中TabNet实现的原始代码库：[https://github.com/google-research/google-research/tree/master/tabnet](https://github.com/google-research/google-research/tree/master/tabnet)。

# 亚马逊DeepAR的时间序列预测

我们已经在*第六章*《时间序列分析与预测》和*第七章*《基于机器学习的时间序列预测方法》中讲解了时间序列分析和预测。这次，我们将看一个深度学习方法在时间序列预测中的应用示例。在本教程中，我们将介绍亚马逊的DeepAR模型。该模型最初是作为一个需求/销售预测工具开发的，旨在处理成百上千个**库存单位**（**SKU**）的规模。

DeepAR的架构超出了本书的范围。因此，我们将只关注模型的一些关键特性。具体如下：

+   DeepAR创建一个用于所有考虑的时间序列的全局模型。它在架构中实现了LSTM单元，这种架构允许同时使用成百上千个时间序列进行训练。该模型还使用了编码器-解码器结构，这是序列到序列模型中常见的做法。

+   DeepAR允许使用与目标时间序列相关的一组协变量（外部回归量）。

+   该模型对特征工程的要求最小。它自动创建相关的时间序列特征（根据数据的粒度，这些特征可能包括月份中的天、年份中的天等），并且它从提供的协变量中学习时间序列的季节性模式。

+   DeepAR提供基于蒙特卡罗采样的概率预测——它计算一致的分位数估计。

+   该模型能够通过学习相似的时间序列来为具有少量历史数据的时间序列生成预测。这是解决冷启动问题的一种潜在方案。

+   该模型可以使用各种似然函数。

在本教程中，我们将使用2020年和2021年的大约100个时间序列的日常股票价格来训练一个DeepAR模型。然后，我们将创建涵盖2021年最后20个工作日的20天前的预测。

在继续之前，我们想强调的是，使用股票价格的时间序列只是为了说明目的。深度学习模型在经过数百甚至数千个时间序列的训练后表现最佳。我们选择了股票价格作为示例，因为这些数据最容易下载。正如我们之前提到的，准确预测股票价格，尤其是在长时间预测的情况下，极其困难，甚至几乎不可能。

## 如何做…

执行以下步骤，使用股票价格作为输入时间序列来训练DeepAR模型：

1.  导入库：

    ```py
    import pandas as pd
    import torch
    import yfinance as yf
    from random import sample, seed
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_forecasting import DeepAR, TimeSeriesDataSet 
    ```

1.  下载标准普尔500指数成分股的股票代码，并从列表中随机抽取100个股票代码：

    ```py
    df = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    df = df[0]
    seed(44)
    sampled_tickers = sample(df["Symbol"].to_list(), 100) 
    ```

1.  下载所选股票的历史股价：

    ```py
    raw_df = yf.download(sampled_tickers,
                         start="2020-01-01",
                         end="2021-12-31") 
    ```

1.  保留调整后的收盘价，并剔除缺失值的股票：

    ```py
    df = raw_df["Adj Close"]
    df = df.loc[:, ~df.isna().any()]
    selected_tickers = df.columns 
    ```

    在剔除在关注期内至少有一个缺失值的股票后，剩下了98只股票。

1.  将数据格式从宽格式转换为长格式，并添加时间索引：

    ```py
    df = df.reset_index(drop=False) 

    df = ( 
        pd.melt(df, 
                id_vars=["Date"], 
                value_vars=selected_tickers, 
                value_name="price"
        ).rename(columns={"variable": "ticker"}) 
    )
    df["time_idx"] = df.groupby("ticker").cumcount() 
    df 
    ```

    执行代码片段后，会生成以下数据框架的预览：

    ![](../Images/B18112_15_10.png)

    图 15.10：DeepAR模型输入数据框架的预览

1.  定义用于设置模型训练的常量：

    ```py
    MAX_ENCODER_LENGTH = 40
    MAX_PRED_LENGTH = 20
    BATCH_SIZE = 128
    MAX_EPOCHS = 30
    training_cutoff = df["time_idx"].max() - MAX_PRED_LENGTH 
    ```

1.  定义训练集和验证集：

    ```py
    train_set = TimeSeriesDataSet(
        df[lambda x: x["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target="price",
        group_ids=["ticker"],
        time_varying_unknown_reals=["price"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PRED_LENGTH,
    )
    valid_set = TimeSeriesDataSet.from_dataset(
        train_set, df, min_prediction_idx=training_cutoff+1
    ) 
    ```

1.  从数据集获取数据加载器：

    ```py
    train_dataloader = train_set.to_dataloader(
        train=True, batch_size=BATCH_SIZE
    )
    valid_dataloader = valid_set.to_dataloader(
        train=False, batch_size=BATCH_SIZE
    ) 
    ```

1.  定义DeepAR模型并找到建议的学习率：

    ```py
    pl.seed_everything(42)
    deep_ar = DeepAR.from_dataset(
        train_set,
        learning_rate=1e-2,
        hidden_size=30,
        rnn_layers=4
    )
    trainer = pl.Trainer(gradient_clip_val=1e-1)
    res = trainer.tuner.lr_find(
        deep_ar,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        min_lr=1e-5,
        max_lr=1e0,
        early_stop_threshold=100,
    )
    fig = res.plot(show=True, suggest=True) 
    ```

    执行代码片段后，会生成以下图表，其中红点表示建议的学习率。

    ![](../Images/B18112_15_11.png)

    图 15.11：训练DeepAR模型时建议的学习率

1.  训练DeepAR模型：

    ```py
    pl.seed_everything(42)
    deep_ar.hparams.learning_rate = res.suggestion()
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10
    )
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback]
    )
    trainer.fit(
        deep_ar,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    ) 
    ```

1.  从检查点中提取最佳的DeepAR模型：

    ```py
    best_model = DeepAR.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    ) 
    ```

1.  创建验证集的预测并绘制其中的5个：

    ```py
    raw_predictions, x = best_model.predict(
        valid_dataloader,
        mode="raw",
        return_x=True,
        n_samples=100
    )
    tickers = valid_set.x_to_index(x)["ticker"]
    for idx in range(5):
        best_model.plot_prediction(
            x, raw_predictions, idx=idx, add_loss_to_title=True
        )
        plt.suptitle(f"Ticker: {tickers.iloc[idx]}") 
    ```

    在代码片段中，我们生成了100个预测并绘制了其中5个以进行视觉检查。为了简洁起见，我们只展示了其中两个。但我们强烈建议检查更多图表，以更好地理解模型的表现。

![](../Images/B18112_15_12.png)

图 15.12：DeepAR对ABMD股票的预测

![](../Images/B18112_15_13.png)

图 15.13：DeepAR对ADM股票的预测

这些图表展示了2021年最后20个工作日两只股票的预测值，以及对应的分位数估计值。虽然预测结果表现一般，但我们可以看到，至少实际值位于提供的分位数估计范围内。

我们不会花更多时间评估模型及其预测的表现，因为主要目的是展示DeepAR模型是如何工作的，以及如何使用它生成预测。然而，我们会提到一些潜在的改进。首先，我们本可以训练更多的周期，因为我们没有检查模型的收敛性。我们使用了早停法，但在训练过程中并没有触发。其次，我们使用了相当多的任意值来定义网络的架构。在实际应用中，我们应该使用自己选择的超参数优化方法来识别适合我们任务的最佳值。

## 它是如何工作的……

在*步骤1*中，我们导入了所需的库。为了使用DeepAR模型，我们决定使用PyTorch Forecasting库。它是建立在PyTorch Lightning之上的库，允许我们轻松使用最先进的深度学习模型进行时间序列预测。这些模型可以使用GPU进行训练，我们还可以参考TensorBoard查看训练日志。

在*步骤2*中，我们下载了包含标准普尔500指数成分股的列表。然后，我们随机抽取了其中100只并将结果存储在列表中。我们随机抽取了股票代码，以加速训练。重复这个过程，使用所有的股票，肯定会对模型产生有益的影响。

在*步骤3*中，我们使用`yfinance`库下载了2020年和2021年的历史股票价格。在下一步中，我们需要进行进一步的预处理。我们只保留了调整后的收盘价，并移除了任何有缺失值的股票。

在*步骤5*中，我们继续进行预处理。我们将数据框从宽格式转换为长格式，然后添加时间索引。DeepAR的实现使用整数时间索引而不是日期，因此我们使用了`cumcount`方法结合`groupby`方法为每只考虑的股票创建了时间索引。

在*步骤6*中，我们定义了用于训练过程的一些常数，例如编码器步骤的最大长度、我们希望预测的未来观测值数量、最大训练周期数等。我们还指定了哪个时间索引将训练与验证数据分开。

在*步骤7*中，我们定义了训练集和验证集。我们使用了`TimeSeriesDataSet`类来完成这一操作，该类的职责包括：

+   变量转换的处理

+   缺失值的处理

+   存储有关静态和时间变化变量的信息（包括已知和未来未知的）

+   随机子抽样

在定义训练数据集时，我们需要提供训练数据（使用先前定义的截止点进行过滤）、包含时间索引的列名称、目标、组ID（在我们的案例中，这些是股票代码）、编码器长度和预测范围。

从 `TimeSeriesDataSet` 生成的每个样本都是完整时间序列的一个子序列。每个子序列包含给定时间序列的编码器和预测时间点。`TimeSeriesDataSet` 创建了一个索引，定义了哪些子序列存在并可以从中进行采样。

在*步骤 8*中，我们使用 `TimeSeriesDataSet` 的 `to_dataloader` 方法将数据集转换为数据加载器。

在*步骤 9*中，我们使用 `DeepAR` 类的 `from_dataset` 方法定义了 DeepAR 模型。这样，我们就不必重复在创建 `TimeSeriesDataSet` 对象时已经指定的内容。此外，我们还指定了学习率、隐藏层的大小以及 RNN 层数。后两者是 DeepAR 模型中最重要的超参数，应该使用一些超参数优化（HPO）框架进行调优，例如 Hyperopt 或 Optuna。然后，我们使用 PyTorch Lightning 的 `Trainer` 类来寻找最优的学习率。

默认情况下，DeepAR 模型使用高斯损失函数。根据任务的不同，我们可以使用一些其他替代方法。对于处理实数数据时，高斯分布是首选。对于正整数计数数据，我们可能需要使用负二项式似然。对于位于单位区间的数据，Beta 似然是一个不错的选择，而对于二值数据，Bernoulli 似然则是理想的选择。

在*步骤 10*中，我们使用确定的学习率训练了 DeepAR 模型。此外，我们还指定了早停回调函数，如果在 10 个 epoch 内验证损失没有显著（由我们定义）改善，训练将停止。

在*步骤 11*中，我们从检查点提取了最佳模型。然后，我们使用最佳模型通过 `predict` 方法生成预测。我们为验证数据加载器中可用的 100 个序列创建了预测。我们指出希望提取原始预测（此选项返回一个包含预测及附加信息（如相应的分位数等）的字典）以及生成这些预测所用的输入。然后，我们使用拟合的 `DeepAR` 模型的 `plot_prediction` 方法绘制了预测图。

## 还有更多内容……

PyTorch Forecasting 还允许我们轻松训练 DeepVAR 模型，它是 DeepAR 的多变量对应模型。最初，Salinas *et al*.（2019）将该模型称为 VEC-LSTM。

DeepAR 和 DeepVAR 都可以在亚马逊的 GluonTS 库中使用。

在本节中，我们展示了如何调整用于训练 DeepAR 模型的代码，改为训练 DeepVAR 模型：

1.  导入库：

    ```py
    from pytorch_forecasting.metrics import MultivariateNormalDistributionLoss
    import seaborn as sns
    import numpy as np 
    ```

1.  再次定义数据加载器：

    ```py
    train_set = TimeSeriesDataSet(
        df[lambda x: x["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target="price",
        group_ids=["ticker"],
        static_categoricals=["ticker"],  
        time_varying_unknown_reals=["price"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PRED_LENGTH,
    )
    valid_set = TimeSeriesDataSet.from_dataset(
        train_set, df, min_prediction_idx=training_cutoff+1
    )
    train_dataloader = train_set.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        batch_sampler="synchronized"
    )
    valid_dataloader = valid_set.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        batch_sampler="synchronized"
    ) 
    ```

    在这一步有两个不同之处。首先，在创建训练数据集时，我们还指定了`static_categoricals`参数。因为我们要预测相关性，使用诸如股票代码等序列特征非常重要。第二，在创建数据加载器时，我们还必须指定`batch_sampler="synchronized"`。使用这个选项可以确保传递给解码器的样本在时间上是对齐的。

1.  定义DeepVAR模型并找到学习率：

    ```py
    pl.seed_everything(42)
    deep_var = DeepAR.from_dataset(
        train_set,
        learning_rate=1e-2,
        hidden_size=30,
        rnn_layers=4,
        loss=MultivariateNormalDistributionLoss()
    )
    trainer = pl.Trainer(gradient_clip_val=1e-1)
    res = trainer.tuner.lr_find(
        deep_var,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        min_lr=1e-5,
        max_lr=1e0,
        early_stop_threshold=100,
    ) 
    ```

    训练DeepVAR和DeepAR模型的最后一个区别是，对于前者，我们使用`MultivariateNormalDistributionLoss`作为损失，而不是默认的`NormalDistributionLoss`。

1.  使用选择的学习率训练DeepVAR模型：

    ```py
    pl.seed_everything(42)
    deep_var.hparams.learning_rate = res.suggestion()
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10
    )
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback]
    )
    trainer.fit(
        deep_var,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    ) 
    ```

1.  从检查点提取最佳DeepVAR模型：

    ```py
    best_model = DeepAR.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    ) 
    ```

1.  提取相关性矩阵：

    ```py
    preds = best_model.predict(valid_dataloader,
                               mode=("raw", "prediction"),
                               n_samples=None)

    cov_matrix = (
        best_model
        .loss
        .map_x_to_distribution(preds)
        .base_dist
        .covariance_matrix
        .mean(0)
    )
    cov_diag_mult = (
        torch.diag(cov_matrix)[None] * torch.diag(cov_matrix)[None].T
    )
    corr_matrix = cov_matrix / torch.sqrt(cov_diag_mult) 
    ```

1.  绘制相关性矩阵及其分布：

    ```py
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    fif, ax = plt.subplots()
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_matrix, mask=mask, cmap=cmap, 
        vmax=.3, center=0, square=True, 
        linewidths=.5, cbar_kws={"shrink": .5}
    )
    ax.set_title("Correlation matrix") 
    ```

    执行该代码片段生成以下图形：

![](../Images/B18112_15_14.png)

图 15.14：从DeepVAR提取的相关性矩阵

为了更好地理解相关性的分布，我们绘制了其直方图：

```py
plt.hist(corr_matrix[corr_matrix < 1].numpy()) 
```

执行该代码片段生成以下图形：

![](../Images/B18112_15_15.png)

图 15.15：直方图展示了提取的相关性分布

在查看直方图时，请记住我们是基于相关性矩阵创建了直方图。这意味着我们实际上对每个值进行了两次计数。

## 另见

+   Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. 2020\. “DeepAR: 基于自回归递归网络的概率预测”，*国际预测学杂志*，36(3)：1181-1191。

+   Salinas, D., Bohlke-Schneider, M., Callot, L., Medico, R., & Gasthaus, J. 2019\. 高维多元预测与低秩高斯哥普拉过程。*神经信息处理系统进展*，32。

# 使用NeuralProphet进行时间序列预测

在*第7章，基于机器学习的时间序列预测方法*中，我们介绍了Meta（前身为Facebook）创建的Prophet算法。在本食谱中，我们将探讨该算法的扩展——NeuralProphet。

简单回顾一下，Prophet的作者强调了模型的良好性能、可解释性和易用性作为其主要优势。NeuralProphet的作者也考虑到了这一点，并在其方法中保留了Prophet的所有优势，同时加入了新组件，提升了准确性和可扩展性。

原始Prophet算法的批评包括其僵化的参数结构（基于广义线性模型）以及它作为一种“曲线拟合器”不足以适应局部模式。

传统上，时间序列模型使用时间序列的滞后值来预测未来的值。Prophet的创造者将时间序列预测重新定义为曲线拟合问题，算法试图找到趋势的函数形式。

在接下来的内容中，我们简要提到 NeuralProphet 的一些重要新增功能：

+   NeuralProphet 在 Prophet 规范中引入了自回归项。

+   自回归通过**自回归网络（AR-Net）**实现。AR-Net 是一个神经网络，用于模拟时间序列信号中的自回归过程。虽然传统 AR 模型和 AR-Net 的输入相同，但后者能够在比前者更大规模下操作。

+   NeuralProphet 使用 PyTorch 作为后端，而 Prophet 算法使用 Stan。这使得训练速度更快，并带来其他一些好处。

+   滞后回归变量（特征）通过前馈神经网络建模。

+   该算法可以与自定义的损失函数和评估指标一起使用。

+   该库广泛使用正则化，并且我们可以将其应用于模型的各个组件：趋势、季节性、假期、自回归项等。尤其对于 AR 项，使用正则化可以让我们在不担心训练时间迅速增加的情况下使用更多的滞后值。

实际上，NeuralProphet 支持几种 AR 项的配置：

+   线性 AR——一个没有偏置项和激活函数的单层神经网络。本质上，它将特定的滞后值回归到特定的预测步长。由于其简洁性，它的解释相对简单。

+   深度 AR——在这种形式下，AR 项通过具有指定数量隐藏层和 ReLU 激活函数的全连接神经网络建模。尽管它增加了复杂性、延长了训练时间并且失去了可解释性，但这种配置通常比线性模型提供更高的预测精度。

+   稀疏 AR——我们可以结合高阶 AR（有更多先前时间步的值）和正则化项。

所提到的每种配置可以应用于目标变量和协变量。

总结一下，NeuralProphet 由以下几个组件构成：

+   趋势

+   季节性

+   假期和特殊事件

+   自回归

+   滞后回归——滞后值的协变量通过前馈神经网络内部建模

+   未来回归——类似于事件/假期，这是我们已知的未来回归值（无论是给定值还是我们对这些值有单独的预测）

在这个示例中，我们将 NeuralProphet 的几种配置拟合到 2010 至 2021 年的每日 S&P 500 股价时间序列中。与之前的示例类似，我们选择资产价格的时间序列是因为数据的可获取性以及其每日频率。使用机器学习/深度学习预测股价可能非常困难，甚至几乎不可能，因此本练习的目的是展示如何使用 NeuralProphet 算法，而不是创造最精确的预测。

## 如何操作...

执行以下步骤，将 NeuralProphet 算法的几种配置拟合到每日 S&P 500 股价的时间序列中：

1.  导入库：

    ```py
    import yfinance as yf
    import pandas as pd
    from neuralprophet import NeuralProphet
    from neuralprophet.utils import set_random_seed 
    ```

1.  下载标准普尔 500 指数的历史价格并准备 DataFrame，以便使用 NeuralProphet 进行建模：

    ```py
    df = yf.download("^GSPC",
                     start="2010-01-01",
                     end="2021-12-31")
    df = df[["Adj Close"]].reset_index(drop=False)
    df.columns = ["ds", "y"] 
    ```

1.  创建训练/测试集拆分：

    ```py
    TEST_LENGTH = 60
    df_train = df.iloc[:-TEST_LENGTH]
    df_test = df.iloc[-TEST_LENGTH:] 
    ```

1.  训练默认的 Prophet 模型并绘制评估指标：

    ```py
    set_random_seed(42)
    model = NeuralProphet(changepoints_range=0.95)
    metrics = model.fit(df_train, freq="B")
    (
        metrics
        .drop(columns=["RegLoss"])
        .plot(title="Evaluation metrics during training",
              subplots=True)
    ) 
    ```

    执行这段代码会生成以下图表：

    ![](../Images/B18112_15_16.png)

    图 15.16：NeuralProphet 训练过程中每个周期的评估指标

1.  计算预测值并绘制拟合结果：

    ```py
    pred_df = model.predict(df)
    pred_df.plot(x="ds", y=["y", "yhat1"],
                 title="S&P 500 - forecast vs ground truth") 
    ```

    执行这段代码会生成以下图表：

    ![](../Images/B18112_15_17.png)

    图 15.17：NeuralProphet 模型的拟合与整个时间序列的实际值对比

    如我们所见，模型的拟合线遵循整体上升趋势（甚至随着时间的推移调整增长速度），但它忽略了极端周期，并未跟随局部尺度上的变化。

    此外，我们还可以放大与测试集对应的时间段：

    ```py
    (
        pred_df
        .iloc[-TEST_LENGTH:]
        .plot(x="ds", y=["y", "yhat1"],
              title="S&P 500 - forecast vs ground truth")
    ) 
    ```

    执行这段代码会生成以下图表：

    ![](../Images/B18112_15_18.png)

    图 15.18：NeuralProphet 模型的拟合与测试集中的实际值对比

    从图表得出的结论与整体拟合的结论非常相似——模型遵循上升趋势，但没有捕捉到局部模式。

    为了评估测试集的表现，我们可以使用以下命令：`model.test(df_test)`。

1.  向 NeuralProphet 中添加 AR 组件：

    ```py
    set_random_seed(42)
    model = NeuralProphet(
        changepoints_range=0.95,
        n_lags=10,
        ar_reg=1,
    )
    metrics = model.fit(df_train, freq="B")
    pred_df = model.predict(df)
    pred_df.plot(x="ds", y=["y", "yhat1"],
                 title="S&P 500 - forecast vs ground truth") 
    ```

    执行这段代码会生成以下图表：

    ![](../Images/B18112_15_19.png)

    图 15.19：NeuralProphet 模型的拟合与整个时间序列的实际值对比

    该拟合效果比之前的要好得多。再次，我们更仔细地查看测试集：

    ```py
    (
        pred_df
        .iloc[-TEST_LENGTH:]
        .plot(x="ds", y=["y", "yhat1"],
              title="S&P 500 - forecast vs ground truth")
    ) 
    ```

    执行这段代码会生成以下图表：

    ![](../Images/B18112_15_20.png)

    图 15.20：NeuralProphet 模型的拟合与测试集中的实际值对比

    我们可以看到一个既熟悉又令人担忧的模式——预测结果滞后于原始序列。这里的意思是，预测值非常接近最后一个已知值。换句话说，预测线与真实值线相似，只不过在时间轴上向右偏移了一个或多个周期。

1.  向 NeuralProphet 中添加 AR-Net：

    ```py
    set_random_seed(42)
    model = NeuralProphet(
        changepoints_range=0.95,
        n_lags=10,
        ar_reg=1,
        num_hidden_layers=3,
        d_hidden=32,
    )
    metrics = model.fit(df_train, freq="B")
    pred_df = model.predict(df)
    (
        pred_df
        .iloc[-TEST_LENGTH:]
        .plot(x="ds", y=["y", "yhat1"],
              title="S&P 500 - forecast vs ground truth")
    ) 
    ```

    执行这段代码会生成以下图表：

    ![](../Images/B18112_15_21.png)

    图 15.21：NeuralProphet 模型的拟合与测试集中的实际值对比

    我们可以看到，使用 AR-Net 后，预测图比没有使用 AR-Net 时的效果更好。尽管模式仍然看起来相差一个周期，但它们不像之前那样过拟合。

1.  绘制模型的组件和参数：

    ```py
    model.plot_components(model.predict(df_train)) 
    ```

    执行这段代码会生成以下图表：

![](../Images/B18112_15_22.png)

图 15.22：拟合的 NeuralProphet 模型组件（包括 AR-Net）

在这些图表中，我们可以看到一些模式：

+   一个上升趋势，并且有几个已识别的变化点。

+   4 月底的季节性高峰和 9 月底及 10 月初的季节性低谷。

+   在工作日内没有出现意外的模式。然而，重要的是要记住，我们不应查看星期六和星期日的周季节性值。由于我们处理的是仅在工作日提供的每日数据，因此预测也应仅针对工作日进行，因为周内季节性不会很好地估算周末的数据。

查看股票价格的年度季节性可以揭示一些有趣的模式。最著名的模式之一是1月效应，它涉及股票价格在1月份可能的季节性上涨。通常，这被归因于资产购买的增加，通常发生在12月的价格下跌之后，投资者倾向于为了税收目的出售部分资产。

接着，我们还绘制了模型的参数：

```py
model.plot_parameters() 
```

执行这段代码会生成以下图表：

![](../Images/B18112_15_23.png)

图15.23：拟合的NeuralProphet模型参数（包括AR-Net）

由于组件和参数的图表之间有很多重叠，因此我们只关注新的元素。首先，我们可以查看描绘趋势变化幅度的图表。我们可以将其与*图15.22*中的趋势组件图表一起考虑。接着，我们可以看到变化率是如何与多年来的趋势相对应的。其次，似乎滞后期2是所有考虑的10个滞后期中最相关的。

## 它是如何工作的……

在导入库之后，我们从2010年到2021年下载了标准普尔500指数的每日价格。我们只保留了调整后的收盘价，并将DataFrame转换为Prophet和NeuralProphet都能识别的格式，即一个包含名为`ds`的时间列和目标时间序列`y`的DataFrame。

在*步骤3*中，我们将测试集大小设置为60，并将DataFrame切分为训练集和测试集。

NeuralProphet还支持在训练模型时使用验证集。我们可以在调用`fit`方法时添加它。

在*步骤4*中，我们实例化了几乎默认的NeuralProphet模型。我们调整的唯一超参数是`changepoints_range`。我们将其值从默认的`0.9`增加到`0.95`，这意味着模型可以在前95%的数据中识别变点，其余部分保持不变，以确保最终趋势的一致性。我们之所以增加默认值，是因为我们将关注相对短期的预测。

在*步骤5*中，我们使用`predict`方法和整个时间序列作为输入来计算预测值。这样，我们得到了拟合值（样本内拟合）和测试集的样本外预测值。此时，我们也可以使用`make_future_dataframe`方法，这在原始Prophet库中是熟悉的。

在*第6步*中，我们添加了线性AR项。我们通过`n_lags`参数指定了考虑的滞后期数。此外，我们通过将`ar_reg`设置为`1`，添加了AR项的正则化。我们本可以指定学习率。然而，当我们不提供值时，库会使用学习率范围测试来找到最佳值。

设置AR项的正则化时（这适用于库中的所有正则化），值为零表示不进行正则化。较小的值（例如，0.001到1之间）表示弱正则化。在AR项的情况下，这意味着会有更多非零的AR系数。较大的值（例如，1到100之间）会显著限制非零系数的数量。

在*第7步*中，我们将AR项的使用从线性AR扩展到了AR-Net。我们保持其他超参数与*第6步*相同，但我们指定了要使用多少个隐藏层（`num_hidden_layers`）以及它们的大小（`d_hidden`）。

在最后一步，我们使用`plot_components`方法绘制了NeuralProphet的组件，并使用`plot_parameters`方法绘制了模型的参数。

## 还有更多……

我们刚刚介绍了使用NeuralProphet的基础知识。在本节中，我们将提到该库的其他一些功能。

### 添加节假日和特殊事件

原始Prophet算法中非常受欢迎的一个功能，在NeuralProphet中也同样可以使用，就是能够轻松添加节假日和特殊日期。例如，在零售工作中，我们可以添加体育赛事（例如世界锦标赛或超级碗）或黑色星期五，后者并非官方假期。在以下代码片段中，我们基于AR-Net将美国的节假日添加到我们的模型中：

```py
set_random_seed(42)
model = NeuralProphet(
    changepoints_range=0.95,
    n_lags=10,
    ar_reg=1,
    num_hidden_layers=3,
    d_hidden=32,
)
model = model.add_country_holidays(
    "US", lower_window=-1, upper_window=1
)
metrics = model.fit(df_train, freq="B") 
```

此外，我们指定节假日也会影响周围的日期，即节假日前一天和节假日后一天。如果我们考虑某些日期的前期准备和后期回落，这一功能可能尤其重要。例如，在零售领域，我们可能希望指定圣诞节前的一个时期，因为那时人们通常会购买礼物。

通过检查组件图，我们可以看到节假日对时间的影响。

![](../Images/B18112_15_24.png)

图15.24：拟合后的NeuralProphet的节假日组件

此外，我们可以检查参数图，从中获取更多关于特定节假日（及其周围日期）影响的见解。

在这种情况下，我们一次性添加了所有的美国节假日。因此，所有节假日也都有相同的周围日期范围（节假日前一天和节假日后一天）。然而，我们也可以手动创建一个包含自定义节假日的DataFrame，并在特定事件级别指定周围的天数，而不是全局设置。

### 下一步预测与多步预测

使用NeuralProphet进行多步预测有两种方法：

+   我们可以递归地创建一步预测。该过程如下：我们预测下一步，将预测值添加到数据中，然后预测下一步。我们重复此过程，直到达到所需的预测时长。

+   我们可以直接预测多个步骤。

默认情况下，NeuralProphet 将使用第一种方法。然而，我们可以通过指定 `NeuralProphet` 类的 `n_forecasts` 超参数来使用第二种方法：

```py
model = NeuralProphet(
    n_lags=10,
    n_forecasts=10,
    ar_reg=1,
    learning_rate=0.01
)
metrics = model.fit(df_train, freq="B")
pred_df = model.predict(df)
pred_df.tail() 
```

下面我们只展示结果 DataFrame 的一部分。

![](../Images/B18112_15_25.png)

图 15.25：包含 10 步预测的 DataFrame 预览

这次，DataFrame 每行将包含 10 个预测值：`yhat1`、`yhat2`、`…`、`yhat10`。要学习如何解读该表，我们可以查看*图 15.25*中展示的最后一行。`yhat2` 值对应于 `2021-12-30` 的预测，预测时间为该日期之前的 2 天。因此，`yhat` 后面的数字表示预测的时间跨度（在此案例中以天为单位）。

或者，我们可以调换这一过程。通过在调用 `predict` 方法时指定 `raw=True`，我们可以获得基于行日期的预测，而不是预测该日期的预测值：

```py
pred_df = model.predict(df, raw=True, decompose=False)
pred_df.tail() 
```

执行该代码片段生成了以下 DataFrame 预览：

![](../Images/B18112_15_26.png)

图 15.26：包含前 5 个 10 步预测的 DataFrame 预览

我们可以轻松地在两个表格中跟踪某些预测，看看它们的结构如何不同。

当绘制多步预测时，我们将看到多条线——每条线都来自不同的预测日期：

```py
pred_df = model.predict(df_test)
model.plot(pred_df)
ax = plt.gca()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title("10-day ahead multi-step forecast") 
```

执行该代码片段会生成以下图表：

![](../Images/B18112_15_27.png)

图 15.27：10 天后的多步预测

由于线条重叠，图表很难读取。我们可以使用`highlight_nth_step_ahead_of_each_forecast`方法突出显示为某一步预测的内容。以下代码片段演示了如何操作：

```py
model = model.highlight_nth_step_ahead_of_each_forecast(1)
model.plot(pred_df)
ax = plt.gca()
ax.set_title("Step 1 of the 10-day ahead multi-step forecast") 
```

执行该代码片段会生成以下图表：

![](../Images/B18112_15_28.png)

图 15.28：10 天多步预测的第一步

在分析*图 15.28*后，我们可以得出结论：模型在预测上仍然存在困难，预测值非常接近最后已知值。

### 其他功能

NeuralProphet 还包含一些其他有趣的功能，包括：

+   广泛的交叉验证和基准测试功能

+   模型的组成部分，如假期/事件、季节性或未来回归因子，既可以是加法的，也可以是乘法的。

+   默认的损失函数是 Huber 损失，但我们可以将其更改为其他流行的损失函数。

## 另见

+   Triebe, O., Laptev, N., & Rajagopal, R. 2019\. *Ar-net: 一种简单的自回归神经网络用于时间序列预测*。arXiv 预印本 arXiv:1911.12436。

+   Triebe, O., Hewamalage, H., Pilyugina, P., Laptev, N., Bergmeir, C., & Rajagopal, R. 2021\. *Neuralprophet: 大规模可解释的预测*。arXiv 预印本 arXiv:2111.15397。

# 摘要

在本章中，我们探讨了如何同时使用深度学习处理表格数据和时间序列数据。我们没有从零开始构建神经网络，而是使用了现代的 Python 库，这些库为我们处理了大部分繁重的工作。

正如我们已经提到的，深度学习是一个快速发展的领域，每天都有新的神经网络架构发表。因此，在一个章节中很难仅仅触及冰山一角。这就是为什么我们现在会引导您了解一些流行且具有影响力的方法/库，您可能会想要自己探索。

## 表格数据

以下是一些相关的论文和 Python 库，它们绝对是进一步探索使用深度学习处理表格数据这一主题的好起点。

进一步阅读：

+   Huang, X., Khetan, A., Cvitkovic, M., & Karnin, Z. 2020\. *Tabtransformer: 使用上下文嵌入进行表格数据建模*。arXiv 预印本 arXiv:2012.06678。

+   Popov, S., Morozov, S., & Babenko, A. 2019\. *神经无知决策集成方法用于深度学习表格数据*。arXiv 预印本 arXiv:1909.06312。

库：

+   `pytorch_tabular`—这个库提供了一个框架，用于在表格数据上应用深度学习模型。它提供了像 TabNet、TabTransformer、FT Transformer 和带类别嵌入的前馈网络等模型。

+   `pytorch-widedeep`—基于谷歌的 Wide and Deep 算法的库。它不仅使我们能够使用深度学习处理表格数据，还方便了将文本和图像与相应的表格数据结合起来。

## 时间序列

在本章中，我们介绍了两种基于深度学习的时间序列预测方法——DeepAR 和 NeuralProphet。我们强烈推荐您还可以查阅以下有关时间序列分析和预测的资源。

进一步阅读：

+   Chen, Y., Kang, Y., Chen, Y., & Wang, Z. (2020). “基于时序卷积神经网络的概率预测”， *神经计算*，399: 491-501。

+   Gallicchio, C., Micheli, A., & Pedrelli, L. 2018\. “深度回声状态网络的设计”， *神经网络*，108: 33-47。

+   Kazemi, S. M., Goel, R., Eghbali, S., Ramanan, J., Sahota, J., Thakur, S., ... & Brubaker, M. 2019\. *Time2vec: 学习时间的向量表示*。arXiv 预印本 arXiv:1907.05321。

+   Lea, C., Flynn, M. D., Vidal, R., Reiter, A., & Hager, G. D. 2017\. 时序卷积网络用于动作分割和检测。见于 *IEEE计算机视觉与模式识别会议论文集*，156-165。

+   Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. 2021\. “用于可解释的多时间跨度时间序列预测的时序融合变换器”， *国际预测期刊*，37(4): 1748-1764。

+   Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. 2019\. *N-BEATS：用于可解释时间序列预测的神经基扩展分析*。arXiv预印本arXiv:1905.10437。

库：

+   `tsai`—这是一个建立在PyTorch和`fastai`之上的深度学习库，专注于各种时间序列相关的任务，包括分类、回归、预测和填补。除了LSTM或GRU等传统方法外，它还实现了一些最先进的架构，如ResNet、InceptionTime、TabTransformer和Rocket。

+   `gluonts`—一个用于使用深度学习进行概率时间序列建模的Python库。它包含像DeepAR、DeepVAR、N-BEATS、Temporal Fusion Transformer、WaveNet等模型。

+   `darts`—一个多功能的时间序列预测库，使用多种方法，从统计模型如ARIMA到深度神经网络。它包含了N-BEATS、Temporal Fusion Transformer和时间卷积神经网络等模型的实现。

## 其他领域

在本章中，我们重点展示了深度学习在表格数据和时间序列预测中的应用。然而，还有许多其他的应用案例和最新进展。例如，FinBERT是一个预训练的NLP模型，用于分析财务文本的情感，如财报电话会议的记录。

另一方面，我们可以利用生成对抗网络的最新进展，为我们的模型生成合成数据。以下，我们提到了一些有趣的起点，供进一步探索深度学习在金融背景下的应用。

进一步阅读：

+   Araci, D. 2019\. *Finbert：使用预训练语言模型进行财务情感分析*。arXiv预印本arXiv:1908.10063。

+   Cao, J., Chen, J., Hull, J., & Poulos, Z. 2021\. “使用强化学习进行衍生品的深度对冲”，*金融数据科学杂志*，3(1)：10-27。

+   Xie, J., Girshick, R., & Farhadi, A. 2016年6月。无监督深度嵌入用于聚类分析。在*国际机器学习大会*，478-487\. PMLR。

+   Yoon, J., Jarrett, D., & Van der Schaar, M. 2019\. 时间序列生成对抗网络。*神经信息处理系统的进展*，32。

库：

+   `tensortrade`—提供一个强化学习框架，用于训练、评估和部署交易代理。

+   `FinRL`—一个包含多种强化学习应用的生态系统，专注于金融领域。它涵盖了最先进的算法、加密货币交易或高频交易等金融应用，以及更多内容。

+   `ydata-synthetic`—一个用于生成合成表格数据和时间序列数据的库，使用的是最先进的生成模型，例如TimeGAN。

+   `sdv`—该名称代表合成数据库，顾名思义，它是另一个用于生成合成数据的库，涵盖表格、关系型和时间序列数据。

+   `transformers`—这是一个 Python 库，使我们能够访问一系列预训练的变换器模型（例如，FinBERT）。这个库背后的公司叫做 Hugging Face，它提供一个平台，允许用户构建、训练和部署机器学习/深度学习模型。

+   `autogluon`—这个库为表格数据、文本和图像提供了 AutoML。它包含了多种最先进的机器学习和深度学习模型。

# 加入我们，和我们一起在 Discord 上交流！

要加入本书的 Discord 社区——在这里你可以分享反馈、向作者提问，并了解最新的版本——请扫描下面的二维码：

![](../Images/QR_Code203602028422735375.png)

[https://packt.link/ips2H](https://packt.link/ips2H)
