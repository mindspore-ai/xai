mindspore_xai.explainer
=================================

解释器。

.. py:class:: mindspore_xai.explainer.Gradient(network)

    `Gradient` 解释方法。

    `Gradient` 是最简单的归因方法，它使用基于输入的梯度作为解释。

    .. math::

        attribution = \frac{\partial{y}}{\partial{x}}

    .. note::

        解析后的 `network` 将通过 `network.set_grad(False)` 和 `network.set_train(False)` 设置为eval模式。如果想在之后训练 `network`，请通过相反的方式将其重置为训练模式。

    **参数：**

    - **network** (Cell) - 要解释的黑盒模型。

    **输入：**

    - **inputs** (Tensor) - 要解释的输入数据，shape 为 :math:`(N, C, H, W)` 的 4D Tensor。
    - **targets** (Tensor, int, tuple, list) - 目标分类， 1D/0D Tensor 或 integer，或integer类型的tuple/list。如果是 1D 的 Tensor、tuple 或 list，其长度应为 :math:`N`。
    - **ret** (str) - 返回对象的类型。'tensor'表示返回 Tensor ，'image'表示返回一个PIL.Image.Image list。默认值：'tensor'。
    - **show** (bool, 可选) - 显示热力图（saliency map)，`None` 表示自动。默认值：`None` 。

    **输出：**

    Tensor，shape 为 :math:`(N, 1, H, W)` 的 4D Tensor，热力图。或如果 `ret` 设置为'image'，则输出 list[list[PIL.Image.Image]]，归一化热力图。

    **异常：**

    - **TypeError** - 在出现任何参数类型问题时抛出。
    - **ValueError** - 在任何输入的值出现问题时抛出。

.. py:class:: mindspore_xai.explainer.Deconvolution(network)

    `Deconvolution` 解释方法。

    `Deconvolution` 方法是梯度方法的改进版本。对于要解释的网络原始 `ReLU` 操作，反卷积将传播规则从直接反向传播梯度修改为反向传播正梯度。

    .. note::

        解析后的 `network` 将通过 `network.set_grad(False)` 和 `network.set_train(False)` 设置为eval模式。如果您想在之后训练 `network` ，请通过相反的方式将其重置为训练模式。要使用 `Deconvolution` 时，网络中的 `ReLU` 操作必须使用 `mindspore.nn.Cell` ，而不是 `mindspore.ops.Operations.ReLU` ，否则，将会导致错误结果。

    **参数：**

    - **network** (Cell) - 要解释的黑盒模型。

    **输入：**

    - **inputs** (Tensor) - 要解释的输入数据，shape 为 :math:`(N, C, H, W)` 的 4D Tensor。
    - **targets** (Tensor, int, tuple, list) - 目标分类，1D 、 0D Tensor、integer 或 integer类型的tuple/list。如果是 1D Tensor、tuple 或 list，其长度应与 `inputs` 一致。
    - **ret** (str) - 返回对象的类型。'tensor'表示返回 Tensor，'image'表示返回一个PIL.Image。默认值：'tensor'。
    - **show** (bool) - 如果在 JupiterLab 或 Notebook 上运行，则自动显示热力图。默认值：`True`。

    **输出：**

    Tensor，shape 为 :math:`(N, 1, H, w)` 的 4D Tensor。或如果 `ret` 设置为 'image'，则输出list[PIL.Image.Image]，归一化热力图。

    **异常：**

    - **TypeError** - 在出现任何参数或输入类型问题时抛出。
    - **ValueError** - 在任何输入的值出现问题时抛出。

.. py:class:: mindspore_xai.explainer.GuidedBackprop(network)

    `GuidedBackprop` 解释方法。

    `GuidedBackprop` 方法是梯度方法的扩展。在要解释的网络原始 `ReLU` 操作之上，引导反向传播引入了另一个 `ReLU` 操作来过滤反向传播期间的梯度。

    .. note::

        解析后的 `network` 将通过 `network.set_grad(False)` 和 `network.set_train(False)` 设置为eval模式。如果想在之后训练 `network` ，请通过相反的方式将其重置为训练模式。要使用 `GuidedBackprop` 时，网络中的 `ReLU` 操作必须使用 `mindspore.nn.Cell` ，而不是 `mindspore.ops.Operations.ReLU` ，否则，将会导致错误结果。

    **参数：**

    - **network** (Cell) - 要解释的黑盒模型。

    **输入：**

    - **inputs** (Tensor) - 要解释的输入数据，shape 为 :math:`(N, C, H, W)` 的 4D Tensor。
    - **targets** (Tensor, int, tuple, list) - 目标分类， 1D/0D Tensor、integer 或 integer类型的tuple/list。如果是 1D Tensor、tuple 或 list，其长度应为 :math:`N` 。
    - **ret** (str) - 返回对象的类型。'tensor'表示返回Tensor，'image'表示返回一个 PIL.Image.Image list。默认值： 'tensor'。
    - **show** (bool, 可选) - 如果在 JupiterLab 或 Notebook 上运行，则自动显示热力图。默认值： `None` 。

    **输出：**

    Tensor，shape 为 :math:`(N, 1, H, W)` 的 4D Tensor，热力图。或如果 `ret` 设置为'image'，则输出list[list[PIL.Image.Image]]，归一化热力图。

    **异常：**

    - **TypeError** - 在出现任何参数或输入类型问题时抛出。
    - **ValueError** - 在任何输入的值出现问题时抛出。

.. py:class:: mindspore_xai.explainer.GradCAM(network, layer="")

    `GradCAM` 解释方法。

    `GradCAM` 在中间层生成热力图。属性获取方式为：

    .. math::

        \alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial{y^c}}{\partial{A_{i,j}^k}}

        attribution = ReLU(\sum_k \alpha_k^c A^k)

    欲了解更多详细信息，请参考原始论文：`GradCAM <https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf>`_。

    .. note::

        解析后的 `network` 将通过 `network.set_grad(False)` 和 `network.set_train(False)` 设置为eval模式。如果想在之后训练 `network` ，请通过相反的方式将其重置为训练模式。

    **参数：**

    - **network** (Cell) - 要解释的黑盒模型。
    - **layer** (str, 可选) - 生成解释的层名称，通常选择为最后一个卷积层以更好地练习。如果是''，则将在输入层生成解释。默认值：''。

    **输入：**

    - **inputs** (Tensor) - 要解释的输入数据，shape 为 :math:`(N, C, H, W)` 的 4D Tensor。
    - **targets** (Tensor, int, tuple, list) - 目标分类， 1D/0D Tensor、integer 或 integer类型的tuple/list。如果是 1D Tensor、tuple 或 list，其长度应为 :math:`N`。
    - **ret** (str) - 返回对象的类型。'tensor'表示返回Tensor，'image'表示返回一个PIL.Image.Image list。默认值：'tensor'。
    - **show** (bool, 可选) - 显示热力图， `None` 表示自动。默认值： `None` 。

    **输出：**

    Tensor，shape 为 :math:`(N, 1, H, W)` 的 4D Tensor，热力图。或如果 `ret` 设置为'image'，则输出 list[list[PIL.Image.Image]]，归一化热力图。

    **异常：**

    - **TypeError** - 在出现任何参数或输入类型问题时抛出。
    - **ValueError** - 在任何输入的值出现问题时抛出。

.. py:class:: mindspore_xai.explainer.SHAPGradient(network, features, feature_names=None, class_names=None, num_neighbours=200, max_features=10)

    `SHAPGradient` 解释方法。

    使用预期梯度（集成梯度的扩展）解释网络。

    .. note::

        解析后的 `network` 将通过 `network.set_grad(False)` 和 `network.set_train(False)` 设置为eval模式。如果想在之后训练 `network` ，请通过相反的方式将其重置为训练模式。

    **参数：**

    - **network** (Cell) - 要解释的 MindSpore cell。对于分类，它接受 shape 为 :math:`(N, K)` 的 2D 数组/Tensor 作为输入并输出 shape 为 :math:`(N, L)` 的 2D 数组/Tensor。对于回归，它接受 shape 为 :math:`(N, K)` 的2D 数组/Tensor作为输入，并输出 shape 为 :math:`(N)` 的 1D 数组/Tensor。
    - **features** (Tensor) - shape 为 :math:`(N, K)` 的 2D Tensor(N 是样本数，而K是特征数)。用于集成特征的背景数据集，接受全部或部分的训练数据集。
    - **feature_names** (list, 可选) - 与训练数据中的列相对应的名称（string）的 list。默认值： `None` 。
    - **class_names** (list, 可选) - 类名的 list，根据分类器使用的内容而排序。如果不存在，类名将为'0'、'1'、... 默认值： `None` 。
    - **num_neighbours** (int, 可选) - 用于估计 shap 数值的子集数。默认值：200。
    - **max_features** (int, 可选) - 最多解释多少个特征。默认值：10。

    **输入：**

    - **inputs** (Tensor) - 要解释的输入数据，shape 为 :math:`(N, K)` 的2D float Tensor。
    - **targets** (Tensor, numpy.ndarray, list, int, 可选) - 要解释的目标分类。当 `target` 为 integer，所有输入都将参考它生成归因图 (attribution map)。而当 `target` 为 Tensor、numpy 数组 或 list 时，它的 shape 则为 :math:`(N, L)` (L是每个样例的标签数量)， :math:`(N,)` 或者 :math:`()` 。默认值：0。
    - **show** (bool, 可选) - 显示解释图像， `None` 表示自动。默认值： `None` 。

    **输出：**

    Tensor，shape 为 :math:`(N, L, K)` 的 3D Tensor。第一个维度表示输入。第二个维度表示目标。第三个维度表示特征权重。

.. py:class:: mindspore_xai.explainer.SHAPKernel(predictor, features, feature_names=None, class_names=None, num_neighbours=5000, max_features=10)

    `SHAPKernel` 解释方法。

    使用 `SHAPKernel` 方法解释任何函数的输出。

    **参数：**

    - **predictor** (Callable) - 要解释的黑盒模型，一个可调用的函数。对于分类模型，它接受 shape 为 :math:`(N, K)` 的2D 数组 / Tensor 作为输入，并输出shape 为 :math:`(N, L)` 的 2D 数组 / Tensor 。对于回归模型， 它接受 shape 为 :math:`(N, K)` 的2D 数组/Tensor作为输入并输出 shape 为 :math:`(N)` 的1D 数组/Tensor。
    - **features** (Tensor, numpy.ndarray) - 2D Tensor 或 :math:`(N, K)` 的2D numpy 数组 (N是样本，K是特征的数量)。用于集成特征的背景数据集，接受全部或部分的训练数据集。
    - **feature_names** (list, 可选) - 与训练数据中的列相对应的名称（string）的 list。默认值： `None` 。
    - **class_names** (list, 可选) - 类名的 list，根据分类器使用的任何内容排序。如果不存在，类名将为‘0’、‘1’、... 默认值： `None` 。
    - **num_neighbours** (int, 可选) - 用于估计 shap 数值的子集数。默认值：5000。
    - **max_features** (int, 可选) - 最多解释多少个特征。默认值：10。

    **输入：**

    - **inputs** (Tensor, numpy.ndarray) - 要解释的输入数据，2D float Tensor 或 shape 为 :math:`(N, K)` 的 2D float numpy 数组 。
    - **targets** (Tensor, numpy.ndarray, list, int, 可选) - 要解释的目标分类。当 `targets` 为integer时，所有输入都将参考该integer生成归因图。而当 `target` 为 Tensor、numpy 数组 或 list 时，它的 shape 则为 :math:`(N, L)`(L是每个样例的标签数量)， :math:`(N,)`或者 :math:`()`。默认值：0。
    - **show** (bool, 可选) - 显示解释图像， `None` 表示自动。默认值：`None` 。

    **输出：**

    Tensor，shape 为 :math:`(N, L, K)` 的 3D Tensor。第一个维度表示输入。第二个维度表示目标。第三个维度表示特征权重。

.. py:class:: mindspore_xai.explainer.Occlusion(network, activation_fn, perturbation_per_eval=32)

    `Occlusion` 解释方法。

    `Occlusion` 使用滑动窗口将像素替换为参考值（例如恒定值），并参考原始输出计算差异。由扰动像素引起的输出差异被指定为特征对这些像素的重要性。对于多个滑动窗口中涉及的像素，特征重要性为多个滑动窗口的平均差异。

    欲了解更多详情，请参考原始文件：`<https://arxiv.org/abs/1311.2901>`_ 。

    .. note::

         目前，每个调用仅支持单个样本（ :math:`N=1` ）。

    **参数：**

    - **network** (Cell) - 要解释的黑盒模型。
    - **activation_fn** (Cell) - 将 logits 转换为预测概率的激活层。对于单标签分类任务，通常应用 `nn.Softmax` 。而对于多标签分类任务，则通常应用 `nn.Sigmoid` 。用户也可以将自定义的 `activation_fn` 与网络结合，最终输出便是输入的概率。
    - **perturbation_per_eval** （int，可选） - 推断扰动样本期间，每个推断的扰动数。在内存容量内，通常此数字越大，越快得到解释。默认值：32。

    **输入：**

    - **inputs** (Tensor) - 要解释的输入数据，shape 为 :math:`(N, C, H, W)` 的 4D Tensor 。
    - **targets** (Tensor, int, tuple, list) - 目标分类，1D、 0D Tensor、integer 或 integer的tuple/list。如果为 1D Tensor、tuple 或 list，其长度应为 :math:`N`。
    - **ret** (str) - 返回对象类型。'tensor'表示返回Tensor，'image'表示返回一个PIL.Image.Image list。默认值：'tensor'。
    - **show** (bool, 可选) - 显示热力图， `None` 表示自动。默认值： `None` 。

    **输出：**

    Tensor，shape 为 :math:`(N, 1, H, W)` 的 4D Tensor ，热力图。或如果 `ret` 设置为'image'，则输出list[list[PIL.Image.Image]]，归一化热力图。

    **异常：**

    - **TypeError** - 在出现任何参数或输入类型问题时抛出。
    - **ValueError** - 在任何输入的值出现问题时抛出。

.. py:class:: mindspore_xai.explainer.RISE(network, activation_fn, perturbation_per_eval=32)

    `RISE` 解释方法：用于解释黑盒模型的随机输入采样。

    `RISE` 是一种基于摄动的方法，通过在多个随机二进制掩码上采样来生成归因图。原始图像被随机屏蔽，然后馈入黑盒模型以获得预测，最后的归因图便是这些随机掩码的加权和，权重是目标的节点：

    .. math::
        attribution = \sum_{i}f_c(I\odot M_i)  M_i

    有关更多详细信息，请参考原始文件：`RISE <https://arxiv.org/abs/1806.07421>`_ 。

    **参数：**

    - **network** (Cell) - 要解释的黑盒模型。
    - **activation_fn** (Cell) - 将 logits 转换为预测概率的激活层。对于单标签分类任务，通常应用 `nn.Softmax` 。而对于多标签分类任务，则通常应用 `nn.Sigmoid` 。用户也可以将自定义的 `activation_fn` 与网络结合，最终输出便是输入的概率。
    - **perturbation_per_eval** (int, 可选) - 推断扰动样本期间，每个推断的扰动数。在内存容量内，通常此数字越大，越快得到解释。默认值：32。

    **输入：**

    - **inputs** (Tensor) - 要解释的输入数据，shape 为 :math:`(N, C, H, W)` 的4DTensor。
    - **targets** (Tensor, int) - 目标分类。当 `targets` 为 integer 时，所有输入都将参考此 integer 生成归因图。而当 `targets` 为 Tensor 时， shape 应为 :math:`(N, l)` （l是每个样本的标签数量）或 :math:`(N,)` :math:`()`。
    - **ret** (str) - 返回对象类型。'tensor'表示返回Tensor，'image'表示返回一个PIL.Image.Image list。默认值：'tensor'。
    - **show** (bool, 可选) - 显示热力图， `None` 表示自动。默认值： `None` 。

    **输出：**

    Tensor，4D Tensor，当目标的 shape 为（N,l）Tensor时，其 shape 为 :math:`(N, l, H, W)` ，否则为 :math:`(N, 1, H, W)` ，热力图。或如果 `ret` 设置为'image'，则输出 list[list[PIL.Image.Image]]，归一化热力图。

    **异常：**

    - **TypeError** - 在出现任何参数或输入类型问题时抛出。
    - **ValueError** - 在任何输入的值出现问题时抛出。

.. py:class:: mindspore_xai.explainer.RISEPlus(ood_net, network, activation_fn, perturbation_per_eval=32)

    `RSPlus` 解释方法。

    `RSPlus` 是一种基于扰动的方法，通过对多个随机二进制进行采样来生成归因图掩码。采用分布外检测器来产生"内值分数"，估计样本的概率从分布生成，然后将内值分数聚合到随机掩码的加权和，权重是目标节点上的相应输出：

    .. math::
        attribution = \sum_{i}s_if_c(I\odot M_i)  M_i

    有关更多详细信息，请参考原始论文： `Resisting Out-of-Distribution Data Problem in Perturbation of XAI <https://arxiv.org/abs/2107.14000>`_ 。

    **参数：**

    - **ood_net** (`OoDNet <https://www.mindspore.cn/xai/docs/zh-CN/master/mindspore_xai.tool.html>`_) - 用于生成内值分数的 OoD 网络。
    - **network** (Cell) - 要解释的黑盒模型。
    - **activation_fn** (Cell) - 将 logits 转换为预测概率的激活层。对于单标签分类任务，通常应用 `nn.Softmax` 。而对于多标签分类任务，则通常应用 `nn.Sigmoid` 。用户还可以将自己自定义的`activation_fn`传递为只要将此函数与网络结合时，最终输出是输入的概率。
    - **perturbation_per_eval** (int, 可选) - 推断扰动样本期间，每个推断的扰动数。在内存容量内，通常此数字越大，越快得到解释。默认值：32。

    **输入：**

    - **inputs** (Tensor) - 要解释的输入数据，shape 为 :math:`(N, C, H, W)` 的4D Tensor。
    - **targets** (Tensor, int) - 要解释的目标分类。当 `targets` 为integer时，所有输入都将参考此integer生成归因图。而当 `targets` 为 Tensor 时，shape 则为 :math:`(N, l)`（l是每个样本的标签数量）或 :math:`(N,)` :math:`()`。
    - **ret** (str) - 返回对象类型。'tensor'表示返回 Tensor，'image'表示返回一个PIL.Image.Image list。默认值：'tensor'。
    - **show** (bool, 可选) - 显示热力图， `None` 表示自动。默认值：`None` 。

    **输出：**

    Tensor，4DTensor，当目标的 shape 为 :math:`(N, l)`时，其 shape 为 :math:`(N, l, H, W)`，否则为 :math:`(N, 1, H, w)`，热力图。或如果 `ret` 设置为'image'，则输出list[list[PIL.Image.Image]]，归一化热力图。

    **异常：**

    - **TypeError** - 在出现任何参数或输入类型问题时抛出。
    - **ValueError** - 在任何输入的值出现问题时抛出。

.. py:class:: mindspore_xai.explainer.LIMETabular(predictor, train_feat_stats, feature_names=None, categorical_features_indexes=None, class_names=None, num_perturbs=5000, max_features=10)

    `Lime Tabular` 解释方法。

    解释表格（即矩阵）数据的预测。对于数值特征，根据训练数据中的平均值和标准差，通过从 Normal(0,1) 采样并进行均值中心化和缩放的逆运算来扰乱它们。而对于分类特征，则根据训练分布采样进行扰动，当它的值与被解释的实例相同时，生成一个数值为 1 的二进制特征。

    **参数：**

    - **predictor** (Callable) - 要解释的黑盒模型，一个可调用的函数。对于分类模型，它接受 shape 为 :math:`(N, K)` 的2D 数组/Tensor作为输入，并输出 shape 为 :math:`(N, L)` 的2D数组/Tensor。对于回归模型，它接受 shape 为 :math:`(N, K)` 的2D数组 /Tensor作为输入并输出 shape 为 :math:`(N)` 的 1D 数组/Tensor。
    - **train_feat_stats** (dict) - 具有训练数据统计详细信息的dict对象。统计可以使用静态方法 `LIMETabular.to_feat_stats(training_data)` 生成。
    - **feature_names** (list, 可选) - 与训练数据中的列相对应的名称（string）的 list。默认值： `None` 。
    - **categorical_features_indexes** (list, 可选) - 与分类列相对应的索引（ints）的 list。其他的一切都将被视为连续的。这些列中的值必须是integer。默认值： `None` 。
    - **class_names** (list, 可选) - 类名的 list，根据分类器使用的任何东西排序。如果不存在，类名将为"0"、"1"、... 默认值： `None` 。
    - **num_perturbs** (int, 可选) - 学习线性模型的邻域大小。默认值：5000。
    - **max_features** (int, 可选) - 最多解释多少个特征。默认值：10。

    **输入：**

    - **inputs** (Tensor, numpy.ndarray) - 要解释的输入数据，2D float Tensor 或 shape 为 :math:`(N, K)` 的 2D float numpy 数组。
    - **targets** (Tensor, numpy.ndarray, list, int, 可选) - 要解释的目标分类。当 `targets` 为integer时，所有输入都将参考此integer生成归因图。而当 `targets` 为 Tensor、numpy 数组 或 list 时，shape 则为 :math:`(N, L)`(L是每个样例的标签数量)，:math:`(N,)`或者 :math:`()`。对于回归模型，此参数将被忽略。默认值：0。
    - **show** (bool, 可选) - 显示解释图像， `None` 表示自动。默认值： `None` 。

    **输出：**

    list[list[list[(str, float)]]]，tuple 的 3D list。第一个维度表示输入。第二个维度表示目标。第三个维度表示特征。tuple 表示特征描述和权重。

    **异常：**

    - **TypeError** - 在出现任何参数或输入类型问题时抛出。
    - **ValueError** - 在任何输入的值出现问题时抛出。

    .. py:method:: load_feat_stats(file)

        从文件加载特征统计信息。

        **参数：**

        - **file** (str, Path, IOBase) - 文件路径或流。

        **返回：**

        dict，训练数据统计信息

    .. py:method:: save_feat_stats(stats, file)

        将特征统计信息保存到文件。

        **参数：**

        - **stats** (dict) - 训练数据统计信息。
        - **file** (str, Path, IOBase) - 文件路径或流。

    .. py:method:: to_feat_stats(features, feature_names=None, categorical_features_indexes=None)

        将特征转换为特征统计信息。

        **参数：**

        - **features** (Tensor, numpy.ndarray) - 训练数据。
        - **feature_names** (list, None) - 特征名称。
        - **categorical_features_indexes** (list, 可选) - 与分类列对应的索引 list(ints)。所有内容都将被视为连续的，这些列中的值必须是integer。默认值：`None` 。

        **返回：**

        dict，训练数据统计信息。