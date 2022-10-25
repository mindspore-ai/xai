mindspore_xai.explainer
=================================

深度神经网络解释器。

.. py:class:: mindspore_xai.explainer.Gradient(network)

    `Gradient` 解释方法。

    `Gradient` 是最简单的归因方法，它使用输出对输入的梯度作为解释。

    .. math::

        attribution = \frac{\partial{y}}{\partial{x}}

    .. note::

        解析后的 `network` 将通过 `network.set_grad(False)` 和 `network.set_train(False)` 设为eval模式。如果想在之后训练 `network`，请通过相反的方式将其重置为训练模式。

    参数：
        - **network** (Cell) - 要解释的黑盒模型。

    输入：
        - **inputs** (Tensor) - 要解释的输入数据，shape为 :math:`(N, C, H, W)` 的4D Tensor。
        - **targets** (Tensor, int, tuple, list) - 目标分类，1D/Scalar Tensor或integer，或integer类型的tuple/list。如果是1D Tensor、tuple或list，其长度应为 :math:`N`。
        - **ret** (str, 可选) - 返回对象的类型。'tensor'表示返回Tensor ，而'image'代表返回PIL.Image.Image的list。默认值： `tensor`。
        - **show** (bool, 可选) - 显示热力图， `None` 代表自动，只会在JupyterLab上显示。默认值： `None`。

    输出：
        Tensor，shape为 :math:`(N, 1, H, W)` 的4D Tensor，热力图。如果 `ret` 设为'image'，输出list[list[PIL.Image.Image]]，归一化热力图。

    异常：
        - **TypeError** - 参数或输入类型错误。
        - **ValueError** - 输入值错误。

.. py:class:: mindspore_xai.explainer.Deconvolution(network)

    `Deconvolution` 解释方法。

    `Deconvolution` 方法是梯度方法的改进版本。它把要解释的网络的 `ReLU` 传播规则由直接反向传播梯度修改为反向传播正梯度。

    .. note::

        解析后的 `network` 将通过 `network.set_grad(False)` 和 `network.set_train(False)` 设为eval模式。如果想在之后训练 `network` ，请通过相反的方式将其重置为训练模式。在使用 `Deconvolution` 时，网络中的 `ReLU` 必须用 `mindspore.nn.Cell` 类来实现，而不是用 `mindspore.ops.Operations.ReLU` 。否则，将会导致错误结果。

    参数：
        - **network** (Cell) - 要解释的黑盒模型。

    输入：
        - **inputs** (Tensor) - 要解释的输入数据，shape为 :math:`(N, C, H, W)` 的4D Tensor。
        - **targets** (Tensor, int, tuple, list) - 目标分类。1D/Scalar Tensor、integer，或integer类型的tuple/list。如果是1D Tensor、tuple或list，其长度应与 `inputs` 一致。
        - **ret** (str, 可选) - 返回对象的类型。'tensor'代表返回 Tensor，而'image'代表返回PIL.Image.Image的list。默认值： `tensor`。
        - **show** (bool, 可选) - 显示热力图， `None` 代表自动，只会在JupyterLab上显示。默认值： `None`。

    输出：
        Tensor，shape为 :math:`(N, 1, H, W)` 的 4D Tensor。如果 `ret` 设为'image'，输出list[list[PIL.Image.Image]]，归一化热力图。

    异常：
        - **TypeError** - 参数或输入类型错误。
        - **ValueError** - 输入值错误。

.. py:class:: mindspore_xai.explainer.GuidedBackprop(network)

    `GuidedBackprop` 解释方法。

    `GuidedBackprop` 方法是梯度方法的扩展。在要解释的网络的原 `ReLU` 上，它引入了另一个 `ReLU` 来过滤反向传播期间的负梯度。

    .. note::

        解析后的 `network` 将通过 `network.set_grad(False)` 和 `network.set_train(False)` 设为eval模式。如果想在之后训练 `network` ，请通过相反的方式将其重置为训练模式。要使用 `GuidedBackprop` 时，网络中的 `ReLU` 必须用 `mindspore.nn.Cell` 类来实现，而不是用 `mindspore.ops.Operations.ReLU` 。否则，将会导致错误结果。

    参数：
        - **network** (Cell) - 要解释的黑盒模型。

    输入：
        - **inputs** (Tensor) - 要解释的输入数据，shape为 :math:`(N, C, H, W)` 的4D Tensor。
        - **targets** (Tensor, int, tuple, list) - 目标分类。1D/Scalar Tensor、integer，或integer类型的tuple/list。如果是1D Tensor、tuple或list，其长度应为 :math:`N` 。
        - **ret** (str, 可选) - 返回对象的类型。'tensor'代表返回Tensor，而'image'代表返回PIL.Image.Image的list。默认值： `tensor`。
        - **show** (bool, 可选) - 显示热力图， `None` 代表自动，只会在JupyterLab上显示。默认值： `None`。

    输出：
        Tensor，shape为 :math:`(N, 1, H, W)` 的4D Tensor，热力图。如果 `ret` 设为'image'，输出list[list[PIL.Image.Image]]，归一化热力图。

    异常：
        - **TypeError** - 参数或输入类型错误。
        - **ValueError** - 输入值错误。

.. py:class:: mindspore_xai.explainer.GradCAM(network, layer="")

    `GradCAM` 解释方法。

    `GradCAM` 会在中间层生成热力图。属性获取方式为：

    .. math::

        \alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial{y^c}}{\partial{A_{i,j}^k}}

        attribution = ReLU(\sum_k \alpha_k^c A^k)

    有关更多详情，请参考原始论文：`GradCAM <https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf>`_。

    .. note::

        解析后的 `network` 将通过 `network.set_grad(False)` 和 `network.set_train(False)` 设为eval模式。如果想在之后训练 `network` ，请通过相反的方式将其重置为训练模式。

    参数：
        - **network** (Cell) - 要解释的黑盒模型。
        - **layer** (str, 可选) - 生成解释的层名称，最好的方法是选择最后一个卷积层。如果设为''，将在输入层生成解释。默认值：''。

    输入：
        - **inputs** (Tensor) - 要解释的输入数据，shape为 :math:`(N, C, H, W)` 的4D Tensor。
        - **targets** (Tensor, int, tuple, list) - 目标分类，1D/Scalar Tensor、integer，或integer类型的tuple/list。如果是1D Tensor、tuple或list，其长度应为 :math:`N`。
        - **ret** (str, 可选) - 返回对象的类型。'tensor'代表返回Tensor，而'image'代表返回PIL.Image.Image的list。默认值： `tensor`。
        - **show** (bool, 可选) - 显示热力图， `None` 代表自动，只会在JupyterLab上显示。默认值： `None`。

    输出：
        Tensor，shape为 :math:`(N, 1, H, W)` 的4D Tensor，热力图。如果 `ret` 设为'image'，输出list[list[PIL.Image.Image]]，归一化热力图。

    异常：
        - **TypeError** - 参数或输入类型错误。
        - **ValueError** - 输入值错误。

.. py:class:: mindspore_xai.explainer.SHAPGradient(network, features, feature_names=None, class_names=None, num_neighbours=200, max_features=10)

    `SHAP gradient` 解释方法。

    使用预期梯度，即为集成梯度的扩展，以解释网络。

    .. note::

        解析后的 `network` 将通过 `network.set_grad(False)` 和 `network.set_train(False)` 设为eval模式。如果想在之后训练 `network` ，请通过相反的方式将其重置为训练模式。

    参数：
        - **network** (Cell) - 要解释的 MindSpore cell。分类模型接受shape为 :math:`(N, K)` 的2D Tensor作为输入，并输出shape为 :math:`(N, L)` 的2D Tensor。而回归模型接受shape为 :math:`(N, K)` 的2D Tensor作为输入，并输出shape为 :math:`(N)` 的1D Tensor。
        - **features** (Tensor) - shape为 :math:`(N, K)` 的2DTensor，N是样本数，而K是特征数。用于集成特征的背景数据集，接受全部或部分的训练数据集。
        - **feature_names** (list, 可选) - 训练数据中的列的名称（string）的list。默认值： `None`。
        - **class_names** (list, 可选) - 类名的list，排序根据分类器的类名排序。如果没有，类名会设为'0'、'1'、...。默认值： `None`。
        - **num_neighbours** (int, 可选) - 用于估计shap数值的子集数。默认值：200。
        - **max_features** (int, 可选) - 最多解释多少个特征。默认值：10。

    输入：
        - **inputs** (Tensor) - 要解释的输入数据，shape为 :math:`(N, K)` 的 2D float Tensor。
        - **targets** (Tensor, numpy.ndarray, list, int, 可选) - 要解释的目标分类。当 `target` 是integer时，生成该目标的归因图(attribution map)。而当 `targets` 为Tensor、numpy数组或list时，shape会是 :math:`(N, L)` ，L是每个样本的标签数量， :math:`(N,)` 或者 :math:`()` 。默认值：0。
        - **show** (bool, 可选) - 显示解释图像，`None` 代表自动，只会在JupyterLab上显示。默认值： `None`。

    输出：
        Tensor，shape为 :math:`(N, L, K)` 的3D Tensor。第一个维度代表输入。第二个维度代表目标。第三个维度代表特征的权重。

.. py:class:: mindspore_xai.explainer.SHAPKernel(predictor, features, feature_names=None, class_names=None, num_neighbours=5000, max_features=10)

    `Kernel SHAP` 解释方法。

    使用Kernel SHAP方法解释任何函数的输出。

    参数：
        - **predictor** (Cell, Callable) - 要解释的黑盒模型，一个网络或函数。分类模型接受shape为 :math:`(N, K)` 的2D 数组/Tensor作为输入，并输出shape为 :math:`(N, L)` 的2D数组/Tensor。而回归模型接受shape为 :math:`(N, K)` 的2D数组/Tensor作为输入，并输出shape为 :math:`(N)` 的1D数组/Tensor。
        - **features** (Tensor, numpy.ndarray) - 2D Tensor或 :math:`(N, K)` 的2D numpy数组，N是样本数，而K是特征数。用于集成特征的背景数据集，接受全部或部分的训练数据集。
        - **feature_names** (list, 可选) - 训练数据中的列的名称（string）的list。默认值： `None`。
        - **class_names** (list, 可选) - 类名的 list，排序根据分类器的类名排序。如果没有，类名会设为‘0’、‘1’、...。默认值： `None`。
        - **num_neighbours** (int, 可选) - 用于估计shap数值的子集数。默认值：5000。
        - **max_features** (int, 可选) - 最多解释多少个特征。默认值：10。

    输入：
        - **inputs** (Tensor, numpy.ndarray) - 要解释的输入数据，2D float Tensor或shape为 :math:`(N, K)` 的2D float numpy数组。
        - **targets** (Tensor, numpy.ndarray, list, int, 可选) - 要解释的目标分类。当 `targets` 是integer时，生成该目标的归因图。而当 `target` 是一个Tensor、numpy数组或list时，shape会是 :math:`(N, L)` ，L是每个样本的标签数量， :math:`(N,)` 或者 :math:`()` 。默认值：0。
        - **show** (bool, 可选) - 显示解释图像，`None` 代表自动，只会在JupyterLab上显示。默认值：`None`。

    输出：
        Tensor，shape为 :math:`(N, L, K)` 的3D Tensor。第一个维度代表输入。第二个维度代表目标。第三个维度代表特征的权重。

.. py:class:: mindspore_xai.explainer.Occlusion(network, activation_fn, perturbation_per_eval=32)

    `Occlusion` 解释方法。

    `Occlusion` 使用滑动窗口将像素换为一个参考值，例如常数，并计算新输出与原输出的差异。像素的重要性就是这些滑动窗口所引致的平均输出差异。

    有关更多详情，请参考原始论文：`Visualizing and Understanding Convolutional Networks <https://arxiv.org/abs/1311.2901>`_ 。

    .. note::

         目前，每个调用仅支持单个样本（ :math:`N=1` ）。

    参数：
        - **network** (Cell) - 要解释的黑盒模型。
        - **activation_fn** (Cell) - 将logits转换为预测概率的激活层。单标签分类任务通常使用 `nn.Softmax` ，而多标签分类任务较常使用 `nn.Sigmoid` 。用户也可以将自定义的 `activation_fn` 与网络结合，最终输出便是输入的概率。
        - **perturbation_per_eval** (int, 可选) - 在推理扰动样本期间，每次推理的扰动数。在内存容许情况下，通常此数字越大，便越快得到解释。默认值：32。

    输入：
        - **inputs** (Tensor) - 要解释的输入数据，shape为 :math:`(N, C, H, W)` 的4D Tensor 。
        - **targets** (Tensor, int, tuple, list) - 目标分类，1D/Scalar Tensor、integer或integer的tuple/list。如果是1D Tensor、tuple 或 list，其长度应为 :math:`N`。
        - **ret** (str, 可选) - 返回对象类型。'tensor'代表返回Tensor，而'image'代表返回PIL.Image.Image的list。默认值： `tensor`。
        - **show** (bool, 可选) - 显示热力图， `None` 代表自动，只会在JupyterLab上显示。默认值： `None`。

    输出：
        Tensor，shape为 :math:`(N, 1, H, W)` 的4D Tensor，热力图。如果 `ret` 设为'image'，输出list[list[PIL.Image.Image]]，归一化热力图。

    异常：
        - **TypeError** - 参数或输入类型错误。
        - **ValueError** - 输入值错误。

.. py:class:: mindspore_xai.explainer.RISE(network, activation_fn, perturbation_per_eval=32)

    `RISE` 解释方法：用随机输入采样来解释黑盒模型。

    `RISE` 是一种基于扰动的方法，通过在多个随机二进制掩码上采样来生成归因图。原始图像 :math:`I` 被随机屏蔽，然后输入到黑盒模型以获取预测概率，最后的归因图便是这些随机掩码 :math:`M_i` 的加权和，而权重是目标节点上的相应输出：

    .. math::
        attribution = \sum_{i}f_c(I\odot M_i)  M_i

    有关更多详情，请参考原始论文：`RISE <https://arxiv.org/abs/1806.07421>`_ 。

    参数：
        - **network** (Cell) - 要解释的黑盒模型。
        - **activation_fn** (Cell) - 将logits转换为预测概率的激活层。单标签分类任务通常使用 `nn.Softmax` ，而多标签分类任务较常使用 `nn.Sigmoid` 。用户也可以将自定义的 `activation_fn` 与网络结合，最终输出便是输入的概率。
        - **perturbation_per_eval** (int, 可选) - 推理扰动样本期间，每次推理的扰动数。在内存容许情况下，通常此数字越大，便越快得到解释。默认值：32。

    输入：
        - **inputs** (Tensor) - 要解释的输入数据，shape为 :math:`(N, C, H, W)` 的 4D Tensor。
        - **targets** (Tensor, int) - 目标分类。当 `targets` 是integer时，生成该目标的归因图。而当 `targets` 是Tensor时，shape会是 :math:`(N, L)` ，L是每个样本的标签数量，或 :math:`(N,)` :math:`()`。
        - **ret** (str, 可选) - 返回对象类型。'tensor'代表返回Tensor，'image'代表返回PIL.Image.Image的list。默认值： `tensor`。
        - **show** (bool, 可选) - 显示热力图， `None` 代表自动，只会在JupyterLab上显示。默认值： `None`。

    输出：
        Tensor，4D Tensor，当目标是shape为 :math:`(N, L)` 的Tensor时，输出的shape便会是 :math:`(N, L, H, W)` ，否则会是 :math:`(N, 1, H, W)` ，热力图。如果 `ret` 设为'image'，输出 list[list[PIL.Image.Image]]，归一化热力图。

    异常：
        - **TypeError** - 参数或输入类型错误。
        - **ValueError** - 输入值错误。

.. py:class:: mindspore_xai.explainer.RISEPlus(ood_net, network, activation_fn, perturbation_per_eval=32)

    `RISEPlus` 解释方法。

    `RISEPlus` 是一种基于扰动的方法，通过在多个随机二进制掩码上采样来生成归因图。它采用分布外检测器来产生"inlier 分数"，并用于估计从分布生成样本的概率，然后将"inlier 分数"聚合到随机掩码的加权和，而权重是目标节点上的相应输出：

    .. math::
        attribution = \sum_{i}s_if_c(I\odot M_i)  M_i

    有关更多详情，请参考原始论文： `Resisting Out-of-Distribution Data Problem in Perturbation of XAI <https://arxiv.org/abs/2107.14000>`_ 。

    参数：
        - **ood_net** (`OoDNet <https://www.mindspore.cn/xai/docs/zh-CN/master/mindspore_xai.tool.html>`_) - 用于生成"inlier 分数"的 OoD 网络。
        - **network** (Cell) - 要解释的黑盒模型。
        - **activation_fn** (Cell) - 将logits转换为预测概率的激活层。单标签分类任务通常使用 `nn.Softmax` ，而多标签分类任务较常使用 `nn.Sigmoid` 。用户还可以将自己自定义的 `activation_fn` 与网络结合，最终输出便是输入的概率。
        - **perturbation_per_eval** (int, 可选) - 在推理扰动样本期间，每次推理的扰动数。在内存容许情况下，通常此数字越大，便越快得到解释。默认值：32。

    输入：
        - **inputs** (Tensor) - 要解释的输入数据，shape为 :math:`(N, C, H, W)` 的4D Tensor。
        - **targets** (Tensor, int) - 要解释的目标分类。当 `targets` 是integer时，生成该目标的归因图。而当 `targets` 是Tensor时，shape为 :math:`(N, L)` ，L是每个样本的标签数量，或 :math:`(N,)` :math:`()`。
        - **ret** (str, 可选) - 返回对象类型。'tensor'代表返回Tensor，'image'代表返回PIL.Image.Image的list。默认值： `tensor`。
        - **show** (bool, 可选) - 显示热力图， `None` 代表自动，只会在JupyterLab上显示。默认值： `None`。

    输出：
        Tensor，4D Tensor，当目标是shape为 :math:`(N, L)` 的Tensor时，输出的shape便会是 :math:`(N, L, H, W)`，否则会是 :math:`(N, 1, H, W)`，热力图。如果 `ret` 设为'image'，输出list[list[PIL.Image.Image]]，归一化热力图。

    异常：
        - **TypeError** - 参数或输入类型错误。
        - **ValueError** - 输入值错误。

.. py:class:: mindspore_xai.explainer.LIMETabular(predictor, train_feat_stats, feature_names=None, categorical_features_indexes=None, class_names=None, num_perturbs=5000, max_features=10)

    `Lime Tabular` 解释方法。

    解释表格（即矩阵）数据的预测。数值特征会根据训练数据中的平均值和标准差，从 Normal(0,1) 分布中采样并以逆向均值中心化和缩放来进行扰动。而分类特征会根据训练分布采样进行扰动，当采样值与被解释的样本相同时，将生成一个数值为1的二进制特征。

    参数：
        - **predictor** (Cell, Callable) - 要解释的黑盒模型，一个网络或函数。分类模型接受shape为 :math:`(N, K)` 的2D 数组/Tensor作为输入，并输出shape为 :math:`(N, L)` 的2D数组/Tensor。而回归模型接受shape为 :math:`(N, K)` 的2D 数组/Tensor作为输入，并输出shape为 :math:`(N)` 的1D数组/Tensor。
        - **train_feat_stats** (dict) - 含有训练数据统计详细信息的dict对象。统计信息可以使用静态方法 `LIMETabular.to_feat_stats(training_data)` 生成。
        - **feature_names** (list, 可选) - 训练数据中的名称（string）的list。默认值： `None`。
        - **categorical_features_indexes** (list, 可选) - 分类列的索引（ints）的list，这些列中的值必须是integer。其他列将被视为连续的。默认值： `None`。
        - **class_names** (list, 可选) - 类名的list，排序根据分类器的类名排序。如果没有，类名会设为'0'、'1'、...。默认值： `None`。
        - **num_perturbs** (int, 可选) - 学习线性模型的邻域大小。默认值：5000。
        - **max_features** (int, 可选) - 最多解释多少个特征。默认值：10。

    输入：
        - **inputs** (Tensor, numpy.ndarray) - 要解释的输入数据，2D float Tensor或shape为 :math:`(N, K)` 的2D float numpy 数组。
        - **targets** (Tensor, numpy.ndarray, list, int, 可选) - 要解释的目标分类。当 `targets` 是integer时，生成该目标的归因图。而当 `targets` 是Tensor、numpy数组或list时，shape会是 :math:`(N, L)`，L是每个样本的标签数量， :math:`(N,)`或者 :math:`()`。对于回归模型，此参数将被忽略。默认值：0。
        - **show** (bool, 可选) - 显示解释图像，`None` 代表自动，只会在JupyterLab上显示。默认值： `None`。

    输出：
        list[list[list[(str, float)]]]，一个tuple类的3D list。第一个维度代表输入。第二个维度代表目标。第三个维度代表特征。tuple代表特征的描述和权重。

    异常：
        - **TypeError** - 参数或输入类型错误。
        - **ValueError** - 输入值错误。

    .. py:method:: load_feat_stats(file)

        从文件加载特征统计信息。

        参数：
            - **file** (str, Path, IOBase) - 文件路径或流。

        返回：
            dict，训练数据统计信息

    .. py:method:: save_feat_stats(stats, file)

        将特征统计信息保存到文件。

        参数：
            - **stats** (dict) - 训练数据统计信息。
            - **file** (str, Path, IOBase) - 文件路径或流。

    .. py:method:: to_feat_stats(features, feature_names=None, categorical_features_indexes=None)

        将特征转换为特征统计信息。

        参数：
            - **features** (Tensor, numpy.ndarray) - 训练数据。
            - **feature_names** (list, 可选) - 特征名称。默认值： `None`。
            - **categorical_features_indexes** (list, 可选) - 分类列的索引（ints）的list，这些列中的值必须是integer。其他列将被视为连续的。默认值：`None`。

        返回：
            dict，训练数据统计信息。

.. py:class:: mindspore_xai.explainer.PseudoLinearCoef(predictor, num_classes, class_names=None, feature_names=None, stepwise=False, threshold=0.5, monte_carlo=1000, riemann=1000, batch_size=2000, eps=1e-9)

    分类器的伪线性系数（PLC）。

    伪线性系数是一个全局归因方法，从数据分布的角度来看，它用来度量分类器决策边界周围的特征敏感度。

    A类的伪线性系数：

    .. math::

        \vec{R}(A)=\int \vec{S}(A,nearest_{A}(x),x))p_{\neg A}(x)dx

    A类（目标类）相对于B类（视点类）的伪线性系数，称为相对伪线性系数:

    .. math::

        \vec{R}(A,B)=\int \vec{S}(A,nearest_{A}(x),x))p_{B}(x)dx

    where:

    .. math::

        nearest_A(x):=\underset{g\in G}{argmin}(\left \| g-x \right \|)\text{ }s.t.\text{ } g\neq x,f_A(g)
        \geq \xi

        \vec{S}(A,a,x)=\left\{\begin{matrix}
        \vec{0} & \text{if }f_A(x)\geq \xi \\
        \frac{a-x}{\left \| a-x \right \|} & \text{if }f_A(\cdot )\text{ is a step function}\\
        \frac{(a-x)(f_{A}(a)-f_A(x))}{\left \| a-x \right \|^{2}\int_{0}^{1}h(f_A(u(t)))dt} & \text{else}
        \end{matrix}\right.

    .. math::

        u(t)=ta+(1-t)x

    .. math::

        h(f_{A})=-f_{A}log_2(f_{A})-(1-f_A)log_2(1-f_A)

    :math:`G` 代表样本全集， :math:`f_A(\cdot )` 代表A类的预测概率， :math:`\xi` 代表决策阀值，通常设为0.5。
    :math:`p_{\neg A}` 和 :math:`p_{B}` 分别代表非A类和B类的样本分布的概率密度函數。请注意在伪线性系数中样本的类别是由分类器决定，
    而不是使用ground truth标签。

    .. note::
        如果 `predictor` 是一个函数， `stepwise` 是 `False` 和在graph mode上运行， `predictor` 必须符合
        `static graph syntax <https://mindspore.cn/docs/en/master/note/static_graph_syntax_support.html>`_ 的语法。
        如果有很多样本被分类到多于一个类别，PLC可能会不准确。

    参数：
        - **predictor** (Cell, Callable) - 要解释的分类器 :math:`f(\cdot )` ，输入只接受一个shape为 :math:`(N, K)` 的Tensor，并输出一个shape为 :math:`(N, L)` 的概率Tensor。 :math:`K` 是特征的数量，输入和输出的Tensor dtype只能是 `ms.float32` 或 `ms.float64`。
        - **num_classes** (int) - 类的数量 :math:`L`。
        - **class_names** (list[str], tuple[str], 可选) - 类名的list/tuple，排序根据分类器的类名排序。如果没有，类名会设为'Class 0'、'Class 1'、...。默认值： `None`。
        - **feature_names** (list[str], tuple[str], 可选) - 训练数据中的名称的list/tuple。如果没有，类名会设为'feature 0'、'feature 1'、...。默认值： `None`。
        - **stepwise** (bool, 可选) - 如果 `predictor` 只输出0和1，请设置为 `True`。默认值： `False`。
        - **threshold** (float, 可选) - 分类的决策阀值 :math:`\xi` 。默认值：0.5。
        - **monte_carlo** (int, 可选) - 计算积分 :math:`\vec{R}` 的蒙特卡洛样本的数量。默认值：1000。数值越大，计算时间就越长和越准确。
        - **riemann** (int, 可选) - 计算积分 :math:`\int_{0}^{1}h(f_A(u(t)))dt` 的黎曼和分割数量。默认值：1000。数值越大，计算时间就越长和越准确。
        - **batch_size** (int, 可选) - 寻找最近的样本时 `predictor` 的批量大小。默认值：2000。
        - **eps** (float, 可选) - 误差范围。数值必须大于0。默认值：1e-9。

    输入：
        - **features** (Tensor) - 样本全集 :math:`G`。实际上，它通常是训练集或其随机子集，shape为 :math:`(|G|, K)`，
          :math:`|G|` 是样本的总数，输入的Tensor dtype只能是 `ms.float32` 或 `ms.float64`。
        - **max_classes** (int, 可选) - 最多显示多少个类。默认值：5。
        - **max_features** (int, 可选) - 最多显示多少个特征。默认值：5。
        - **show** (bool, 可选) - 显示解释图像，`None` 代表自动，只会在JupyterLab上显示。默认值： `None`。

    输出：
        - **plc** (Tensor) - shape为 :math:`(L, K)` 的伪线性系数。
        - **relative plc** (Tensor) - shape为 :math:`(L, L, K)` 的相对伪线性系数。第一个 :math:`L` 轴代表目标类，而第二个代表视点类。

    异常：
        - **TypeError** - 参数或输入类型错误。
        - **ValueError** - 输入值错误。
        - **AttributeError** - underlying缺少必需的属性。

    .. py:method:: normalize(plc, per_vector=False, eps=1e-9)

        归一化伪线性系数到[-1, 1]范围。

        .. warning::
            把从未归一化特征产生的伪线性系数归一化可能会引致误导结果。

        参数：
            - **plc** (Tensor) - 要归一化的伪线性系数或相对伪线性系数。
            - **per_vector** (bool, 可选) - 归一化 :math:`\vec{R}` 向量。默认值： `False`。
            - **eps** (float, 可选) - 误差范围。数值必须大于0。默认值：1e-9。

        返回：
            Tensor，归一化的数值。

    .. py:method:: plot(plc, title=None, feature_names=None, max_features=5)

        显示指定的伪线性系数或相对伪线性系数向量的图表。

        参数：
            - **plc** (Tensor) - 要显示的伪线性系数或相对伪线性系数向量，shape为 :math:`(K,)`。
            - **title** (str, 可选) - 图表标题。如果没有，则不会显示图表标题。默认值：`None`。
            - **feature_names** (list[str], tuple[str], 可选) - 特征名称。如果没有，特征名称将为'feature 0'、'feature 1'、...。默认值：`None`。
            - **max_features** (int, 可选) - 最多显示多少个特征。默认值：5。

        异常：
            - **ValueError** - 输入值错误。
