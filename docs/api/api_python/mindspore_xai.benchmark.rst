mindspore_xai.benchmark
=================================

预定义的XAI指标。

.. py:class:: mindspore_xai.benchmark.ClassSensitivity

    类敏感度（ClassSensitivity）用于度量归因类的解释器。

    合理的归因类解释器应为不同标签生成不同的热力图，特别是对"高置信度"和"低置信度"的标签，类敏感度通过计算这两种标签之间的热力图相关性来评估解释器，类敏感度较好的解释器将获得较低的相关性分数。而为了使评估结果直观，返回的分数将取相关性的负值并归一化。

    .. py:method:: evaluate(explainer, inputs)

        评估解释器的类敏感度。

        .. note::
             目前，每个调用仅支持单个样本（ :math:`N=1` ）。

        参数：
            - **explainer** (Explainer) - 要评估的解释器，请参见 `mindspore_xai.explainer` 。
            - **inputs** (Tensor) - 数据样本，shape为 :math:`(N, C, H, W)` 的4D Tensor。

        返回：
            numpy.ndarray，shape为 :math:`(N,)` 的1D数组，为 `explainer` 的类敏感度评估结果。

        异常：
            - **TypeError** - 参数或输入类型错误。
            - **ValueError** - :math:`N` 不是1。

.. py:class:: mindspore_xai.benchmark.Faithfulness(num_labels, activation_fn, metric="NaiveFaithfulness")

    评估XAI解释的忠实度（Faithfulness）。

    支持三个量化指标："NaiveFaithfulness"，"DeletionAUC" 和 "InsertionAUC"。

    "NaiveFaithfulness"指标是通过修改原始图像上的像素来创建一系列扰动图像，把这些图像输入模型會令预测概率下降，而在概率下降和热力图数值两者之间的相关性便是忠实度数值，最后我们会归一化相关性，使它们在[0, 1]的范围内。

    "DeletionAUC"指标是通过将原始图像的像素累积地修改为基本数值，例如用一个常数，来创建一系列扰动图像，扰动会從高显著值的像素開始再到低显著值的像素，并将这些图像按顺序输入到模型，从而得到输出概率的下降曲线，"DeletionAUC"为该曲线下的面积。

    "InsertionAUC"指标是通过将原始图像的像素累积地插入参考图像，例如用黑色图像，来创建一系列扰动图像，插入会从高显着值的像素开始再到低显着值的像素，并将这些图像按顺序输入到模型，从而得到输出概率的增长曲线，"InsertionAUC"为该曲线下的面积。

    对于这三个指标，值越高表示忠诚度越高。

    参数：
        - **num_labels** (int) - 标签的数量。
        - **activation_fn** (Cell) - 将logits转换为预测概率的激活层。单标签分类任务通常使用 `nn.Softmax` ，而多标签分类任务较常使用 `nn.Sigmoid` 。用户也可以将自定义的 `activation_fn` 与网络结合，而最终的输出便是输入的概率。
        - **metric** (str, 可选) - 量化忠诚度的特定指标。可选项："DeletionAUC"，"InsertionAUC"，"NaiveFaithfulness"。默认值："NaiveFaithfulness"。

    异常：
        - **TypeError** - 参数或输入类型错误。

    .. py:method:: evaluate(explainer, inputs, targets, saliency=None)

        评估解释器的忠诚度。

        .. note::
            目前，每个调用仅支持单个样本（ :math:`N=1` ）。

        参数：
            - **explainer** (Explainer) - 要评估的解释器，请参见 `mindspore_xai.explainer` 。
            - **inputs** (Tensor) - 数据样本，shape为 :math:`(N, C, H, W)` 的4D Tensor。
            - **targets** (Tensor, int) - 目标分类，1D/Scalar Tensor或integer。如果 `targets` 是1D Tensor，其长度应为 :math:`N` 。
            - **saliency** (Tensor, 可选) - 要评估的热力图，shape为 :math:`(N, 1, H, W)` 的4D Tensor。如果设置为 `None` ，解析后的 `explainer` 将生成具有 `inputs` 和 `targets` 的热力图，并继续评估。默认值：`None`。

        返回：
            numpy.ndarray，shape为 :math:`(N,)` 的1D数组，为 `explainer` 的忠实度评估结果。

        异常：
            - **TypeError** - 参数或输入类型错误。
            - **ValueError** - :math:`N` 不是1。

.. py:class:: mindspore_xai.benchmark.Localization(num_labels, metric="PointingGame")

    评估XAI方法的定位性（Localization）能力。

    支持两个量化指标："PointingGame" 和 "IoSR"（显著区域的相交）。

    "PointingGame"指标会计算热力图峰值位置位于边界框内的比例。具体来说，如果单个样本的热力图的峰值位置位于边界框内，结果为1，否则为0。

    "IoSR"指标是边界框和显着区域的相交面积除以显着区域面积。显着区域是指显着值高于 :math:`\theta * \max{saliency}`。

    参数：
        - **num_labels** (int) - 数据集中的类数。
        - **metric** (str，可选) - 计算定位性能力的特定指标。可选项："PointingGame"和"IoSR"。默认值："PointingGame"。

    异常：
        - **TypeError** - 参数或输入类型错误。

    .. py:method:: evaluate(explainer, inputs, targets, saliency=None, mask=None)

        评估解释器的定位性。

        .. note::

            目前，每个调用仅支持单个样本（ :math:`N=1` ）。

        参数：
            - **explainer** (Explainer) - 要评估的解释器，请参见 `mindspore_xai.explainer` 。
            - **inputs** (Tensor) - 数据样本，shape为 :math:`(N, C, H, W)` 的4D Tensor。
            - **targets** (Tensor, int) - 目标分类，1D/Scalar Tensor或integer。如果 `targets` 是1D Tensor，其长度应为 :math:`N` 。
            - **saliency** (Tensor, 可选) - 要评估的热力图，shape为 :math:`(N, 1, H, W)` 的4D Tensor。如果设置为 `None` ，解析后的 `explainer` 将生成具有 `inputs` 和 `targets` 的热力图，并继续评估。默认值： `None` 。
            - **mask** (Tensor, numpy.ndarray, 可选) - 基于 `targets` 给于 `inputs` 的ground truth边界框/掩码，4D Tensor或shape为 :math:`(N, 1, H, W)` 的 `numpy.ndarray` 。默认值： `None` 。

        返回：
            numpy.ndarray，shape为 :math:`(N,)` 的1D数组，为 `explainer` 的定位性评估结果。

        异常：
            - **TypeError** - 参数或输入类型错误。
            - **ValueError** - :math:`N` 不是1。

.. py:class:: mindspore_xai.benchmark.Robustness(num_labels, activation_fn)

    鲁棒性 (Robustness) 通过添加随机噪音来扰动输入，并从扰动中选择最大灵敏度作为评估分数。

    参数：
        - **num_labels** (int) - 数据集中的类数。
        - **activation_fn** (Cell) - 将logits转换为预测概率的激活层。单标签分类任务通常使用 `nn.Softmax` ，而多标签分类任务较常使用 `nn.Sigmoid` 。用户也可以将自定义的 `activation_fn` 与网络结合，最终的输出便是输入的概率。

    异常：
        - **TypeError** - 参数或输入类型错误。

    .. py:method:: evaluate(explainer, inputs, targets, saliency=None)

        评估解释器的鲁棒性。

        .. note::

            目前，每个调用仅支持单个样本（ :math:`N=1` ）。

        参数：
            - **explainer** (Explainer) - 要评估的解释器，请参见 `mindspore_xai.explainer` 。
            - **inputs** (Tensor) - 数据样本，shape为 :math:`(N, C, H, W)` 的4D Tensor。
            - **targets** (Tensor, int) - 目标分类，1D/Scalar Tensor或integer。如果 `targets` 是1D Tensor，其长度应为 :math:`N`。
            - **saliency** (Tensor, 可选) - 要评估的热力图，shape为 :math:`(N, 1, H, W)` 的4D Tensor。如果设置为 `None` ，解析后的 `explainer` 将生成具有 `inputs` 和 `targets` 的热力图，并继续评估。默认值： `None` 。

        返回：
            numpy.ndarray，shape为 :math:`(N,)` 的1D数组，为 `explainer` 的鲁棒性评估结果。

        异常：
            - **TypeError** - 参数或输入类型错误。
            - **ValueError** - :math:`N` 不是1。