mindspore_xai.benchmark
=================================

预定义的XAI指标。

.. py:class:: mindspore_xai.benchmark.ClassSensitivity

    类敏感度(ClassSensitivity)度量用于评估基于归因的解释。

    一个合理的基于归因的解释器应该为不同标签生成不同的热力图，特别是对"高置信度"和"低置信度"两种标签，类敏感度通过计算这两种标签的热力图之间的相关性来评估解释器，而类敏感度较好的解释器将获得较低的相关分数。为了使评估结果直观，返回的分数将取相关性的负值并被归一化。

    .. py:function:: evaluate(explainer, inputs)

        在单个数据样本上评估类敏感度。

        .. note::
             目前，每个调用仅支持单个样本（ :math:`N=1` ）。

        **参数：**

        - **explainer** (Explainer) - 要评估的解释器，请参见 `mindspore_xai.explainer` 。
        - **inputs** (Tensor) - 数据样本，shape 为 :math:`(N, C, H, W)` 的4D Tensor。

        **返回：**

        numpy.ndarray，shape 为 :math:`(N,)` 的1D数组，在 `explainer` 上评估类敏感度的结果。

        **异常：**

        - **TypeError** - 在出现任何参数类型问题时抛出。
        - **ValueError** - 在 :math:`N` 不是1时抛出。

.. py:class:: mindspore_xai.benchmark.Faithfulness(num_labels, activation_fn, metric="NaiveFaithfulness")

    提供对XAI解释的忠实度进行评估。

    支持三个特定指标来获得量化结果："NaiveFaithfulness"，"DeletionAUC"和"InsertionAUC"。

    对于度量"NaiveFaithfulness"，通过修改原始图像上的像素来创建一系列扰动图像。扰动图像将被馈送到模型中以令输出预测概率下降，而在概率下降和热力图数值两者之间的相关性便是忠实度数值，然后我们会进一步归一化相关性，使它们在[0, 1]的范围内。

    对于度量"DeletionAUC"，通过将原始图像的累积像素修改为基本数值（例如：常数）来创建一系列扰动图像。扰动会根据像素的显著值从高至低依次进行，并将扰动图像按顺序馈入模型中，从而得到输出概率的下降曲线，"DeletionAUC" 为该曲线下的面积。

    对于度量"InsertionAUC"，通过将原始图像的累积像素插入参考图像（例如：黑色图像）来创建一系列扰动图像。插入会根据像素的显著值从高至低依次进行，并将扰动图像按顺序馈入模型中，从而得到输出概率的生长曲线，"InsertionAUC" 为该曲线下的面积。

    对于所有三个指标，值越高表示忠诚度越高。

    **参数：**

    - **num_labels** (int) - 标签的数量。
    - **activation_fn** (Cell) - 将logits转换为预测概率的激活层。对于单个标签分类任务，通常应用 `nn.Softmax` 。而对于多标签分类任务，则通常应用 `nn.Sigmoid` 。用户也可以将自定义的 `activation_fn` 与网络结合，最终输出便是输入的概率。
    - **metric** (str, 可选) - 量化忠诚度的特定度量。可选项："DeletionAUC"，"InsertionAUC"，"NaiveFaithfulness"。默认值："NaiveFaithfulness"。

    **异常：**

    - **TypeError** - 在出现任何参数类型问题时抛出。

    .. py:function:: evaluate(explainer, inputs, targets, saliency=None)

        评估单个数据样本的忠诚度。

        .. note::
            目前，每个调用仅支持单个样本（ :math:`N=1` ）。

        **参数：**

        - **explainer** (Explainer) - 要评估的解释器，请参见 `mindspore_xai.explainer` 。
        - **inputs** (Tensor) - 数据样本，shape 为 :math:`(N, C, H, W)` 的 4D Tensor。
        - **targets** (Tensor, int) - 目标分类， 1D/0D Tensor 或 integer。如果 `targets` 为 1D Tensor，则其长度应为 :math:`N` 。
        - **saliency** (Tensor, 可选) - 要评估的热力图，shape 为 :math:`(N, 1, H, W)` 的4D Tensor。如果为 `None` ，解析后的 `explainer` 则将生成具有 `inputs` 和 `targets` 的热力图，并且继续评估。默认值：`None`。

        **返回：**

        numpy.ndarray，shape 为 :math:`(N,)` 的1D数组，在 `explainer` 上评估的忠实度结果。

        **异常：**

        - **TypeError** - 在出现任何参数类型问题时抛出。
        - **ValueError** - 在 :math:`N` 不是1时抛出。

.. py:class:: mindspore_xai.benchmark.Localization(num_labels, metric="PointingGame")

    提供对XAI方法的定位性（Localization）能力评估。

    支持两个特定指标来获得量化结果："PointingGame" 和 "IoSR"（显著区域的相交）。

    对于度量"PointingGame"，定位性能力会计算图最大位置位于边界框内的相交比例。具体来说，对于单个基准，如果热力图的最大位置位于边界框内，计算结果为1，否则为0。

    对于度量"IoSR"（显著区域的相交），定位性能力会计算在显著区域上边界框和显著区域之间的相交面积。如果它的值超过 :math:`\theta * \max{saliency}` ，将会被定义为显著区域。

    **参数：**

    - **num_labels** （int） - 数据集中的类数。
    - **metric** （str，可选） - 计算定位性能力的特定度量。可选项："PointingGame"和"IoSR"。默认值："PointingGame"。

    **异常：**

    - **TypeError** - 在出现任何参数类型问题时抛出。

    .. py:function:: evaluate(explainer, inputs, targets, saliency=None, mask=None)

        在单个数据样本上评估定位性。

        .. note::

             目前，每个调用仅支持单个样本（ :math:`N=1` ）。

        **参数：**

        - **explainer** (Explainer) - 要评估的解释器，请参见 `mindspore_xai.explainer` 。
        - **inputs** (Tensor) - 数据样本，shape 为 :math:`(N, C, H, W)` 的 4D Tensor。
        - **targets** (Tensor, int) - 目标分类， 1D/0D Tensor 或 integer。如果 `targets` 为 1D Tensor，则其长度应为:math:`N` 。
        - **saliency** (Tensor, 可选) - 要评估的热力图，shape 为 :math:`(N, 1, H, W)` 的 4D Tensor。如果为 `None` ，则解析的 `explainer` 将生成具有 `inputs` 和 `targets` 的热力图，并且继续评估。默认值： `None` 。
        - **mask** （Tensor,numpy.ndarray） - 参考目标给输入的 ground truth边界框/掩码，4D Tensor 或 shape 为 :math:`(N, 1, H, W)` 的 `numpy.ndarray` 。

        **返回：**

        numpy.ndarray，shape 为 :math:`(N,)` 的 1D 数组，在 `explainer` 上评估的定位性结果。

        **异常：**

        - **TypeError** - 在出现任何参数类型问题时抛出。
        - **ValueError** - 在 :math:`(N,)` 不是1时抛出。

.. py:class:: mindspore_xai.benchmark.Robustness(num_labels, activation_fn)

    鲁棒性 (Robustness) 通过添加随机噪声来扰动输入，并从扰动中选择最大灵敏度作为评估分数。

    **参数：**

    - **num_labels** (int) - 数据集中的类数。
    - **activation_fn** (Cell) - 将 logits 转换为预测概率的激活层。对于单标签分类任务，通常应用 `nn.Softmax` 。而对于多标签分类任务，则通常应用 `nn.Sigmoid` 。用户也可以将自定义的 `activation_fn` 与网络结合，最终输出便是输入的概率。

    **异常：**

    - **TypeError** - 在出现任何参数类型问题时抛出。

    .. py:function:: evaluate(explainer, inputs, targets, saliency=None)

        评估单个样品的鲁棒性。

        .. note::

            目前，每个调用仅支持单个样本（ :math:`N=1` ）。

        **参数：**

        - **explainer** (Explainer) - 要评估的解释器，请参见 `mindspore_xai.explainer` 。
        - **inputs** (Tensor) - 数据样本，shape 为 :math:`(N, C, H, W)` 的4D Tensor。
        - **targets** (Tensor, int) - 目标分类，1D/0D Tensor 或 integer。如果 `targets` 为 1D Tensor，则其长度应为 :math:`N`。
        - **saliency** (Tensor, 可选) - 要评估的热力图，shape 为 :math:`(N, 1, H, W)` 的4D Tensor。如果为 `None` ，则解析的 `explainer` 将生成带有 `inputs` 和 `targets` 的热力图，并继续计算。默认值： `None` 。

        **返回：**

        numpy.ndarray，shape 为 1D 数组 :math:`(N,)` ，在 `explainer` 上评估的鲁棒性结果。

        **异常：**

        - **TypeError** - 在出现任何参数类型问题时抛出。
        - **ValueError** - 在 :math:`N` 不是1时抛出。