mindspore_xai.tool
=================================

CV类工具。

.. py:class:: mindspore_xai.tool.cv.OoDNet(underlying, num_classes)

    分布外检测网络。

    OoDNet需要一个下游分类器，并会输出样本的分布外分数。

    .. note::
       为了给出正确的分布外分数，OoDNet需要使用分类器的训练数据集来进行训练。

    **参数：**

    - **underlying** (Cell) - 下游分类器，它必须具有 `num_features` (int)和 `output_features` (bool)的属性，具体详情请参见样例。
    - **num_classes** (int) - 分类器的类数。

    **返回：**

    Tensor，如果 `set_train(True)` 被调用，将返回分类logits。而如果 `set_train(False)` 被调用，则返回分布外分数。返回的shape均为 :math:`(N, L)` ，L 是类数。

    **异常：**

    - **TypeError** - 在出现任何参数或输入类型问题时抛出。
    - **ValueError** - 在任何输入的值出现问题时抛出。
    - **AttributeError** - 在缺少任何必需的属性时抛出。

    .. py:method:: construct(x)

        向前推理分类logits或分布外分数。

        **参数：**

        **x** (Tensor) - 下游分类器的输入。

        **返回：**

        Tensor，如果 `set_train(True)` 被调用，将返回logits of softmax with temperature。而如果 `set_train(False)` 被调用，则返回分布外分数。返回的shape均为 :math:`(N, L)` ，L 是类数。

    .. py:method:: get_train_parameters(train_underlying=False)

        获取训练参数。

        **参数：**

        **train_underlying** (bool) - 如需包含下游分类器的参数，请设置为 `True` 。默认值： `False` 。

        **返回：**

        list[Parameter]，训练参数。

    .. py:method:: num_classes
        :property:

        获取类的数量。

        **返回：**

        int，类的数量。

    .. py:method:: prepare_train(learning_rate=0.1, momentum=0.9, weight_decay=0.0001, lr_base_factor=0.1, lr_epoch_denom=30, train_underlying=False)

        准备训练参数。

        **参数：**

        - **learning_rate** (float) - 优化器的学习率。默认值：0.1。
        - **momentum** (float) - 优化器的Momentum。默认值：0.9。
        - **weight_decay** (float) - 优化器的权重衰减。默认值：0.0001。
        - **lr_base_factor** (float) - 学习率调度器的基本比例因子。默认值：0.1。
        - **lr_epoch_denom** (int) - 学习率调度器的epoch分母。默认值：30。
        - **train_underlying** (bool) - 如需训练下游分类器，请设置为 `True` 。默认值：`False`。

        **返回：**

        - Optimizer，优化器。
        - LearningRateScheduler，学习率调度器。

    .. py:method:: set_train(mode=True)

        选择训练模式。

        **参数：**

        - **mode** (bool) - 训练模式。默认值： `True` 。

    .. py:method:: train(dataset, loss_fn, callbacks=None, epoch=90, optimizer=None, scheduler=None, **kwargs)

        训练分布外网络。

        **参数：**

        - **dataset** (Dataset) - 训练数据集，预期格式为（数据, one-hot标签）。
        - **loss_fn** (Cell) - loss 函数，如果分类器选择的激活函数是 `nn.Softmax`，请使用 `nn.SoftmaxCrossEntropyWithLogits`，而如果选择的是 `nn.Sigmod`，则使用 `nn.BCEWithLogitsLoss`。
        - **callbacks** (Callback, 可选) - 训练时的回调。默认值： `None` 。
        - **epoch** (int, 可选) - 训练时的epoch数量。默认值：90。
        - **optimizer** (Optimizer, 可选) - 优化器。如果设置为 `None` ，将使用 `prepare_train()` 预定义的参数。默认值： `None` 。
        - **scheduler** (LearningRateScheduler, 可选) - 学习率调度器。如果设置为 `None` ，将使用 `prepare_train()` 预定义的参数。默认值： `None` 。
        - ****kwargs** (any, 可选) - 在 `prepare_train()` 定义的关键参数。

    .. py:method:: underlying
        :property:

        获取下游分类器。

        **返回：**

        `nn.Cell`，下游分类器。