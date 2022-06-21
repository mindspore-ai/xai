mindspore_xai.tool
=================================

CV类工具。

.. py:class:: mindspore_xai.tool.cv.OoDNet(underlying, num_classes)

    分布外检测网络。

    OoDNet 需要一个下游分类器并输出样本的分布外分数。

    .. note::
       为了给出正确的分布外分数，需要使用分类器的训练数据集对 OoDNet 进行训练。

    **参数：**

    - **underlying** (Cell) - 下游分类器，它必须具有 `num_features` (int) 和 `output_features` (bool) 的属性，具体请参见示例代码。
    - **num_classes** (int) - 分类器的类数。

    **返回：**

    Tensor，如果 `set_train(True)` 被调用，返回classification logits。而如果 `set_train(False)` 被调用，返回分布外分数。 Shape 则为 :math:`(batch\_size,num\_classes)` 。

    **异常：**

    - **TypeError** - 在出现任何参数或输入类型问题时抛出。
    - **ValueError** - 在任何输入的值出现问题时抛出。
    - **AttributeError** - 在缺少任何必需的属性时抛出。

    .. py:method:: construct(x)

        向前推断 classification logits 或分布外分数.

        **参数：**

        **x** (Tensor) - 下游分类器的输入。

        **返回：**

        Tensor，如果 `set_train(True)` 被调用，返回有温度 softmax 的 logits。而如果 `set_train(False)` 被调用，返回分布外分数。 Shape 则为 :math:`(batch\_size,num\_classes)` 。

    .. py:method:: get_train_parameters(train_underlying=False)

        训练参数。

        **参数：**

        **train_underlying** (bool) - 设置为 `True` 以包括下游分类器参数。

        **返回：**

        list[Parameter]，参数。

    .. py:method:: num_classes
        :property:

        类的数量。

        **返回：**

        int，类的数量。

    .. py:method:: prepare_train(learning_rate=0.1, momentum=0.9, weight_decay=0.0001, lr_base_factor=0.1, lr_epoch_denom=30, train_underlying=False)

        准备训练。

        **参数：**

        - **learning_rate** (float) - 优化器的学习率。
        - **momentum** (float) - 优化器的 Momentum。
        - **weight_decay** (float) - 优化器的权重衰减。
        - **lr_base_factor** (float) - 学习率调度器的基本比例因子。
        - **lr_epoch_denom** (int) - 学习率调度器的 epoch 分母。
        - **train_underlying** (bool) - 如果训练下游分类器，则为 `True` 。

        **返回：**

        - Optimizer，优化器。
        - LearningRateScheduler，学习率调度器。

    .. py:method:: set_train(mode=True)

        训练模式。

        **参数：**

        - **mode** (bool) - 训练模式。

    .. py:method:: train(dataset, loss_fn, callbacks=None, epoch=90, optimizer=None, scheduler=None, **kwargs)

        训练分布外网络。

        **参数：**

        - **dataset** (Dataset) - 训练数据集，预期（数据, one-hot 标签）项目。
        - **loss_fn** (Cell) - loss 函数，如果分类器的激活函数是 `nn.Softmax`，使用 `nn.SoftmaxCrossEntropyWithLogits`，而如果激活函数是 `nn.Sigmod`，则使用 `nn.BCEWithLogitsLoss`。
        - **callbacks** (Callback, 可选) - 训练回调。
        - **epoch** (int, 可选) - 训练时期 epoch 的数量。默认值：90。
        - **optimizer** (Optimizer, 可选) - 优化器。如果设置为 `None` ，则将使用 `prepare_train()`。
        - **scheduler** (LearningRateScheduler, 可选) - 学习率调度器。如果设置为 `None` ，则将使用 `prepare_train()`。
        - ****kwargs** (any, 可选) - `prepare_train()` 的关键字参数。

    .. py:method:: underlying
        :property:

        下游分类器。

        **返回：**

        `nn.Cell`，下游分类器。