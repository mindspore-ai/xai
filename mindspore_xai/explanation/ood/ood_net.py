# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Out Of Distribution Network."""

import mindspore as ms
from mindspore import ops, nn
from mindspore.common.initializer import HeNormal
from mindspore.train._utils import check_value_type
from mindspore.train.callback import Callback
from mindspore.train.callback import LearningRateScheduler


class _GradNet(nn.Cell):
    """
    Network for gradient calculation.

    Args:
        network (Cell): The network to generate backpropagated gradients.
    """

    def __init__(self, network):
        super(_GradNet, self).__init__()
        self.network = network
        self.grad_op = ops.GradOperation()

    def construct(self, x):
        """
        Get backpropgated gradients.

        Returns:
            Tensor, output gradients.
        """
        grad_func = self.grad_op(self.network)
        return grad_func(x)


class OoDNet(nn.Cell):
    """
    Out of distribution network.

    OoDNet takes a underlying classifier and outputs the out of distribution scores of samples.

    Note:
        A training of OoDNet is required with the classifier's training dataset inorder to give the correct OoD scores.

    Args:
        underlying (Cell): The underlying classifier, it must has the 'num_features' (int) and 'output_features'
            (bool) attributes, please check the example code for the details.
        num_classes (int): The number of classes for the classifier.

    Returns:
        Tensor, classification logits (if set_train(True) was called) or
            OOD scores (if set_train(False) was called). In the shape of [batch_size, num_classes].

    Raises:
        TypeError: Be raised for any argument or input type problem.
        ValueError: Be raised for any input value problem.
        AttributeError: Be raised for underlying is missing any required attribute.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import context, nn
        >>> from mindspore_xai.explanation import OoDNet
        >>> from mindspore.common.initializer import Normal
        >>>
        >>>
        >>> class MyLeNet5(nn.Cell):
        >>>
        >>>    def __init__(self, num_class, num_channel):
        >>>        super(MyLeNet5, self).__init__()
        >>>
        >>>        # must add the following 2 attributes to your model
        >>>        self.num_features = 84 # no. of features, int
        >>>        self.output_features = False # output features flag, bool
        >>>
        >>>        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        >>>        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        >>>        self.relu = nn.ReLU()
        >>>        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        >>>        self.flatten = nn.Flatten()
        >>>        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        >>>        self.fc2 = nn.Dense(120, self.num_features, weight_init=Normal(0.02))
        >>>        self.fc3 = nn.Dense(self.num_features, num_class, weight_init=Normal(0.02))
        >>>
        >>>    def construct(self, x):
        >>>        x = self.conv1(x)
        >>>        x = self.relu(x)
        >>>        x = self.max_pool2d(x)
        >>>        x = self.conv2(x)
        >>>        x = self.relu(x)
        >>>        x = self.max_pool2d(x)
        >>>        x = self.flatten(x)
        >>>        x = self.relu(self.fc1(x))
        >>>        x = self.relu(self.fc2(x))
        >>>
        >>>        # return the features tensor if output_features is True
        >>>        if self.output_features:
        >>>            return x
        >>>
        >>>        x = self.fc3(x)
        >>>        return x
        >>>
        >>> context.set_context(mode=context.PYNATIVE_MODE)
        >>> # prepare trained classifier
        >>> net = MyLeNet5(10, num_channel=3)
        >>> param_dict = load_checkpoint('mylenet5.ckpt')
        >>> load_param_into_net(net, param_dict)
        >>> # prepare train_dataset and your OoD network
        >>> train_dataset = create_dataset_cifar10("/path/to/cifar/dataset")
        >>> ood_net = OoDNet(net, 10)
        >>> ood_net.train(train_dataset, nn.SoftmaxCrossEntropyWithLogits())
        >>> inputs = ms.Tensor(np.random.rand(1, 3, 32, 32), ms.float32)
        >>> ood_map = ood_net(inputs)
        >>> print(ood_map.shape)
        (1, 10)
    """

    def __init__(self, underlying, num_classes):
        super(OoDNet, self).__init__()

        check_value_type('num_classes', num_classes, int)
        if num_classes < 1:
            raise ValueError('num_classes is less then 1!')
        check_value_type('underlying', underlying, nn.Cell)
        try:
            check_value_type('underlying.num_features', underlying.num_features, int)
            if underlying.num_features < 1:
                raise ValueError('underlying.num_features is less then 1!')
            check_value_type('underlying.output_features', underlying.output_features, bool)
            underlying.output_features = False  # assignment test
        except AttributeError:
            raise AttributeError('underlying has no num_features or output_features attribute!')

        self._num_classes = num_classes
        self._underlying = underlying

        self._h = nn.Dense(in_channels=self._underlying.num_features,
                           out_channels=num_classes,
                           has_bias=False,
                           weight_init=HeNormal(nonlinearity='relu'))
        self._expand_dims = ops.ExpandDims()
        self._g_fc = nn.Dense(in_channels=self._underlying.num_features, out_channels=1)
        # BatchNorm1d is not working on GPU, workaround with BatchNorm2d
        self._g_bn2d = nn.BatchNorm2d(num_features=1)
        self._g_squeeze = ops.Squeeze(axis=(2, 3))
        self._g_sigmoid = nn.Sigmoid()

        self._matmul_weight = ops.MatMul(transpose_a=False, transpose_b=True)
        self._norm = nn.Norm(axis=(1,))
        self._transpose = ops.Transpose()
        self._num_features = self._underlying.num_features
        self._tile = ops.Tile()
        self._reduce_max = ops.ReduceMax(keep_dims=True)
        self._ge = ops.GreaterEqual()

        self._grad_net = None
        self._output_max_score = False
        self._is_train = False
        self.set_train(False)
        self.set_grad(False)

    @property
    def underlying(self):
        """
        Get the underlying classifier.

        Returns:
            nn.Cell, the underlying classifier.
        """
        return self._underlying

    @property
    def num_classes(self):
        """
        Get the number of classes.

        Returns:
            int, the number of classes.
        """
        return self._num_classes

    def set_train(self, mode=True):
        """
        Set training mode.

        Args:
            mode (bool): It is in training mode.
        """
        super(OoDNet, self).set_train(mode)
        self._is_train = mode

    def construct(self, x):
        """
        Forward inferences the classification logits or OOD scores.

        Returns:
            Tensor, logits of softmax with temperature (if set_train(True) was called) or
                OOD scores (if set_train(False) was called). In the shape of [batch_size, num_classes].
        """
        self._underlying.output_features = True
        feat = self._underlying(x)
        self._underlying.output_features = False
        if len(feat.shape) != 2:
            raise ValueError('The underlying output features is not 2 dimensional!')
        if feat.shape[1] != self._num_features:
            raise ValueError(f'The underlying output feature count:{feat.shape[1]} is different '
                             f'from underlying.num_features:{self._num_features}.')
        scores = self._ood_scores(feat)
        if self._is_train:
            feat = self._g_fc(feat)
            feat = self._expand_dims(feat, 2)
            feat = self._expand_dims(feat, 2)
            feat = self._g_bn2d(feat)
            feat = self._g_squeeze(feat)

            # logits of softmax with temperature
            temperature = self._g_sigmoid(feat)
            logits = scores / temperature
            return logits

        if self._output_max_score:
            scores = self._reduce_max(scores, 1)
        return scores

    def get_train_parameters(self, train_underlying=False):
        """
        Get the training parameters.

        Returns:
            list[Parameter], parameters.
        """
        if train_underlying:
            parameters = list(self._underlying.get_parameters())
        else:
            parameters = list()
        parameters.extend(self._h.get_parameters())
        parameters.extend(self._g_fc.get_parameters())
        return parameters

    def prepare_train(self,
                      learning_rate=0.1,
                      momentum=0.9,
                      weight_decay=0.0001,
                      lr_base_factor=0.1,
                      lr_epoch_denom=30,
                      train_underlying=False):
        """
        Creates necessities for training.

        Args:
            learning_rate (float): The optimizer learning rate.
            momentum (float): The optimizer momentum.
            weight_decay (float): The optimizer weight decay.
            lr_base_factor (float): The base scaling factor of learning rate scheduler.
            lr_epoch_denom (int): The epoch denominator of learning rate scheduler.
            train_underlying (bool): True to train the underlying classifier as well.

        Returns:
            - Optimizer, optimizer.
            - LearningRateScheduler, learning rate scheduler.
        """
        parameters = self.get_train_parameters(train_underlying)
        scheduler = _EpochLrScheduler(learning_rate, lr_base_factor, lr_epoch_denom)
        optimizer = nn.SGD(parameters, learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay)
        return optimizer, scheduler

    def train(self,
              dataset,
              loss_fn,
              callbacks=None,
              epoch=90,
              optimizer=None,
              scheduler=None,
              **kwargs):
        """
        Trains this OOD net.

        Args:
            dataset (Dataset): The training dataset, expecting (data, one-hot label) items.
            loss_fn (Cell): The loss function, if the classifier's activation function is nn.Softmax(), then use
                nn.SoftmaxCrossEntropyWithLogits(), if the activation function is nn.Sigmod(), then use
                nn.BCEWithLogitsLoss().
            callbacks (Callback, optional): The train callbacks.
            epoch (int, optional): The number of epochs to be trained. Default: 90.
            optimizer (Optimizer, optional): The optimizer. The one from prepare_train() will be used if which is set
                to None.
            scheduler (LearningRateScheduler, optional): The learning rate scheduler. The one from prepare_train() will
                be used if which is set to None.
            **kwargs (any, optional): Keyword arguments for prepare_train().
        """
        self.set_train(True)
        self.set_grad(True)

        if optimizer is None or scheduler is None:
            auto_optimizer, auto_scheduler = self.prepare_train(**kwargs)
            if optimizer is None:
                optimizer = auto_optimizer
            if scheduler is None:
                scheduler = auto_scheduler

        model = ms.Model(self, loss_fn=loss_fn, optimizer=optimizer)
        if callbacks is None:
            callbacks = [scheduler]
        elif isinstance(callbacks, list):
            callbacks_ = [scheduler]
            callbacks_.extend(callbacks)
            callbacks = callbacks_
        elif isinstance(callbacks, Callback):
            callbacks = [scheduler, callbacks]
        else:
            raise ValueError('invalid callbacks type')
        model.train(epoch, dataset, callbacks=callbacks)
        self.set_train(False)
        self.set_grad(False)

    def _ood_scores(self, feat):
        """Forward inferences the OOD scores."""
        norm_f = self._normalize(feat)
        norm_w = self._normalize(self._h.weight)
        scores = self._matmul_weight(norm_f, norm_w)
        return scores

    def _normalize(self, x):
        """Normalizes an tensor."""
        norm = self._norm(x)
        tiled_norm = self._tile((norm + 1e-4), (self._num_features, 1))
        tiled_norm = self._transpose(tiled_norm, (1, 0))
        x = x / tiled_norm
        return x


class _EpochLrScheduler(LearningRateScheduler):
    """
    Epoch based learning rate scheduler.

    Args:
        base_lr (float): The base learning rate.
        base_factor (float): The base scaling factor.
        denominator (int): The epoch denominator.
    """
    def __init__(self, base_lr, base_factor, denominator):
        super(_EpochLrScheduler, self).__init__(self._lr_function)
        self.base_lr = base_lr
        self.base_factor = base_factor
        self.denominator = denominator
        self._cur_epoch_num = 1

    def epoch_end(self, run_context):
        """On an epoch was ended."""
        cb_params = run_context.original_args()
        self._cur_epoch_num = cb_params.cur_epoch_num

    def _lr_function(self, lr, cur_step_num):
        """Returns the dynamic learning rate."""
        del lr
        del cur_step_num
        return self.base_lr * (self.base_factor ** (self._cur_epoch_num // self.denominator))
