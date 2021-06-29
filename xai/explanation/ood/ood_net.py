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

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import HeNormal
from mindspore.train.callback import Callback
from mindspore.train.callback import LearningRateScheduler


class OODUnderlying(nn.Cell):
    """The base class of underlying classifier."""

    def __init__(self):
        super(OODUnderlying, self).__init__()
        self.output_feature = False

    @property
    def feature_count(self):
        """
        The Number of features.

        Returns:
            int, the number of features.
        """
        raise NotImplementedError


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


class OODNet(nn.Cell):
    """
    Out of distribution network.

    Args:
        underlying (OODUnderlying, optional): The underlying classifier. None means using OODResNet50 as underlying.
        num_classes (int): Number of classes for the classifier.

     Returns:
        Tensor, classification logits (if set_train(True) was called) or
            OOD scores (if set_train(False) was called). In the shape of [batch_size, num_classes].
    """

    def __init__(self, underlying, num_classes):
        super(OODNet, self).__init__()

        self._num_classes = num_classes

        if underlying is None:
            from .ood_resnet import OODResNet50
            self._underlying = OODResNet50(num_classes)
            self._train_partial = False
        else:
            self._underlying = underlying
            self._train_partial = True

        self._h = nn.Dense(in_channels=self._underlying.feature_count,
                           out_channels=num_classes,
                           has_bias=False,
                           weight_init=HeNormal(nonlinearity='relu'))
        self._expand_dims = ops.ExpandDims()
        self._g_fc = nn.Dense(in_channels=self._underlying.feature_count, out_channels=1)
        # BatchNorm1d is not working on GPU, workaround with BatchNorm2d
        self._g_bn2d = nn.BatchNorm2d(num_features=1)
        self._g_squeeze = ops.Squeeze(axis=(2, 3))
        self._g_sigmoid = nn.Sigmoid()

        self._matmul_weight = ops.MatMul(transpose_a=False, transpose_b=True)
        self._norm = nn.Norm(axis=(1,))
        self._transpose = ops.Transpose()
        self._feature_count = self._underlying.feature_count
        self._tile = ops.Tile()
        self._reduce_max = ops.ReduceMax(keep_dims=True)
        self._ge = ops.GreaterEqual()

        self._grad_net = None
        self._output_max_score = False
        self._is_train = False
        self.set_train(False)
        self.set_grad(False)

    def set_train(self, mode=True):
        """
        Set training mode.

        Args:
            mode (bool): It is in training mode.
        """
        super(OODNet, self).set_train(mode)
        self._is_train = mode

    def score(self, x, noise_mag=0.08, channel_sd=(0.229, 0.224, 0.225)):
        """
        Compute OOD scores with noise added to input image tensor.

        Args:
            x (Tensor): Image tensor of shape [N,C,H,W]
            noise_mag (float): Noise magnitude.
            channel_sd (tuple): Channel standard deviations.

        Returns:
            Tensor, OOD scores of shape [N,num_classes].
        """
        if self._is_train:
            self.set_train(False)

        if noise_mag == 0:
            return self(x)

        self.set_grad(True)
        self._output_max_score = True
        if self._grad_net is None:
            self._grad_net = _GradNet(self)
        grad = self._grad_net(x)
        self._output_max_score = False
        self.set_grad(False)

        noise = self._ge(grad, 0).asnumpy().copy().astype(np.float32)
        noise = (noise - 0.5) * 2

        channel_delta = noise_mag / np.array(channel_sd)

        for c, delta in enumerate(channel_delta):
            noise[:, c, :, :] *= delta

        x = x + ms.Tensor(noise, dtype=ms.float32)
        scores = self(x)
        return scores

    def construct(self, x):
        """
        Forward inferences the classification logits or OOD scores.

        Returns:
            Tensor, logits of softmax with temperature (if set_train(True) was called) or
                OOD scores (if set_train(False) was called). In the shape of [batch_size, num_classes].
        """
        self._underlying.output_feature = True
        feat = self._underlying(x)
        self._underlying.output_feature = False
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

    def prepare_train(self,
                      learning_rate=0.1,
                      momentum=0.9,
                      weight_decay=0.0001,
                      lr_base_factor=0.1,
                      lr_epoch_denom=30):
        """
        Creates necessities for training.

        Args:
            learning_rate (float): The optimizer learning rate.
            momentum (float): The optimizer momentum.
            weight_decay (float): The optimizer weight decay.
            lr_base_factor (float): The base scaling factor of learning rate scheduler.
            lr_epoch_denom (int): The epoch denominator of learning rate scheduler.

        Returns:
            - Optimizer, optimizer.
            - LearningRateScheduler, learning rate scheduler.
        """
        if self._train_partial:
            parameters = []
        else:
            parameters = list(self._underlying.get_parameters())
        parameters.extend(self._h.get_parameters())
        parameters.extend(self._g_fc.get_parameters())
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
            loss_fn (Cell): The loss function.
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
        tiled_norm = self._tile((norm + 1e-4), (self._feature_count, 1))
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
