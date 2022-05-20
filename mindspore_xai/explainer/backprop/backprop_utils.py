# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""Providing utility functions."""

from mindspore.nn import Cell
from mindspore.ops.composite import GradOperation

from mindspore_xai.common.utils import generate_one_hot


def get_bp_weights(num_classes, targets, weights=None):
    r"""
    Compute the gradient of output w.r.t input.

    Args:
        num_classes (int): The number of classes.
        targets (Tensor): Target label id specifying which category to compute gradient.
        weights (Tensor, optional): Custom weights for computing gradients. The shape of weights should match the model
            outputs. If None is provided, an one-hot weights with one in targets positions will be used instead.
            Default: None.

    Returns:
        Tensor, signal to be back-propagated to the input.
    """
    if weights is None:
        weights = generate_one_hot(targets, num_classes)
    return weights


class GradNet(Cell):
    """
    Network for gradient calculation.

    Args:
        network (Cell): The network to generate backpropagated gradients.
        sens_param (bool): Enable GradOperation sens_params.
    """

    def __init__(self, network, sens_param=True):
        super(GradNet, self).__init__()
        self.network = network
        self.grad = GradOperation(get_all=True, sens_param=sens_param)(network)

    def construct(self, *input_data):
        """
        Get backpropgated gradients.

        Returns:
            Tensor, output gradients.
        """
        gout = self.grad(*input_data)[0]
        return gout
