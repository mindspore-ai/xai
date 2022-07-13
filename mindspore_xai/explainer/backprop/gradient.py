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
"""Gradient explainer."""
from copy import deepcopy

from mindspore import Tensor
from mindspore.train._utils import check_value_type

from mindspore_xai.common.utils import abs_max, unify_inputs, unify_targets
from mindspore_xai.common.attribution import Attribution
from .backprop_utils import get_bp_weights, GradNet


class Gradient(Attribution):
    r"""
    Provides Gradient explanation method.

    Gradient is the simplest attribution method which uses the naive gradients of outputs w.r.t inputs as the
    explanation.

    .. math::

        attribution = \frac{\partial{y}}{\partial{x}}

    Note:
        The parsed `network` will be set to eval mode through `network.set_grad(False)` and `network.set_train(False)`.
        If you want to train the `network` afterwards, please reset it back to training mode through the opposite
        operations.

    Args:
        network (Cell): The black-box model to be explained.

    Inputs:
        - **inputs** (Tensor) - The input data to be explained, a 4D tensor of shape :math:`(N, C, H, W)`.
        - **targets** (Tensor, int, tuple, list) - The label of interest. It should be a 1D or scalar tensor, or an
          integer, or a tuple/list of integers. If it is a 1D tensor, tuple or list, its length should be :math:`N`.
        - **ret** (str): The return object type. 'tensor' means returns a Tensor object, 'image' means return a
          PIL.Image.Image list. Default: `tensor`.
        - **show** (bool, optional): Show the saliency images, `None` means automatically show the saliency images if it
          is running on JupyterLab. Default: `None`.

    Outputs:
        Tensor, a 4D tensor of shape :math:`(N, 1, H, W)`, saliency maps. Or list[list[PIL.Image.Image]], the
        normalized saliency images if `ret` was set to 'image'.

    Raises:
        TypeError: Be raised for any argument or input type problem.
        ValueError: Be raised for any input value problem.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import set_context, PYNATIVE_MODE
        >>> from mindspore_xai.explainer import Gradient
        >>>
        >>> set_context(mode=PYNATIVE_MODE)
        >>> # The detail of LeNet5 is shown in model_zoo.official.cv.lenet.src.lenet.py
        >>> net = LeNet5(10, num_channel=3)
        >>> gradient = Gradient(net)
        >>> inputs = ms.Tensor(np.random.rand(1, 3, 32, 32), ms.float32)
        >>> label = 5
        >>> saliency = gradient(inputs, label)
        >>> print(saliency.shape)
        (1, 1, 32, 32)
    """

    def __init__(self, network):
        super(Gradient, self).__init__(network)
        self._backward_model = deepcopy(network)
        self._backward_model.set_train(False)
        self._backward_model.set_grad(False)
        self._grad_net = GradNet(self._backward_model)
        self._aggregation_fn = abs_max
        self._num_classes = None

    def __call__(self, inputs, targets, ret='tensor', show=None):
        """Call function for `Gradient`."""
        self._verify_data(inputs, targets)
        self._verify_other_args(ret, show)

        inputs = unify_inputs(inputs)
        targets = unify_targets(inputs[0].shape[0], targets)

        weights = self._get_bp_weights(inputs, targets)
        gradient = self._grad_net(*inputs, weights)
        saliency = self._aggregation_fn(gradient)

        return self._postproc_saliency(saliency, ret, show)

    def _get_bp_weights(self, unified_inputs, unified_targets):
        if self._num_classes is None:
            output = self._backward_model(*unified_inputs)
            self._num_classes = output.shape[-1]
        return get_bp_weights(self._num_classes, unified_targets)

    @staticmethod
    def _verify_data(inputs, targets):
        """
        Verify the validity of the parsed inputs.

        Args:
            inputs (Tensor): The inputs to be explained.
            targets (Tensor, int): The label of interest. It should be a 1D or 0D tensor, or an integer.
          If it is a 1D tensor, its length should be the same as `inputs`.
        """
        check_value_type('inputs', inputs, Tensor)
        if len(inputs.shape) != 4:
            raise ValueError(f'Argument inputs must be 4D Tensor. But got {len(inputs.shape)}D Tensor.')
        check_value_type('targets', targets, (Tensor, int, tuple, list))
        if isinstance(targets, Tensor):
            if len(targets.shape) > 1 or (len(targets.shape) == 1 and len(targets) != len(inputs)):
                raise ValueError('Argument targets must be a 1D or 0D Tensor. If it is a 1D Tensor, '
                                 'it should have the same length as inputs.')
