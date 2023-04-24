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
"""Occlusion explainer."""

from typing import Tuple
import numpy as np
import mindspore as ms
import mindspore.nn as nn

from mindspore_xai.common.utils import abs_max, unify_targets
from .ablation import Ablation
from .perturbation import PerturbationAttribution
from .replacement import Constant


def _generate_patches(array, window_size: Tuple, strides: Tuple):
    """Generate patches from image w.r.t given window_size and strides."""
    window_strides = array.strides
    slices = tuple(slice(None, None, stride) for stride in strides)
    indexing_strides = array[slices].strides
    win_indices_shape = (np.array(array.shape) - np.array(window_size)) // np.array(strides) + 1

    patches_shape = tuple(win_indices_shape) + window_size
    strides_in_memory = indexing_strides + window_strides
    patches = np.lib.stride_tricks.as_strided(array, shape=patches_shape, strides=strides_in_memory, writeable=False)
    patches = patches.reshape((-1,) + window_size)
    return patches


class Occlusion(PerturbationAttribution):
    r"""
    Provides Occlusion explanation method.

    Occlusion uses a sliding window to replace the pixels with a reference value (e.g. constant value), and computes
    the output difference w.r.t the original output. The output difference caused by perturbed pixels are assigned as
    feature importance to those pixels. For pixels involved in multiple sliding windows, the feature importance is the
    averaged differences from multiple sliding windows.

    For more details, please refer to the original paper via:
    `Visualizing and Understanding Convolutional Networks <https://arxiv.org/abs/1311.2901>`_.

    Note:
         Currently only single sample (:math:`N=1`) at each call is supported.

    Args:
        network (Cell): The black-box model to be explained.
        activation_fn (Cell): The activation layer that transforms logits to prediction probabilities. For
            single label classification tasks, `nn.Softmax` is usually applied. As for multi-label classification
            tasks, `nn.Sigmoid` is usually be applied. Users can also pass their own customized `activation_fn` as long
            as when combining this function with network, the final output is the probability of the input.
        perturbation_per_eval (int, optional): Number of perturbations for each inference during inferring the
            perturbed samples. Within the memory capacity, usually the larger this number is, the faster the
            explanation is obtained. Default: ``32``.

    Inputs:
        - **inputs** (Tensor) - The input data to be explained, a 4D tensor of shape :math:`(N, C, H, W)`.
        - **targets** (Tensor, int, tuple, list) - The label of interest. It should be a 1D or scalar tensor, or an
          integer, or a tuple/list of integers. If it is a 1D tensor, tuple or list, its length should be :math:`N`.
        - **ret** (str, optional): The return object type. ``'tensor'`` means returns a Tensor object, ``'image'``
          means return a PIL.Image.Image list. Default: ``'tensor'``.
        - **show** (bool, optional): Show the saliency images, ``None`` means automatically show the saliency images
          if it is running on JupyterLab. Default: ``None``.

    Outputs:
        Tensor, a 4D tensor of shape :math:`(N, 1, H, W)`, saliency maps. Or list[list[PIL.Image.Image]], the
        normalized saliency images if `ret` was set to ``'image'``.

    Raises:
        TypeError: Be raised for any argument or input type problem.
        ValueError: Be raised for any input value problem.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore_xai.explainer import Occlusion
        >>> from mindspore import set_context, PYNATIVE_MODE
        >>>
        >>> set_context(mode=PYNATIVE_MODE)
        >>> # The detail of LeNet5 is shown in models.official.cv.lenet.src.lenet.py
        >>> net = LeNet5(10, num_channel=3)
        >>> # initialize Occlusion explainer with the pretrained model and activation function
        >>> activation_fn = ms.nn.Softmax() # softmax layer is applied to transform logits to probabilities
        >>> occlusion = Occlusion(net, activation_fn=activation_fn)
        >>> input_x = ms.Tensor(np.random.rand(1, 3, 32, 32), ms.float32)
        >>> label = ms.Tensor([1], ms.int32)
        >>> saliency = occlusion(input_x, label)
        >>> print(saliency.shape)
        (1, 1, 32, 32)
    """

    def __init__(self, network, activation_fn, perturbation_per_eval=32):
        super().__init__(network, activation_fn, perturbation_per_eval)

        self._ablation = Ablation(perturb_mode='Deletion')
        self._aggregation_fn = abs_max
        self._get_replacement = Constant(base_value=0.0)
        self._num_sample_per_dim = 32  # specify the number of perturbations each dimension.

    def __call__(self, inputs, targets, ret='tensor', show=None):
        """Call function for 'Occlusion'."""
        self._verify_data(inputs, targets)
        self._verify_other_args(ret, show)

        inputs = inputs.asnumpy()
        targets = unify_targets(inputs.shape[0], targets).asnumpy()

        batch_size = inputs.shape[0]
        window_size, strides = self._get_window_size_and_strides(inputs)

        full_network = nn.SequentialCell([self._network, self._activation_fn])

        original_outputs = full_network(ms.Tensor(inputs, ms.float32)).asnumpy()[np.arange(batch_size), targets]

        masks = Occlusion._generate_masks(inputs, window_size, strides)

        saliency = self._perturbate(batch_size, full_network, (original_outputs, masks, inputs, targets))
        return self._postproc_saliency(saliency, ret, show)

    def _perturbate(self, batch_size, full_network, data):
        """Perform perturbations."""
        original_outputs, masks, inputs, targets = data
        total_attribution = np.zeros_like(inputs)
        weights = np.ones_like(inputs)
        num_perturbations = masks.shape[1]
        reference = self._get_replacement(inputs)

        count = 0
        while count < num_perturbations:
            ith_masks = masks[:, count:min(count+self._perturbation_per_eval, num_perturbations)]
            actual_num_eval = ith_masks.shape[1]
            num_samples = batch_size * actual_num_eval
            occluded_inputs = self._ablation(inputs, reference, ith_masks)
            occluded_inputs = occluded_inputs.reshape((-1, *inputs.shape[1:]))
            targets_repeat = np.repeat(targets, repeats=actual_num_eval, axis=0)
            occluded_outputs = full_network(
                ms.Tensor(occluded_inputs, ms.float32)).asnumpy()[np.arange(num_samples), targets_repeat]
            original_outputs_repeat = np.repeat(original_outputs, repeats=actual_num_eval, axis=0)
            outputs_diff = original_outputs_repeat - occluded_outputs
            total_attribution += (
                outputs_diff.reshape(ith_masks.shape[:2] + (1,) * (len(masks.shape) - 2)) * ith_masks).sum(axis=1)
            weights += ith_masks.sum(axis=1)
            count += actual_num_eval
        attribution = self._aggregation_fn(ms.Tensor(total_attribution / weights, ms.float32))
        return attribution

    def _get_window_size_and_strides(self, inputs):
        """
        Return window_size and strides.

        # If spatial size of input data is smaller than self._num_sample_per_dim, window_size and strides will set to
        # `(C, 3, 3)` and `(C, 1, 1)` separately. Otherwise, the window_size and strides will generated adaptively to
        match self._num_sample_per_dim.
        """
        window_size = tuple(
            [inputs.shape[1]]
            + [x // self._num_sample_per_dim if x > self._num_sample_per_dim else 3 for x in inputs.shape[2:]])
        strides = tuple(
            [inputs.shape[1]]
            + [x // self._num_sample_per_dim if x > self._num_sample_per_dim else 1 for x in inputs.shape[2:]])
        return window_size, strides

    @staticmethod
    def _generate_masks(inputs, window_size, strides):
        """Generate masks to perturb contiguous regions."""
        total_dim = np.prod(inputs.shape[1:]).item()
        template = np.arange(total_dim).reshape(inputs.shape[1:])
        indices = _generate_patches(template, window_size, strides)
        num_perturbations = indices.shape[0]
        indices = indices.reshape(num_perturbations, -1)

        mask = np.zeros((num_perturbations, total_dim), dtype=bool)
        for i in range(num_perturbations):
            mask[i, indices[i]] = True
        mask = mask.reshape((num_perturbations,) + inputs.shape[1:])

        masks = np.tile(mask, reps=(inputs.shape[0],) + (1,) * len(mask.shape))
        return masks
