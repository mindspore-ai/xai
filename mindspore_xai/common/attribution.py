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
"""Attribution."""

from typing import Callable
from math import ceil
import itertools

import mindspore as ms
import mindspore.nn as nn
from mindspore.train._utils import check_value_type
import matplotlib.pyplot as plt

from mindspore_xai.common import utils
from mindspore_xai.visual.cv.saliency import np_saliency_to_image


# max no. of image per row for display
_MAX_IMG_PER_ROW = 4


class Attribution:
    """
    Basic class of attributing the salient score

    The explainers which explanation through attributing the relevance scores should inherit this class.

    Args:
        network (nn.Cell): The black-box model to be explained.
    """
    def __init__(self, network):
        check_value_type("network", network, nn.Cell)
        self._network = network
        self._network.set_train(False)
        self._network.set_grad(False)

    def _postproc_saliency(self, saliency, ret, show):
        """Post-process saliency."""
        if show is None:
            show = utils.is_notebook()

        if ret == 'tensor' and not show:
            return saliency

        saliency_images = self._to_saliency_images(saliency)

        if show:
            self._show_saliency_images(saliency_images)

        if ret == 'image':
            return saliency_images

        return saliency

    @staticmethod
    def _to_saliency_images(saliency):
        """Convert saliency tensor to PIL images."""
        saliency_np = saliency.asnumpy()
        saliency_images = []
        for saliency_sample in saliency_np:
            images = []
            for saliency_label in saliency_sample:
                img = np_saliency_to_image(saliency_label)
                images.append(img)
            saliency_images.append(images)
        return saliency_images

    @staticmethod
    def _show_saliency_images(saliency_images):
        """Show saliency PIL images."""
        sample_count = len(saliency_images)
        label_count = len(saliency_images[0])

        if label_count == 1:
            if sample_count == 1:
                saliency_images[0][0].show()
                return
            saliency_images = list(itertools.chain(*saliency_images))
            col_count = min(sample_count, _MAX_IMG_PER_ROW)
            row_count = int(ceil(sample_count / col_count))
            fig = plt.figure()
            for i, saliency_image in enumerate(saliency_images):
                fig.add_subplot(row_count, col_count, i + 1)
                plt.imshow(saliency_image)
                plt.title(f'[{i}][0]')
                plt.axis('off')
            plt.show()
        else:
            fig = plt.figure()
            col_count = min(label_count, _MAX_IMG_PER_ROW)
            row_per_sample = int(ceil(label_count / col_count))
            row_count = sample_count * row_per_sample
            for i, saliency_images0 in enumerate(saliency_images):
                offset = i * row_per_sample * col_count
                for j, saliency_image in enumerate(saliency_images0):
                    fig.add_subplot(row_count, col_count, offset + j + 1)
                    plt.imshow(saliency_image)
                    plt.title(f'[{i}][{j}]')
                    plt.axis('off')
            plt.show()

    @staticmethod
    def _verify_network(network):
        """Verify the input `network` for __init__ function."""
        if not isinstance(network, nn.Cell):
            raise TypeError("The parsed `network` must be a `mindspore.nn.Cell` object.")

    __call__: Callable
    """
    The explainers return the explanations by calling directly on the explanation.
    Derived class should overwrite this implementations for different
    algorithms.

    Args:
        input (ms.Tensor): Input tensor to be explained.

    Returns:
        - saliency map (ms.Tensor): saliency map of the input.
    """

    @property
    def network(self):
        """Return the model."""
        return self._network

    @staticmethod
    def _verify_other_args(ret, show):
        """Verify the validity of other arguments."""
        check_value_type('ret', ret, str)
        if ret not in ('tensor', 'image'):
            raise ValueError(f"Unrecognized argument ret:'{ret}', must be either 'tensor' or 'image'.")
        if show is not None:
            check_value_type('show', show, bool)

    @staticmethod
    def _verify_data(inputs, targets):
        """Verify the validity of the parsed inputs."""
        check_value_type('inputs', inputs, ms.Tensor)
        if len(inputs.shape) != 4:
            raise ValueError('Argument inputs must be 4D Tensor')
        check_value_type('targets', targets, (ms.Tensor, int, tuple, list))
        if isinstance(targets, ms.Tensor):
            if len(targets.shape) > 1 or (len(targets.shape) == 1 and len(targets) != len(inputs)):
                raise ValueError('Argument targets must be a 1D or 0D Tensor. If it is a 1D Tensor, '
                                 'it should has the same length as inputs.')
        elif isinstance(targets, (tuple, list)):
            if len(targets) != len(inputs):
                raise ValueError('Argument targets must has the same length as inputs.')
        elif inputs.shape[0] != 1:
            raise ValueError('If targets have type of int, batch_size of inputs should equals 1. Receive batch_size {}'
                             .format(inputs.shape[0]))
