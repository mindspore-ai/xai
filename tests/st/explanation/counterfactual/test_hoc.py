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
"""Tests of xai.explanation.hierarchical_occlusion."""

import numpy as np
import pytest
import mindspore as ms
import mindspore.ops as P
from mindspore import nn

from xai.explanation import hierarchical_occlusion as hoc


class PseudoNet(nn.Cell):
    """ Pseudo model for the unit test. """
    def __call__(self, x):
        slice = P.Slice()
        batch_size = x.shape[0]
        output = np.zeros((batch_size, 1))
        for i in range(batch_size):
            sliced0 = slice(x, (i, 0, 0, 0), (1, 3, 2, 4)).asnumpy()
            sliced1 = slice(x, (i, 0, 8, 6), (1, 3, 2, 4)).asnumpy()
            output[i, 0] = np.mean((np.mean(sliced0), np.mean(sliced1)))
        return ms.Tensor(output, dtype=ms.float32)


def _assert_result(root_step):
    """ Assert the search result nodes. """
    nodes = root_step.get_layer_steps(1)
    assert len(nodes) == 4
    nodes.sort(key=lambda n: n.x)
    expected_xy = [(0, 0), (2, 0), (6, 8), (8, 8)]
    for i, node in enumerate(nodes):
        assert node.x == expected_xy[i][0]
        assert node.y == expected_xy[i][1]
        assert node.width == 2
        assert node.height == 2


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_searcher():
    """ Unit test for single threaded search. """

    net = PseudoNet()
    searcher = hoc.Searcher(network=net,
                            win_sizes=[5, 2],
                            strides=[2, 1],
                            threshold=0.01,
                            by_masking=True)
    image = np.ones((3, 10, 10), dtype=np.float32)
    root_step, _ = searcher.search(image, 0, mask=0.0)

    _assert_result(root_step)
