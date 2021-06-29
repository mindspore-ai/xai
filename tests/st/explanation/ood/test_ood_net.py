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
"""Tests of OoD networks."""

import numpy as np
from numpy import random
import pytest
import mindspore as ms
from mindspore import context
from mindspore import nn
import mindspore.dataset as de

from xai.explanation import OODNet, OODUnderlying

context.set_context(mode=context.PYNATIVE_MODE)

num_classes = 3
num_samples = 8
batch_size = 2
C, H, W = 3, 16, 16
seed = 23412


class CustomOODUnderlying(OODUnderlying):
    """Simple OOD underlying for unit test."""
    def __init__(self):
        super(CustomOODUnderlying, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(in_channels=C*H*W, out_channels=5)
        self.fc2 = nn.Dense(in_channels=5, out_channels=num_classes)

    def construct(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        if self.output_feature:
            return x
        x = self.fc2(x)
        return x

    @property
    def feature_count(self):
        return 5


def _ds_generator():
    """Dataset generator."""
    for _ in range(num_samples):
        image = random.random((C, H, W)).astype(np.float32)
        labels = random.randint(0, num_classes, 2)
        one_hot = np.zeros(num_classes, dtype=np.float32)
        for label in labels:
            one_hot[label] = 1.0
        yield image, one_hot


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ood_net_2d():
    """Test  of OoD networks with 2d input images."""
    random.seed(seed)
    ms.set_seed(seed)

    dataset = de.GeneratorDataset(source=_ds_generator, num_samples=num_samples,
                                  column_names=['data', 'label'], column_types=[ms.float32, ms.float32])
    dataset = dataset.batch(batch_size)

    classifier = CustomOODUnderlying()
    ood_net = OODNet(classifier, num_classes)
    ood_net.train(dataset, epoch=1, loss_fn=nn.BCEWithLogitsLoss())

    batch_x = ms.Tensor(random.random((batch_size, C, H, W)), dtype=ms.float32)
    # invoke ood.net.score() after ood_net.train() may cause error in MS1.3, use odd_net() for the moment.
    ood_scores = ood_net(batch_x)
    assert tuple(ood_scores.shape) == (batch_size, num_classes)
    assert np.any(ood_scores.asnumpy() != 0)
