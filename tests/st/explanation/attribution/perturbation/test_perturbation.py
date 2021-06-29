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
"""Tests of perturbation explainers of xai.explanation."""

import numpy as np
from numpy import random
import pytest
import mindspore as ms
from mindspore import context
from mindspore import nn

from xai.explanation import RISE, RISEPlus, Occlusion, OODNet, OODUnderlying

context.set_context(mode=context.PYNATIVE_MODE)

C, H, W = 3, 16, 16
num_classes = 2
seed = 24232


class CustomNet(nn.Cell):
    """Simple net for unit test."""
    def __init__(self):
        super(CustomNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(in_channels=C*H*W, out_channels=num_classes)

    def construct(self, x):
        x = self.flatten(x)
        out = self.fc(x)
        return out


class ActivationFn(nn.Cell):
    """Simple activation function for unit test."""
    def construct(self, x):
        return x


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


class TestPerturbation:
    """Unit test for perturbation explainers."""

    def setup_method(self):
        """Setup method."""
        self.net = CustomNet()
        self.activation_fn = ActivationFn()

    @staticmethod
    def _test_2d(batch_size, explainer, test_multi_targets=True):
        """Generic testcase to make sure no runtime error."""
        random.seed(seed)
        x = ms.Tensor(random.random((batch_size, C, H, W)), ms.float32)

        if test_multi_targets:
            # multiple targets
            targets = [list(range(num_classes)) for _ in range(batch_size)]
            targets = ms.Tensor(targets, ms.int32)
            saliency = explainer(x, targets)
            assert tuple(saliency.shape) == (batch_size, num_classes, H, W)
            assert np.any(saliency.asnumpy() != 0)

        # single integer target
        saliency = explainer(x, 0)
        assert tuple(saliency.shape) == (batch_size, 1, H, W)
        assert np.any(saliency.asnumpy() != 0)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_rise_2d(self):
        """Test for RISE."""
        explainer = RISE(self.net, self.activation_fn, perturbation_per_eval=1)
        return self._test_2d(2, explainer)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_riseplus_2d(self):
        """Test for RISEPlus."""
        explainer = RISEPlus(OODNet(CustomOODUnderlying(), num_classes), self.net,
                             self.activation_fn, perturbation_per_eval=1)
        # RISEPlus doesn't support batch size > 1, may need to change in the future
        return self._test_2d(1, explainer)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_occlusion_2d(self):
        """Test for Occlusion."""
        explainer = Occlusion(self.net, self.activation_fn, perturbation_per_eval=1)
        return self._test_2d(1, explainer, test_multi_targets=False)
