# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Tests of SHAP gradient explainer."""
import pytest
import mindspore as ms
from mindspore import context

from mindspore_xai.explainer import SHAPGradient
from .conftest import NUM_INPUTS, NUM_FEATURES

context.set_context(mode=context.PYNATIVE_MODE)


@pytest.fixture(scope='module', name="classification_net_shap")
def fixture_classification_net_shap(classification_net, training_data_tensor):
    """fixture classification net shap."""
    return SHAPGradient(classification_net, training_data_tensor, num_neighbours=10)


@pytest.fixture(scope='module', name="regression_net_shap")
def fixture_regression_net_shap(regression_net, training_data_tensor):
    """fixture regression net shap."""
    return SHAPGradient(regression_net, training_data_tensor, num_neighbours=10)


class TestSHAPGradient:
    """Unit test for SHAP gradient explainer."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_targets_int(self, classification_net_shap, inputs):
        """targets is int."""
        targets = 0
        exps = classification_net_shap(inputs, targets)
        assert exps.shape == (NUM_INPUTS, 1, NUM_FEATURES)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_targets_1d_tensor(self, classification_net_shap, inputs):
        """targets is 1d tensor."""
        targets = ms.Tensor([0, 1], ms.int32)
        exps = classification_net_shap(inputs, targets)
        assert exps.shape == (NUM_INPUTS, 1, NUM_FEATURES)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_targets_2d_tensor(self, classification_net_shap, inputs):
        """targets is 2d tensor."""
        targets = ms.Tensor([[0, 1, 2], [0, 1, 2]], ms.int32)
        exps = classification_net_shap(inputs, targets)
        assert exps.shape == (NUM_INPUTS, 3, NUM_FEATURES)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_network_regression(self, regression_net_shap, inputs):
        """regression network."""
        targets = 0
        exps = regression_net_shap(inputs, targets)
        assert exps.shape == (NUM_INPUTS, 1, NUM_FEATURES)
