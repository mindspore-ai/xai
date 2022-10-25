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
"""Tests of Pseudo Linear Coefficients of mindspore_xai.explainer."""

import pytest
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, set_context, PYNATIVE_MODE, GRAPH_MODE
from mindspore_xai.explainer import PseudoLinearCoef


class Classifier(nn.Cell):
    def construct(self, x):
        y = ops.Zeros()((x.shape[0], 3), ms.float32)
        y[:, 0] = -x[:, 0] + x[:, 1] + x[:, 2] - 0.5
        y[:, 1] = x[:, 0] - x[:, 1] + x[:, 2] - 0.5
        y[:, 2] = x[:, 0] + x[:, 1] - x[:, 2] - 0.5
        return ops.Sigmoid()(y * 10)


net = Classifier()


def classifier_fn(x):
    return net(x)


class TestPseudoLinearCoef:
    """Unit test for Pseudo Linear Coefficients."""

    def compute_plc(self, classifier_input):
        """Compute PLC, Relative PLC and test their outputs."""
        ms.set_seed(123)
        num_classes = 3
        num_samples = 100
        num_features = 3
        explainer = PseudoLinearCoef(classifier_input, num_classes=num_classes)
        features = Tensor(np.random.uniform(size=(num_samples, num_features)), dtype=ms.float32)
        plc, relative_plc = explainer(features)
        assert isinstance(plc, Tensor)
        assert plc.shape == (num_classes, num_features)
        assert isinstance(relative_plc, Tensor)
        assert relative_plc.shape == (num_classes, num_classes, num_features)

        assert (plc[0, 0] < 0 < plc[0, 1]) and (plc[0, 2] > 0)
        assert (plc[1, 0] > 0) and (plc[1, 1] < 0 < plc[1, 2])
        assert (plc[2, 0] > 0) and (plc[2, 1] > 0 > plc[2, 2])

        assert (relative_plc[0, 1, 0] < 0) and (relative_plc[0, 2, 0] < 0)
        assert (relative_plc[1, 0, 1] < 0) and (relative_plc[1, 2, 1] < 0)
        assert (relative_plc[2, 0, 2] < 0) and (relative_plc[2, 1, 2] < 0)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_plc_pynative_cell(self):
        """Test PLC with Cell input under pynative mode."""
        set_context(mode=PYNATIVE_MODE)
        self.compute_plc(net)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_plc_graph_cell(self):
        """Test PLC with Cell input under graph mode."""
        set_context(mode=GRAPH_MODE)
        self.compute_plc(net)
        set_context(mode=PYNATIVE_MODE)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_plc_pynative_callable(self):
        """Test PLC with callable input under pynative mode."""
        set_context(mode=PYNATIVE_MODE)
        self.compute_plc(classifier_fn)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_plc_graph_callable(self):
        """Test PLC with callable input under graph mode."""
        set_context(mode=GRAPH_MODE)
        self.compute_plc(classifier_fn)
        set_context(mode=PYNATIVE_MODE)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_plc_normalize_without_vec(self):
        """Test normalize function of PLC."""
        plc = Tensor([[0.1, 0.6, 0.8], [-2, 0.2, 0.4], [0.4, 0.1, -0.1]])
        norm_plc = PseudoLinearCoef.normalize(plc)
        assert isinstance(norm_plc, Tensor)
        assert plc.shape == norm_plc.shape
        assert np.array_equal(norm_plc, Tensor([[0.05, 0.3, 0.4], [-1, 0.1, 0.2], [0.2, 0.05, -0.05]]))

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_plc_normalize_with_vec(self):
        """Test normalize function of PLC."""
        plc = Tensor([[0.1, 0.6, 0.8], [-2, 0.2, 0.4], [0.4, 0.1, -0.1]])
        norm_plc = PseudoLinearCoef.normalize(plc, per_vector=True)
        assert isinstance(norm_plc, Tensor)
        assert plc.shape == norm_plc.shape
        assert np.array_equal(norm_plc, Tensor([[0.125, 0.75, 1.], [-1, 0.1, 0.2], [1., 0.25, -0.25]]))
