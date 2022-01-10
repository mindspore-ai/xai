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
"""Tests of LIME Tabular explainers of xai.explanation."""
import numpy as np
from numpy import random
import mindspore as ms
from mindspore import nn

from mindspore_xai.explainer import LIMETabular

num_training_data = 10
num_inputs = 2
num_features = 4
num_classes = 3


class ClassificationNet(nn.Cell):
    """classification net for unit test."""

    def __init__(self):
        super(ClassificationNet, self).__init__()
        self.fc = nn.Dense(num_features, num_classes, activation=nn.Softmax())

    def construct(self, x):
        x = self.fc(x)
        return x


class RegressionNet(nn.Cell):
    """regression net for unit test."""

    def __init__(self):
        super(RegressionNet, self).__init__()
        self.fc = nn.Dense(num_features, 1)

    def construct(self, x):
        x = self.fc(x)
        return x


def eval_exps_dims(exps, _num_inputs, _num_classes, _num_features):
    assert len(exps) == _num_inputs
    for sample_exp in exps:
        assert len(sample_exp) == _num_classes
        for class_exp in sample_exp:
            assert len(class_exp) == _num_features
            for feature_exp in class_exp:
                assert type(feature_exp) == tuple
                assert len(feature_exp) == 2
                assert type(feature_exp[0]) == str
                assert type(feature_exp[1]) == float


class TestLIMETabular:
    """Unit test for perturbation explainers."""

    def setup_method(self):
        """Setup method."""
        self.classification_net = ClassificationNet()
        self.regression_net = RegressionNet()
        self.training_data_np = np.random.rand(num_training_data, num_features)
        self.training_data = ms.Tensor(self.training_data_np, ms.float32)
        self.inputs = ms.Tensor(np.random.rand(num_inputs, num_features), ms.float32)
        self.lime = LIMETabular(self.classification_net, self.training_data)

    def test_training_data_tensor(self):
        LIMETabular(self.classification_net, self.training_data)

    def test_training_data_numpy(self):
        LIMETabular(self.classification_net, self.training_data_np)

    def test_targets_int(self):
        targets = 0
        exps = self.lime(self.inputs, targets,  num_samples=10)
        eval_exps_dims(exps, num_inputs, 1, num_features)

    def test_targets_1d_tensor(self):
        targets = ms.Tensor([0, 1], ms.int32)
        exps = self.lime(self.inputs, targets, num_samples=10)
        eval_exps_dims(exps, num_inputs, 1, num_features)

    def test_targets_2d_tensor(self):
        targets = ms.Tensor([[0, 1, 2], [0, 1, 2]], ms.int32)
        exps = self.lime(self.inputs, targets,  num_samples=10)
        eval_exps_dims(exps, num_inputs, 3, num_features)

    def test_network_regression(self):
        lime = LIMETabular(self.regression_net, self.training_data)
        targets = 0
        exps = lime(self.inputs, targets,  num_samples=10)
        eval_exps_dims(exps, num_inputs, 1, num_features)
