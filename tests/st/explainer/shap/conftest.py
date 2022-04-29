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
"""SHAP explainer test fixtures."""
import numpy as np
import pytest
import mindspore as ms
from mindspore import nn

NUM_TRAINING_DATA = 10
NUM_INPUTS = 2
NUM_FEATURES = 4
NUM_CLASSES = 3


class ClassificationNet(nn.Cell):
    """classification net for unit test."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(NUM_FEATURES, NUM_CLASSES, activation=nn.Softmax())

    def construct(self, x):
        """construct function."""
        x = self.fc(x)
        return x


class RegressionNet(nn.Cell):
    """regression net for unit test."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(NUM_FEATURES, 1)

    def construct(self, x):
        """construct function."""
        x = self.fc(x)
        return x


@pytest.fixture(scope='package', name="classification_net")
def fixture_classification_net():
    """fixture classification net."""
    return ClassificationNet()


@pytest.fixture(scope='package', name="regression_net")
def fixture_regression_net():
    """fixture regression net."""
    return RegressionNet()


@pytest.fixture(scope='package', name="training_data_np")
def fixture_training_data_np():
    """fixture training data in numpy array format."""
    return np.random.rand(NUM_TRAINING_DATA, NUM_FEATURES)


@pytest.fixture(scope='package', name="training_data_tensor")
def fixture_training_data_tensor(training_data_np):
    """fixture training data in Mindspore Tensor format."""
    return ms.Tensor(training_data_np, ms.float32)


@pytest.fixture(scope='package', name="inputs")
def fixture_inputs():
    """fixture inputs."""
    return ms.Tensor(np.random.rand(NUM_INPUTS, NUM_FEATURES), ms.float32)
