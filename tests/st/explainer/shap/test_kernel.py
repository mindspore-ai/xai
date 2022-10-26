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
"""Tests of SHAP kernel explainer."""
import pytest
import mindspore as ms
from mindspore import set_context, PYNATIVE_MODE, GRAPH_MODE
import sklearn
import numpy as np

from mindspore_xai.explainer import SHAPKernel
from .conftest import NUM_INPUTS, NUM_FEATURES, NUM_TRAINING_DATA, NUM_CLASSES

set_context(mode=PYNATIVE_MODE)


@pytest.fixture(scope='module', name="classification_net_shap")
def fixture_classification_net_shap(classification_net, training_data_tensor):
    """fixture classification net shap."""
    return SHAPKernel(classification_net, training_data_tensor, num_neighbours=10)


@pytest.fixture(scope='module', name="regression_net_shap")
def fixture_regression_net_shap(regression_net, training_data_tensor):
    """fixture regression net shap."""
    return SHAPKernel(regression_net, training_data_tensor, num_neighbours=10)


class TestSHAPKernel:
    """Unit test for SHAP kernel explainer."""

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

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_network_sklearn_classifier(self, training_data_np):
        """sklearn.ensemble.RandomForestClassifier."""
        # labels_train has 3 classes
        ms.set_seed(123)
        labels_train = np.zeros(NUM_TRAINING_DATA)
        for i in range(NUM_TRAINING_DATA):
            labels_train[i] = i % NUM_CLASSES

        model = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
        model.fit(training_data_np, labels_train)

        shap = SHAPKernel(model.predict_proba, training_data_np, num_neighbours=10)

        inputs = np.random.rand(NUM_INPUTS, NUM_FEATURES)
        targets = np.array([[0, 1, 2], [0, 1, 2]])
        exps = shap(inputs, targets)
        assert exps.shape == (NUM_INPUTS, 3, NUM_FEATURES)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_network_sklearn_regressor(self, training_data_np):
        """sklearn.ensemble.RandomForestRegressor."""
        ms.set_seed(123)
        labels_train = np.random.rand(NUM_TRAINING_DATA)
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=10)
        model.fit(training_data_np, labels_train)

        shap = SHAPKernel(model.predict, training_data_np, num_neighbours=10)

        inputs = np.random.rand(NUM_INPUTS, NUM_FEATURES)
        targets = 0
        exps = shap(inputs, targets)
        assert exps.shape == (NUM_INPUTS, 1, NUM_FEATURES)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_graph_mode(self, classification_net_shap, inputs):
        """mode is GRAPH_MODE."""
        set_context(mode=GRAPH_MODE)
        targets = ms.Tensor([[0, 1, 2], [0, 1, 2]], ms.int32)
        exps = classification_net_shap(inputs, targets)
        assert exps.shape == (NUM_INPUTS, 3, NUM_FEATURES)
        set_context(mode=PYNATIVE_MODE)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_callable_function_with_tensor_input(self, training_data_tensor, inputs):
        """predictor is a callable function, input is tensor."""
        linear = ms.nn.Dense(4, 3, activation=ms.nn.Softmax())

        def predict_fn(x):
            return linear(x)
        shap = SHAPKernel(predict_fn, training_data_tensor)
        targets = 0
        exps = shap(inputs, targets)
        assert exps.shape == (NUM_INPUTS, 1, NUM_FEATURES)
