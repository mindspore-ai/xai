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
"""Tests of LIME Tabular explainers."""
from pathlib import Path
import tempfile
import numpy as np
import pytest
import mindspore as ms
from mindspore import nn, set_context, PYNATIVE_MODE, GRAPH_MODE
import sklearn.ensemble

from mindspore_xai.explainer import LIMETabular

set_context(mode=PYNATIVE_MODE)

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


def eval_exps_dims(exps, num_inputs, num_classes, num_features):
    """evaluate the explanation dimensions."""
    assert len(exps) == num_inputs
    for sample_exp in exps:
        assert len(sample_exp) == num_classes
        for class_exp in sample_exp:
            assert len(class_exp) == num_features
            for feature_exp in class_exp:
                assert isinstance(feature_exp, tuple)
                assert len(feature_exp) == 2
                assert isinstance(feature_exp[0], str)
                assert isinstance(feature_exp[1], float)


@pytest.fixture(scope='session', name="classification_net")
def fixture_classification_net():
    """fixture classification net."""
    return ClassificationNet()


@pytest.fixture(scope='session', name="regression_net")
def fixture_regression_net():
    """fixture regression net."""
    return RegressionNet()


@pytest.fixture(scope='session', name="training_data_np")
def fixture_training_data_np():
    """fixture training data in numpy array format."""
    return np.random.rand(NUM_TRAINING_DATA, NUM_FEATURES)


@pytest.fixture(scope='session', name="training_data_tensor")
def fixture_training_data_tensor(training_data_np):
    """fixture training data in Mindspore Tensor format."""
    return ms.Tensor(training_data_np, ms.float32)


@pytest.fixture(scope='session', name="training_data_stats")
def fixture_training_data_stats(training_data_np):
    """fixture training data stats."""
    return LIMETabular.to_feat_stats(training_data_np)


@pytest.fixture(scope='session', name="inputs")
def fixture_inputs():
    """fixture inputs."""
    return ms.Tensor(np.random.rand(NUM_INPUTS, NUM_FEATURES), ms.float32)


@pytest.fixture(scope='session', name="classification_net_lime")
def fixture_classification_net_lime(classification_net, training_data_stats):
    """fixture classification net lime."""
    return LIMETabular(classification_net, training_data_stats, num_perturbs=10)


@pytest.fixture(scope='session', name="regression_net_lime")
def fixture_regression_net_lime(regression_net, training_data_stats):
    """fixture regression net lime."""
    return LIMETabular(regression_net, training_data_stats, num_perturbs=10)


class TestLIMETabular:
    """Unit test for perturbation explainers."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_tensor_to_feat_stats(self, training_data_tensor):
        """training data is a tensor."""
        assert isinstance(LIMETabular.to_feat_stats(training_data_tensor), dict)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_numpy_to_feat_stats(self, training_data_np):
        """training data is a numpy array."""
        assert isinstance(LIMETabular.to_feat_stats(training_data_np), dict)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_save_feat_stats_to_str(self, training_data_stats):
        """save training data stats to str file"""
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp_file:
            f = str(tmp_file)
            LIMETabular.save_feat_stats(training_data_stats, f)
            assert isinstance(LIMETabular.load_feat_stats(f), dict)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_save_feat_stats_to_path(self, training_data_stats):
        """save training data stats to pathlib file"""
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp_file:
            f = Path(str(tmp_file))
            LIMETabular.save_feat_stats(training_data_stats, f)
            assert isinstance(LIMETabular.load_feat_stats(f), dict)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_save_feat_stats_to_filestream(self, training_data_stats):
        """save training data stats to filestream"""
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp_file:
            with open(str(tmp_file), "w", encoding="utf-8") as f:
                LIMETabular.save_feat_stats(training_data_stats, f)
            with open(str(tmp_file), "r", encoding="utf-8") as f:
                assert isinstance(LIMETabular.load_feat_stats(f), dict)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_targets_int(self, classification_net_lime, inputs):
        """targets is int."""
        targets = 0
        exps = classification_net_lime(inputs, targets)
        eval_exps_dims(exps, NUM_INPUTS, 1, NUM_FEATURES)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_targets_1d_tensor(self, classification_net_lime, inputs):
        """targets is 1d tensor."""
        targets = ms.Tensor([0, 1], ms.int32)
        exps = classification_net_lime(inputs, targets)
        eval_exps_dims(exps, NUM_INPUTS, 1, NUM_FEATURES)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_targets_2d_tensor(self, classification_net_lime, inputs):
        """targets is 2d tensor."""
        targets = ms.Tensor([[0, 1, 2], [0, 1, 2]], ms.int32)
        exps = classification_net_lime(inputs, targets)
        eval_exps_dims(exps, NUM_INPUTS, 3, NUM_FEATURES)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_network_regression(self, regression_net_lime, inputs):
        """regression network lime."""
        targets = 0
        exps = regression_net_lime(inputs, targets)
        eval_exps_dims(exps, NUM_INPUTS, 1, NUM_FEATURES)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_sklearn_model(self, training_data_np, training_data_stats):
        """SKlearn model"""
        # labels_train has 3 classes
        labels_train = np.zeros(NUM_TRAINING_DATA)
        for i in range(NUM_TRAINING_DATA):
            labels_train[i] = i % NUM_CLASSES

        model = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
        model.fit(training_data_np, labels_train)
        lime = LIMETabular(model.predict_proba, training_data_stats, num_perturbs=10)

        inputs = np.random.rand(NUM_INPUTS, NUM_FEATURES)
        targets = np.array([[0, 1, 2], [0, 1, 2]])
        exps = lime(inputs, targets)
        eval_exps_dims(exps, NUM_INPUTS, 3, NUM_FEATURES)
        # test 1D numpy array targets
        targets = np.array([0, 1])
        lime(inputs, targets)
        # test int targets
        targets = 0
        lime(inputs, targets)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_graph_mode(self, classification_net_lime, inputs):
        """mode is GRAPH_MODE."""
        set_context(mode=GRAPH_MODE)
        targets = ms.Tensor([[0, 1, 2], [0, 1, 2]], ms.int32)
        exps = classification_net_lime(inputs, targets)
        eval_exps_dims(exps, NUM_INPUTS, 3, NUM_FEATURES)
        set_context(mode=PYNATIVE_MODE)
