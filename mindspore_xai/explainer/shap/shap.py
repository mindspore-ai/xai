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
"""SHAP explainers base class."""
import mindspore as ms
from mindspore import Tensor
from mindspore.common.initializer import Zero
from mindspore.train._utils import check_value_type
import numpy as np
import matplotlib.pyplot as plt

from mindspore_xai.third_party.shap.shap.plots import waterfall


class _SHAP:
    """SHAP explainers base class."""
    def __init__(self, predictor, data, feature_names=None, class_names=None):
        """SHAP explainer base class."""
        if issubclass(type(predictor), ms.nn.Cell):
            check_value_type("data", data, ms.Tensor)
            predictor.set_train(False)
            predictor.set_grad(False)

        if len(data.shape) != 2:
            raise ValueError('Dimension invalid. `data` should be 2D. '
                             'But got {}D.'.format(len(data.shape)))

        self._predictor = predictor

        outputs = self._get_predictor_output(predictor, data[:1])
        if len(outputs.shape) > 1:
            self._predictor_num_outputs = outputs.shape[1]
        else:
            self._predictor_num_outputs = 1
        if self._predictor_num_outputs > 1:
            self._mode = "classification"
        else:
            self._mode = "regression"

        if feature_names is None:
            feature_names = ['feature {}'.format(i) for i in range(data.shape[1])]
        self._feature_names = feature_names

        if class_names is None:
            class_names = [str(i) for i in range(self._predictor_num_outputs)]
        self._class_names = class_names

    @staticmethod
    def _get_predictor_output(predictor, data):
        """Get predictor output."""
        if hasattr(predictor, "predict_proba"):
            return predictor.predict_proba(data)
        if hasattr(predictor, "predict"):
            return predictor.predict(data)
        return predictor(data)

    def _unify_targets(self, inputs, targets):
        """To unify targets to be 2D numpy.ndarray."""
        if self._mode == 'regression':
            targets = 0
        if isinstance(targets, int):
            return np.array([[targets] for _ in inputs]).astype(int)
        if isinstance(targets, Tensor):
            if not targets.shape:
                return np.array([[targets.asnumpy()] for _ in inputs]).astype(int)
            if len(targets.shape) == 1:
                return np.array([[t.asnumpy()] for t in targets]).astype(int)
            if len(targets.shape) == 2:
                return np.array([t.asnumpy() for t in targets]).astype(int)
        if isinstance(targets, np.ndarray):
            if not targets.shape:
                return np.array([[targets] for _ in inputs]).astype(int)
            if len(targets.shape) == 1:
                return np.array([[t] for t in targets]).astype(int)
        if isinstance(targets, list):
            return np.array(targets)
        return targets

    def _show_all(self, exps, targets, base_values, num_features):
        """Show the explanation."""
        if isinstance(exps, Tensor):
            exps = exps.asnumpy()

        for i, _ in enumerate(exps):
            for j, _ in enumerate(exps[i]):
                phis = exps[i][j]
                label = targets[i][j]
                class_name = self._class_names[label]

                if isinstance(base_values, float):
                    base_value = base_values
                else:
                    base_value = base_values[label]

                waterfall(base_value, phis, self._feature_names, phis, i, class_name, self._mode,
                          max_display=num_features)

        plt.show()

    @staticmethod
    def _reshape_output(output, targets):
        """From [(samples, features), ...] to (samples, labels, features)."""
        num_labels = targets.shape[1]
        num_inputs, num_features = output[0].shape
        exps = Tensor(shape=(num_inputs, num_labels, num_features), dtype=ms.float32,
                      init=Zero()).init_data()
        for i in range(num_inputs):
            sample_exp = Tensor(shape=(num_labels, num_features), dtype=ms.float32,
                                init=Zero()).init_data()
            for j in range(num_labels):
                label = targets[i][j]
                sample_exp[j] = Tensor(output[label][i])

            exps[i] = sample_exp

        return exps
