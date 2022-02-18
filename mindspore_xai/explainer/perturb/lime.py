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
"""LIME Tabular explainer."""
import json
from io import IOBase

import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore.train._utils import check_value_type

from mindspore_xai.common.utils import is_notebook
from mindspore_xai.third_party.lime.lime.lime_tabular import LimeTabularExplainer


class LIMETabular:
    r"""
    Provides Lime Tabular explanation method.

    Explains predictions on tabular (i.e. matrix) data. For numerical features, perturb them by sampling from a
    Normal(0,1) and doing the inverse operation of mean-centering and scaling, according to the means and stds in the
    training data. For categorical features, perturb by sampling according to the training distribution, and making a
    binary feature that is 1 when the value is the same as the instance being explained.

    Note:
        The parsed `predictor` will be set to eval mode through `predictor.set_grad(False)` and
        `predictor.set_train(False)`. If you want to train the `predictor` afterwards, please reset it back to training
        mode through the opposite operations.

    Args:
        predictor (Cell, Callable): The black-box model to be explained, or a callable function.
        training_data_stats (dict): a dict object having the details of training data statistics. The stats can be
            generated using static method LIETabular.to_training_data_stats(training_data).
        feature_names (list, optional): list of names (strings) corresponding to the columns in the training data.
            Default: `None`.
        categorical_features (list, optional): list of indices (ints) corresponding to the categorical columns.
            Everything else will be considered continuous. Values in these columns MUST be integers. Default: `None`.
        class_names (list, optional): list of class names, ordered according to whatever the classifier is using. If
            not present, class names will be '0', '1', ... Default: `None`.
        random_state (int, optional): an integer that will be used to generate random numbers. If None, the random
            state will be initialized using the internal numpy seed. Default: `None`.

    Inputs:
        - **inputs** (Tensor, numpy.ndarray) - The input data to be explained, a 2D tensor or 2D numpy array of
          shape :math:`(N, K)`.
        - **targets** (Tensor, numpy.ndarray, list, int, optional) - The labels of interest to be explained. When
          `targets` is an integer, all the inputs will generate attribution map w.r.t this integer. When `targets` is a
          tensor or numpy array or list, it should be of shape :math:`(N, l)` (l being the number of labels for each
          sample) or :math:`(N,)` :math:`()`. Default: 0.
        - **num_samples** (int, optional): size of the neighborhood to learn the linear model. Default: 5000.
        - **num_features** (int, optional): Maximum number of features present in explanation. Default: 10.
        - **show** (bool, optional): Show the explanation figures, `None` means auto. Default: `None`.

    Outputs:
        list[list[list[(str, float)]]], for nth sample, jth label, kth feature description and weight.

    Raises:
        TypeError: Be raised for any argument or input type problem.
        ValueError: Be raised for any input value problem.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>>
        >>> from mindspore_xai.explainer import LIMETabular
        >>>
        >>> # Linear classification model
        >>> class LinearNet(nn.Cell):
        >>>     def __init__(self, num_inputs, num_class):
        >>>         super(LinearNet, self).__init__()
        >>>         self.fc = nn.Dense(num_inputs, num_class, activation=nn.Softmax())
        >>>
        >>>     def construct(self, x):
        >>>         x = self.fc(x)
        >>>         return x
        >>>
        >>> net = LinearNet(4, 3)
        >>> # use iris data as example
        >>> feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        >>> class_names = ['setosa', 'versicolor', 'virginica']
        >>> training_data = ms.Tensor(np.random.rand(10, 4), ms.float32)
        >>> # initialize LIMETabular explainer with the model and training data
        >>> lime = LIMETabular(net, training_data, feature_names=feature_names, class_names=class_names)
        >>> inputs = ms.Tensor(np.random.rand(2, 4), ms.float32)
        >>> targets = ms.Tensor([[1, 2], [1, 2]], ms.int32)
        >>> exps = lime(inputs, targets)
        >>> for i, sample_exp in enumerate(exps):
        >>>     for j, class_exp in enumerate(sample_exp):
        >>>         print('Local explanation for sample {} class {}'.format(i, class_names[targets[i][j]]))
        >>>         print(class_exp)
        Local explanation for sample 0 class versicolor
        [('sepal width (cm) <= 0.25', 0.001606624848945185), ...]
        Local explanation for sample 0 class virginica
        [('petal length (cm) <= 0.29', 0.0033004810806473504), ...]
        Local explanation for sample 1 class versicolor
        [('petal width (cm) <= 0.20', 0.002856020026411598), ...]
        Local explanation for sample 1 class virginica
        [('petal length (cm) > 0.80', -0.0031887318983352536), ...]
"""

    def __init__(self, predictor,
                 training_data_stats,
                 feature_names=None,
                 categorical_features=None,
                 class_names=None,
                 random_state=None):
        if not callable(predictor):
            raise ValueError("predictor must be callable.")
        check_value_type("training_data_stats", training_data_stats, dict)
        check_value_type("feature_names", feature_names, [list, type(None)])
        check_value_type("categorical_features", categorical_features, [list, type(None)])
        check_value_type("class_names", class_names, [list, type(None)])
        check_value_type("random_state", random_state, [int, type(None)])

        num_features = len(training_data_stats['feature_values'].keys())
        # create dummy training_data
        training_data = np.zeros((1, num_features))

        self._impl = LimeTabularExplainer(predictor,
                                          training_data,
                                          feature_names=feature_names,
                                          categorical_features=categorical_features,
                                          class_names=class_names,
                                          random_state=random_state,
                                          training_data_stats=training_data_stats)

    def __call__(self,
                 inputs,
                 targets=0,
                 num_samples=5000,
                 num_features=10,
                 show=None):
        check_value_type("inputs", inputs, [Tensor, np.ndarray])
        check_value_type("targets", targets, [Tensor, np.ndarray, list, int])
        check_value_type("show", show, [bool, type(None)])
        check_value_type("num_features", num_features, int)
        check_value_type("num_samples", num_samples, int)

        if self._impl.mode == "regression":
            targets = 0

        if show is None:
            show = is_notebook()

        if len(inputs.shape) != 2:
            raise ValueError('Dimension invalid. `training_data` should be 2D. '
                             'But got {}D.'.format(len(inputs.shape)))

        targets = self._unify_targets(inputs, targets)

        exps = []
        for sample_index, data in enumerate(inputs):
            if isinstance(data, Tensor):
                data = data.asnumpy()
            labels = targets[sample_index]
            class_exp = self._impl.explain_instance(data, labels, None, num_features, num_samples)
            sample_exp = []
            for label in labels:
                sample_exp.append(class_exp.as_list(label))
                if show:
                    class_exp.show_in_notebook(sample_index, labels=[label])
            exps.append(sample_exp)

        return exps

    @staticmethod
    def to_training_data_stats(training_data, categorical_features=None, feature_names=None):
        """
        Convert training data to training data stats.

        Args:
            training_data (Tensor, numpy.ndarray): training data.
            categorical_features (list, None): categorical features.
            feature_names (list, None): feature names.

        Returns:
            dict, training data stats
        """

        # dummy model
        def func(array):
            return array

        if isinstance(training_data, Tensor):
            data = training_data.asnumpy()
        else:
            data = training_data
        explainer = LimeTabularExplainer(func, data, categorical_features=categorical_features,
                                         feature_names=feature_names)
        stats = {
            "means": explainer.discretizer.means,
            "mins": explainer.discretizer.mins,
            "maxs": explainer.discretizer.maxs,
            "stds": explainer.discretizer.stds,
            "feature_values": explainer.feature_values,
            "feature_frequencies": explainer.feature_frequencies,
            "bins": explainer.discretizer.bins_value
        }

        return stats

    @staticmethod
    def save_training_data_stats(stats, f):
        """
        Save training data stats to disk.

        Args:
            stats (dict): training data stats.
            f (str, Path, IOBase): Target path.

        """

        def convert_to_float(data):
            if isinstance(data, dict):
                new_data = {}
                for key, value in data.items():
                    new_data[int(key)] = [float(x) for x in value]
            else:
                new_data = [x.tolist() for x in data]
            return new_data

        stats_float = {k: convert_to_float(v) for k, v in stats.items()}

        if isinstance(f, IOBase):
            json.dump(stats_float, f)
        else:
            with open(f, 'w') as file_handler:
                json.dump(stats_float, file_handler)

    @staticmethod
    def load_training_data_stats(f):
        """
        Load training data stats to disk.

        Args:
            f (str, Path, IOBase): Target path.

        Returns:
            dict, training data stats
        """

        def convert_to_numpy(data):
            if isinstance(data, dict):
                new_data = {}
                for key, value in data.items():
                    new_data[int(key)] = [np.float32(x) for x in value]
            else:
                new_data = [np.array(x, dtype=np.float32) for x in data]
            return new_data

        if isinstance(f, IOBase):
            stats_float = json.load(f)
        else:
            with open(f, 'r') as file_handler:
                stats_float = json.load(file_handler)

        stats_numpy = {k: convert_to_numpy(v) for k, v in stats_float.items()}

        return stats_numpy

    @staticmethod
    def _unify_targets(inputs, targets):
        """To unify targets to be 2D numpy.ndarray."""
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
        return targets
