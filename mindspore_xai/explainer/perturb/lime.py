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

    Args:
        predictor (Callable): The black-box model to be explained, should be a callable function. For classification
            model, it accepts a 2D array/tensor of shape :math:`(N, K)` as input and outputs a 2D array/tensor of
            shape :math:`(N, L)`. For regression model, it accepts a 2D array/tensor of shape :math:`(N, K)` as input
            and outputs a 1D array/tensor of shape :math:`(N)`.
        train_feat_stats (dict): a dict object having the details of training data statistics. The stats can be
            generated using static method LIMETabular.to_feat_stats(training_data).
        feature_names (list, optional): list of names (strings) corresponding to the columns in the training data.
            Default: `None`.
        categorical_features_indexes (list, optional): list of indices (ints) corresponding to the categorical columns,
            their values MUST be integers. Other columns will be considered continuous. Default: `None`.
        class_names (list, optional): list of class names, ordered according to whatever the classifier is using. If
            not present, class names will be '0', '1', ... Default: `None`.
        num_perturbs (int, optional): size of the neighborhood to learn the linear model. Default: 5000.
        max_features (int, optional): Maximum number of features present in explanation. Default: 10.

    Inputs:
        - **inputs** (Tensor, numpy.ndarray) - The input data to be explained, a 2D float tensor or 2D float
          numpy array of shape :math:`(N, K)`.
        - **targets** (Tensor, numpy.ndarray, list, int, optional) - The labels of interest to be explained. When
          `targets` is an integer, all the inputs will generate attribution map w.r.t this integer. When `targets` is a
          tensor, numpy array or list, it should be of shape :math:`(N, L)` (L being the number of labels for each
          sample), :math:`(N,)` or :math:`()`. For regression model, this parameter will be ignored. Default: 0.
        - **show** (bool, optional): Show the explanation figures, `None` means automatically show the explanation
          figures if it is running on JupyterLab. Default: `None`.

    Outputs:
        list[list[list[(str, float)]]], a 3-dimension list of tuple. The first dimension represents inputs.
        The second dimension represents targets. The third dimension represents features.
        The tuple represents feature description and weight.

    Raises:
        TypeError: Be raised for any argument or input type problem.
        ValueError: Be raised for any input value problem.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindspore_xai.explainer import LIMETabular
        >>> # Linear classification model
        >>> class LinearNet(nn.Cell):
        ...     def __init__(self, num_inputs, num_class):
        ...         super(LinearNet, self).__init__()
        ...         self.fc = nn.Dense(num_inputs, num_class, activation=nn.Softmax())
        ...     def construct(self, x):
        ...         x = self.fc(x)
        ...         return x
        >>> net = LinearNet(4, 3)
        >>> # use iris data as example
        >>> feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        >>> class_names = ['setosa', 'versicolor', 'virginica']
        >>> train = ms.Tensor(np.random.rand(10, 4), ms.float32)
        >>> stats = LIMETabular.to_feat_stats(train, feature_names=feature_names)
        >>> lime = LIMETabular(net, stats, feature_names=feature_names, class_names=class_names)
        >>> inputs = ms.Tensor(np.random.rand(2, 4), ms.float32)
        >>> targets = ms.Tensor([[1, 2], [1, 2]], ms.int32)
        >>> exps = lime(inputs, targets)
        >>> # output is a 3-dimension list of tuple
        >>> print((len(exps), len(exps[0]), len(exps[0][0])))
        (2, 2, 4)
"""

    def __init__(self, predictor,
                 train_feat_stats,
                 feature_names=None,
                 categorical_features_indexes=None,
                 class_names=None,
                 num_perturbs=5000,
                 max_features=10):
        if not callable(predictor):
            raise ValueError("predictor must be callable.")
        check_value_type("train_feat_stats", train_feat_stats, dict)
        check_value_type("feature_names", feature_names, [list, type(None)])
        check_value_type("categorical_features_indexes", categorical_features_indexes, [list, type(None)])
        check_value_type("class_names", class_names, [list, type(None)])
        check_value_type("num_perturbs", num_perturbs, int)
        check_value_type("max_features", max_features, int)

        num_features = len(train_feat_stats['feature_values'].keys())
        # create dummy training_data
        training_data = np.zeros((1, num_features))

        self._num_perturbs = num_perturbs
        self._max_features = max_features

        self._impl = LimeTabularExplainer(predictor,
                                          training_data,
                                          feature_names=feature_names,
                                          categorical_features=categorical_features_indexes,
                                          class_names=class_names,
                                          training_data_stats=train_feat_stats)

    def __call__(self,
                 inputs,
                 targets=0,
                 show=None):
        check_value_type("inputs", inputs, [Tensor, np.ndarray])
        check_value_type("targets", targets, [Tensor, np.ndarray, list, int])
        check_value_type("show", show, [bool, type(None)])

        if self._impl.mode == "regression":
            targets = 0

        if show is None:
            show = is_notebook()

        if len(inputs.shape) != 2:
            raise ValueError('Dimension invalid. `inputs` should be 2D. '
                             'But got {}D.'.format(len(inputs.shape)))

        targets = self._unify_targets(inputs, targets)

        exps = []
        for sample_index, data in enumerate(inputs):
            if isinstance(data, Tensor):
                data = data.asnumpy()
            labels = targets[sample_index]
            class_exp = self._impl.explain_instance(data, labels, None, self._max_features, self._num_perturbs)
            sample_exp = []
            for label in labels:
                sample_exp.append(class_exp.as_list(label))
                if show:
                    class_exp.show_in_notebook(sample_index, labels=[label])
            exps.append(sample_exp)

        return exps

    @staticmethod
    def to_feat_stats(features, feature_names=None, categorical_features_indexes=None):
        """
        Convert features to feature stats.

        Args:
            features (Tensor, numpy.ndarray): training data.
            feature_names (list, optional): feature names. Default: `None`.
            categorical_features_indexes (list, optional): list of indices (ints) corresponding to the categorical
                columns, their values MUST be integers. Other columns will be considered continuous. Default: `None`.

        Returns:
            dict, training data stats
        """
        check_value_type("feature_names", feature_names, [list, type(None)])
        check_value_type("categorical_features_indexes", categorical_features_indexes, [list, type(None)])

        # dummy model
        def func(array):
            return array

        if isinstance(features, Tensor):
            data = features.asnumpy()
        else:
            data = features
        explainer = LimeTabularExplainer(func, data, categorical_features=categorical_features_indexes,
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
    def save_feat_stats(stats, file):
        """
        Save feature stats to disk.

        Args:
            stats (dict): training data stats.
            file (str, Path, IOBase): File path or stream.

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

        if isinstance(file, IOBase):
            json.dump(stats_float, file)
        else:
            with open(file, 'w', encoding='utf-8') as file_handler:
                json.dump(stats_float, file_handler)

    @staticmethod
    def load_feat_stats(file):
        """
        Load feature stats from disk.

        Args:
            file (str, Path, IOBase): File path or stream.

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

        if isinstance(file, IOBase):
            stats_float = json.load(file)
        else:
            with open(file, 'r', encoding='utf-8') as file_handler:
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
