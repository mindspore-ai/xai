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
"""Shap gradient explainer."""
import numpy as np
import mindspore as ms
from mindspore.train._utils import check_value_type

from mindspore_xai.common.utils import is_notebook
from mindspore_xai.third_party.shap.shap import GradientExplainer
from .shap import _SHAP


class SHAPGradient(_SHAP):
    r"""
    Provides SHAP gradient explanation method.

    Explains a network using expected gradients (an extension of integrated gradients).

    Note:
        The parsed `network` will be set to eval mode through `network.set_grad(False)` and `network.set_train(False)`.
        If you want to train the `network` afterwards, please reset it back to training mode through the opposite
        operations.

    Args:
        network (Cell): The mindspore cell to be explained. For classification, it accepts a 2D tensor of
            shape :math:`(N, K)` as input and outputs a 2D tensor of shape :math:`(N, L)`. For regression, it
            accepts a 2D tensor of shape :math:`(N, K)` as input and outputs a 1D tensor of
            shape :math:`(N)`.
        features (Tensor): 2D tensor of shape :math:`(N, K)` (N being the number of samples, K being the number of
            features). The background dataset to use for integrating out features, accept (whole or part of) training
            dataset.
        feature_names (list, optional): list of names (strings) corresponding to the columns in the training data.
            Default: `None`.
        class_names (list, optional): list of class names, ordered according to whatever the classifier is using. If
            not present, class names will be '0', '1', ... Default: `None`.
        num_neighbours (int, optional): Number of subsets used for the estimation of the shap values. Default: 200.
        max_features (int, optional): Maximum number of features present in explanation. Default: 10.

    Inputs:
        - **inputs** (Tensor) - The input data to be explained, a 2D float tensor of shape :math:`(N, K)`.
        - **targets** (Tensor, numpy.ndarray, list, int, optional) - The labels of interest to be explained. When
          `targets` is an integer, all the inputs will generate attribution map w.r.t this integer. When `targets` is a
          tensor or numpy array or list, it should be of shape :math:`(N, L)` (L being the number of labels for each
          sample), :math:`(N,)` or :math:`()`. Default: 0.
        - **show** (bool, optional): Show the explanation figures, `None` means automatically show the explanation
          figures if it is running on JupyterLab. Default: `None`.

    Outputs:
        Tensor, a 3D tensor of shape :math:`(N, L, K)`. The first dimension represents inputs.
        The second dimension represents targets. The third dimension represents feature weight.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindspore import set_context, PYNATIVE_MODE
        >>> from mindspore_xai.explainer import SHAPGradient
        >>>
        >>> set_context(mode=PYNATIVE_MODE)
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
        >>> training_data = ms.Tensor(np.random.rand(10, 4), ms.float32)
        >>> shap = SHAPGradient(net, training_data, feature_names=feature_names, class_names=class_names)
        >>> inputs = ms.Tensor(np.random.rand(2, 4), ms.float32)
        >>> targets = ms.Tensor([[1, 2], [1, 2]], ms.int32)
        >>> exps = shap(inputs, targets)
        >>> print(exps.shape)
        (2, 2, 4)
"""

    def __init__(self, network, features, feature_names=None, class_names=None, num_neighbours=200, max_features=10):
        check_value_type("network", network, ms.nn.Cell)
        check_value_type("features", features, ms.Tensor)
        check_value_type("num_neighbours", num_neighbours, int)
        check_value_type("max_features", max_features, int)
        check_value_type("feature_names", feature_names, [list, type(None)])
        check_value_type("class_names", class_names, [list, type(None)])

        super().__init__(network, features, feature_names, class_names)

        self._impl = GradientExplainer(network, features, self._mode, self._predictor_num_outputs)
        self._num_neighbours = num_neighbours
        self._max_features = max_features

    def __call__(self, inputs, targets=0, show=None):
        check_value_type("inputs", inputs, ms.Tensor)
        check_value_type("targets", targets, [ms.Tensor, np.ndarray, list, int])
        check_value_type("show", show, [bool, type(None)])

        if len(inputs.shape) != 2:
            raise ValueError('Dimension invalid. `inputs` should be 2D. '
                             'But got {}D.'.format(len(inputs.shape)))

        targets = self._unify_targets(inputs, targets)

        exps = self._impl.shap_values(inputs, targets, self._num_neighbours)

        if show is None:
            show = is_notebook()
        if show:
            self._show_all(exps, targets, self._impl.expected_value, self._max_features)

        return exps
