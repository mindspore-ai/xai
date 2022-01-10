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
import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import Tensor
import numpy as np

from mindspore_xai.common.utils import is_notebook
from mindspore_xai.third_party.lime.lime_tabular import LimeTabularExplainer


class LIMETabular:
    r"""
    Provides Lime Tabular explanation method.

    Explains predictions on tabular (i.e. matrix) data. For numerical features, perturb them by sampling from a
    Normal(0,1) and doing the inverse operation of mean-centering and scaling, according to the means and stds in the
    training data. For categorical features, perturb by sampling according to the training distribution, and making a
    binary feature that is 1 when the value is the same as the instance being explained.

    For more details, please refer to the original paper via: `"Why Should I Trust You?": Explaining the Predictions of
    Any Classifier <https://arxiv.org/abs/1602.04938>`_.

    Note:
        The parsed `network` will be set to eval mode through `network.set_grad(False)` and `network.set_train(False)`.
        If you want to train the `network` afterwards, please reset it back to training mode through the opposite
        operations.

    Args:
        network (Cell, function): The black-box model to be explained, or a prediction function. For classifiers, the
            function should take a tensor and outputs prediction probabilities. For regressors, this takes a tensor and
            returns the predictions.
        training_data (Tensor, numpy.ndarray): 2D tensor or 2D numpy array of shape (N, K) (N being the number of
            samples, K being the number of features)
        training_labels (list): labels for training data. Not required, but may be
            used by discretizer.
        feature_names (list, optional): list of names (strings) corresponding to the columns
            in the training data.
        categorical_features (list, optional): list of indices (ints) corresponding to the
            categorical columns. Everything else will be considered
            continuous. Values in these columns MUST be integers.
        categorical_names (dict, optional): map from int to list of names, where
            categorical_names[x][y] represents the name of the yth value of
            column x.
        kernel_width (float, optional): kernel width for the exponential kernel.
            If None, defaults to sqrt (number of columns) * 0.75
        kernel (function, optional): similarity kernel that takes euclidean distances and kernel
            width as input and outputs weights in (0,1). If None, defaults to
            an exponential kernel.
        verbose (bool, optional): if true, print local prediction values from linear model
        class_names (list, optional): list of class names, ordered according to whatever the
            classifier is using. If not present, class names will be '0',
            '1', ...
        feature_selection (str, optional): feature selection method. can be
            'forward_selection', 'lasso_path', 'none' or 'auto'.
            See function 'explain_instance_with_data' in lime_base.py for
            details on what each of the options does.
        discretize_continuous (bool, optional): if True, all non-categorical features will
            be discretized into quartiles.
        discretizer (str, optional): only matters if discretize_continuous is True
            and data is not sparse. Options are 'quartile', 'decile',
            'entropy' or a BaseDiscretizer instance.
        sample_around_instance (bool, optional): if True, will sample continuous features
            in perturbed samples from a normal centered at the instance
            being explained. Otherwise, the normal is centered on the mean
            of the feature data.
        random_state (int, numpy.RandomState, optional): an integer or numpy.RandomState that will be used to
            generate random numbers. If None, the random state will be
            initialized using the internal numpy seed.
        training_data_stats (dict, optional): a dict object having the details of training data
            statistics. If None, training data information will be used, only matters
            if discretize_continuous is True. Must have the following keys:
            means", "mins", "maxs", "stds", "feature_values",
            "feature_frequencies"

    Inputs:
        - **inputs** (Tensor) - The input data to be explained, a 2D tensor of shape :math:`(N, K)`.
        - **targets** (Tensor, int) - The labels of interest to be explained. When `targets` is an integer,
          all the inputs will generate attribution map w.r.t this integer. When `targets` is a tensor, it
          should be of shape :math:`(N, l)` (l being the number of labels for each sample) or :math:`(N,)` :math:`()`.
        - **show** (bool, optional): Show the explanation figures, `None` means auto. Default: `None`.
        - **num_features** (int, optional): Maximum number of features present in explanation.
        - **num_samples** (int, optional): size of the neighborhood to learn the linear model.
        - **distance_metric** (str, optional): the distance metric to use for weights.
        - **model_regressor** (object, optional): sklearn regressor to use in explanation. Defaults
          to Ridge regression in LimeBase. Must have model_regressor.coef_
          and 'sample_weight' as a parameter to model_regressor.fit()

    Outputs:
        list[list[list[(str, float)]]], nth sample, jth label, kth feature (description, weight).

    Raises:
        TypeError: Be raised for any argument or input type problem.
        ValueError: Be raised for any input value problem.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindspore import context
        >>> from mindspore_xai.explainer import LIMETabular
        >>>
        >>> context.set_context(mode=context.PYNATIVE_MODE)
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
        >>> # initialize LIMETabular explainer with the pretrained model and training data
        >>> lime = LIMETabular(net, training_data, feature_names=feature_names, class_names=class_names)
        >>> inputs = ms.Tensor(np.random.rand(2, 4), ms.float32)
        >>> # when `targets` is an integer
        >>> targets = 0
        >>> exps = lime(inputs, targets)
        >>> for i, sample_exp in enumerate(exps):
        >>>     for j, class_exp in enumerate(sample_exp):
        >>>         print('Local explanation for sample {} class {}'.format(i, class_names[targets]))
        >>>         print(class_exp, '\n')
        Local explanation for sample 0 class setosa
        [('petal width (cm) <= 0.17', 0.0015253320612389082), ...]

        Local explanation for sample 1 class setosa
        [('petal width (cm) > 0.57', -0.0016295816592064624), ...]
        >>> # `targets` can also be a 2D tensor
        >>> targets = ms.Tensor([[1, 2], [1, 2]], ms.int32)
        >>> exps = lime(inputs, targets)
        >>> for i, sample_exp in enumerate(exps):
        >>>     for j, class_exp in enumerate(sample_exp):
        >>>         print('Local explanation for sample {} class {}'.format(i, class_names[targets[i][j]]))
        >>>         print(class_exp, '\n')
        Local explanation for sample 0 class versicolor
        [('sepal width (cm) <= 0.25', 0.001606624848945185), ...]

        Local explanation for sample 0 class virginica
        [('petal length (cm) <= 0.29', 0.0033004810806473504), ...]

        Local explanation for sample 1 class versicolor
        [('petal width (cm) <= 0.20', 0.002856020026411598), ...]

        Local explanation for sample 1 class virginica
        [('petal length (cm) > 0.80', -0.0031887318983352536), ...]
"""

    def __init__(self, network,
                 training_data,
                 training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 discretize_continuous=True,
                 discretizer='quartile',
                 sample_around_instance=False,
                 random_state=None,
                 training_data_stats=None):

        self._impl = LimeTabularExplainer(network,
                                          training_data,
                                          training_labels,
                                          feature_names,
                                          categorical_features,
                                          categorical_names,
                                          kernel_width,
                                          kernel,
                                          verbose,
                                          class_names,
                                          feature_selection,
                                          discretize_continuous,
                                          discretizer,
                                          sample_around_instance,
                                          random_state,
                                          training_data_stats)

    def __call__(self,
                 inputs,
                 targets,
                 show=None,
                 num_features=10,
                 num_samples=5000,
                 distance_metric='euclidean',
                 model_regressor=None):
        if self._impl.mode == "regression":
            targets = 0

        if show is None:
            show = is_notebook()

        if isinstance(inputs, Tensor):
            if len(inputs.shape) != 2:
                raise ValueError('Dimension invalid. If `targets` is a Tensor, it should be 2D. '
                                 'But got {}D.'.format(len(inputs.shape)))

        targets = self._unify_targets(inputs, targets)
        exps = []
        for idx, data in enumerate(inputs):
            data = data.asnumpy()
            labels = targets[idx]
            class_exp = self._impl.explain_instance(data, labels, None, num_features, num_samples, distance_metric,
                                                    model_regressor)
            sample_exp = []
            for label in labels:
                sample_exp.append(class_exp.as_list(label))
                if show:
                    class_exp.as_pyplot_figure(label, idx, (8, 4))
            exps.append(sample_exp)

        if show:
            plt.show()

        return exps

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
        return targets
