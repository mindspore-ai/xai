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
"""Pseudo Linear Coefficients (PLC)."""
from tqdm import tqdm

import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore import ms_function
import numpy as np

from mindspore_xai.tool.tab.neighbor import SimpleNN


_squeeze = ops.Squeeze()
_square = ops.Square()
_sqrt = ops.Sqrt()
_zeros = ops.Zeros()
_log = ops.Log()
_not = ops.LogicalNot()
_is_finite = ops.IsFinite()


class _StepwiseComputer(nn.Cell):
    """Helper of computing PLC for stepwise classifiers."""
    def __init__(self, eps):
        super().__init__()
        self._eps = eps

    @ms_function
    def construct(self, queries, nearests):
        """Computation."""
        displaces = nearests - queries
        dists = _sqrt(_square(displaces).sum(1))
        is_same = dists < self._eps
        dists += is_same.astype(ms.float32)
        dists = dists.reshape((-1, 1))
        unit_vecs = displaces / dists
        unit_vecs *= _not(is_same).reshape((-1, 1))
        plc = unit_vecs.sum(0) / queries.shape[0]
        return plc


class _Computer(nn.Cell):
    """Helper of computing PLC."""
    def __init__(self, classifier, riemann, eps):
        super().__init__()
        self._classifier = classifier
        self._riemann = riemann
        self._t = ms.Tensor([p / riemann for p in range(riemann)], dtype=ms.float32)
        self._t = self._t.reshape((-1, 1))
        self._eps = eps

    @ms_function
    def _pre_compute(self, query, nearest):
        """Prepare for the actual computation."""
        displace = _squeeze(nearest - query)
        sq_dist = _square(displace).sum()
        u = query * (1 - self._t) + nearest * self._t
        return displace, sq_dist, u

    @ms_function
    def _compute(self, displace, sq_dist, fu, plc_sum):
        """Do compute the PLC."""
        # Riemann sum
        minus_fu = 1 - fu
        log_fu = _log(fu + self._eps)
        log_minus_fu = _log(minus_fu + self._eps)
        h = -(fu * log_fu) - (minus_fu * log_minus_fu)
        ig_h = h.sum() / self._riemann
        ig_h /= 0.69314718056  # ln(2)=0.69314718056, change to base 2

        sample_plc = (displace * (fu[-1] - fu[0])) / (sq_dist * ig_h)
        sample_plc = sample_plc.masked_fill(_not(_is_finite(sample_plc)), 0.0)

        return plc_sum + sample_plc

    def construct(self, target, query, nearest, plc_sum):
        """Computation."""
        displace, sq_dist, u = self._pre_compute(query, nearest)
        if sq_dist < self._eps:
            return plc_sum
        fu = self._classifier(u)[:, target]
        return self._compute(displace, sq_dist, fu, plc_sum)


class PseudoLinearCoef:
    r"""
        Pseudo Linear Coefficients (PLC) for classifiers.

        PLC is a global attribution method, it is a measure of feature sensitivities around the classifier's decision
        boundaries from the data distribution's point of view.

        Authors: NG Ngai Fai, WANG Shendi, LI Xiaohui (2022 Huawei)

        PLC of class A:

        .. math::

            \vec{R}(A)=\int \vec{S}(A,nearest_{A}(x),x))p_{\neg A}(x)dx

        PLC of class A (target class) relative to class B (view point class), it is called Relative PLC:

        .. math::

            \vec{R}(A,B)=\int \vec{S}(A,nearest_{A}(x),x))p_{B}(x)dx

        Where:

        .. math::

            nearest_A(x):=\underset{g\in G}{argmin}(\left \| g-x \right \|)\text{ }s.t.\text{ } g\neq x,f_A(g)
            \geq \xi

            \vec{S}(A,a,x)=\left\{\begin{matrix}
            \vec{0} & \text{if }f_A(x)\geq \xi \\
            \frac{a-x}{\left \| a-x \right \|} & \text{if }f_A(\cdot )\text{ is a step function}\\
            \frac{(a-x)(f_{A}(a)-f_A(x))}{\left \| a-x \right \|^{2}\int_{0}^{1}h(f_A(u(t)))dt} & \text{else}
            \end{matrix}\right.

            u(t)=ta+(1-t)x

            h(f_{A})=-f_{A}log_2(f_{A})-(1-f_A)log_2(1-f_A)

        :math:`G` is the universal sample set, :math:`f_A(\cdot )` is the predicted probability of class A,
        :math:`\xi ` is the decision threshold (usually 0.5). :math:`p_{\neg A}` and :math:`p_{B}` are the PDF of
        sample's distribution of non A class(es) and class B representatively. Beware that the ground truth labels take
        no part in PLC, a sample's classes are determined by the classifier.

        Note:
            If `classifier` is a function, `stepwise` is `False` and it is running in graph mode then
            `classifier` must complies with the
            `static graph syntax <https://mindspore.cn/docs/en/master/note/static_graph_syntax_support.html>`_. PLC may
            not be accurate if there are many samples classified to more than one class.

        Args:
            classifier (Cell, Callable): The classifier :math:`f(\cdot )` to be explained, it must take an input tensor
                with shape :math:`(N, K)` and output a probability tensor with shape :math:`(N, L)`. :math:`K` is the
                number of features,Both input and output tensors should has dtype `ms.float32`.
            num_classes (int): The number of classes :math:`L`.
            stepwise (bool): Set to `True` if `classifier` outputs 0s and 1s only. Default: `False`.
            threshold (float): Decision threshold :math:`\xi` of classification. Default: 0.5.
            monte_carlo (int): The number of Monte Carlo samples for computing the integrals :math:`\vec{R}`.
                Default: 1000. Higher the number more lengthy and accurate the computation.
            riemann (int): The number of Riemann sum partitions for computing the integrals
                :math:`\int_{0}^{1}h(f_A(u(t)))dt`. Default: 1000. Higher the number more lengthy and accurate the
                computation.
            batch_size(int): Batch size for `classifier` when finding nearest neighbors. Default: 2000.
            eps (float): Epsilon. Default: 1e-9.

        Inputs:
            - **features** (Tensor) - The universal sample set :math:`G`. Practically, it is often the training set or
              its random subset. The shape must be :math:`(|G|, K)`, :math:`|G|` is the total number of samples.

        Outputs:
            - **plc** (Tensor) - Pseudo Linear Coefficients in shape of :math:`(L, K)`.
            - **relative plc** (Tensor) - Relative Pseudo Linear Coefficients in shape of :math:`(L, L, K)`. The first
              :math:`L` axis is for the target classes and the second one is for the view point classes.

        Raises:
            TypeError: Be raised for any argument or input type problem.
            ValueError: Be raised for any input value problem.
            AttributeError: Be raised for underlying is missing any required attribute.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import nn
            >>> from mindspore import ops
            >>> from mindspore_xai.explainer import PseudoLinearCoef
            >>>
            >>> class Classifier(nn.Cell):
            ...     def construct(self, x):
            ...         y = ops.Zeros()((x.shape[0], 3), ms.float32)
            ...         y[:, 0] = -x[:, 0] + x[:, 1] + x[: ,2] - 0.5
            ...         y[:, 1] =  x[:, 0] - x[:, 1] + x[: ,2] - 0.5
            ...         y[:, 2] =  x[:, 0] + x[:, 1] - x[: ,2] - 0.5
            ...         return ops.Sigmoid()(y * 10)
            >>>
            >>> classifier = Classifier()
            >>> explainer = PseudoLinearCoef(classifier, num_classes=3)
            >>> features = ms.Tensor(np.random.uniform(size=(10000, 5)), dtype=ms.float32)  # 5 features
            >>> plc, relative_plc = explainer(features)
            >>> print(str(plc.shape))
            (3, 5)
            >>> print(str(relative_plc.shape))
            (3, 3, 5)
    """
    def __init__(self, classifier, num_classes, stepwise=False, threshold=0.5,
                 monte_carlo=1000, riemann=1000, batch_size=2000, eps=1e-9):
        self._classifier = classifier
        self._num_classes = num_classes
        self._stepwise = stepwise
        self._threshold = threshold
        self._monte_carlo = monte_carlo
        self._batch_size = batch_size
        self._eps = eps
        if self._stepwise:
            self._computer = _StepwiseComputer(eps)
        else:
            self._computer = _Computer(classifier, riemann, eps)
        self._computer.set_train(False)

    def __call__(self, features):
        """Compute PLC and Relative PLC."""
        nn_finder = SimpleNN(features, self._classifier, self._num_classes,
                             batch_size=self._batch_size, threshold=self._threshold)

        plc = _zeros((self._num_classes, features.shape[1]), ms.float32)
        relative_plc = _zeros((self._num_classes, self._num_classes, features.shape[1]), ms.float32)

        # may different from features.shape[0]
        all_finder_samples_count = sum([nn_finder.sample_count(c) for c in range(self._num_classes)])
        pairs = [(t, vp) for t in range(self._num_classes) for vp in range(self._num_classes) if t != vp]
        for target, view_point in tqdm(pairs, desc='Compute Pseudo Linear Coef.'):
            vp_samples = nn_finder.sample_count(view_point)
            if vp_samples == 0:
                continue
            total_vp_samples = all_finder_samples_count - nn_finder.sample_count(target)
            plc_ele = self._relative(target, view_point, features, nn_finder)
            relative_plc[target, view_point] = plc_ele
            vp_weight = vp_samples / total_vp_samples
            plc[target] += plc_ele * vp_weight

        return plc, relative_plc

    @classmethod
    def normalize(cls, plc, per_vec=False, eps=1e-9):
        r"""
        Normalize Pseudo Linear Coefficients to range [-1, 1].

        Warning:
            Normalizing PLC from unnormalized features may lead to misleading results.

        Args:
            plc (Tensor): The PLC or Relative PLC to be normalized.
            per_vec (bool): Normalize within each :math:`\vec{R}` vector. Default: False.
            eps (float): Epsilon. Default: 1e-9.

        Returns:
            Tensor, the normalized values.

        Examples:
            >>> from mindspore import Tensor
            >>> from mindspore_xai.explainer import PseudoLinearCoef
            >>>
            >>> plc = Tensor([[0.1, 0.6, 0.8], [-2, 0.2, 0.4], [0.4, 0.1, -0.1]])
            >>> PseudoLinearCoef.normalize(plc)
            [[ 0.05  0.3   0.4 ]
             [-1.    0.1   0.2 ]
             [ 0.2   0.05 -0.05]]
            >>> PseudoLinearCoef.normalize(plc, per_vec=True)
            [[ 0.125  0.75   1.   ]
             [-1.     0.1    0.2  ]
             [ 1.     0.25  -0.25 ]]
        """
        if not plc.shape:
            return plc

        if per_vec and plc.shape[-1] > 0:
            scale = plc.abs().max(axis=-1, keepdims=True)
            scale = scale.masked_fill(scale < eps, 1)
            return plc / scale

        scale = plc.abs().max()
        if scale > eps:
            return plc / scale
        return plc

    def _relative(self, target, view_point, features, nn_finder):
        """Compute Relative PLC."""
        vp_samples_count = nn_finder.sample_count(view_point)
        if self._monte_carlo < vp_samples_count:
            picked = np.random.choice(np.arange(vp_samples_count), size=self._monte_carlo, replace=False)
            queries = features[nn_finder.sample_idxs(view_point)[ms.Tensor(picked, dtype=ms.int32)]]
        else:
            queries = features[nn_finder.sample_idxs(view_point)]

        nearests = nn_finder(queries, target)

        if self._stepwise:
            return self._computer(queries, nearests)

        plc_sum = _zeros(features.shape[1], ms.float32)
        target = ms.Tensor(target, dtype=ms.int32)
        for query, nearest in tqdm(zip(queries, nearests), leave=False, total=queries.shape[0],
                                   desc=f'Class {target} Relative to Class {view_point}'):
            plc_sum = self._computer(target, query, nearest, plc_sum)

        return plc_sum / queries.shape[0]
