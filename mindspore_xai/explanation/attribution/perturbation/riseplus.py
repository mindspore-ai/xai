# Copyright 2021 Huawei Technologies Co., Ltd
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
"""RISEPlus."""
import math

import numpy as np
from mindspore.train._utils import check_value_type

from mindspore_xai.common import operators as op
from mindspore_xai.explanation.ood.ood_net import OoDNet
from .rise import RISE


class RISEPlus(RISE):
    r"""
    RISEPlus is a perturbation-based method that generates attribution maps by sampling on multiple random binary
    masks. An OoD detector is adopted to produce an 'inlier score', estimating the probability that a sample is
    generated from the distribution. Then the inlier score is aggregated to the weighted sum of the random masks, with
    the weights being the corresponding output on the node of interest:

    .. math::
        attribution = \sum_{i}s_if_c(I\odot M_i)  M_i

    For more details, please refer to the original paper: Resisting Out-of-Distribution Samples for Perturbation-based
    XAI.

    Args:
        ood_net (OoDNet): The OoD network for generating inlier score.
        network (Cell): The black-box model to be explained.
        activation_fn (Cell): The activation layer that transforms logits to prediction probabilities. For
            single label classification tasks, `nn.Softmax` is usually applied. As for multi-label classification
            tasks, `nn.Sigmoid` is usually be applied. Users can also pass their own customized `activation_fn` as
            long as when combining this function with network, the final output is the probability of the input.
        perturbation_per_eval (int, optional): Number of perturbations for each inference during inferring the
            perturbed samples. Within the memory capacity, usually the larger this number is, the faster the
            explanation is obtained. Default: 32.

    Inputs:
        - **inputs** (Tensor) - The input data to be explained, a 4D tensor of shape :math:`(N, C, H, W)`.
        - **targets** (Tensor, int) - The labels of interest to be explained. When `targets` is an integer,
          all of the inputs will generates attribution map w.r.t this integer. When `targets` is a tensor, it
          should be of shape :math:`(N, l)` (l being the number of labels for each sample) or :math:`(N,)` :math:`()`.

    Outputs:
        Tensor, a 4D tensor of shape :math:`(N, l, H, W)` when targets is a tensor of shape (N, l), otherwise a tensor
        of shape (N, 1, H, w), saliency maps.

    Raises:
        TypeError: Be raised for any argument or input type problem.
        ValueError: Be raised for any input value problem.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import nn, context, load_checkpoint, load_param_into_net
        >>> from mindspore_xai.explanation import RISEPlus, OoDNet
        >>>
        >>>
        >>> class MyLeNet5(nn.Cell):
        >>>
        >>>    def __init__(self, num_class, num_channel):
        >>>        super(MyLeNet5, self).__init__()
        >>>
        >>>        # must add the following 2 attributes to your model
        >>>        self.num_features = 84 # no. of features, int
        >>>        self.output_features = False # output features flag, bool
        >>>
        >>>        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        >>>        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        >>>        self.relu = nn.ReLU()
        >>>        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        >>>        self.flatten = nn.Flatten()
        >>>        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        >>>        self.fc2 = nn.Dense(120, self.num_features, weight_init=Normal(0.02))
        >>>        self.fc3 = nn.Dense(self.num_features, num_class, weight_init=Normal(0.02))
        >>>
        >>>    def construct(self, x):
        >>>        x = self.conv1(x)
        >>>        x = self.relu(x)
        >>>        x = self.max_pool2d(x)
        >>>        x = self.conv2(x)
        >>>        x = self.relu(x)
        >>>        x = self.max_pool2d(x)
        >>>        x = self.flatten(x)
        >>>        x = self.relu(self.fc1(x))
        >>>        x = self.relu(self.fc2(x))
        >>>
        >>>        # return the features tensor if output_features is True
        >>>        if self.output_features:
        >>>            return x
        >>>
        >>>        x = self.fc3(x)
        >>>        return x
        >>>
        >>> context.set_context(mode=context.PYNATIVE_MODE)
        >>> # prepare trained classifier
        >>> net = MyLeNet5(10, num_channel=3)
        >>> param_dict = load_checkpoint('mylenet5.ckpt')
        >>> load_param_into_net(net, param_dict)
        >>> # prepare train_dataset and your OoD network
        >>> train_dataset = create_dataset_cifar10("/path/to/cifar/dataset")
        >>> ood_net = OoDNet(net, 10)
        >>> ood_net.train(train_dataset, nn.SoftmaxCrossEntropyWithLogits())
        >>> # initialize RISEPlus explainer with the pretrained model and activation function
        >>> activation_fn = ms.nn.Softmax() # softmax layer is applied to transform logits to probabilities
        >>> riseplus = RISEPlus(ood_net, net, activation_fn=activation_fn)
        >>> # given an instance of RISEPlus, saliency map can be generate
        >>> inputs = ms.Tensor(np.random.rand(2, 3, 32, 32), ms.float32)
        >>> # when `targets` is an integer
        >>> targets = 5
        >>> saliency = riseplus(inputs, targets)
        >>> print(saliency.shape)
        (2, 1, 32, 32)
    """

    def __init__(self,
                 ood_net,
                 network,
                 activation_fn,
                 perturbation_per_eval=32):
        super(RISEPlus, self).__init__(network, activation_fn, perturbation_per_eval)
        check_value_type('ood_net', ood_net, OoDNet)
        self._ood_net = ood_net
        self._ood_net.set_train(mode=False)
        self._num_classes = ood_net.num_classes

    def __call__(self, inputs, targets):
        """Generates attribution maps for inputs."""
        self._verify_data(inputs, targets)
        height, width = inputs.shape[2], inputs.shape[3]

        # Due to the unsupported Op of slice assignment, we use numpy array here
        targets = self._unify_targets(inputs, targets)

        attr_np = np.zeros(shape=(inputs.shape[0], targets.shape[1], height, width))

        cal_times = math.ceil(self._num_masks / self._perturbation_per_eval)

        for idx, data in enumerate(inputs):
            data = op.reshape(data, (1, -1, height, width))
            empty_data = op.Tensor(np.zeros(data.shape), dtype=data.dtype)

            min_score = np.amax(self._ood_net(empty_data).asnumpy())
            max_score = np.amax(self._ood_net(data).asnumpy())

            for j in range(cal_times):
                bs = min(self._num_masks - j * self._perturbation_per_eval, self._perturbation_per_eval)
                data = op.reshape(data, (1, -1, height, width))
                masks = self._generate_masks(data, bs)

                masked_input = masks * (data - self._base_value) + self._base_value

                inlier_scores = self._calc_inlier_score(masked_input, min_score, max_score)

                weights = self._activation_fn(self.network(masked_input)) * inlier_scores

                while len(weights.shape) > 2:
                    weights = op.mean(weights, axis=2)

                weights = np.expand_dims(np.expand_dims(weights.asnumpy()[:, targets[idx]], 2), 3)

                attr_np[idx] += np.sum(weights * masks.asnumpy(), axis=0)

        attr_np = attr_np / self._num_masks

        return op.Tensor(attr_np, dtype=inputs.dtype)

    def _calc_inlier_score(self, masked_input, min_score, max_score):
        """Calculate inlier scores for masked_input"""
        image_h_scores = np.amax(self._ood_net(masked_input).asnumpy(), axis=1)
        clipped_h_scores = np.clip(image_h_scores, min_score, max_score)
        normalized_h_scores = (clipped_h_scores - min_score) / (max_score - min_score + np.finfo(float).eps)
        normalized_h_scores = normalized_h_scores.reshape(len(normalized_h_scores), 1)

        inlier_scores = op.Tensor(np.repeat(normalized_h_scores, self._num_classes, axis=1), dtype=masked_input.dtype)
        return inlier_scores
