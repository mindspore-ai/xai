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
import numpy as np
import mindspore as ms
from mindspore import set_context, PYNATIVE_MODE
import mindspore.nn as nn
import sklearn.datasets

from mindspore_xai.explainer import LIMETabular, SHAPGradient, SHAPKernel


class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet, self).__init__()
        # input features: 4
        # output classes: 3
        self.linear = nn.Dense(4, 3, activation=nn.Softmax())

    def construct(self, x):
        x = self.linear(x)
        return x


if __name__ == "__main__":

    iris = sklearn.datasets.load_iris()

    # feature_names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    feature_names = iris.feature_names
    # class_names: ['setosa', 'versicolor', 'virginica']
    class_names = list(iris.target_names)

    # convert data and labels from numpy array to mindspore tensor
    # use the first 100 samples
    data = ms.Tensor(iris.data, ms.float32)[:100]
    labels = ms.Tensor(iris.target, ms.int32)[:100]

    # explain the first sample
    inputs = data[:1]
    # explain the label 'setosa'(class index 0)
    targets = 0

    net = LinearNet()

    # load pre-trained parameters
    weight = np.array([[0.648, 1.440, -2.05, -0.977], [0.507, -0.276, -0.028, -0.626], [-1.125, -1.183, 2.099, 1.605]])
    bias = np.array([0.308, 0.343, -0.652])
    net.linear.weight.set_data(ms.Tensor(weight, ms.float32))
    net.linear.bias.set_data(ms.Tensor(bias, ms.float32))

    # convert features to feature stats
    feature_stats = LIMETabular.to_feat_stats(data, feature_names=feature_names)
    # initialize the explainer
    lime = LIMETabular(net, feature_stats, feature_names=feature_names, class_names=class_names)
    # explain
    lime_outputs = lime(inputs, targets)
    print("LIMETabular:")
    for i, exps in enumerate(lime_outputs):
        for exp in exps:
            print("Explanation for sample {} class {}:".format(i, class_names[targets]))
            print(exp, '\n')

    # initialize the explainer
    shap_kernel = SHAPKernel(net, data, feature_names=feature_names, class_names=class_names)
    # explain
    shap_kernel_outputs = shap_kernel(inputs, targets)
    print("SHAPKernel:")
    for i, exps in enumerate(shap_kernel_outputs):
        for exp in exps:
            print("Explanation for sample {} class {}:".format(i, class_names[targets]))
            print(exp, '\n')

    # Gradient only works under PYNATIVE_MODE.
    set_context(mode=PYNATIVE_MODE)
    # initialize the explainer
    shap_gradient = SHAPGradient(net, data, feature_names=feature_names, class_names=class_names)
    # explain
    shap_gradient_outputs = shap_gradient(inputs, targets)
    print("SHAPGradient:")
    for i, exps in enumerate(shap_gradient_outputs):
        for exp in exps:
            print("Explanation for sample {} class {}:".format(i, class_names[targets]))
            print(exp, '\n')
