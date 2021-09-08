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
import mindspore.nn as nn
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net
from mindspore_xai.explanation import GradCAM, GuidedBackprop
from mindspore_xai.benchmark import Faithfulness
from mindspore_xai.runner import ImageClassificationRunner

from common.resnet import resnet50
from common.dataset import classes, load_dataset


if __name__ == "__main__":
    # Preparing

    # only PYNATIVE_MODE is supported
    context.set_context(mode=context.PYNATIVE_MODE)
    num_classes = 20

    net = resnet50(num_classes)
    param_dict = load_checkpoint('xai_examples_data/ckpt/resnet50.ckpt')
    load_param_into_net(net, param_dict)

    # initialize explainers with the loaded black-box model
    gradcam = GradCAM(net, layer='layer4')
    guidedbackprop = GuidedBackprop(net)

    # initialize benchmarkers to evaluate the chosen explainers
    # for Faithfulness, the initialization needs an activation function that transforms the output of the network to a
    # probability is also needed
    activation_fn = nn.Sigmoid()  # for multi-label classification
    faithfulness = Faithfulness(num_labels=num_classes, metric='InsertionAUC', activation_fn=activation_fn)

    # returns the dataset to be explained, when localization is chosen, the dataset is required to provide bounding box
    # the columns of the dataset should be in [image], [image, labels], or [image, labels, bbox] (order matters)
    # You may refer to 'mindspore.dataset.project' for columns managements
    test_dataset = load_dataset('xai_examples_data/test')

    data = (test_dataset, classes)
    explainers = [gradcam, guidedbackprop]
    benchmarkers = [faithfulness]

    # initialize runner with specified summary_dir
    runner = ImageClassificationRunner(summary_dir='./summary_dir', network=net, activation_fn=activation_fn, data=data)
    runner.register_saliency(explainers, benchmarkers)

    # Run

    # execute runner.run to generate explanation and evaluation results to save it to summary_dir
    runner.run()
