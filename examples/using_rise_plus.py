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
from mindspore import context, save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.nn import Softmax, SoftmaxCrossEntropyWithLogits
from mindspore_xai.explanation import RISEPlus, OoDNet

from common.dataset import load_dataset, load_image_tensor
from common.resnet import resnet50


if __name__ == "__main__":
    # Preparing

    # only PYNATIVE_MODE is supported
    context.set_context(mode=context.PYNATIVE_MODE)

    num_classes = 20

    # classifier training dataset
    train_ds = load_dataset('xai_examples_data/train').batch(4)

    # load the trained classifier
    net = resnet50(num_classes)
    param_dict = load_checkpoint('xai_examples_data/ckpt/resnet50.ckpt')
    load_param_into_net(net, param_dict)

    # Training OoDNet

    ood_net = OoDNet(underlying=net, num_classes=num_classes)

    # use SoftmaxCrossEntropyWithLogits as loss function if the activation function of
    # the classifier is Softmax, use BCEWithLogitsLoss if the activation function is Sigmod
    ood_net.train(train_ds, loss_fn=SoftmaxCrossEntropyWithLogits())

    save_checkpoint(ood_net, 'ood_net.ckpt')

    # Using RISEPlus

    ood_net = OoDNet(underlying=resnet50(num_classes), num_classes=num_classes)
    param_dict = load_checkpoint('ood_net.ckpt')
    load_param_into_net(ood_net, param_dict)

    rise_plus = RISEPlus(ood_net=ood_net, network=net, activation_fn=Softmax())
    boat_image = load_image_tensor("xai_examples_data/test/boat.jpg")
    saliency = rise_plus(boat_image, targets=5)
    print(f"saliency.shape:{saliency.shape}")
