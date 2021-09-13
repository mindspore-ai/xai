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
import numpy as np
import mindspore as ms
from mindspore import context, Tensor, load_checkpoint, load_param_into_net
from mindspore.nn import Softmax
from mindspore_xai.explanation import GradCAM
from mindspore_xai.benchmark import Robustness, Localization

from common.resnet import resnet50
from common.dataset import load_image_tensor


if __name__ == "__main__":
    # Preparing

    # only PYNATIVE_MODE is supported
    context.set_context(mode=context.PYNATIVE_MODE)

    # 20 classes
    num_classes = 20

    # load the trained classifier
    net = resnet50(num_classes)
    param_dict = load_checkpoint("xai_examples_data/ckpt/resnet50.ckpt")
    load_param_into_net(net, param_dict)

    # [1, 3, 224, 224] Tensor
    boat_image = load_image_tensor('xai_examples_data/test/boat.jpg')
    print(f"boat_image.shape:{boat_image.shape}")

    # Generate a saliency map

    # explainer
    grad_cam = GradCAM(net, layer='layer4')

    # 5 is the class id of 'boat'
    saliency = grad_cam(boat_image, targets=5)
    print(f"saliency.shape:{saliency.shape}")

    # Evaluate with Robustness

    # the classifier use Softmax as activation function
    robustness = Robustness(num_classes, activation_fn=Softmax())
    # the 'saliency' argument is optional
    score = robustness.evaluate(grad_cam, boat_image, targets=5, saliency=saliency)
    print(f"robustness:{score}")

    # Using Localization

    # top-left:80,66 bottom-right:223,196 is the bounding box of a boat
    mask = np.zeros([1, 1, 224, 224])
    mask[:, :, 66:196, 80:223] = 1

    mask = Tensor(mask, dtype=ms.float32)

    localization = Localization(num_classes)
    score = localization.evaluate(grad_cam, boat_image, targets=5, mask=mask)
    print(f"localization:{score}")
