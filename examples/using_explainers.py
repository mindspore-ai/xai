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
import mindspore as ms
from mindspore import Tensor, context, load_checkpoint, load_param_into_net
from mindspore_xai.explanation import GradCAM

from common.dataset import load_dataset, load_image_tensor
from common.resnet import resnet50


if __name__ == "__main__":
    # Preparing

    # only PYNATIVE_MODE is supported
    context.set_context(mode=context.PYNATIVE_MODE)

    # 20 classes
    num_classes = 20

    # load the trained classifier
    net = resnet50(num_classes)
    param_dict = load_checkpoint('xai_examples_data/ckpt/resnet50.ckpt')
    load_param_into_net(net, param_dict)

    # [1, 3, 224, 224] Tensor
    boat_image = load_image_tensor('xai_examples_data/test/boat.jpg')
    print(f'boat_image.shape: {boat_image.shape}')


    # Using GradCAM

    # usually specify the last convolutional layer
    grad_cam = GradCAM(net, layer='layer4')

    # 5 is the class id of 'boat'
    saliency = grad_cam(boat_image, targets=5)
    print(f'saliency.shape: {saliency.shape}')


    # Batch Explanation

    test_ds = load_dataset('xai_examples_data/test').batch(4)

    for images, labels in test_ds:
        saliencies = grad_cam(images, targets=Tensor([5, 5, 5, 5], dtype=ms.int32))
        # [4, 1, 224, 224] Tensor
        print(f'saliencies.shape: {saliencies.shape}')
        # other custom operations ...
