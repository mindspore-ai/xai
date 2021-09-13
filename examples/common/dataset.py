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
import os
from functools import partial

import numpy as np
from PIL import Image

import mindspore as ms
from mindspore.dataset import GeneratorDataset
import mindspore.dataset.vision.c_transforms as C
from mindspore.dataset.transforms.c_transforms import Compose


classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

trans = Compose([
    C.Resize(224),
    C.CenterCrop(224),
    C.Normalize(mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375]),
    C.HWC2CHW()
])


def load_image_tensor(path):
    image = Image.open(path)
    image = trans(image)
    image = ms.Tensor(np.expand_dims(np.asarray(image), axis=0))
    return image


def load_dataset(path):
    the_dataset = GeneratorDataset(source=partial(_ds_generator, path),
                                   column_names=['image', 'label'])
    the_dataset = the_dataset.map(input_columns=['image'], operations=trans)
    return the_dataset


def _ds_generator(dir_path):
    annote_path = os.path.join(dir_path, 'annotations.txt')
    with open(annote_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            splited = line.split(';')

            if len(splited) != 2:
                continue
            filename = splited[0]
            labels = splited[1].split(',')
            if not filename or not labels:
                continue
            try:
                image_path = os.path.join(dir_path, filename)
                image = Image.open(os.path.join(dir_path, filename))
            except IOError:
                print('cannot open image:' + image_path)
                continue

            one_hot = np.zeros(len(classes))
            one_hot[classes.index(labels[0])] = 1
            one_hot = one_hot.astype(np.float32)
            image = np.asarray(image)

            yield image, one_hot
