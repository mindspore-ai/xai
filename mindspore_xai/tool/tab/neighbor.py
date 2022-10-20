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
"""Nearest Neighbor."""
import math
from tqdm import tqdm

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import ops
from mindspore import ms_function
import numpy as np

_square = ops.Square()
_zeros = ops.Zeros()
_concat = ops.Concat()


class NearestNeighbor:
    """Nearest neighbor."""
    def __init__(self, samples, classifier, num_classes, batch_size, threshold):
        self._classes_sample_idxs = None
        self._split_by_class(samples, classifier, num_classes, batch_size, threshold)

    @property
    def num_classes(self):
        """Number of classes."""
        return len(self._classes_sample_idxs)

    def sample_count(self, class_id):
        """Number of samples."""
        return self._classes_sample_idxs[class_id].shape[0]

    def sample_idxs(self, class_id):
        """Sample indices of the class."""
        return self._classes_sample_idxs[class_id]

    def __call__(self, queries, class_id):
        """Find the nearest neighbor."""
        raise NotImplementedError

    def _split_by_class(self, samples, classifier, num_classes, batch_size, threshold):
        """Split samples by class."""
        if classifier is None:
            num_classes = 1

        self._classes_sample_idxs = [[] for _ in range(num_classes)]

        if classifier is None:
            idxs = ms.Tensor(mnp.arange(samples.shape[0]), dtype=ms.int32)
            self._classes_sample_idxs[0] = idxs
            return

        num_samples = samples.shape[0]
        batch_count = int(math.ceil(num_samples/batch_size))
        for i in tqdm(range(batch_count), total=batch_count, leave=False, desc='Classify Samples'):
            start = i * batch_size
            if start >= num_samples:
                break
            aligned_end = start + batch_size
            if aligned_end > num_samples:
                end = num_samples
            else:
                end = aligned_end
            probs = classifier(samples[start:end])
            larger = (probs >= threshold).asnumpy()
            indices = np.arange(start, end, dtype=int)
            for cls in range(num_classes):
                self._classes_sample_idxs[cls].append(ms.Tensor(indices[larger[:, cls]], dtype=ms.int32))

        for cls, idxs in enumerate(self._classes_sample_idxs):
            if idxs:
                self._classes_sample_idxs[cls] = _concat(idxs)
            else:
                self._classes_sample_idxs[cls] = _zeros(0, dtype=ms.int32)


@ms_function
def _simple_nn_idx(query, samples):
    """Find index of the nearest neighbor."""
    diff = query.reshape((1, -1)) - samples
    sq_dist = _square(diff)
    sq_dist = sq_dist.sum(1)
    return sq_dist.argmin()


class SimpleNN(NearestNeighbor):
    """Nearest neighbor by brute force."""
    def __init__(self, samples, classifier, num_classes, batch_size=10000, threshold=0.5):
        super().__init__(samples, classifier, num_classes, batch_size, threshold)
        self._classes_samples = [samples[self.sample_idxs(cls)] for cls in range(self.num_classes)]

    def __call__(self, queries, class_id):
        """Find the nearest neighbor."""
        samples = self._classes_samples[class_id]
        min_idxs = [0] * queries.shape[0]
        for i, query in tqdm(enumerate(queries), total=queries.shape[0], leave=False, desc='Find Nearest Neighbors'):
            min_idxs[i] = int(_simple_nn_idx(query, samples))
        return ops.gather(samples, ms.Tensor(min_idxs, dtype=ms.int32), 0)
