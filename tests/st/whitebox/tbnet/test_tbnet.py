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
"""Initialization of tests of TB-Net."""
import pytest
import numpy as np
import mindspore as ms

from mindspore_xai.whitebox.tbnet import TBNet, Recommender


class TestTBNet:
    """Test PathGen."""

    def setup_method(self):
        """Setup the test case."""
        ms.set_seed(1234)

        self.item_count = 1000
        self.ref_count = 20
        self.rel_count = 5
        self.top_k = 3

        tb_net = TBNet(num_items=self.item_count, num_references=self.ref_count,
                       num_relations=self.rel_count, embedding_dim=32)

        id_maps = {
            'item': {i + 1: f'i{i + 1}' for i in range(self.item_count)},
            'reference': {i + 1: f'e{i + 1}' for i in range(self.ref_count)},
            'relation': {i: f'r{i}' for i in range(self.rel_count)}
        }
        self.recommender = Recommender(tb_net, id_maps=id_maps, top_k=self.top_k)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_tb_net(self):
        """Test TB-Net infer."""
        cand_count = 256  # candidate count
        per_item_paths = 24
        batch_size = 128

        batch_count = cand_count // batch_size

        candidates = np.random.choice(np.arange(self.item_count), size=cand_count, replace=False)
        candidates += 1  # begin from 1

        for bi in range(batch_count):
            offset = bi*batch_size
            items = candidates[offset:offset+batch_size]
            rl1s = np.random.randint(self.rel_count, size=(batch_size, per_item_paths))
            refs = np.random.randint(self.ref_count + 1, size=(batch_size, per_item_paths))
            rl2s = np.random.randint(self.rel_count, size=(batch_size, per_item_paths))
            hist_items = np.random.randint(self.item_count + 1, size=(batch_size, per_item_paths))

            items = ms.Tensor(items, dtype=ms.int32)
            rl1s = ms.Tensor(rl1s, dtype=ms.int32)
            refs = ms.Tensor(refs, dtype=ms.int32)
            rl2s = ms.Tensor(rl2s, dtype=ms.int32)
            hist_items = ms.Tensor(hist_items, dtype=ms.int32)

            self.recommender(items, rl1s, refs, rl2s, hist_items)

        suggestions = self.recommender.suggest()
        assert len(suggestions) == self.top_k
        for suggestion in suggestions:
            assert len(suggestion.paths) == per_item_paths
