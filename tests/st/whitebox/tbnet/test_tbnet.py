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
from functools import partial

import pytest
import numpy as np
import mindspore as ms
from mindspore.dataset import GeneratorDataset

from mindspore_xai.whitebox.tbnet import TBNet, NetWithLossCell, TrainStepWrapCell, EvalNet
from mindspore_xai.whitebox.tbnet import AUC, ACC, Recommender


class TestTBNet:
    """Test TB-Net."""

    def setup_method(self):
        """Setup the test case."""
        ms.set_seed(1234)

        self.item_count = 1000
        self.ref_count = 20
        self.rel_count = 5
        self.top_k = 3
        self.per_item_paths = 24

        self.id_maps = {
            'item': {i + 1: f'i{i + 1}' for i in range(self.item_count)},
            'reference': {i + 1: f'e{i + 1}' for i in range(self.ref_count)},
            'relation': {i: f'r{i}' for i in range(self.rel_count)}
        }

        self.tb_net = TBNet(num_items=self.item_count, num_references=self.ref_count,
                            num_relations=self.rel_count, embedding_dim=32)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_tbnet_train(self):
        """Test TB-Net training."""

        column_names = ['item', 'relation1', 'reference', 'relation2', 'hist_item', 'label']
        max_entity = self.item_count + self.ref_count + 1

        def gen_data(size):
            for _ in range(size):
                item = np.array(np.random.randint(1, max_entity), dtype=np.int32)
                relation1 = np.random.randint(self.rel_count, size=self.per_item_paths, dtype=np.int32)
                reference = np.random.randint(1, max_entity, size=self.per_item_paths, dtype=np.int32)
                relation2 = np.random.randint(self.rel_count, size=self.per_item_paths, dtype=np.int32)
                hist_item = np.random.randint(1, max_entity, size=self.per_item_paths, dtype=np.int32)
                label = np.array(np.random.randint(2), dtype=np.float32)

                yield item, relation1, reference, relation2, hist_item, label

        train_ds = GeneratorDataset(column_names=column_names, source=partial(gen_data, 64*10)).batch(64)
        eval_ds = GeneratorDataset(column_names=column_names, source=partial(gen_data, 64*2)).batch(64)
        loss_net = NetWithLossCell(self.tb_net, 0.01, 0.01, 0.01)
        train_net = TrainStepWrapCell(loss_net, 0.001)
        train_net.set_train()
        eval_net = EvalNet(self.tb_net)
        model = ms.Model(network=train_net, eval_network=eval_net, metrics={'auc': AUC(), 'acc': ACC()})
        for _ in range(2):
            model.train(epoch=1, train_dataset=train_ds, dataset_sink_mode=False)
            train_out = model.eval(train_ds, dataset_sink_mode=False)
            eval_out = model.eval(eval_ds, dataset_sink_mode=False)
            assert train_out["auc"] >= 0
            assert train_out["acc"] >= 0
            assert eval_out["auc"] >= 0
            assert eval_out["acc"] >= 0

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_tbnet_infer(self):
        """Test TB-Net inference."""
        cand_count = 256  # candidate count

        batch_size = 128

        batch_count = cand_count // batch_size

        candidates = np.random.choice(np.arange(self.item_count), size=cand_count, replace=False)
        candidates += 1  # begin from 1

        recommender = Recommender(self.tb_net, id_maps=self.id_maps, top_k=self.top_k)

        for bi in range(batch_count):
            offset = bi * batch_size
            items = candidates[offset:offset + batch_size]
            rl1s = np.random.randint(self.rel_count, size=(batch_size, self.per_item_paths))
            refs = np.random.randint(self.ref_count + 1, size=(batch_size, self.per_item_paths))
            rl2s = np.random.randint(self.rel_count, size=(batch_size, self.per_item_paths))
            hist_items = np.random.randint(self.item_count + 1, size=(batch_size, self.per_item_paths))

            items = ms.Tensor(items, dtype=ms.int32)
            rl1s = ms.Tensor(rl1s, dtype=ms.int32)
            refs = ms.Tensor(refs, dtype=ms.int32)
            rl2s = ms.Tensor(rl2s, dtype=ms.int32)
            hist_items = ms.Tensor(hist_items, dtype=ms.int32)

            recommender(items, rl1s, refs, rl2s, hist_items)

        suggestions = recommender.suggest()
        assert len(suggestions) == self.top_k
        for suggestion in suggestions:
            assert len(suggestion.paths) == self.per_item_paths
