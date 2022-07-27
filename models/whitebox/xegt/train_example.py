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
"""XEGT training example."""

# This script should be run directly with 'python <script> <args>'.

import time

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import ops
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from src.xegt import XEGT, LossNet, TrainOneStepCellWithGradClipping, InputCompiler
from src.graph import GraphTopo


EPOCHS = 3
NUM_NODE_FEAT = 16
NUM_HIDDEN_DIM = 64
NUM_CLASSES = 3
BATCH_SIZE = 1000
LR = 0.001

graph = GraphTopo.random(num_nodes=2000,
                         num_edges=5000,
                         num_node_types=8,
                         num_edge_types=3)
edge_lookup = graph.build_edge_lookup()
train_set = np.arange(int(graph.num_nodes * 0.9))
test_set = np.arange(train_set.shape[0], graph.num_nodes)
node_features = np.random.uniform(size=(graph.num_nodes, NUM_NODE_FEAT))
ground_truths = np.random.choice(NUM_CLASSES, size=graph.num_nodes)


def train_xegt():
    xegt = XEGT(num_node_types=graph.num_node_types,
                num_edge_types=graph.num_edge_types,
                num_node_feat=NUM_NODE_FEAT,
                hidden_size=NUM_HIDDEN_DIM,
                output_size=NUM_CLASSES)
    loss_net = LossNet(xegt)
    optimizer = nn.optim.AdamWeightDecay(xegt.trainable_params(), learning_rate=LR)
    train_net = TrainOneStepCellWithGradClipping(loss_net, optimizer, clip_val=0.1)

    compiler = InputCompiler(max_num_nodes=2000, max_per_type_num_edges=2000)
    softmax = ops.Softmax()

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}...")
        start = time.time()
        last_loss = 0
        np.random.shuffle(train_set)
        for i in range(0, train_set.shape[0], BATCH_SIZE):
            end = min(i + BATCH_SIZE, train_set.shape[0])
            centers = train_set[i:end]
            subgraph, node_map = graph.fast_k_hop_subgraph(edge_lookup, centers,
                                                           num_hops=2, ret_node_map=True)
            out_nidx = [node_map[c] for c in centers]
            h = node_features[list(node_map.keys())]
            inputs = compiler.compile(h, subgraph, out_nidx)

            labels = ms.Tensor(ground_truths[centers])
            last_loss = train_net(*inputs, labels)

        elapsed = time.time() - start
        print(f"train loss:{last_loss} elapsed: {elapsed}s")

        test_truths = np.empty(test_set.shape[0], int)
        test_preds = np.empty_like(test_truths)
        test_probs = np.empty((test_set.shape[0], NUM_CLASSES), float)
        for i in range(0, test_set.shape[0], BATCH_SIZE):
            end = min(i + BATCH_SIZE, test_set.shape[0])
            centers = test_set[i:end]
            subgraph, node_map = graph.fast_k_hop_subgraph(edge_lookup, centers,
                                                           num_hops=2, ret_node_map=True)
            out_nidx = [node_map[c] for c in centers]
            h = node_features[list(node_map.keys())]
            inputs = compiler.compile(h, subgraph, out_nidx)

            out, h_atts, edge_atts = xegt(*inputs)

            out = softmax(out).asnumpy()
            test_probs[i:end] = out
            test_preds[i:end] = out.argmax(axis=1)
            test_truths[i:end] = ground_truths[centers]

        acc = (test_preds == test_truths).sum() / test_truths.shape[0]
        print(f"test accuracy:{acc}")

        if NUM_CLASSES == 2:
            test_probs = test_probs[:, 1]
        auc = roc_auc_score(test_truths, test_probs, multi_class='ovr')
        print(f"test roc auc:{auc}")

        pre, rec, f1, sup = precision_recall_fscore_support(test_truths, test_preds)
        print(f"test precision:\n", pre, "\ntest recall:\n", rec,
              "\ntest f1-score:\n", f1, "\ntest support:\n", sup)


if __name__ == '__main__':
    train_xegt()
