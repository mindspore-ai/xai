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
"""eXplainable Embedding Graph Transformer."""
import mindspore as ms
from mindspore import Tensor
import numpy as np


class InputCompiler:

    def __init__(self, max_num_nodes, max_per_type_num_edges, trim=True):
        self.max_num_nodes = max_num_nodes
        self.max_per_type_num_edges = max_per_type_num_edges
        self.trim = trim
        self._nodes_trimmed = False
        self._edges_trimmed = False

    @property
    def node_in_len(self):
        """Node input length."""
        return self.max_num_nodes + 1  # add one for the dummy node

    @property
    def edge_in_len(self):
        """Edge input length."""
        return self.max_per_type_num_edges

    @property
    def trimmed(self):
        """Is the compiled graph trimmed."""
        return self._nodes_trimmed or self._edges_trimmed

    @property
    def nodes_trimmed(self):
        """Are nodes of the compiled graph trimmed."""
        return self._nodes_trimmed

    @property
    def edges_trimmed(self):
        """Are edges of the compiled graph trimmed."""
        return self._edges_trimmed

    def warmup(self, num_node_feat, num_node_types, num_edge_types, num_out_node_idxs):
        """Generate dummy inputs for network warmup."""
        zeros = ms.ops.Zeros()
        node_in_len = self.node_in_len
        edge_in_len = self.edge_in_len
        adj = zeros((2, edge_in_len * num_edge_types), ms.int32)
        etype = zeros(edge_in_len * num_edge_types, ms.int32)
        h = zeros((node_in_len, num_node_feat), ms.float32)
        pt_nidx = zeros((num_node_types, node_in_len), ms.int32)
        pt_adj = zeros((num_edge_types, 2, edge_in_len), ms.int32)
        pt_eidx = zeros((num_edge_types, edge_in_len), ms.int32)
        out_nidx = zeros((num_out_node_idxs,), ms.int32)
        return h, adj, etype, pt_nidx, pt_adj, pt_eidx, out_nidx

    def compile(self, node_feat, graph, out_node_idxs):
        """Generate the inputs."""
        if graph.num_nodes <= 0:
            raise ValueError('The graph contains no node.')

        graph.compile()

        node_in_len = self.node_in_len
        edge_in_len = self.edge_in_len
        max_num_nodes = self.max_num_nodes
        max_per_type_num_edges = self.max_per_type_num_edges
        dum_node_idx = max_num_nodes
        dum_edge_idx = edge_in_len * graph.num_edge_types

        self._nodes_trimmed = False
        self._edges_trimmed = False

        if not self.trim:
            if graph.num_nodes > max_num_nodes:
                raise OverflowError(f'Too many nodes, max:{max_num_nodes}.')
            if max(graph.per_type_num_edges) > max_per_type_num_edges:
                raise OverflowError(f'Too many per-type edges, max:{max_per_type_num_edges}.')

        pt_nidx = np.full((graph.num_node_types, node_in_len), dum_node_idx, dtype=np.int32)
        for i in range(graph.num_node_types):
            if graph.per_type_num_nodes[i] > 0:
                if node_in_len < graph.per_type_num_nodes[i]:
                    assign_len = node_in_len
                    self._nodes_trimmed = True
                else:
                    assign_len = graph.per_type_num_nodes[i]
                pt_nidx[i, 0:assign_len] = graph.per_type_node_idxs[i][0:assign_len]
        pt_nidx[pt_nidx > dum_node_idx] = dum_node_idx
        pt_nidx = Tensor(pt_nidx, dtype=ms.int32)

        pt_adj = np.full((graph.num_edge_types, 2, edge_in_len), dum_node_idx, dtype=np.int32)
        for i in range(graph.num_edge_types):
            if graph.per_type_num_edges[i] > 0:
                edge_idxs = graph.per_type_edge_idxs[i]
                if edge_in_len < edge_idxs.shape[0]:
                    assign_len = edge_in_len
                    edge_idxs = edge_idxs[0:assign_len]
                    self._edges_trimmed = True
                else:
                    assign_len = edge_idxs.shape[0]
                pt_adj[i, :, 0:assign_len] = graph.adj[:, edge_idxs]
        pt_adj[pt_adj > dum_node_idx] = dum_node_idx
        pt_adj = Tensor(pt_adj, dtype=ms.int32)

        pt_eidx = np.full((graph.num_edge_types, edge_in_len), dum_edge_idx, dtype=np.int32)
        for i, idxs in enumerate(graph.per_type_edge_idxs):
            if idxs.shape[0] > 0:
                if edge_in_len < idxs.shape[0]:
                    assign_len = edge_in_len
                    edge_idxs = idxs[0:assign_len]
                else:
                    assign_len = idxs.shape[0]
                    edge_idxs = idxs[0:assign_len]
                pt_eidx[i, 0:assign_len] = edge_idxs
        pt_eidx[pt_eidx > dum_edge_idx] = dum_edge_idx
        pt_eidx = Tensor(pt_eidx, dtype=ms.int32)

        if node_feat.shape[0] > max_num_nodes:
            node_feat = node_feat[0:max_num_nodes]
            self._nodes_trimmed = True
        h = np.concatenate((node_feat, np.zeros((node_in_len - node_feat.shape[0], node_feat.shape[1]))))
        h = Tensor(h, dtype=ms.float32)

        adj = np.full((2, edge_in_len * graph.num_edge_types), dum_node_idx, dtype=np.int32)
        size = min(adj.shape[1], graph.adj.shape[1])
        adj[:, 0:size] = graph.adj[:, 0:size]
        adj[adj > dum_node_idx] = dum_node_idx
        adj = Tensor(adj, dtype=ms.int32)

        if graph.edge_types is None:
            etype = ms.ops.Zeros()(edge_in_len * graph.num_edge_types, ms.int32)
        else:
            etype = np.zeros(edge_in_len * graph.num_edge_types, dtype=np.int32)
            size = min(etype.shape[0], graph.edge_types.shape[0])
            etype[0:size] = graph.edge_types[0:size]
            etype = Tensor(etype, dtype=ms.int32)

        if isinstance(out_node_idxs, Tensor):
            out_nidx = out_node_idxs
        else:
            if not isinstance(out_node_idxs, (tuple, list, np.ndarray)):
                out_node_idxs = (out_node_idxs,)
            out_nidx = Tensor(out_node_idxs, ms.int32)

        return h, adj, etype, pt_nidx, pt_adj, pt_eidx, out_nidx
