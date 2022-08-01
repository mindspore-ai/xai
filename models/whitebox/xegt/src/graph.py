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
"""Graph Topology."""

from collections import OrderedDict
import numpy as np


class GraphTopo:
    """
    Topology of a directed graph.
    """
    @classmethod
    def random(cls,
               num_nodes,
               num_edges,
               num_node_types=1,
               num_edge_types=1,
               dtype=int,
               tdtype=None):
        """Generate a random graph."""
        if tdtype is None:
            if num_node_types > 255 or num_edge_types > 255:
                tdtype = int
            else:
                tdtype = np.uint8
        adj = np.random.randint(low=0, high=num_nodes, size=(2, num_edges), dtype=dtype)
        if num_node_types > 1:
            node_types = np.random.randint(low=0, high=num_node_types, size=num_nodes, dtype=tdtype)
        else:
            node_types = None
        if num_edge_types > 1:
            edge_types = np.random.randint(low=0, high=num_edge_types, size=num_edges, dtype=tdtype)
        else:
            edge_types = None
        return cls(adj, node_types=node_types, edge_types=edge_types, num_nodes=num_nodes, dtype=dtype, tdtype=tdtype)

    def __init__(self,
                 adj,
                 node_types=None,
                 edge_types=None,
                 num_nodes=-1,
                 num_node_types=-1,
                 num_edge_types=-1,
                 dtype=None,
                 tdtype=None):
        """
        Args:
            adj(np.array): node indices of edges' ends in COO format with shape of [2, E], `adj[0]` are source node
                indices.
            node_types(np.array, optional): types of all nodes, in shape of [N], if None or empty is provided then it
                is assumed all nodes shared the same type of 0.
            edge_types(np.array, optional): types of all edges, in shape of [E], if None or empty is provided then it
                is assumed all edges shared the same type of 0.
            num_nodes(int): Total number of nodes in this graph, if -1 is provided then it will be the length of
                `num_node_types` (if provided) or the max. value in `adj` plus one (otherwise).
            num_node_types(int): Total number of node types in the complete graph, it will be the max. value in
                `node_types` plus one if -1 is provided.
            num_edge_types(int): Total number of edge types in the complete graph, it will be the  max. value in
                `edge_types` plus one if -1 is provided.

        Note:
            E is the total number of edges and N is the total number of nodes.
        """
        if dtype is None:
            self._dtype = int
        else:
            self._dtype = dtype

        if tdtype is None:
            self._tdtype = np.uint8
        else:
            self._tdtype = tdtype

        if isinstance(adj, np.ndarray):
            self.adj = adj
        else:
            self.adj = np.array(adj, dtype=self._dtype)

        if node_types is None or isinstance(node_types, np.ndarray):
            self.node_types = node_types
        else:
            self.node_types = np.array(node_types, dtype=self._tdtype)

        if edge_types is None or isinstance(edge_types, np.ndarray):
            self.edge_types = edge_types
        else:
            self.edge_types = np.array(edge_types, dtype=self._tdtype)

        if self.node_types is not None and self.node_types.size == 0:
            self.node_types = None

        if self.edge_types is not None and self.edge_types.size == 0:
            self.edge_types = None

        if num_nodes >= 0:
            self._num_nodes = num_nodes
        elif self.node_types is not None:
            self._num_nodes = int(self.node_types.shape[0])
        else:
            self._num_nodes = int(self.adj.max() + 1)

        if num_node_types > 0:
            self._num_node_types = num_node_types
        else:
            if self.node_types is None:
                self._num_node_types = 1
            else:
                self._num_node_types = int(self.node_types.max() + 1)

        if num_edge_types > 0:
            self._num_edge_types = num_edge_types
        else:
            if self.edge_types is None:
                self._num_edge_types = 1
            else:
                self._num_edge_types = int(self.edge_types.max() + 1)

        # available after compile() was called:
        self.per_type_node_idxs = None
        self.per_type_num_nodes = None
        self.per_type_edge_idxs = None
        self.per_type_num_edges = None

    @property
    def dtype(self):
        """Indices' data type."""
        return self._dtype

    @property
    def tdtype(self):
        """Node/edge types' data type."""
        return self._tdtype

    @property
    def num_edges(self):
        """Number of edges."""
        return self.adj.shape[1]

    @property
    def num_edge_types(self):
        """Number of edge types."""
        return self._num_edge_types

    @property
    def num_nodes(self):
        """Number of nodes."""
        return self._num_nodes

    @property
    def num_node_types(self):
        """Number of node types."""
        return self._num_node_types

    @property
    def homogeneous(self):
        """Is homogeneous."""
        return self._num_node_types == 1 and self._num_edge_types == 1

    @property
    def is_compiled(self):
        """Is compiled."""
        return bool(self.per_type_node_idxs) and bool(self.per_type_num_nodes) and \
               bool(self.per_type_edge_idxs) and bool(self.per_type_num_edges)

    def get_per_type_nodes(self, force_compile=False):
        """Get per-type nodex indices."""
        if force_compile or self.per_type_node_idxs is None or self.per_type_num_nodes is None:
            self.per_type_node_idxs, self.per_type_num_nodes = \
                self._compile_item_types(self.node_types, self.num_nodes, self._num_node_types)
        return self.per_type_node_idxs, self.per_type_num_nodes

    def get_per_type_edges(self, force_compile=False):
        """Get per-type edge indices."""
        if force_compile or self.per_type_edge_idxs is None or self.per_type_num_edges is None:
            self.per_type_edge_idxs, self.per_type_num_edges = \
                self._compile_item_types(self.edge_types, self.num_edges, self._num_edge_types)
        return self.per_type_edge_idxs, self.per_type_num_edges

    def build_edge_lookup(self, flow='src_to_dst', dtype=None):
        """Build edge lookup (Complexity: O(N+E))."""
        if flow == 'dst_to_src':
            local = self.adj[0]
        else:
            local = self.adj[1]

        if dtype is None:
            dtype = self._dtype

        begins = np.zeros(self._num_nodes + 1, dtype=dtype)
        indices = np.zeros(self.num_edges, dtype=dtype)
        edge_counts = np.zeros(self._num_nodes + 1, dtype=dtype)

        for node_idx in local:
            edge_counts[node_idx] += 1

        pos = 0
        for node_idx in range(begins.shape[0]):
            begins[node_idx] = pos
            pos += edge_counts[node_idx]

        edge_counts.fill(0)
        for edge_idx, node_idx in enumerate(local):
            pos = begins[node_idx] + edge_counts[node_idx]
            indices[pos] = edge_idx
            edge_counts[node_idx] += 1

        return begins, indices

    def fast_k_hop_subgraph(self, edge_lookup, centers, num_hops, flow='src_to_dst',
                            reindex_nodes=True, ret_node_map=False, ret_edge_map=False):
        """K-hop subgraph from extra-large graph, complexity: O(Q^K), Q is avg. degree for the flow,
        K is no. of hops."""
        if flow == 'dst_to_src':
            remote = self.adj[1]
        else:
            remote = self.adj[0]

        if isinstance(centers, (list, tuple)):
            centers = np.array(centers).flatten()
        elif not isinstance(centers, np.ndarray):
            centers = np.array((centers,))

        lookup_begins, lookup_idxs = edge_lookup

        node_idxs = centers
        node_subsets = [centers]
        edge_subsets = []
        for _ in range(num_hops):
            new_node_idxs = []
            for node_idx in node_idxs:
                if lookup_begins[node_idx] >= lookup_begins[node_idx + 1]:
                    continue
                edges = lookup_idxs[lookup_begins[node_idx]:lookup_begins[node_idx + 1]]
                if edges.size:
                    edge_subsets.append(edges)
                    new_node_idxs.append(remote[edges])
            if not new_node_idxs:
                break
            node_idxs = np.concatenate(new_node_idxs)
            node_subsets.append(node_idxs)

        node_subset = np.unique(np.concatenate(node_subsets))
        edge_subset = np.unique(np.concatenate(edge_subsets)) if edge_subsets else np.array([], dtype=int)

        subgraph_adj = self.adj[:, edge_subset]

        if reindex_nodes:
            node_map = OrderedDict(((o, n) for n, o in enumerate(node_subset)))
            for i in range(subgraph_adj.shape[0]):
                for j in range(subgraph_adj.shape[1]):
                    subgraph_adj[i, j] = node_map[subgraph_adj[i, j]]
        elif ret_node_map:
            node_map = OrderedDict(((o, o) for o in node_subset))

        node_types = None if self.node_types is None else self.node_types[node_subset]
        edge_types = None if self.edge_types is None else self.edge_types[edge_subset]

        subgraph = GraphTopo(adj=subgraph_adj,
                             node_types=node_types,
                             edge_types=edge_types,
                             num_nodes=node_subset.shape[0],
                             num_node_types=self.num_node_types,
                             num_edge_types=self.num_edge_types,
                             dtype=self._dtype,
                             tdtype=self._tdtype)

        if ret_edge_map:
            edge_map = OrderedDict(zip(edge_subset, range(edge_subset.shape[0])))

        if ret_node_map and ret_edge_map:
            return subgraph, node_map, edge_map
        if ret_node_map:
            return subgraph, node_map
        if ret_edge_map:
            return subgraph, edge_map
        return subgraph

    def k_hop_subgraph(self, centers, num_hops, reindex_nodes=True, flow='src_to_dst',
                       ret_node_map=False, ret_edge_map=False):
        """Complexity: O(KN), K is no. of hops."""
        if flow == 'dst_to_src':
            local, remote = self.adj
        else:
            remote, local = self.adj

        node_mask = np.empty(self.num_nodes, dtype=bool)

        if isinstance(centers, (list, tuple)):
            centers = np.array(centers).flatten()
        elif not isinstance(centers, np.ndarray):
            centers = np.array((centers,))

        subsets = [centers]

        for _ in range(num_hops):
            node_mask.fill(False)
            node_mask[subsets[-1]] = True
            edge_mask = np.take(node_mask, local)
            edge_remotes = remote[edge_mask]
            if edge_remotes.shape[0] > 0:
                subsets.append(remote[edge_mask])

        subset = np.concatenate(subsets)
        subset = np.unique(subset)

        node_mask.fill(False)
        node_mask[subset] = True

        edge_mask = node_mask[local] & node_mask[remote]

        subgraph_adj = self.adj[:, edge_mask]

        if reindex_nodes:
            node_idx = np.full((self._num_nodes,), -1)
            new_node_idxs = np.arange(subset.shape[0], dtype=self._dtype)
            node_idx[subset] = new_node_idxs
            subgraph_adj = node_idx[subgraph_adj]
        else:
            new_node_idxs = subset

        node_types = None if self.node_types is None else self.node_types[subset]
        edge_types = None if self.edge_types is None else self.edge_types[edge_mask]

        subgraph = GraphTopo(adj=subgraph_adj,
                             node_types=node_types,
                             edge_types=edge_types,
                             num_nodes=subset.shape[0],
                             num_node_types=self.num_node_types,
                             num_edge_types=self.num_edge_types,
                             dtype=self._dtype,
                             tdtype=self._tdtype)

        if ret_node_map:
            node_map = OrderedDict(zip(subset, new_node_idxs))
        if ret_edge_map:
            edge_map = OrderedDict(zip(np.where(edge_mask)[0], range(subgraph.num_edges)))

        if ret_node_map and ret_edge_map:
            return subgraph, node_map, edge_map
        if ret_node_map:
            return subgraph, node_map
        if ret_edge_map:
            return subgraph, edge_map
        return subgraph

    def copy(self):
        """Copy to a new graph."""
        cloned = GraphTopo(adj=self.adj.copy(),
                           node_types=None if self.node_types is None else self.node_types.copy(),
                           edge_types=None if self.edge_types is None else self.edge_types.copy(),
                           num_nodes=self.num_nodes,
                           num_node_types=self.num_node_types,
                           num_edge_types=self.num_edge_types,
                           dtype=self.dtype,
                           tdtype=self.tdtype)

        if self.per_type_edge_idxs is not None:
            cloned.per_type_edge_idxs = [i.copy() for i in self.per_type_edge_idxs]
        if self.per_type_num_edges is not None:
            cloned.per_type_num_edges = self.per_type_num_edges.copy()
        if self.per_type_node_idxs is not None:
            cloned.per_type_node_idxs = [i.copy() for i in self.per_type_node_idxs]
        if self.per_type_num_nodes is not None:
            cloned.per_type_num_nodes = self.per_type_num_nodes.copy()

        return cloned

    def remove_edges(self, edge_idxs):
        """Remove edges."""
        self.adj = np.delete(self.adj, edge_idxs, axis=1)
        if self.edge_types is not None:
            self.edge_types = np.delete(self.edge_types, edge_idxs)
        self.per_type_edge_idxs = None
        self.per_type_num_edges = None

    def compile(self, force=False):
        """Compile the graph."""
        self.get_per_type_nodes(force)
        self.get_per_type_edges(force)

    def _compile_item_types(self, item_types, num_items, num_types):
        """Compile a type list."""
        if num_types == 1:
            return [np.arange(num_items, dtype=self._dtype)], [num_items]

        per_type_idxs = [None] * num_types
        per_type_num = [0] * num_types
        indices = np.arange(num_items, dtype=self._dtype)
        for item_type in range(num_types):
            idxs_of_type = indices[item_types == item_type]
            per_type_idxs[item_type] = idxs_of_type
            per_type_num[item_type] = idxs_of_type.shape[0]
        return per_type_idxs, per_type_num
