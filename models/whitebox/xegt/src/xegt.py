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
import mindspore.ops.functional as F
from mindspore import nn, Tensor
from mindspore.common.initializer import initializer
from mindspore.common.initializer import XavierUniform
from mindspore_gl import GNNCell, Graph
import numpy as np

_EPS = 1e-9
clip_grad = ms.ops.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Tensor", "Tensor")
def _clip_grad(clip_value, grad):
    return ms.ops.clip_by_value(grad, -clip_value, clip_value)


class HomoHGTLayer(GNNCell):
    """homo HGT layer"""

    def __init__(self, n_heads, d_k):
        super().__init__()
        gain = np.sqrt(2)
        self.pri = ms.Parameter(ms.ops.Ones()((n_heads, 1), ms.float32))
        self.msg = ms.Parameter(initializer(XavierUniform(gain), [n_heads, d_k * d_k], ms.float32),
                                name="relation_msg")
        self.att = ms.Parameter(initializer(XavierUniform(gain), [n_heads, d_k * d_k], ms.float32),
                                name="relation_att")
        self.n_heads = n_heads
        self.d_k = d_k
        self.sqrt_dk = np.sqrt(d_k)
        self.exp = ms.ops.Exp()
        self.reduce_sum = ms.ops.ReduceSum(keep_dims=True)
        self.reduce = ms.ops.ReduceMax(keep_dims=True)
        self.norm = ms.ops.Div()

    def construct(self, k, v, q, edge_mask, g: Graph):
        """homo HGT layer forward."""
        k_tran = ms.ops.Transpose()(ms.ops.BatchMatMul()(ms.ops.Transpose()(k, (1, 0, 2)),
                                                         ms.ops.Reshape()(self.att, (-1, self.d_k, self.d_k))),
                                    (1, 0, 2))
        v_tran = ms.ops.Transpose()(ms.ops.BatchMatMul()(ms.ops.Transpose()(v, (1, 0, 2)),
                                                         ms.ops.Reshape()(self.msg, (-1, self.d_k, self.d_k))),
                                    (1, 0, 2))
        g.set_vertex_attr({"qe": q, "ke": k_tran, "ve": v_tran})

        edge_mask = ms.ops.Reshape()(edge_mask, (-1, 1, 1))
        g.set_edge_attr({"em": edge_mask})

        for v in g.dst_vertex:
            e_origin = [ms.ops.ReduceSum(keep_dims=True)(v.qe * u.ke, -1) * self.pri / self.sqrt_dk
                        for u in v.innbs]
            e_max = self.reduce(e_origin, -1) + _EPS
            e = self.exp(self.norm(e_origin, e_max))
            attn_score = [c / g.sum(e) for c in e]
            a = [u.ve * e.em for u, e in v.inedges]
            v.ret = g.sum(attn_score * a)
        ret = [v.ret for v in g.dst_vertex]
        return ret


class HeteroHGTLayer(nn.Cell):
    """Hetero HGT layer."""

    def __init__(self,
                 num_node_types: int,
                 num_edge_types: int,
                 hidden_size: int,
                 output_size: int,
                 dropout: float = 0.2,
                 n_heads: int = 4,
                 use_norm=True):
        super().__init__()
        self.num_ntypes = num_node_types
        self.num_etypes = num_edge_types
        self.output_size = output_size
        self.n_heads = n_heads
        self.use_norm = use_norm
        cl_k_tmp = []
        cl_q_tmp = []
        cl_v_tmp = []
        cl_a_tmp = []
        if use_norm:
            cl_norm_tmp = []
        for i in range(num_node_types):
            cl_k_tmp.append(ms.nn.Dense(hidden_size, output_size))
            cl_q_tmp.append(ms.nn.Dense(hidden_size, output_size))
            cl_v_tmp.append(ms.nn.Dense(hidden_size, output_size))
            cl_a_tmp.append(ms.nn.Dense(output_size, output_size))
            if use_norm:
                cl_norm_tmp.append(ms.nn.LayerNorm((output_size,)))
        self.cl_k = ms.nn.CellList(cl_k_tmp)
        self.cl_q = ms.nn.CellList(cl_q_tmp)
        self.cl_v = ms.nn.CellList(cl_v_tmp)
        self.cl_a = ms.nn.CellList(cl_a_tmp)
        self.skip = ms.Parameter(ms.ops.Ones()((num_node_types,), ms.float32), name="skip{}".format(i))
        if use_norm:
            self.cl_norm = ms.nn.CellList(cl_norm_tmp)
        self.d_k = output_size // n_heads
        self.drop = ms.nn.Dropout(dropout)
        layer = []
        for _ in range(self.num_etypes):
            layer.append(HomoHGTLayer(n_heads, self.d_k))
        self.layers = ms.nn.CellList(layer)

        self.expand_dims = ms.ops.ExpandDims()

    def construct(self, h, pt_nidx, pt_adj, edge_mask=None, pt_eidx=None):
        """Hetero HGT layer forward"""

        shape = ms.ops.Shape()
        num_nodes = shape(h)[0]

        out = ms.ops.Zeros()((num_nodes, self.n_heads, self.d_k), ms.float32)

        k = ms.ops.Zeros()((num_nodes, self.n_heads, self.d_k), ms.float32)
        v = ms.ops.Zeros()((num_nodes, self.n_heads, self.d_k), ms.float32)
        q = ms.ops.Zeros()((num_nodes, self.n_heads, self.d_k), ms.float32)

        reshape = ms.ops.Reshape()

        for ntype in range(self.num_ntypes):
            nidx = pt_nidx[ntype]
            h_of_ntype = h[nidx]
            k[nidx] = reshape(self.cl_k[ntype](h_of_ntype), (-1, self.n_heads, self.d_k))
            v[nidx] = reshape(self.cl_v[ntype](h_of_ntype), (-1, self.n_heads, self.d_k))
            q[nidx] = reshape(self.cl_q[ntype](h_of_ntype), (-1, self.n_heads, self.d_k))

        for etype in range(self.num_etypes):
            num_edges = shape(pt_adj)[2]
            src_idx = pt_adj[etype, 0]
            dst_idx = pt_adj[etype, 1]
            if edge_mask is None:
                homo_em = ms.ops.Ones()(num_edges, ms.float32)
            else:
                homo_em = edge_mask[pt_eidx[etype]]
            ret = self.layers[etype](k, v, q, homo_em,
                                     src_idx, dst_idx, num_nodes, num_edges)
            out += ret

        new_h = ms.ops.Zeros()(h.shape, ms.float32)

        for ntype in range(self.num_ntypes):
            nidx = pt_nidx[ntype]
            ntype_out = out[nidx]
            alpha = ms.ops.Sigmoid()(self.skip[ntype])
            t = ms.ops.Reshape()(ntype_out, (-1, self.output_size))
            emb = self.cl_a[ntype](t)
            dropped = self.drop(emb)
            trans_out = dropped * alpha + h[nidx] * (1 - alpha)
            if self.use_norm:
                new_h[nidx] = self.cl_norm[ntype](trans_out)
            else:
                new_h[nidx] = trans_out

        return new_h


class XEGT(nn.Cell):
    """eXplainable Embedding Graph Transformer."""
    def __init__(self,
                 num_node_types: int,
                 num_edge_types: int,
                 num_node_feat: int,
                 hidden_size: int,
                 output_size: int,
                 dropout: float = 0.2,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 use_norm=True):
        super().__init__()
        self.num_ntypes = num_node_types
        self.num_etypes = num_edge_types
        self.num_node_feat = num_node_feat
        self.hidden_size = hidden_size
        self.edge_mask = []
        self.node_mask = []
        cl = []
        for _ in range(num_node_types):
            cl.append(ms.nn.Dense(num_node_feat, hidden_size))
        self.cl = ms.nn.CellList(cl)

        cl2 = []
        for _ in range(num_node_types):
            cl2.append(ms.nn.Dense(num_node_feat, hidden_size))
        self.cl2 = ms.nn.CellList(cl2)

        layers = []
        for _ in range(n_layers):
            layers.append(
                HeteroHGTLayer(num_node_types, num_edge_types, hidden_size, hidden_size, dropout, n_heads, use_norm))
        self.layers = ms.nn.CellList(layers)

        layers2 = []
        for _ in range(n_layers):
            layers2.append(
                HeteroHGTLayer(num_node_types, num_edge_types, hidden_size, hidden_size, dropout, n_heads, use_norm))
        self.layers2 = ms.nn.CellList(layers2)

        self.out = ms.nn.Dense(hidden_size, output_size)
        self.gelu = ms.ops.GeLU()
        self.gather = ms.ops.Gather()
        self.concat = ms.ops.Concat()

        relu = ms.nn.ReLU()
        elu = ms.nn.ELU(alpha=1.0)

        # construct the network to calculate the weights
        for_h_node = []
        for _ in range(num_node_types):
            h_att_nets = [ms.nn.Dense(self.num_node_feat + self.hidden_size, self.num_node_feat), elu,
                          ms.nn.Dense(self.num_node_feat, self.num_node_feat), elu]
            h_att_nets = ms.nn.SequentialCell(h_att_nets)
            for_h_node.append(h_att_nets)
        self.h_att_nets = ms.nn.CellList(for_h_node)

        emb_channels = self.num_node_feat * 2 + num_edge_types
        h_edge_nets = [ms.nn.Dense(emb_channels, emb_channels), relu,
                       ms.nn.Dense(emb_channels, 1)]
        self.h_edge_nets = ms.nn.SequentialCell(h_edge_nets)
        self.one_hot = ms.nn.OneHot(depth=num_edge_types)

    def construct(self, h, adj, etype, pt_nidx, pt_adj, pt_eidx, out_nidx):
        """Forward."""

        # 1st pass
        edge_mask = None
        new_h = ms.ops.Zeros()((ms.ops.Shape()(h)[0], self.hidden_size), ms.float32)
        for ntype in range(self.num_ntypes):
            nidx = pt_nidx[ntype]
            new_h[nidx] = self.cl[ntype](h[nidx])

        for i in range(len(self.layers)):
            new_h = self.layers[i](new_h, pt_nidx, pt_adj, edge_mask, pt_eidx)

        # compute attention
        concat = ms.ops.Concat(axis=1)
        h_atts = ms.ops.Ones()(ms.ops.Shape()(h), ms.float32)
        for ntype in range(self.num_ntypes):
            nidx = pt_nidx[ntype]
            h_atts[nidx] = self.h_att_nets[ntype](concat((h[nidx], new_h[nidx])))

        h *= h_atts

        src_idx = adj[0]
        dst_idx = adj[1]
        src_feat = h[src_idx]
        dst_feat = h[dst_idx]

        etype_onehot = self.one_hot(etype)
        concat = ms.ops.Concat(1)
        edge_emb = concat((src_feat, dst_feat, etype_onehot))
        edge_mask = self.h_edge_nets(edge_emb)
        # append a 0 for the dummy edge
        edge_mask = ms.ops.Concat(0)((edge_mask.squeeze(), ms.ops.Zeros()((1, ), ms.float32)))

        # 2nd pass
        new_h = ms.ops.Zeros()((ms.ops.Shape()(h)[0], self.hidden_size), ms.float32)
        for ntype in range(self.num_ntypes):
            nidx = pt_nidx[ntype]
            # another set of MLP
            new_h[nidx] = self.cl2[ntype](h[nidx])

        for i in range(len(self.layers)):
            # another set of conv layer
            new_h = self.layers2[i](new_h, pt_nidx, pt_adj, edge_mask, pt_eidx)

        out_h = new_h[out_nidx]
        out = self.out(out_h)

        return out, h_atts, edge_mask


class LossNet(nn.Cell):
    """loss definition"""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = ms.nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, h, adj, etype, pt_nidx, pt_adj, pt_eidx, out_nidx, ground_truth):
        """Network with loss function"""
        predict, atts, edge_w = self.net(h, adj, etype, pt_nidx, pt_adj, pt_eidx, out_nidx)
        loss = self.loss_fn(predict, ground_truth)
        return loss


class TrainOneStepCellWithGradClipping(ms.nn.TrainOneStepCell):
    """Train one step with gradient clipping."""
    def __init__(self, net, optimizer, clip_val=1.0):
        super().__init__(net, optimizer)
        self.clip_val = Tensor(clip_val, dtype=ms.float32)
        self.hyper_map = ms.ops.HyperMap()
        self.one = Tensor(1.0, dtype=ms.float32)

    def construct(self, h, adj, etype, pt_nidx, pt_adj, pt_eidx, out_nidx, ground_truth):
        weights = self.weights
        loss = self.network(h, adj, etype, pt_nidx, pt_adj, pt_eidx, out_nidx, ground_truth)
        grads = self.grad(self.network, weights)(h, adj, etype, pt_nidx, pt_adj, pt_eidx,
                                                 out_nidx, ground_truth, self.one)
        grads = self.hyper_map(F.partial(clip_grad, self.clip_val), grads)
        grads = self.grad_reducer(grads)
        succ = self.optimizer(grads)
        return F.depend(loss, succ)


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
