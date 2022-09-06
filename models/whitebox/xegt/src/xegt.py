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
