import networkx as nx
import torch as t
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import dgl
import dgl.function as fn
import math

def message_func(edges):
    return {'m' : edges.src['n_f'] + edges.data['e_f']}

class MODEL(nn.Module):
    def __init__(self, args, userNum, itemNum, hide_dim, maxTime, interactionNum=5, layer=[16,16]):
        super(MODEL, self).__init__()
        self.userNum = userNum
        self.itemNum = itemNum
        self.hide_dim = hide_dim
        self.layer = [hide_dim] + layer
        self.embedding_dict = self.init_weight(userNum, itemNum*interactionNum, hide_dim)
        self.args = args
        slope = self.args.slope
        self.addTime = args.time
        # GCN activation is leakyReLU
        self.act = t.nn.LeakyReLU(negative_slope=slope)
        if self.args.fuse == "weight":
            self.w = nn.Parameter(t.Tensor(itemNum, interactionNum, 1))
            init.xavier_uniform_(self.w)

        if self.addTime == 1:
            self.t_e = TimeEncoding(self.hide_dim, maxTime)
        self.layers = nn.ModuleList()
        for i in range(0, len(self.layer)-1):
                self.layers.append(GCNLayer(self.layer[i], self.layer[i+1], self.addTime, weight=True, bias=False, activation=self.act))
    
    def init_weight(self, userNum, itemNum, hide_dim):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(t.empty(userNum, hide_dim))),
            'item_emb': nn.Parameter(initializer(t.empty(itemNum, hide_dim))),
        })
        return embedding_dict
    
    def forward(self, graph, time_seq, out_dim, rClass=5, isTrain=True):
        all_user_embeddings = [self.embedding_dict['user_emb']]
        all_item_embeddings = [self.embedding_dict['item_emb']]
        if len(self.layers) == 0:
            item_embedding = self.embedding_dict['item_emb'].view(-1, rClass, out_dim)
            ret_item_embedding = t.div(t.sum(item_embedding, dim=1), rClass)
            return self.embedding_dict['user_emb'], ret_item_embedding
        if self.addTime == 1:
            edge_feat = self.t_e(time_seq)
        else:
            edge_feat = None

        for i, layer in enumerate(self.layers):
            if i == 0:
                embeddings = layer(graph, self.embedding_dict['user_emb'], self.embedding_dict['item_emb'], edge_feat)
            else:
                embeddings = layer(graph, embeddings[: self.userNum], embeddings[self.userNum: ], edge_feat)
            
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            all_user_embeddings += [norm_embeddings[: self.userNum]]
            all_item_embeddings += [norm_embeddings[self.userNum: ]]


        user_embedding = t.cat(all_user_embeddings, 1)
        item_embedding = t.cat(all_item_embeddings, 1)
        if rClass == 1:
            return user_embedding, item_embedding
        item_embedding = item_embedding.view(-1, rClass, out_dim)
        if self.args.fuse == "mean":
            ret_item_embedding = t.div(t.sum(item_embedding, dim=1), rClass)
        elif self.args.fuse == "weight":
            weight = t.softmax(self.w, dim=1)
            item_embedding = item_embedding*weight
            ret_item_embedding = t.sum(item_embedding, dim=1)
        return user_embedding, ret_item_embedding


class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 timeEdge,
                 weight=True,
                 bias=False,
                 activation=None):
        super(GCNLayer, self).__init__()
        self.bias = bias
        self.addTime = timeEdge
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = weight
        if self.weight:
            self.u_w = nn.Parameter(t.Tensor(in_feats, out_feats))
            self.v_w = nn.Parameter(t.Tensor(in_feats, out_feats))
            # self.e_w = nn.Parameter(t.Tensor(in_feats, out_feats))
            init.xavier_uniform_(self.u_w)
            init.xavier_uniform_(self.v_w)
            # init.xavier_uniform_(self.e_w)
        self._activation = activation

    # def forward(self, graph, feat):
    def forward(self, graph, u_f, v_f, e_f):
        with graph.local_scope():
            if self.weight:
                u_f = t.mm(u_f, self.u_w)
                v_f = t.mm(v_f, self.v_w)
                # e_f = t.mm(e_f, self.e_w)
            node_f = t.cat([u_f, v_f], dim=0)
            # D^-1/2
            # degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
            degs = graph.out_degrees().to(u_f.device).float().clamp(min=1)
            norm = t.pow(degs, -0.5).view(-1, 1)
            # norm = norm.view(-1,1)
            # shp = norm.shape + (1,) * (feat.dim() - 1)
            # norm = t.reshape(norm, shp)

            node_f = node_f * norm

            graph.ndata['n_f'] = node_f
            if self.addTime == 1:
                graph.edata['e_f'] = e_f
                graph.update_all(message_func=message_func, reduce_func=fn.sum(msg='m', out='n_f'))
            else:
                graph.update_all(message_func=fn.copy_src(src='n_f', out='m'), reduce_func=fn.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = t.pow(degs, -0.5).view(-1, 1)
            # shp = norm.shape + (1,) * (feat.dim() - 1)
            # norm = t.reshape(norm, shp)
            rst = rst * norm

            if self._activation is not None:
                rst = self._activation(rst)

            return rst
            
class TimeEncoding(nn.Module):
    def __init__(self, n_hid, max_len = 240, dropout = 0.2):
        super(TimeEncoding, self).__init__()
        position = t.arange(0., max_len).unsqueeze(1)
        # ref self-attention
        div_term = 1 / (10000 ** (t.arange(0., n_hid * 2, 2.)) / n_hid / 2)
        
        # initializer = nn.init.xavier_uniform_
        # self.emb = nn.Parameter(initializer(t.empty(max_len, n_hid)))
        self.emb = nn.Embedding(max_len, n_hid * 2)
        self.emb.weight.data[:, 0::2] = t.sin(position * div_term) / math.sqrt(n_hid)
        self.emb.weight.data[:, 1::2] = t.cos(position * div_term) / math.sqrt(n_hid)
        self.emb.weight.requires_grad = False
        # 0 is useless, 1 is self->self
        self.emb.weight.data[0] = t.zeros_like(self.emb.weight.data[-1])
        self.emb.weight.data[1] = t.zeros_like(self.emb.weight.data[-1])
        self.lin = nn.Linear(n_hid * 2, n_hid)

    def forward(self, time):
        return self.lin(self.emb(time))