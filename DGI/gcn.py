"""
This code was copied from the GCN implementation in DGL examples.
"""
import torch
import torch.nn as nn
#DGL implement https://github.com/dmlc/dgl
from DGI.graphconv import GraphConv

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 activation):
        super(GCN, self).__init__()
        self.g = g
        self.layer = GraphConv(in_feats, n_hidden, weight=False, activation=activation)

    def forward(self, features):
        h = features
        h = self.layer(self.g, h)
        return h
