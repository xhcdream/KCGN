import torch
import torch.nn as nn
import math
from DGI.gcn import GCN

class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, activation):
        super(Encoder, self).__init__()
        self.g = g
        self.conv = GCN(g, in_feats, n_hidden, activation)

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())
            features = features[perm]
        features = self.conv(features)
        return features


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_hidden, n_hidden)))

    def forward(self, node_embedding, graph_embedding):
        res = torch.sum(node_embedding * graph_embedding, dim=1)
        return res


class DGI(nn.Module):
    def __init__(self, g, in_feats, n_hidden, gcnAct, graphAct):
        super(DGI, self).__init__()
        self.encoder = Encoder(g, in_feats, n_hidden, gcnAct)
        self.discriminator = Discriminator(n_hidden)
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.graphAct = graphAct

    def forward(self, features, subGraphAdj, subGraphNorm, nodeSubGraph, nodeList):
        positive = self.encoder(features, corrupt=False)
        negative = self.encoder(features, corrupt=True)
        #get every sub graph embedding
        graphEmbeddings = torch.sparse.mm(subGraphAdj, positive) / subGraphNorm
        #activate by sigmoid or tanh
        graphEmbeddings = self.graphAct(graphEmbeddings)
        summary = graphEmbeddings[nodeSubGraph]
        positive_score = self.discriminator(positive, summary)
        negative_score = self.discriminator(negative, summary)
        pos_loss = self.loss(positive_score, torch.ones_like(positive_score))
        neg_loss = self.loss(negative_score, torch.zeros_like(negative_score))
        return pos_loss, neg_loss
            

