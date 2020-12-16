import pickle as pk
from ToolScripts.TimeLogger import log
import torch as t
import scipy.sparse as sp
import numpy as np
import os
import networkx as nx

def mkdir(dataset):
    DIR = os.path.join(os.getcwd(), "History", dataset)
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    DIR = os.path.join(os.getcwd(), "Model", dataset)
    if not os.path.exists(DIR):
        os.makedirs(DIR)


def buildSubGraph(mat, subNode):
    node_num = mat.shape[0]
    graph = nx.Graph(mat)
    subGraphList = list(nx.connected_components(graph))
    subGraphCount = len(subGraphList)
    node_subGraph = [-1 for i in range(node_num)]
    adjMat = sp.dok_matrix((subGraphCount, node_num), dtype=np.int)
    node_list = []
    for i in range(len(subGraphList)):
        subGraphID = i
        subGraph = subGraphList[i]
        if len(subGraph) > subNode:
            node_list += list(subGraph)
        for node_id in subGraph:
            assert node_subGraph[node_id] == -1
            node_subGraph[node_id] = subGraphID
            adjMat[subGraphID, node_id] = 1
    node_subGraph = np.array(node_subGraph)
    assert np.sum(node_subGraph == -1) == 0 
    adjMat = adjMat.tocsr()
    return subGraphList, node_subGraph, adjMat, node_list


def loadData(datasetStr):
    DIR = os.path.join(os.getcwd(), "dataset", datasetStr)
    log(DIR)
    with open(DIR + '/train.pkl', 'rb') as fs:
        trainMat = pk.load(fs)
    with open(DIR + '/test_data.pkl', 'rb') as fs:
        testData = pk.load(fs)
    with open(DIR + '/valid_data.pkl', 'rb') as fs:
        validData = pk.load(fs)
    with open(DIR + '/train_time.pkl', 'rb') as fs:
        trainTimeMat = pk.load(fs)
    with open(DIR + '/trust.pkl', 'rb') as fs:
        trustMat = pk.load(fs)
    return (trainMat, testData, validData, trainTimeMat, trustMat)
    
def load(path):
    with open(path, 'rb') as fs:
        data = pk.load(fs)
    return data

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if type(sparse_mx) != sp.coo_matrix:
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = t.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = t.from_numpy(sparse_mx.data)
    shape = t.Size(sparse_mx.shape)
    return t.sparse.FloatTensor(indices, values, shape)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()




    