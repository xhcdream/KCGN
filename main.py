# coding=UTF-8
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ToolScripts.TimeLogger import log
import pickle
import os
import sys
import gc
import random
import argparse
import scipy.sparse as sp
from ToolScripts.utils import loadData
from ToolScripts.utils import load
from ToolScripts.utils import buildSubGraph
from ToolScripts.utils import sparse_mx_to_torch_sparse_tensor
from ToolScripts.utils import mkdir
from dgl import DGLGraph
import dgl
from MyGCN import MODEL
from BPRData import BPRData
import torch.utils.data as dataloader
from DGI.dgi import DGI
import evaluate
import time
import networkx as nx


device_gpu = t.device("cuda")
modelUTCStr = str(int(time.time()))[4:]

isLoadModel = False
LOAD_MODEL_PATH = ""

class Model():

    def __init__(self, args, isLoad=False):
        self.args = args
        self.datasetDir = os.path.join(os.getcwd(), "dataset", args.dataset)
        trainMat, validData, multi_adj_time, uuMat, iiMat = self.getData(args)
        self.userNum, self.itemNum = trainMat.shape
        log("uu num = %d"%(uuMat.nnz))
        log("ii num = %d"%(iiMat.nnz))
        self.trainMat = trainMat
        if self.args.dgi == 1:
            log("dgi process...")

            # self.uu_graph = DGLGraph(uuMat)
            uuMat_edge_src, uuMat_edge_dst = uuMat.nonzero()
            self.uu_graph = dgl.graph(data=(uuMat_edge_src, uuMat_edge_dst),
                              idtype=t.int32,
                              num_nodes=uuMat.shape[0],
                              device=device_gpu)
            # self.ii_graph = DGLGraph(iiMat)
            iiMat_edge_src, iiMat_edge_dst = iiMat.nonzero()
            self.ii_graph = dgl.graph(data=(iiMat_edge_src, iiMat_edge_dst),
                              idtype=t.int32,
                              num_nodes=iiMat.shape[0],
                              device=device_gpu)

            #get sub graph message
            uu_subGraph_data = self.datasetDir + '/uuMat_subGraph_data.pkl'
            if self.args.clear == 1:
                if os.path.exists(uu_subGraph_data):
                    log("clear uu sub graph message")
                    os.remove(uu_subGraph_data)

            if os.path.exists(uu_subGraph_data):
                data = load(uu_subGraph_data)
                self.uu_node_subGraph, self.uu_subGraph_adj, self.uu_dgi_node = data
            else:
                log("rebuild uu sub graph message")
                _, self.uu_node_subGraph, self.uu_subGraph_adj, self.uu_dgi_node = buildSubGraph(uuMat, self.args.subNode)
                data = (self.uu_node_subGraph, self.uu_subGraph_adj, self.uu_dgi_node)
                with open(uu_subGraph_data, 'wb') as fs:
                    pickle.dump(data, fs)
            
            ii_subGraph_data = self.datasetDir + '/iiMat_subGraph_data.pkl'

            if self.args.clear == 1:
                if os.path.exists(ii_subGraph_data):
                    log("clear ii sub graph message")
                    os.remove(ii_subGraph_data)

            if os.path.exists(ii_subGraph_data):
                data = load(ii_subGraph_data)
                self.ii_node_subGraph, self.ii_subGraph_adj, self.ii_dgi_node = data
            else:
                log("rebuild ii sub graph message")
                _, self.ii_node_subGraph, self.ii_subGraph_adj, self.ii_dgi_node = buildSubGraph(iiMat, self.args.subNode)
                data = (self.ii_node_subGraph, self.ii_subGraph_adj, self.ii_dgi_node)
                with open(ii_subGraph_data, 'wb') as fs:
                    pickle.dump(data, fs)

            self.uu_subGraph_adj_tensor = sparse_mx_to_torch_sparse_tensor(self.uu_subGraph_adj).cuda()
            self.uu_subGraph_adj_norm = t.from_numpy(np.sum(self.uu_subGraph_adj, axis=1)).float().cuda()

            self.ii_subGraph_adj_tensor = sparse_mx_to_torch_sparse_tensor(self.ii_subGraph_adj).cuda()
            self.ii_subGraph_adj_norm = t.from_numpy(np.sum(self.ii_subGraph_adj, axis=1)).float().cuda()

            self.uu_dgi_node_mask = np.zeros(self.userNum)
            self.uu_dgi_node_mask[self.uu_dgi_node] = 1
            self.uu_dgi_node_mask = t.from_numpy(self.uu_dgi_node_mask).float().cuda()

            self.ii_dgi_node_mask = np.zeros(self.itemNum)
            self.ii_dgi_node_mask[self.ii_dgi_node] = 1
            self.ii_dgi_node_mask = t.from_numpy(self.ii_dgi_node_mask).float().cuda()
        
        #norm time value
        if self.args.time == 1:
            log("time process")
            self.time_step = self.args.time_step
            log("time step = %.1f hour"%(self.time_step))
            time_step = 3600*self.time_step
            row, col = multi_adj_time.nonzero()
            data = multi_adj_time.data
            minUTC = data.min()
            #data.min = 2
            data = ((data-minUTC)/time_step).astype(np.int)+2
            assert np.sum(row == col) == 0
            multi_adj_time_norm = sp.coo_matrix((data, (row, col)), dtype=np.int, shape=multi_adj_time.shape).tocsr()
            self.maxTime = multi_adj_time_norm.max() + 1
            log("max time = %d"%(self.maxTime))
            num = multi_adj_time_norm.shape[0]
            multi_adj_time_norm = multi_adj_time_norm + sp.eye(num)
            print("uv graph link num = %d"%(multi_adj_time_norm.nnz))
        else:
            self.maxTime = None
            num = multi_adj_time.shape[0]
            multi_adj_time = multi_adj_time + sp.eye(num)
            multi_adj_time_norm = (multi_adj_time!=0) * 1

        
        # multi_adj = (multi_adj_time_norm!=0)
        # self.v_u_adj = multi_adj[self.userNum:, 0:self.userNum]
        # self.multi_item_mask = t.from_numpy(np.sum(self.v_u_adj,axis=1).A!=0).float().to(device_gpu)
        
        edge_src, edge_dst = multi_adj_time_norm.nonzero()
        # edge_src = multi_adj_time_norm.tocoo().row
        # edge_dst = multi_adj_time_norm.tocoo().col
        if self.args.time == 1:
            time_seq = multi_adj_time_norm.tocoo().data
            self.time_seq_tensor = t.from_numpy(time_seq.astype(np.float)).long().to(device_gpu)
        
        self.ratingClass = np.unique(trainMat.data).size
        log("user num =%d, item num =%d"%(self.userNum, self.itemNum))

        # self.uv_g = DGLGraph()
        # self.uv_g.add_nodes(multi_adj_time_norm.shape[0])
        # self.uv_g.add_edges(edge_src, edge_dst)
        self.uv_g = dgl.graph(data=(edge_src, edge_dst),
                              idtype=t.int32,
                              num_nodes=multi_adj_time_norm.shape[0],
                              device=device_gpu)


        #train data
        train_u, train_v = self.trainMat.nonzero()
        assert np.sum(self.trainMat.data ==0) == 0
        log("train data size = %d"%(train_u.size))
        train_data = np.hstack((train_u.reshape(-1,1), train_v.reshape(-1,1))).tolist()
        train_dataset = BPRData(train_data, self.itemNum, self.trainMat, self.args.num_ng, True)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=self.args.batch, shuffle=True, num_workers=0)
        #valid data
        valid_dataset = BPRData(validData, self.itemNum, self.trainMat, 0, False)
        self.valid_loader  = dataloader.DataLoader(valid_dataset, batch_size=args.test_batch*101, shuffle=False, num_workers=0)
        
        self.lr = self.args.lr #0.001
        self.curEpoch = 0
        self.isLoadModel = isLoad
        #history
        self.train_loss = []
        self.his_hr = []
        self.his_ndcg  = []
        gc.collect()
        log("gc.collect()")

    def setRandomSeed(self):
        np.random.seed(self.args.seed)
        t.manual_seed(self.args.seed)
        t.cuda.manual_seed(self.args.seed)
        random.seed(self.args.seed)
    
    def getData(self, args):
        data = loadData(args.dataset)
        trainMat, _, validData, _, _ = data
        with open(self.datasetDir + '/multi_item_adj.pkl', 'rb') as fs:
            multi_adj_time = pickle.load(fs)
        with open(self.datasetDir + '/uu_vv_graph.pkl', 'rb') as fs:
            uu_vv_graph = pickle.load(fs)
        uuMat = uu_vv_graph['UU'].astype(np.bool)
        iiMat = uu_vv_graph['II'].astype(np.bool)
        return trainMat, validData, multi_adj_time, uuMat, iiMat

    #初始化参数
    def prepareModel(self):
        self.modelName = self.getModelName() 
        self.setRandomSeed()

        self.layer = eval(self.args.layer)
        self.hide_dim = args.hide_dim
        self.out_dim = sum(self.layer) + self.hide_dim
        # self.out_dim = self.hide_dim
        
        # self.act = t.nn.ReLU()
        self.model = MODEL(self.args, self.userNum, self.itemNum, self.hide_dim, \
            self.maxTime, self.ratingClass, self.layer).cuda()


        if self.args.dgi == 1:
            if self.args.dgi_graph_act == "sigmoid":
                dgiGraphAct = nn.Sigmoid()
            elif self.args.dgi_graph_act == "tanh":
                dgiGraphAct = nn.Tanh()

            self.uu_dgi = DGI(self.uu_graph, self.out_dim, self.out_dim, nn.PReLU(), dgiGraphAct).cuda()
            self.ii_dgi = DGI(self.ii_graph, self.out_dim, self.out_dim, nn.PReLU(), dgiGraphAct).cuda()
            
            self.opt = t.optim.Adam([
                {'params': self.model.parameters(), 'weight_decay': 0},
                {'params': self.uu_dgi.parameters(), 'weight_decay': 0},
                {'params': self.ii_dgi.parameters(), 'weight_decay': 0},
            ], lr=self.args.lr)
        else:
            self.opt = t.optim.Adam(self.model.parameters(), lr = self.args.lr, weight_decay=0)

    def adjust_learning_rate(self, opt, epoch):
        for param_group in opt.param_groups:
            param_group['lr'] = max(param_group['lr'] * self.args.decay, self.args.minlr)
            # log("cur lr = %.6f"%(param_group['lr']))
    
    def innerProduct(self, u, i, j):
        pred_i = t.sum(t.mul(u,i), dim=1)
        pred_j = t.sum(t.mul(u,j), dim=1)
        return pred_i, pred_j
    
    def run(self):
        self.prepareModel()
        if self.isLoadModel == True:
            self.loadModel(LOAD_MODEL_PATH)
            HR, NDCG = self.test()
            return
        cvWait = 0
        best_HR = 0.1
        for e in range(self.curEpoch, self.args.epochs+1):
            self.curEpoch = e
            log("**************************************************************")
            log("start train")
            epoch_loss, epoch_uu_dgi_loss, epoch_ii_dgi_loss = self.trainModel()
            log("end train")
            self.train_loss.append(epoch_loss)
            log("epoch %d/%d, epoch_loss=%.2f, dgi_uu_loss=%.4f, dgi_ii_loss=%.4f"% \
                (e, self.args.epochs, epoch_loss, epoch_uu_dgi_loss, epoch_ii_dgi_loss))
            
            if e < self.args.startTest:
                HR, NDCG = 0, 0
                cvWait = 0
            else:
                HR, NDCG = self.validModel(self.valid_loader)
                self.his_hr.append(HR)
                self.his_ndcg.append(NDCG)
                log("epoch %d/%d, valid HR = %.4f, valid NDCG = %.4f"%(e, self.args.epochs, HR, NDCG))
            
            if e%10 == 0 and e != 0:
                testHR, testNDCG = self.test()
                log("test HR = %.4f, test NDCG = %.4f"%(testHR, testNDCG))

            self.adjust_learning_rate(self.opt, e)
            if HR > best_HR:
                best_HR = HR
                cvWait = 0
                best_epoch = self.curEpoch
                self.saveModel()
            else:
                cvWait += 1
                log("cvWait = %d"%(cvWait))

            self.saveHistory()

            if cvWait == self.args.patience:
                log('Early stopping! best epoch = %d'%(best_epoch))
                self.loadModel(self.modelName)
                testHR, testNDCG = self.test()
                log("test HR = %.4f, test NDCG = %.4f"%(testHR, testNDCG))
                break
        
        
    def test(self):
        #load test dataset
        with open(self.datasetDir + "/test_data.pkl", 'rb') as fs:
            test_data = pickle.load(fs)
        test_dataset = BPRData(test_data, self.itemNum, self.trainMat, 0, False)
        self.test_loader  = dataloader.DataLoader(test_dataset, batch_size=args.test_batch*101, shuffle=False, num_workers=0)
        HR, NDCG = self.validModel(self.test_loader)
        return HR, NDCG

    

    def trainModel(self):
        train_loader = self.train_loader
        log("start negative sample...")
        train_loader.dataset.ng_sample()
        log("finish negative sample...")
        epoch_loss = 0
        epoch_uu_dgi_loss = 0
        epoch_ii_dgi_loss = 0
        for user, item_i, item_j in train_loader:
            user = user.long().cuda()
            item_i = item_i.long().cuda()
            item_j = item_j.long().cuda()
            if self.args.time == 1:
                user_embed, item_embed = self.model(self.uv_g, self.time_seq_tensor, self.out_dim, self.ratingClass, True)
            else:
                user_embed, item_embed = self.model(self.uv_g, None, self.out_dim, self.ratingClass, True)
            # item_muliti_embed = item_muliti_embed.view(-1, self.ratingClass, self.out_dim)
            
            # item_muliti_embed = item_muliti_embed*self.multi_item_mask
            # item_embed = t.sum(item_muliti_embed, dim=1) / t.sum(self.multi_item_mask.view(-1,5), dim=1, keepdim=True)
            
            #TODO try self attention
            # item_embed = t.div(t.sum(item_embed, dim=1), self.ratingClass)
            
            userEmbed = user_embed[user]
            posEmbed = item_embed[item_i]
            negEmbed = item_embed[item_j]

            pred_i, pred_j = self.innerProduct(userEmbed, posEmbed, negEmbed)

            bprloss = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log().sum()
            regLoss = (t.norm(userEmbed) ** 2 + t.norm(posEmbed) ** 2 + t.norm(negEmbed) ** 2)

            loss = 0.5*(bprloss + self.args.reg * regLoss)/self.args.batch

            if self.args.dgi == 1:
                uu_dgi_loss = 0
                ii_dgi_loss = 0
                if self.args.lam[0] != 0:
                    uu_dgi_pos_loss, uu_dgi_neg_loss = self.uu_dgi(user_embed, self.uu_subGraph_adj_tensor, \
                        self.uu_subGraph_adj_norm, self.uu_node_subGraph, self.uu_dgi_node)
                    userMask = t.zeros(self.userNum).cuda()
                    userMask[user] = 1
                    userMask = userMask * self.uu_dgi_node_mask
                    uu_dgi_loss = ((uu_dgi_pos_loss*userMask).sum() + (uu_dgi_neg_loss*userMask).sum())/t.sum(userMask)
                    epoch_uu_dgi_loss += uu_dgi_loss.item()

                if self.args.lam[1] != 0:
                    ii_dgi_pos_loss, ii_dgi_neg_loss = self.ii_dgi(item_embed, self.ii_subGraph_adj_tensor, \
                        self.ii_subGraph_adj_norm, self.ii_node_subGraph, self.ii_dgi_node)
                    iiMask = t.zeros(self.itemNum).cuda()
                    iiMask[item_i] = 1
                    iiMask[item_j] = 1
                    iiMask = iiMask * self.ii_dgi_node_mask
                    ii_dgi_loss = ((ii_dgi_pos_loss*iiMask).sum() + (ii_dgi_neg_loss*iiMask).sum())/t.sum(iiMask)
                    epoch_ii_dgi_loss += ii_dgi_loss.item()
                loss = loss + self.args.lam[0] * uu_dgi_loss + self.args.lam[1] * ii_dgi_loss

            epoch_loss += bprloss.item()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            # log('setp %d/%d, step_loss = %f'%(i, loss.item()), save=False, oneline=True)
        return epoch_loss, epoch_uu_dgi_loss, epoch_ii_dgi_loss

    def validModel(self, data_loader, save=False):
        HR, NDCG = [], []
        data = {}
        if self.args.time == 1:
            user_embed, item_embed = self.model(self.uv_g, self.time_seq_tensor, self.out_dim, self.ratingClass, True)
        else:
            user_embed, item_embed = self.model(self.uv_g, None, self.out_dim, self.ratingClass, True)
        for user, item_i in data_loader:
            user = user.long().cuda()
            item_i = item_i.long().cuda()
            userEmbed = user_embed[user]
            testItemEmbed = item_embed[item_i]
            pred_i = t.sum(t.mul(userEmbed, testItemEmbed), dim=1)

            batch = int(user.cpu().numpy().size/101)
            assert user.cpu().numpy().size % 101 ==0
            for i in range(batch):
                batch_scores = pred_i[i*101: (i+1)*101].view(-1)
                _, indices = t.topk(batch_scores, self.args.top_k)
                tmp_item_i = item_i[i*101: (i+1)*101]
                recommends = t.take(tmp_item_i, indices).cpu().numpy().tolist()
                gt_item = tmp_item_i[0].item()
                HR.append(evaluate.hit(gt_item, recommends))
                NDCG.append(evaluate.ndcg(gt_item, recommends))
        if save:
            return HR, NDCG
        else:
            return np.mean(HR), np.mean(NDCG)


    def getModelName(self):
        title = "KCGN_"
        ModelName = title + self.args.dataset + "_" + modelUTCStr + \
        "_reg_" + str(self.args.reg)+ \
        "_batch_" + str(self.args.batch) + \
        "_lr_" + str(self.args.lr) + \
        "_decay_" + str(self.args.decay) + \
        "_hide_" + str(self.args.hide_dim) + \
        "_Layer_" + self.args.layer +\
        "_slope_" + str(self.args.slope) +\
        "_top_" + str(self.args.top_k) +\
        "_fuse_" + self.args.fuse
        if self.args.time == 1:
            ModelName += "_timeStep_" + str(self.args.time_step)
            ModelName += "_time"
        if self.args.dgi == 1:
            ModelName += "_dgi"
            ModelName += str(self.args.lam)
            ModelName += str(self.args.dgi_graph_act)
        return ModelName


    def saveHistory(self):
        #保存历史数据，用于画图
        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        ModelName = self.modelName

        with open(r'./History/' + args.dataset + r'/' + ModelName + '.his', 'wb') as fs:
            pickle.dump(history, fs)

    def saveModel(self):
        # ModelName = self.getModelName()
        ModelName = self.modelName
        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        savePath = r'./Model/' + self.args.dataset + r'/' + ModelName + r'.pth'
        params = {
            'epoch': self.curEpoch,
            'lr': self.lr,
            'model': self.model,
            'reg':self.args.reg,
            'history':history,
            }
        t.save(params, savePath)


    def loadModel(self, modelPath):
        checkpoint = t.load(r'./Model/' + args.dataset + r'/' + modelPath + r'.pth')
        self.curEpoch = checkpoint['epoch'] + 1
        self.lr = checkpoint['lr']
        self.model = checkpoint['model']
        self.args.reg = checkpoint['reg']
        #恢复history
        history = checkpoint['history']
        self.train_loss = history['loss']
        self.his_hr = history['HR']
        self.his_ndcg = history['NDCG']
        log("load model %s in epoch %d"%(modelPath, checkpoint['epoch']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KCGN main.py')
    #dataset params
    parser.add_argument('--dataset', type=str, default="Yelp", help="Epinions,Yelp")
    parser.add_argument('--seed', type=int, default=29)

    parser.add_argument('--hide_dim', type=int, default=64)
    parser.add_argument('--layer', type=str, default="[64]")
    parser.add_argument('--slope', type=float, default=0.4)

    parser.add_argument('--reg', type=float, default=0.05)
    parser.add_argument('--decay', type=float, default=0.98)
    parser.add_argument('--batch', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--minlr', type=float, default=0.0001)
    parser.add_argument('--test_batch', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=180)
    #early stop params
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_ng', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--fuse', type=str, default="mean", help="mean or weight")

    parser.add_argument('--dgi', type=int, default=1)
    parser.add_argument('--dgi_graph_act', type=str, default="sigmoid", help="sigmoid or tanh")
    parser.add_argument('--lam', type=str, default='[0.1,0.001]')
    parser.add_argument('--clear', type=int, default=0)
    parser.add_argument('--subNode', type=int, default=10)

    parser.add_argument('--time', type=int, default=1)
    parser.add_argument('--time_step', type=float, default=360)
    parser.add_argument('--startTest', type=int, default=0)

    args = parser.parse_args()
    args.lam = eval(args.lam)
    assert len(args.lam) == 2
    print(args)
    mkdir(args.dataset)
    hope = Model(args, isLoadModel)

    modelName = hope.getModelName()
    
    print('ModelNmae = ' + modelName)

    hope.run()
    hope.test()

