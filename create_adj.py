import numpy as np
import pickle 
import scipy.sparse as sp
import os

from ToolScripts.utils import loadData

def creatMultiItemUserAdj(dataset, cv):
    trainMat, _, _, trainMat_time, _ = loadData(dataset, cv)

    ratingClass = np.unique(trainMat.data).size
    userNum, itemNum = trainMat.shape
    multi_adj = sp.lil_matrix((ratingClass*itemNum, userNum), dtype=np.int)
    uidList = trainMat.tocoo().row
    iidList = trainMat.tocoo().col
    rList = trainMat.tocoo().data
    # time = trainMat_time.tocoo().data

    for i in range(uidList.size):
        uid = uidList[i]
        iid = iidList[i]
        r = rList[i]
        multi_adj[iid*ratingClass+r-1, uid] = trainMat_time[uid, iid]
        assert trainMat_time[uid, iid] != 0

    a = sp.csr_matrix((multi_adj.shape[1], multi_adj.shape[1]))
    b = sp.csr_matrix((multi_adj.shape[0], multi_adj.shape[0]))
    multi_adj2 = sp.vstack([sp.hstack([a, multi_adj.T]), sp.hstack([multi_adj,b])])

    DIR = os.path.join(os.getcwd(), "dataset", dataset, 'implicit', "cv{0}".format(cv))
    path = DIR + '/multi_item_adj.pkl'
    with open(path, 'wb') as fs:
        pickle.dump(multi_adj2.tocsr(), fs)
    print("create multi_item_feat")



# if __name__ == '__main__':
#     creatMultiItemUserAdj("CiaoDVD_time", 1)
#     print("Done")