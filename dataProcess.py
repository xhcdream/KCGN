import pickle 
import numpy as np
import scipy.sparse as sp
import random
import os
import argparse
from create_adj import creatMultiItemUserAdj
import networkx as nx



def splitData(dataset, cv):
    DIR = os.path.join(os.getcwd(), "dataset", dataset)
    with open(DIR + "/category.pkl", 'rb') as fs:
        category = pickle.load(fs)
    with open(DIR + "/ratings.pkl", 'rb') as fs:
        data = pickle.load(fs)
    with open(DIR + "/times.pkl", 'rb') as fs:
        time = pickle.load(fs)
    with open(DIR + "/trust.pkl", 'rb') as fs:
        trust = pickle.load(fs)
    assert np.sum(data.tocoo().row != time.tocoo().row) == 0
    assert np.sum(data.tocoo().col != time.tocoo().col) == 0
    row, col = data.shape
    print("user num = %d, item num = %d"%(row, col))

    train_row, train_col, train_data, train_data_time = [], [], [], []
    test_row, test_col = [], []
    valid_row, valid_col = [], []

    # userList = np.where(np.sum(data!=0, axis=1)>=2)[0]
    for i in range(row):
        tmp_data = data[i].toarray()[0]
        if np.sum(tmp_data != 0) < 3:
            continue
        tmp_data_time = time[i].toarray()[0]
        uid = [i] * col 
        num = data[i].nnz
        #降序排序
        idx = np.argsort(-tmp_data_time).tolist()
        idx = idx[: num]
        rating_data = tmp_data[idx].tolist()
        time_data = tmp_data_time[idx].tolist()
        assert np.sum(tmp_data[idx] == 0) == 0
        assert np.sum(tmp_data_time[idx] == 0) == 0
        
        test_num = 1
        valid_num = 1
        train_num = num - 2

        test_row += [uid[0]]
        test_col += [idx[0]]

        valid_row += [uid[1]]
        valid_col += [idx[1]]

        train_row += uid[0: train_num]
        train_col += idx[2:]
        train_data += rating_data[2:]
        train_data_time += time_data[2:]
        assert (0 in train_data) == False
        assert (0 in train_data_time) == False


    train = sp.csc_matrix((train_data, (train_row, train_col)), shape=data.shape)
    train_time = sp.csc_matrix((train_data_time, (train_row, train_col)), shape=data.shape)

    test  = sp.csc_matrix(([1]*len(test_row), (test_row, test_col)), shape=data.shape)
    valid  = sp.csc_matrix(([1]*len(valid_row), (valid_row, valid_col)), shape=data.shape)

    print("train num = %d, train rate = %.2f"%(train.nnz, train.nnz/data.nnz))
    print("test num = %d, test rate = %.2f"%(test.nnz, test.nnz/data.nnz))
    print("valid num = %d, valid rate = %.2f"%(valid.nnz, valid.nnz/data.nnz))

    with open(DIR + "/implicit/train.pkl", 'wb') as fs:
        pickle.dump(train.tocsr(), fs)
    with open(DIR + "/implicit/train_time.pkl", 'wb') as fs:
        pickle.dump(train_time.tocsr(), fs)

    with open(DIR + "/implicit/test.pkl", 'wb') as fs:
        pickle.dump(test.tocsr(), fs)
    with open(DIR + "/implicit/valid.pkl", 'wb') as fs:
        pickle.dump(valid.tocsr(), fs)

    with open(DIR + "/implicit/trust.pkl", 'wb') as fs:
        pickle.dump(trust.tocsr(), fs)
    with open(DIR + "/implicit/category.pkl", 'wb') as fs:
        pickle.dump(category.tocsr(), fs)


def filterData(dataset, cv):
    DIR = os.path.join(os.getcwd(), "dataset", dataset)
    #filter
    with open(DIR + "/implicit/train.pkl", 'rb') as fs:
        train = pickle.load(fs)
    with open(DIR + "/implicit/test.pkl", 'rb') as fs:
        test = pickle.load(fs)
    with open(DIR + "/implicit/valid.pkl", 'rb') as fs:
        valid = pickle.load(fs)
    with open(DIR + "/implicit/category.pkl", 'rb') as fs:
        category = pickle.load(fs)

    with open(DIR + "/implicit/train_time.pkl", 'rb') as fs:
        train_time = pickle.load(fs)

    with open(DIR + "/implicit/trust.pkl", 'rb') as fs:
        trust = pickle.load(fs)

    trust = trust + trust.transpose()
    trust = (trust != 0)*1

    a = np.sum(np.sum(train != 0, axis=1) ==0)
    b = np.sum(np.sum(train != 0, axis=0) ==0)
    c = np.sum(np.sum(trust, axis=1) == 0)
    while a != 0 or b != 0 or c != 0:
        if a != 0:
            idx, _ = np.where(np.sum(train != 0, axis=1) != 0)
            train = train[idx]
            test = test[idx]
            valid = valid[idx]
            train_time = train_time[idx]
            trust = trust[idx][:, idx]
        elif b != 0:
            _, idx = np.where(np.sum(train != 0, axis=0) != 0)
            train = train[:, idx]
            test = test[:, idx]
            valid = valid[:, idx]
            train_time = train_time[:, idx]
            category = category[idx]
        elif c != 0:
            idx, _ = np.where(np.sum(trust, axis=1) != 0)
            train = train[idx]
            test = test[idx]
            valid = valid[idx]
            train_time = train_time[idx]
            trust = trust[idx][:, idx]
        a = np.sum(np.sum(train != 0, axis=1) ==0)
        b = np.sum(np.sum(train != 0, axis=0) ==0)
        c = np.sum(np.sum(trust, axis=1) == 0)

    nums = train.nnz+test.nnz+valid.nnz
    print("train num = %d, train rate = %.2f"%(train.nnz, train.nnz/nums))
    print("test num = %d, test rate = %.2f"%(test.nnz, test.nnz/nums))
    print("valid num = %d, valid rate = %.2f"%(valid.nnz, valid.nnz/nums))

    with open(DIR + "/implicit/train.pkl", 'wb') as fs:
        pickle.dump(train, fs)
    with open(DIR + "/implicit/test.pkl", 'wb') as fs:
        pickle.dump(test, fs)
    with open(DIR + "/implicit/valid.pkl", 'wb') as fs:
        pickle.dump(valid, fs)
    with open(DIR + "/implicit/train_time.pkl", 'wb') as fs:
        pickle.dump(train_time, fs)
    with open(DIR + "/implicit/trust.pkl", 'wb') as fs:
        pickle.dump(trust, fs)
    with open(DIR + "/implicit/category.pkl", 'wb') as fs:
        pickle.dump(category, fs)

def splitAgain(dataset, cv):
    DIR = os.path.join(os.getcwd(), "dataset", dataset)
    with open(DIR + "/implicit/train.pkl", 'rb') as fs:
        train = pickle.load(fs)
    with open(DIR + "/implicit/test.pkl", 'rb') as fs:
        test = pickle.load(fs)
    print(train.nnz)
    print(test.nnz)

    with open(DIR + "/implicit/train_time.pkl", 'rb') as fs:
        train_time = pickle.load(fs)

    train = train.tolil()
    test = test.tolil()
    train_time = train_time.tolil()
    
    idx = np.where(np.sum(test!=0, axis=1).A == 0)[0]
    for i in idx:
        uid = i
        tmp_data = train[i].toarray()[0]
        if np.sum(tmp_data != 0) < 2:
            continue
        num = train[i].nnz
        tmp_data_time = train_time[i].toarray()[0]
        l = np.argsort(-tmp_data_time).tolist()
        l = l[: num]
        # test[uid, l[0]] = train[uid, l[0]]
        test[uid, l[0]] = 1
        train[uid, l[0]] = 0
        train_time[uid, l[0]] = 0
    
    train = train.tocsr()
    train_time = train_time.tocsr()
    test = test.tocsr()
    assert  np.sum(train.tocoo().data == 0)==0
    assert  np.sum(test.tocoo().data == 0)==0
    assert  (train+test).nnz == train.nnz+test.nnz

    with open(DIR + "/implicit/train.pkl", 'wb') as fs:
        pickle.dump(train, fs)
    with open(DIR + "/implicit/test.pkl", 'wb') as fs:
        pickle.dump(test, fs)
    with open(DIR + "/implicit/train_time.pkl", 'wb') as fs:
        pickle.dump(train_time, fs)


def generateGraph(dataset, cv):
    DIR = os.path.join(os.getcwd(), "dataset", dataset)
    with open(DIR + "/implicit/train.pkl", 'rb') as fs:
        train = pickle.load(fs)
    with open(DIR + "/implicit/trust.pkl", 'rb') as fs:
        trustMat = pickle.load(fs)
    with open(DIR + "/implicit/category.pkl", 'rb') as fs:
        categoryMat= pickle.load(fs)
    with open(DIR + "/implicit/categoryDict.pkl", 'rb') as fs:
        categoryDict = pickle.load(fs)
    
    userNum, itemNum =  train.shape
    assert categoryMat.shape[0] == train.shape[1]
    mat = (trustMat.T + trustMat) + sp.eye(userNum)
    UU_mat = (mat != 0)*1

    ITI_mat = sp.dok_matrix((itemNum, itemNum))
    categoryMat = categoryMat.toarray()
    for i in range(categoryMat.shape[0]):
        itemTypeList = np.where(categoryMat[i])[0]
        for itemType in itemTypeList:
            itemList = categoryDict[itemType]
            itemList = np.array(itemList)
            if itemList.size < 100:
                rate = 0.1
            elif itemList.size < 1000:
                rate = 0.01
            else:
                rate = 0.001
            itemList2 = np.random.choice(itemList, size=int(itemList.size*rate/2), replace=False)
            itemList2 = itemList2.tolist()
            tmp = [i for _ in range(len(itemList2))]
            ITI_mat[tmp, itemList2] = 1

    ITI_mat = ITI_mat.tocsr()
    ITI_mat = ITI_mat + ITI_mat.T + sp.eye(itemNum)
    ITI_mat = (ITI_mat != 0)*1

    uu_vv_graph = {}
    uu_vv_graph['UU'] = UU_mat
    uu_vv_graph['II'] = ITI_mat
    with open(DIR + '/implicit/uu_vv_graph.pkl', 'wb') as fs:
        pickle.dump(uu_vv_graph, fs)

    
def createCategoryDict(dataset, cv):
    DIR = os.path.join(os.getcwd(), "dataset", dataset)
    with open(DIR + "/implicit/train.pkl", 'rb') as fs:
        train = pickle.load(fs)
    with open(DIR + "/implicit/category.pkl", 'rb') as fs:
        category = pickle.load(fs)
    
    assert category.shape[0] == train.shape[1]
    categoryDict = {}
    categoryData = category.toarray()
    for i in range(categoryData.shape[0]):
        iid = i
        typeList = np.where(categoryData[i])[0]
        # typeid = categoryData[i]
        for typeid in typeList:
            if typeid in categoryDict:
                categoryDict[typeid].append(iid)
            else:
                categoryDict[typeid] = [iid]
    with open(DIR + "/implicit/categoryDict.pkl", 'wb') as fs:
        pickle.dump(categoryDict, fs)

def testNegSample(dataset, cv):
    DIR = os.path.join(os.getcwd(), "dataset", dataset)
    #filter
    with open(DIR + "/implicit/train.pkl", 'rb') as fs:
        train = pickle.load(fs)
    with open(DIR + "/implicit/test.pkl", 'rb') as fs:
        test = pickle.load(fs)
    with open(DIR + "/implicit/valid.pkl", 'rb') as fs:
        valid = pickle.load(fs)

    train = train.todok()
    test_u = test.tocoo().row
    test_v = test.tocoo().col
    valid_u = valid.tocoo().row
    valid_v = valid.tocoo().col
    assert test_u.size == test_v.size
    assert valid_u.size == valid_v.size
    n = test_u.size
    test_data = []
    for i in range(n):
        u = test_u[i]
        v = test_v[i]
        test_data.append([u, v])
        for t in range(100):
            j = np.random.randint(test.shape[1])
            while (u, j) in train or j == v:
                j = np.random.randint(test.shape[1])
            test_data.append([u, j])
    
    n = valid_u.size
    valid_data = []
    for i in range(n):
        u = valid_u[i]
        v = valid_v[i]
        valid_data.append([u, v])
        for t in range(100):
            j = np.random.randint(valid.shape[1])
            while (u, j) in train or j == v:
                j = np.random.randint(valid.shape[1])
            valid_data.append([u, j])
    
    with open(DIR + "/implicit/test_data.pkl", 'wb') as fs:
        pickle.dump(test_data, fs)
    with open(DIR + "/implicit/valid_data.pkl", 'wb') as fs:
        pickle.dump(valid_data, fs)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #dataset params
    parser.add_argument('--dataset', type=str, default="Epinions", help="CiaoDVD,Epinions,Douban")
    parser.add_argument('--cv', type=int, default=1, help="1,2,3,4,5")
    args = parser.parse_args()

    dataset = args.dataset+ "_time"

    splitData(dataset, args.cv)
    filterData(dataset, args.cv)
    splitAgain(dataset, args.cv)
    filterData(dataset, args.cv)

    testNegSample(dataset, args.cv)

    createCategoryDict(dataset, args.cv)
    creatMultiItemUserAdj(dataset, args.cv)
    generateGraph(dataset, args.cv)
    
    print("Done")