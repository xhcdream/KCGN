# KCGN
AAAI-2021
《Knowledge-aware Coupled Graph Neural Network for Social Recommendation》


## Environments

- python 3.8
- pytorch-1.6
- DGL 0.5.3 (https://github.com/dmlc/dgl)

## Example to run the codes		

train model:

```
Dataset: Yelp, Result: test HR = 0.8026, test NDCG = 0.5326
python main.py --dataset Yelp --reg 0.05 --lr 0.01 --batch 2048 --hide_dim 64 --layer [64] --slope 0.4 --time 1 --time_step 360 --dgi 1 --lam [0.1,0.001] --clear 1 --fuse mean

Dataset: Epinions, Result: test HR = 0.7468, test NDCG = 0.5160
python main.py --dataset Epinions --lr 0.005 --reg 0.01 --batch 8192 --hide_dim 32 --layer [32] --slope 0.4 --time 1 --time_step 360 --dgi 1 --lam [0,0.8] --clear 1 --fuse mean

```


