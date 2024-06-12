#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn.functional as F
import torch.nn as nn
from skmultilearn.dataset import load_dataset
from torch.utils.data import TensorDataset, DataLoader
from skmultilearn.model_selection import IterativeStratification
from torch.utils.tensorboard import SummaryWriter
import pdb
import torch
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
import matplotlib.pyplot as plt
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.dataset import load_from_arff
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import itertools
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.preprocessing import MinMaxScaler
import math
import pdb
import warnings
warnings.filterwarnings("ignore")
import time
from skmultilearn.model_selection import IterativeStratification
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from skmultilearn.ensemble import RakelD
from skmultilearn.adapt import MLkNN
from skmultilearn.adapt import BRkNNbClassifier
from sklearn.neighbors import NearestNeighbors
from enum import Enum
from skmultilearn.dataset import load_dataset
from skmultilearn.dataset import load_from_arff
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB
from skmultilearn.ensemble import RakelO
from skmultilearn.adapt import MLTSVM
from skmultilearn.adapt import MLARAM
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from scipy import sparse
import random
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
from sklearn.preprocessing import MinMaxScaler
def ImR(X,y):
    Imr=[]
    for i in range(y.shape[1]):
        count0=0
        count1=0
        for j in range(y.shape[0]):
            if y[j,i]==1:
                count1+=1
            else:
                count0+=1
        if count1<=count0:
            Imr.append(count0/count1)
        else:
            Imr.append(count1/count0)
    return Imr
def Imbalance(X,y):
    countmatrix=[]
    for i in range(y.shape[1]):
        count0=0
        count1=0
        for j in range(y.shape[0]):
            if y[j,i]==1:
                count1+=1
            else:
                count0+=1
        countmatrix.append(count1)
    maxcount=max(countmatrix)
#     pdb.set_trace()
    ImbalanceRatioMatrix=[maxcount/i for i in countmatrix]
    MaxIR=max(ImbalanceRatioMatrix)
    MeanIR=sum(ImbalanceRatioMatrix)/len(ImbalanceRatioMatrix)
    return ImbalanceRatioMatrix,MeanIR,countmatrix
def Labeltype(X,y):
    ImbalanceRatioMatrix,MeanIR,_=Imbalance(X,y)
    DifferenceImbalanceRatioMatrix=[i-MeanIR for i in ImbalanceRatioMatrix]
    MinLabelIndex=[]
    MajLabelIndex=[]
    count=0
    for i in (DifferenceImbalanceRatioMatrix):
        if i>0:
            MinLabelIndex.append(count)
        else:
            MajLabelIndex.append(count)
        count+=1
    MinLabelName=[]
    MajLabelName=[]
    for i in MinLabelIndex:
        MinLabelName.append(label_names[i][0])
    for i in MajLabelIndex:
        MajLabelName.append(label_names[i][0])
    MinLabeldic=dict(zip(MinLabelIndex,MinLabelName))
    MajLabeldic=dict(zip(MajLabelIndex,MajLabelName))
    return MinLabeldic,MajLabeldic
def minority_instance(train_x, train_y,k):
    ImbalanceRatioMatrix,MeanIR,countmatrix=Imbalance(train_x,train_y)

    imbalance_ratios = ImR(train_x, train_y)
    high_imbalance_indices = sorted(range(len(imbalance_ratios)), key=lambda i: imbalance_ratios[i], reverse=True)[:k]

    MinLabeldic, MajLabeldic = Labeltype(train_x, train_y)
    MinLabelindex = list(MinLabeldic.keys())
#     intersection_indices = list(set(MinLabelindex) & set(high_imbalance_indices))

    rows_with_ones = np.any(train_y[:, MinLabelindex] == 1, axis=1)
    result_indices = np.where(rows_with_ones)[0]
    np.random.shuffle(result_indices)
    new_train_x = train_x[result_indices, :]
    new_train_y = train_y[result_indices, :]
    return new_train_x,new_train_y,result_indices
def set_topk_to_ones(y_pred, k):
    y_sigmoid = torch.sigmoid(y_pred)
    topk_values, topk_indices = torch.topk(y_sigmoid, k, dim=0) 
    zero_tensor = torch.zeros_like(y_sigmoid)
    zero_tensor.scatter_(0, topk_indices, 1)
    return zero_tensor.cpu().detach().numpy()
def set_bottomk_to_zeros(x, k):
    x_sigmoid = x
    bottomk_values, bottomk_indices = torch.topk(x_sigmoid, k, dim=0, largest=False) 
    x_transformed = x_sigmoid.clone()
    x_transformed.scatter_(0, bottomk_indices, 0)
    return x_transformed.cpu().detach().numpy()
def process_feature(c2ae_mod, x_syn, activation=None):
    feat = c2ae_mod.Fd_x(x_syn)
    if activation == 'sigmoid':
        return F.sigmoid(feat)
    elif activation == 'softmax':
        return F.softmax(feat)
    elif activation=='tanh':
        return F.Tanh(feat)
    else:
        return feat
def CardAndDens(X,y):
    cardmatrix=[]
    for i in range(X.shape[0]):
        count=0
        for j in range(y.shape[1]):
            if y[i,j]==1:
                count+=1
        cardmatrix.append(count)
    Card=sum(cardmatrix)/len(cardmatrix)
    Dens=Card/y.shape[1]
    return Card,Dens

