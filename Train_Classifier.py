#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def FeatureSelect(X,p):
    if p==1:
        new_x=X.toarray()/ X.toarray().sum(axis=0)
        return new_x,feature_names
    else:
        featurecount=int(X.shape[1]*p)
        Selectfeatureindex=[x[0] for x in (sorted(enumerate(X.sum(axis=0).tolist()[0]),key=lambda x: x[1],reverse=True))][:featurecount]
        Allfeatureindex=[i for i in range(X.shape[1])]
        featureindex=[i for i in Allfeatureindex if i not in Selectfeatureindex]
        new_x=np.delete(X.toarray(),featureindex,axis=1)
        new_featurename=[feature_names[i] for i in Selectfeatureindex] 
#         new_x = new_x / new_x.sum(axis=0)
        return new_x,new_featurename
def LabelSelect(y):
    b=[]
    new_labelname=[i for i in label_names]
    for i in range(y.shape[1]):
        if y[:,i].sum()<=30:
            b.append(i)
            new_labelname.remove(label_names[i])
    new_y=np.delete(y.toarray(),b,axis=1)
    return new_y,new_labelname
def macro_averaging_auc(Y, P, O):
    n = (Y.shape[0] + O.shape[0]) // 2
    l = (Y.shape[1] + O.shape[1]) // 2

    p = np.zeros(l)
    q = np.sum(Y, 0)

    zero_column_count = np.sum(q == 0)
#     print(f"all zero for label: {zero_column_count}")
    r, c = np.nonzero(Y)
    for i, j in zip(r, c):
        p[j] += np.sum((Y[ : , j] < 0.5) * (O[ : , j] <= O[i, j]))

    i = (q > 0) * (q < n)

    return np.sum(p[i] / (q[i] * (n - q[i]))) / l
def hamming_loss(Y, P, O):
    n = (Y.shape[0] + P.shape[0]) // 2
    l = (Y.shape[1] + P.shape[1]) // 2

    s1 = np.sum(Y, 1)
    s2 = np.sum(P, 1)
    ss = np.sum(Y * P, 1)

    return np.sum(s1 + s2 - 2 * ss) / (n * l)
def one_error(Y, P, O):
    n = (Y.shape[0] + O.shape[0]) // 2

    i = np.argmax(O, 1)

    return np.sum(1 - Y[range(n), i]) / n
def ranking_loss(Y, P, O):
    n = (Y.shape[0] + O.shape[0]) // 2
    l = (Y.shape[1] + O.shape[1]) // 2

    p = np.zeros(n)
    q = np.sum(Y, 1)

    r, c = np.nonzero(Y)
    for i, j in zip(r, c): 
        p[i] += np.sum((Y[i, : ] < 0.5) * (O[i, : ] >= O[i, j]))

    i = (q > 0) * (q < l)

    return np.sum(p[i] / (q[i] * (l - q[i]))) / n
# def macro_f1(y_true, y_prob):
#     th = []
#     for i in range(40, 61, 1):
#         # 将每个整数除以100来得到相应的小数，并添加到列表中
#         th.append(i / 100.0)
#     MacroF=[]
#     for i in th:
#         y_pred = (y_prob >= i).astype(int)
#         TP = np.sum((y_true == 1) & (y_pred == 1), axis=0)
#         FP = np.sum((y_true == 0) & (y_pred == 1), axis=0)
#         FN = np.sum((y_true == 1) & (y_pred == 0), axis=0)
#         precision = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
#         recall = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
#         f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
#         macro_f1 = np.nanmean(f1_scores)       
#         MacroF.append(macro_f1)

#     return max(MacroF)
# def micro_f1(y_true, y_prob):
#     th = []
#     for i in range(40, 71, 1):
#         # 将每个整数除以100来得到相应的小数，并添加到列表中
#         th.append(i / 100.0)
#     MicroF=[]
#     for i in th:
#         y_pred = (y_prob >= i).astype(int)
#         TP = np.sum((y_true == 1) & (y_pred == 1), axis=0)
#         FP = np.sum((y_true == 0) & (y_pred == 1), axis=0)
#         FN = np.sum((y_true == 1) & (y_pred == 0), axis=0)
#         precision = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
#         recall = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
#         f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
#         micro_f1 = np.sum(TP) / (np.sum(TP) + 0.5 * (np.sum(FP) + np.sum(FN)))
#         MicroF.append(micro_f1)

#     return max(MicroF)

def training(params,c_idx,X,y):
    Macro_F=[]
    Micro_F=[]
    Hamming_loss=[]
    Ranking_loss=[]
    One_error=[]
    Macro_AUC=[]
    k_fold = IterativeStratification(n_splits=5,order=1,random_state=42)
    for train,test in k_fold.split(X,y):
#     cross = model_selection.KFold(n_splits=5, shuffle=True)
#     for train, test in cross.split(X, y):
        fold=1
        if c_idx==1:
            classifier =BinaryRelevance(
                classifier = DecisionTreeClassifier(random_state=41),
                require_dense = [False, True]
            )
        elif c_idx==2:
            classifier =MLkNN(k=10)
        elif c_idx==3:
            classifier =ClassifierChain(
                classifier = DecisionTreeClassifier(random_state=20),
                require_dense = [False, True]
            ) 
        elif c_idx==4:
            classifier = RakelD(
                base_classifier=DecisionTreeClassifier(random_state=20),
                base_classifier_require_dense=[False, True],
                labelset_size=3
            )
        dataname=params["dataname"]+str(fold)       
        parameters = {
            "dataname": dataname,
            "feat_dim": X[train].shape[1],
            "num_labels": y[train].shape[1],
            "latent_dim": params["latent_dim"],
            "fx_h_dim": params["fx_h_dim"],
            "fe_h_dim": params["fe_h_dim"],
            "fd_h_dim": params["fd_h_dim"],
            "X_train": X[train],
            "y_train": y[train],
            "epoch": params["epoch"],
            "batch_size": params["batch_size"],
            "learningrate": params["learningrate"]
        }
        train(parameters)
        new_x, new_y = generate_instance(dataname,torch.device('cuda'),200,X[train],y[train],params["num_min_labels"],params["samplingrate"],params["activate"],apply_process_feature=False,apply_process_label=False)
        
        if c_idx==1:
            print(new_x.shape[0]-X[train].shape[0])
        classifier.fit(new_x,new_y)
        X2,y2=X[test],y[test]
        ypred = classifier.predict(X2)
        if scipy.sparse.issparse(ypred):
            ypred = ypred.toarray()
        yprob = classifier.predict_proba(X2)
        if scipy.sparse.issparse(yprob):
            yprob = yprob.toarray()
        Macro_F.append(metrics.f1_score(y2, ypred,average='macro'))
        Micro_F.append(metrics.f1_score(y2, ypred,average='micro'))
        Ranking_loss.append(ranking_loss(y2, ypred, yprob))                     
        Macro_AUC.append(macro_averaging_auc(y2, ypred, yprob))  
        Hamming_loss.append(metrics.hamming_loss(y2, ypred)) 
        One_error.append(one_error(y2, ypred, yprob))
        fold+=1
    means = np.array([
    np.mean(Macro_F),
    np.mean(Micro_F),
    np.mean(Macro_AUC),
    np.mean(Ranking_loss),
    np.mean(Hamming_loss),
    np.mean(One_error)
    ])
    stds = np.array([
        np.std(Macro_F),
        np.std(Micro_F),
        np.std(Macro_AUC),
        np.std(Ranking_loss),
        np.std(Hamming_loss),
        np.std(One_error)
    ])
    rounded_means = np.round(means, 4)
    rounded_stds = np.round(stds, 4)
    print(tuple(rounded_means) + tuple(rounded_stds)) 

