#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def generate_instance(dataname,device,epoch,train_x,train_y,k,p,activation,apply_process_feature=False,apply_process_label=False):
    new_x, new_y,result_indices=minority_instance(train_x, train_y,k)
    Card,Dens=CardAndDens(train_x, train_y)
    np.random.seed(21)
    n_neighbors = 5
    nbs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', algorithm='kd_tree').fit(train_x)
    distances, indices = nbs.kneighbors(train_x)
    num_samples=int(train_x.shape[0]*p)
    mod = AEMLO(Fx_dim, Fy_dim, Fd_yy, Fd_xx,beta=0.1, alpha=1, emb_lambda=0.1, latent_dim=latent_dim,device=device).to(device))
    state_dict = torch.load(f"/home/tt/vaesampling/label_assign_model/{dataname}/best_model.pt")
    mod.load_state_dict(state_dict)
    feat_list=[]
    label_list=[]
    with torch.no_grad():
        mod.eval()
        x_encode,_,_=mod(torch.tensor(train_x, device=device, dtype=torch.float))
        for idx in range(num_samples):
            if idx>=new_x.shape[0]:
                s_id=idx % new_x.shape[0] 
            else:
                s_id=idx
            seed_idx=result_indices[s_id]
            feat=process_feature(mod, x_encode[seed_idx], activation=activation).cpu().detach().numpy()
            y_pred=mod.Fd_y(x_encode[seed_idx])
            y_pred=torch.round(torch.sigmoid(y_pred)).cpu().detach().numpy()           
            feat_list.append(feat)
            label_list.append(y_pred)
        featarray= np.vstack(feat_list)
        labelarray=np.vstack(label_list)
        zero_rows_index = np.where(np.all(labelarray == 0, axis=1))[0]
        unique_rows = np.unique(labelarray, axis=0)
        columns_to_delete = np.where(np.all(labelarray == 0, axis=1))[0]
        featarray = np.delete(featarray, columns_to_delete, axis=0)
        labelarray = np.delete(labelarray, columns_to_delete, axis=0)
    train_x = np.concatenate((train_x,featarray), axis=0)
    train_y= np.concatenate((train_y, labelarray), axis=0)
    return train_x,train_y

