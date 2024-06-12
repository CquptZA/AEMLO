#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def train(params):
    np.random.seed(42)
    dataname = params["dataname"]
    feat_dim = params["feat_dim"]
    num_labels = params["num_labels"]
    latent_dim = params["latent_dim"]
    fx_h_dim = params["fx_h_dim"]
    fe_h_dim = params["fe_h_dim"]
    fd_h_dim = params["fd_h_dim"]
    X = params["X_train"]
    y = params["y_train"]
    num_epochs = params["epoch"]
    batch_size = params["batch_size"]
    lr = params["learningrate"]
    train_dataset = TensorDataset(torch.tensor(X, device=device, dtype=torch.float),torch.tensor(y, device=device,dtype=torch.float))
    Fx_dim = Fx(feat_dim, fx_h_dim, latent_dim)
    Fy_dim = Fy(num_labels, fe_h_dim, latent_dim)
    Fd_yy = Fd_y(latent_dim, fd_h_dim, num_labels)
    Fd_xx= Fd_x(latent_dim, fx_h_dim, feat_dim)
    net = AEMLO(Fx_dim, Fy_dim, Fd_yy, Fd_xx,beta=0.1, alpha=1, emb_lambda=0.1, latent_dim=latent_dim,device=device).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    writer = SummaryWriter(comment=f'{dataname}-c2ae')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    prev_loss = float('inf')  
    no_improvement_count = 0
    print_interval=10
    min_loss = float('inf') 
    for epoch in range(num_epochs+1): 
        net.train()
        loss_tracker = 0.0
        latent_loss_tracker = 0.0
        cor_loss_tracker = 0.0
        mse_loss_tracker = 0.0
        for x, y in train_dataloader:
            optimizer.zero_grad()  
            fe_x, fe_y, fd_y,fd_x = net(x,y)         
            l_loss, c_loss,mse_loss= net.losses(fe_x, fe_y, fd_y,fd_x,x,y)
            l_loss /= x.shape[0]
            c_loss /= x.shape[0]
            mse_loss /= x.shape[0] 
            loss = l_loss + mse_loss+c_loss
            loss.backward()
            optimizer.step()
            loss_tracker+=loss.item()
            latent_loss_tracker+=l_loss.item()
            cor_loss_tracker+=c_loss.item()
            mse_loss_tracker+=mse_loss.item()
        if loss_tracker < min_loss:
            min_loss = loss_tracker 
            best_epoch = epoch
            # Save model with the lowest loss
#             save_dir = f"/home/tt/vaesampling/label_assign_model/{dataname}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, "best_model.pt")
            torch.save(net.state_dict(), save_path)

