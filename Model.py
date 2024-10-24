#!/usr/bin/env python
# coding: utf-8

# In[ ]:
class AttentionLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(AttentionLayer, self).__init__()
        self.attention_weights = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        scores = torch.sigmoid(self.attention_weights(x))
        return x * scores

class Fx(torch.nn.Module):
    def __init__(self, in_dim, H, out_dim, dropout_rate=0.5):
        super(Fx, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, H)
        self.fc2 = torch.nn.Linear(H, H)
        self.fc3 = torch.nn.Linear(H, out_dim)
        self.fc_mu = torch.nn.Linear(H, out_dim)
        self.fc_logvar = torch.nn.Linear(H, out_dim)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        x = F.leaky_relu(self.fc3(x))            
        return x,mu,logvar
class Fy(torch.nn.Module):
    def __init__(self, in_dim, H, out_dim):
        super(Fy, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, H)
        self.attention1 = AttentionLayer(H, H)
        self.fc2 = torch.nn.Linear(H, out_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.attention1(x)
        x = self.fc2(x)
        return x
class Fd_y(torch.nn.Module):
    def __init__(self, in_dim, H, out_dim):
        super(Fd_y, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, H)
        self.fc2 = torch.nn.Linear(H, out_dim)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x
class Fd_x(torch.nn.Module):
    def __init__(self, in_dim, H, out_dim):
        super(Fd_x, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, H)
        self.fc2 = torch.nn.Linear(H, out_dim)
    def forward(self, x):       
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x   
class AEMLO(torch.nn.Module):
    def __init__(self, Fx, Fy, Fd_y, Fd_x, beta=1, alpha=.5, emb_lambda=.1, latent_dim=6,
                 device=None):
        super(AEMLO, self).__init__()
        self.Fx = Fx
        self.Fy = Fy
        self.Fd_y = Fd_y
        self.Fd_x = Fd_x
        self.alpha = alpha
        self.beta = beta
        self.emb_lambda = emb_lambda
        self.latent_I = torch.eye(latent_dim).to(device)
    def forward(self, x,y):
        if self.training:
            fe_x, _, _ = self.Fx(x)  # Unpack and only use the first output from Fx
            fe_y = self.Fy(y)
            fd_y = self.Fd_y(fe_y)
            fd_x = self.Fd_x(fe_x)  # Now fe_x is a single Tensor, as expected
            return fe_x, fe_y, fd_y,fd_x
        else:      
            return self.Fx(x),self.predict_y(x),self.predict_x(x)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def predict_y(self, x):
        return self.Fd_y(self.Fx(x))
    def predict_x(self, x):
        return self.Fd_x(self.Fx(x))
    def corr_loss(self, preds, y):
        ones = (y == 1)
        zeros = (y == 0)
        ix_matrix = ones[:, :, None] & zeros[:, None, :]
        diff_matrix = torch.exp(-(preds[:, :, None] - preds[:, None, :]))
        losses = torch.flatten(diff_matrix*ix_matrix, start_dim=1).sum(dim=1)
        losses /= (ones.sum(dim=1)*zeros.sum(dim=1) + 1e-4)
        losses[losses == float('Inf')] = 0
        losses[torch.isnan(losses)] = 0
        return losses.sum()
    def latent_loss(self, fe_x, fe_y):
        c1 = fe_x - fe_y
        c2 = fe_x.T@fe_x - self.latent_I
        c3 = fe_y.T@fe_y - self.latent_I
        latent_loss = torch.trace(
            c1@c1.T) + self.emb_lambda*torch.trace(c2@c2.T + c3@c3.T)
        return latent_loss 
    def mae_loss_weighted(self, fd_x, x, weight=1.0, sim_lambda=1):
        mae = torch.mean(torch.abs(fd_x - x))
        n = x.size(0)
        x_diff = x[:, None, :] - x[None, :, :]
        x_norm = torch.sum(x_diff ** 2, dim=-1)
        fd_x_diff = fd_x[:, None, :] - fd_x[None, :, :]
        fd_x_norm = torch.sum(fd_x_diff ** 2, dim=-1)
        similarity_loss = sim_lambda * torch.sum((x_norm - fd_x_norm) ** 2) / (n * (n - 1))
        # Combined loss
        total_loss = mae + similarity_loss
        return total_loss
    def losses(self, fe_x, fe_y, fd_y, fd_x, x, y):
        l_loss = self.latent_loss(fe_x, fe_y)
        c_loss = self.corr_loss(fd_y, y)
        mseloss = self.mae_loss_weighted(fd_x, x)
        return l_loss, c_loss ,mseloss   

