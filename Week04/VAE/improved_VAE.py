import torch.nn as nn
import torch
import torch.nn.functional as F
# VAE 정의
class my_VAE(nn.Module):
    def __init__(self):
        super(my_VAE, self).__init__()
        #TODO
        input_dim = 784
        hidden_dim = 512
        latent_dim = 50

        #encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        #decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.bn1(self.fc1(x)))
        h2 = F.relu(self.bn2(self.fc2(h1)))
        mu = self.fc_mu(h2)
        logvar = self.fc_logvar(h2)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.bn3(self.fc3(z)))
        h4 = F.relu(self.bn4(self.fc4(h3)))
        return torch.sigmoid(self.fc5(h4))  # Output between 0 and 1 for image generation

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))  # Flatten input
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# loss function 정의
def improved_loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD
