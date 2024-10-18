import torch.nn as nn
import torch
import torch.nn.functional as F
# VAE 정의
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        #TODO
        input_dim = 784
        hidden_dim = 400
        latent_dim = 20

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h3))  # Output between 0 and 1 for image generation

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))  # Flatten input
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
        

# loss function 정의
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD