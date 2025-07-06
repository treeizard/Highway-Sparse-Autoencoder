import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, 
                 input_dim, hidden_dim=2304, 
                 sparsity_lambda=5e-3, target_sparsity = 0.05):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        self.sparsity_lambda = sparsity_lambda
        self.target_sparsity = target_sparsity

    def forward(self, x):
        z = self.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

    def loss(self, x, x_hat, z):
        recon_loss = nn.functional.mse_loss(x_hat, x)
        sparsity_loss = self.sparsity_lambda * torch.mean(torch.abs(z))
        return recon_loss, sparsity_loss
    
    def kl_divergence_sparsity_loss(self, x, x_hat, z):
        sparsity_weight = self.sparsity_lambda
        recon_loss = torch.nn.functional.mse_loss(x_hat, x)
        target_sparsity = self.target_sparsity 

        # Compute average activation per unit across batch
        rho_hat = torch.mean(z, dim=0).clamp(min=1e-8, max=1.0 - 1e-8)
        rho = target_sparsity

        # KL divergence sparsity loss
        kl = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        sparsity_loss = torch.sum(kl)

        return recon_loss, sparsity_weight * sparsity_loss
