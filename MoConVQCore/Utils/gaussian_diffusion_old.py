import torch
import math
import MoConVQCore.Utils.pytorch_utils as ptu

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DiffusionHelper():
    def __init__(self, timesteps, **args):
        self.timesteps = timesteps
        self.betas = cosine_beta_schedule(self.timesteps).type(torch.float32).to(ptu.device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim = 0).to(ptu.device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).type(torch.float32).to(ptu.device)
        self.one_minus_sqrt_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod).type(torch.float32).to(ptu.device)
        self.one_minus_alphas_cumprod_inv = (1.0/(1 - self.alphas_cumprod)).type(torch.float32).to(ptu.device)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1).to(ptu.device), self.alphas_cumprod[:-1]])
        # self.k = self.alphas_cumprod.sqrt() * (1-self.alphas_cumprod.sqrt())
        self.k = torch.zeros_like(self.alphas_cumprod)
        # self.k = torch.exp(self.alphas_cumprod*20 - 20)
        # self.visualize_k()
        
    def visualize_k(self):
        import matplotlib.pyplot as plt
        import numpy as np
        x = np.arange(0, self.timesteps)
        y = ptu.to_numpy(self.alphas)
        z = ptu.to_numpy(self.alphas_cumprod)
        plt.plot(x, y, label = "k")
        plt.plot(x, z, label = "alphas_cumprod")
        plt.legend()
        plt.savefig("k.png")
        # exit()
    
    def add_noise(self, x, t, mu, using_zero = False):
        
        x = x 
        noise =  torch.randn_like(x)
        y = self.sqrt_alphas_cumprod[t].view(-1,1,1) * x + self.one_minus_sqrt_alphas_cumprod[t].view(-1,1,1) * (noise) + self.k[t].view(-1,1,1)*mu
        return y 
    
    def ddim_step(self, x, x_0, t, dt, mu):
        eta = 0
        x = x 
        x_0 = x_0 
        prev = (t - dt).clamp(min = 0)
        sigma = eta * torch.sqrt(
                (1 - self.alphas_cumprod[prev]) / (1 - self.alphas_cumprod[t]) * (1 - self.alphas_cumprod[t] / self.alphas_cumprod[prev])).view(-1,1,1)
        pred_noise = (x - self.k[t].view(-1,1,1)*mu - self.alphas_cumprod[t].view(-1,1,1).sqrt() * x_0)/ ((1-self.alphas_cumprod[t]).sqrt().view(-1,1,1))
        pred_dir_xt = (1-self.alphas_cumprod[prev].view(-1,1,1) - sigma**2).sqrt() * pred_noise
        x_prev = self.alphas_cumprod[prev].view(-1,1,1).sqrt() * x_0 + pred_dir_xt + sigma * torch.randn_like(x) + self.k[prev].view(-1,1,1)*mu
        return x_prev
    
    def ddim_step_conditioned(self, x, x_01, x_02, t, dt, mu, s= 2.5):
        prev = (t - dt).clamp(min = 0)
        sigma = 0
        
        pred_noise1 = (x - self.k[t].view(-1,1,1)*mu - self.alphas_cumprod[t].view(-1,1,1).sqrt() * x_01)
        pred_noise2 = (x - self.k[t].view(-1,1,1)*mu - self.alphas_cumprod[t].view(-1,1,1).sqrt() * x_02)
        
        scaled_pred_noise = pred_noise2 + s * (pred_noise1 - pred_noise2)
        tmp_x = x - self.k[t].view(-1,1,1)*mu
        # x_prev = (1/self.alphas[t].view(-1,1,1).sqrt()) * (tmp_x - ((1-self.alphas[t])/(1-self.alphas_cumprod[t]).sqrt()).view(-1,1,1) * pred_noise ) + self.k[prev].view(-1,1,1)*mu
        x_0 = (1/self.alphas_cumprod[t].view(-1,1,1).sqrt()) * (tmp_x - scaled_pred_noise)
        pred_dir_xt = ( (1-self.alphas_cumprod[prev].view(-1,1,1) - sigma**2)/(1-self.alphas_cumprod[t]).view(-1,1,1) ).sqrt() * scaled_pred_noise
        x_prev = self.alphas_cumprod[prev].view(-1,1,1).sqrt() * x_0 + pred_dir_xt + sigma * torch.randn_like(x) + self.k[prev].view(-1,1,1)*mu
        return x_prev