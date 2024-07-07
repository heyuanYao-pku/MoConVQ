from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
import math
from MoConVQCore.Utils.pytorch_utils import *
# from MoConVQCore.Utils.gaussian_diffusion import *
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].view(1, x.size(1), -1) # batch first
        return self.dropout(x)


class CausalTransformer(nn.Module):
    def __init__(self, **kargs) -> None:
        super().__init__()
        
        self.input_dim = kargs['input_dim']
        layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=8, batch_first=True, dropout=0.1)
        self.seq_length = kargs['seq_length']
        self.model = nn.TransformerEncoder(layer, num_layers=4)
        self.obs2latent = nn.Linear(kargs['obs_dim'], kargs['input_dim'])
        self.positive_encoding = PositionalEncoding(self.input_dim, dropout=0, max_len=self.seq_length * 2 + 1)
        
        self.clip2latent = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512))
        
        self.project = nn.Linear(512, kargs['input_dim'])
        
        # self.mask = self.build_mask(2*self.seq_length+1).to(ptu.device)
        self.mask = self.build_mask_one_obs(2*self.seq_length+1).to(ptu.device)
        
    def build_mask(self, seq_length):
        mask = torch.ones(seq_length, seq_length).triu(diagonal=1).bool()
        return mask
    
    def build_mask_one_obs(self, seq_length):
        mask = torch.ones(seq_length, seq_length).triu(diagonal=1).bool()
        for i in range(2, seq_length, 2):
            for j in range(2, i - 1, 2):
                mask[i, j] = True
        return mask
                
    def drop_out_mask(self):
        
        # random_mask with 10% probability
        rand_mask = ( torch.randint_like(self.int_mask, 10) < 2 ).tril(-2)
        return self.mask | rand_mask
    
    
    def build_padding_mask(self, batch, seq_length, cur_frame):
        mask = torch.zeros(batch, seq_length*2+1).bool()
        mask[:, cur_frame*2+3:] = True
        return mask
    
    def forward(self, obs, latent, time_embedding, clip_feature, cur_frame = None):
        '''
        obs: [N, L, D]
        latent: [N, L, D]
        '''
        assert len(obs.shape) == 3, "obs should be [N, L, D]"
        assert len(latent.shape) == 3, "latent should be [N, L, D]"
        assert len(time_embedding.shape) == 3, "time_embedding should be [N, 1, D]"
        
        if clip_feature is not None:
            clip_feature = clip_feature.view(time_embedding.shape)
            
            norm = torch.norm(clip_feature, dim=-1, keepdim=True)
            norm[norm==0] = 1
            clip_feature/=norm
            
            time_embedding = time_embedding + self.clip2latent(clip_feature)
        
            
        
        time_embedding = self.project(time_embedding)
        
        
        obs = self.obs2latent(obs)
        input = torch.cat([obs, latent], dim=-1).view(-1, self.seq_length * 2, self.input_dim)
        input = torch.cat([time_embedding, input], dim=1)
        
        input = self.positive_encoding(input)

        input = input.contiguous()
        output = self.model(input, 
                            self.mask,
                            self.build_padding_mask(input.shape[0], self.seq_length, cur_frame).to(input.device) if cur_frame is not None else None
                            )
        output = output[:, 1:, :]
        output = output.view(-1, self.seq_length, 2 * self.input_dim)
        output_z = output[:, :, self.input_dim:].contiguous()
        return output_z

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        max_len = 1500
        
        pe = torch.zeros(max_len, latent_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, latent_dim, 2).float() * (-np.log(10000.0) / latent_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        self.latent_dim = latent_dim

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
    def forward(self, timesteps):
        return self.time_embed(self.pe[timesteps])

class CausalEncoder(nn.Module):
    def __init__(self, **kargs) -> None:
        super().__init__()
        self.diffusion_step = kargs.get('diffusion_step', 1000)
        self.ddim_skip = kargs.get('ddim_skip', 20)
        self.time_embedder = TimestepEmbedder(512)
        self.diffusion_helper = DiffusionHelper(self.diffusion_step)
        self.model = CausalTransformer(**kargs)
    
    def generate_random_step(self, batch_size):
        return torch.randint(0, self.diffusion_step, (batch_size,1), device=ptu.device, dtype = torch.long)
    
    def generate_deterministic_step(self, batch_size, step_int):
        return torch.ones( (batch_size,1), device=ptu.device, dtype = torch.long) * step_int
    
    def get_time_embedding(self, steps):
        time_embedding = self.time_embedder(steps)
        return time_embedding
        
    def _predict_x0(self, obs, latent, steps, clip_feature = None):
        # assert False, "Deprecated"
        time_embedding = self.get_time_embedding(steps)
        return self.model(obs, latent, time_embedding, clip_feature)
    
    def _predict_x0_guidance(self, obs, latent, steps, clip_feature, s = 2.5):
        # assert False, "Deprecated"
        time_embedding = self.get_time_embedding(steps)
        conditioned = self.model(obs, latent, time_embedding, clip_feature)
        
        unconditioned = self.model(obs, latent, time_embedding, None)
        return unconditioned + s*(conditioned - unconditioned)
    
    def ddim_step(self, obs, latent, steps, dt, mu, clip_feature, s = 2.5):
        if clip_feature is None:
            x0 = self._predict_x0(obs, latent, steps)
            return self.diffusion_helper.ddim_step(latent, x0, steps, dt, mu)
        else:
            return self.ddim_step_conditioned(obs, latent, steps, dt, mu, clip_feature, s)
    
    def ddim_step_conditioned(self, obs, latent, steps, dt, mu, clip_feature, s):
        x01 = self._predict_x0(obs, latent, steps, clip_feature)
        x02 = self._predict_x0(obs, latent, steps, None)
        return self.diffusion_helper.ddim_step_conditioned(latent, x01, x02, steps, dt, mu, s)
    
    def pred_x0(self, condition, seq_x, noised_x, steps, clip_feature=None):
        cur_length = condition.shape[1]
        pad_length = self.model.seq_length - cur_length
        condition = F.pad(condition, (0, 0, 0, pad_length), value=0)
        x = torch.cat([seq_x, noised_x], dim=1) if seq_x is not None else noised_x
        x = F.pad(x, (0, 0, 0, pad_length - 1), value=0)
        return self._predict_x0(condition, x, steps, clip_feature)
    
    def add_noise(self, x, noise_level, mus):
        return self.diffusion_helper.add_noise(x, noise_level, mus)
    
    def denoise_to_x0(self, condition, seq_x, noised_x, dt, mu, clip_feature=None):
        cur_length = condition.shape[1]
        pad_length = self.model.seq_length - cur_length
        condition = F.pad(condition, (0, 0, 0, pad_length), value=0)
        x = torch.cat([seq_x, noised_x], dim=1) if seq_x is not None else noised_x
        x = F.pad(x, (0, 0, 0, pad_length), value=0)
        mu = F.pad(mu, (0,0,0, pad_length), value=0 )
        # print(mu.shape)
        for i in range(self.diffusion_step -1, -1, -dt):
            if seq_x is not None:
                noised_seq = self.diffusion_helper.add_noise(seq_x, self.generate_deterministic_step(x.shape[0], i), mu[:,:cur_length-1])
                x[:, :cur_length-1] = noised_seq
            latent_new = self.ddim_step(condition, x, self.generate_deterministic_step(x.shape[0], i), dt, mu, clip_feature)
            noised_x = latent_new[:, cur_length-1:cur_length, :]
            x = torch.cat([seq_x, noised_x], dim=1) if seq_x is not None else noised_x
            x = F.pad(x, (0, 0, 0, pad_length), value=0)
        return x
    
    
    
    
        
        
    
    