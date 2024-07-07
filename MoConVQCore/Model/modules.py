from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
import math
from MoConVQCore.Utils.pytorch_utils import *


class ConditionalEncoder(nn.Module):
    def __init__(self, input_size, condition_size, output_size, var, **params) -> None:
        super().__init__()
        self.activation = str_to_activation[params['activation']]
        self.hidden_size = params['hidden_layer_size']
        self.fc_layers = []
        self.fc_layers.append(nn.Linear(input_size + condition_size, self.hidden_size))
        for _ in range(params['hidden_layer_num']):
            self.fc_layers.append(nn.Linear(input_size + self.hidden_size, self.hidden_size))    
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.mu = nn.Linear(input_size + self.hidden_size, output_size)
        self.var = var
        if self.var is not None:
            self.log_var = math.log(var)*2
            
    def encode(self, x, c):
        res = c
        for layer in self.fc_layers:
            if res is not None:
                res = layer(torch.cat([x,res], dim = -1))
            else:
                res = layer(x)
            res = self.activation(res)
            
        latent = torch.cat([x,res], dim = -1)
        mu = self.mu(latent)
        if self.var is not None:
            logvar = torch.ones_like(mu)*self.log_var
        else:
            logvar = self.logvar(latent)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        exp = torch.randn_like(std)
        return mu + exp * std
    
    def forward(self, x, c):
        mu, logvar = self.encode(x,c)
        z = self.reparameterize(mu, logvar)
        return z, mu
    
class GatingMixedDecoder(nn.Module):
    # improved by Zhenhua Song
    def __init__(
            self,
            latent_size: int,
            condition_size: int,
            output_size: int,
            actor_hidden_layer_size: int,
            actor_num_experts: int,
            actor_hidden_layer_num: int,
            actor_gate_hidden_layer_size: int,
            init_mode="origin"  # 
    ):
        super().__init__()
        input_size = latent_size + condition_size
        hidden_size = actor_hidden_layer_size
        inter_size = latent_size + hidden_size
        num_experts = actor_num_experts
        num_layer = actor_hidden_layer_num

        # put in list then initialize and register
        for i in range(num_layer + 1):
            wdim1 = inter_size if i != 0 else input_size
            wdim2 = hidden_size if i != num_layer else output_size
            weight = nn.Parameter(torch.empty(num_experts, wdim1, wdim2))
            bias = nn.Parameter(torch.empty(num_experts, wdim2))
            if init_mode == "origin":  # initialize by Yuanshen's code
                stdv = 1. / math.sqrt(weight.size(1))
                weight.data.uniform_(-stdv, stdv)
                bias.data.uniform_(-stdv, stdv)
            elif init_mode == "soccer":  # initialize following Zhaoming Xie Soccer SIGGRAPH 2022
                nn.init.orthogonal_(weight, 0.1)
                nn.init.orthogonal_(bias.view(1, -1), 0.1)
            elif init_mode == "linear":  # initialize by default manner in PyTorch. Last linear not work..
                nn.init.kaiming_uniform_(weight, a=math.sqrt(5), )
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(bias, -bound, bound)
            elif init_mode == "mann":  # initialize by MANN in SIGGRAPH 2018, kaiming uniform.
                alpha = math.sqrt(6.0 / (wdim1 * wdim2))
                weight.data.uniform_(-alpha, alpha)
                alpha = math.sqrt(6.0 / wdim2)
                bias.data.uniform_(-alpha, alpha)
            elif init_mode == "kaiming_normal":
                weight.data.normal_(0.0, math.sqrt(2.0 / (wdim1 * wdim2)))
                bias.data.normal_(0.0, math.sqrt(2.0 / wdim2))
            else:
                raise NotImplementedError("Here we should test different initialize mode.")

            self.register_parameter(f"w{i}", weight)
            self.register_parameter(f"b{i}", bias)

        # add layer norm
        for i in range(num_layer + 1):
            self.add_module(f"ln{i}", nn.LayerNorm(inter_size if i != 0 else input_size))
            
        
        gate_hsize = actor_gate_hidden_layer_size
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            nn.LeakyReLU(inplace=False),
            nn.Linear(gate_hsize, gate_hsize),
            nn.LeakyReLU(inplace=False),
            nn.Linear(gate_hsize, num_experts)
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        coefficients = F.softmax(self.gate(torch.cat((z, c), dim=-1)), dim=-1)  # (batch_size, num_experts)
        # layer 0
        input_x = torch.cat((z, c), dim=-1)  # (batch_size, hid)
        # input_x = F.layer_norm(input_x, input_x.shape[1:])
        input_x = self.ln0(input_x)
        mixed_bias: torch.Tensor = torch.einsum('be,ek->bk', coefficients, self.b0)  # (batch_size, 512), contract
        mixed_input: torch.Tensor = torch.einsum('be,bj,ejk->bk', coefficients, input_x, self.w0)
        layer_out: torch.Tensor = F.leaky_relu(mixed_input + mixed_bias, inplace=False)

        # layer 1
        input_x = torch.cat((z, layer_out), dim=-1)  # (batch_size, hid)
        # input_x = F.layer_norm(input_x, input_x.shape[1:])
        input_x = self.ln1(input_x)
        mixed_bias: torch.Tensor = torch.einsum('be,ek->bk', coefficients, self.b1)  # (batch_size, 512), contract
        mixed_input: torch.Tensor = torch.einsum('be,bj,ejk->bk', coefficients, input_x, self.w1)
        layer_out: torch.Tensor = F.leaky_relu(layer_out + mixed_input + mixed_bias, inplace=False)

        # layer 2
        input_x = torch.cat((z, layer_out), dim=-1)  # (batch_size, hid)
        # input_x = F.layer_norm(input_x, input_x.shape[1:])
        input_x = self.ln2(input_x)
        mixed_bias: torch.Tensor = torch.einsum('be,ek->bk', coefficients, self.b2)  # (batch_size, 512), contract
        mixed_input: torch.Tensor = torch.einsum('be,bj,ejk->bk', coefficients, input_x, self.w2)
        layer_out: torch.Tensor = F.leaky_relu(layer_out + mixed_input + mixed_bias, inplace=False)

        # layer 3
        input_x = torch.cat((z, layer_out), dim=-1)  # (batch_size, hid)
        # input_x = F.layer_norm(input_x, input_x.shape[1:])
        input_x = self.ln3(input_x)
        mixed_bias: torch.Tensor = torch.einsum('be,ek->bk', coefficients, self.b3)  # (batch_size, 512), contract
        mixed_input: torch.Tensor = torch.einsum('be,bj,ejk->bk', coefficients, input_x, self.w3)
        layer_out: torch.Tensor = F.leaky_relu(layer_out + mixed_input + mixed_bias, inplace=False)
        
        # layer 4
        input_x = torch.cat((z, layer_out), dim=-1)  # (batch_size, hid)
        # input_x = F.layer_norm(input_x, input_x.shape[1:])
        input_x = self.ln4(input_x)
        mixed_bias: torch.Tensor = torch.einsum('be,ek->bk', coefficients, self.b4)  # (batch_size, 512), contract
        mixed_input: torch.Tensor = torch.einsum('be,bj,ejk->bk', coefficients, input_x, self.w4)
        layer_out: torch.Tensor = mixed_input + mixed_bias

        return layer_out

class ConditionalEncoder(nn.Module):
    def __init__(self, input_size, condition_size, output_size, var, **params) -> None:
        super().__init__()
        self.activation = str_to_activation[params['activation']]
        self.hidden_size = params['hidden_layer_size']
        self.fc_layers = []
        self.fc_layers.append(nn.Linear(input_size + condition_size, self.hidden_size))
        for _ in range(params['hidden_layer_num']):
            self.fc_layers.append(nn.Linear(input_size + self.hidden_size, self.hidden_size))    
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.mu = nn.Linear(input_size + self.hidden_size, output_size)
        self.var = var
        if self.var is not None:
            self.log_var = math.log(var)*2
            
    def encode(self, x, c):
        res = c
        for layer in self.fc_layers:
            if res is not None:
                res = layer(torch.cat([x,res], dim = -1))
            else:
                res = layer(x)
            res = self.activation(res)
            
        latent = torch.cat([x,res], dim = -1)
        mu = self.mu(latent)
        if self.var is not None:
            logvar = torch.ones_like(mu)*self.log_var
        else:
            logvar = self.logvar(latent)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        exp = torch.randn_like(std)
        return mu + exp * std
    
    def forward(self, x, c):
        mu, logvar = self.encode(x,c)
        z = self.reparameterize(mu, logvar)
        return z, mu


