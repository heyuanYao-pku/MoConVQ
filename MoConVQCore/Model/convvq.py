from torch.autograd import Function
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from mpi4py import MPI
from sklearn.cluster import KMeans
mpi_comm = MPI.COMM_WORLD
mpi_world_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

def find_max_min(inputs, codebook, top_num):
    with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            minn, _ = torch.min(distances, dim=1)
            _, top_index = torch.topk(minn, top_num, largest=True)

    return inputs_flatten[top_index]

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply

from sklearn.cluster import KMeans

class VQEmbeddingMovingAverage(nn.Module):
    def __init__(self, K, D, training, decay=0.99):
        super().__init__()
        embedding = torch.zeros(K, D)
        embedding.uniform_(-1./K, 1./K)
        self.decay = decay

        # self.embedding = nn.Parameter(embedding, requires_grad=False)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.ones(K))
        self.register_buffer("ema_w", self.embedding.clone())
        self.register_buffer("initialized", torch.zeros(1))
        
        self.is_training = training
        if self.is_training:
            assert mpi_rank == 0, "Only rank 0 can train VQ embedding"
            print("VQ embedding is trainable")
            
    def forward(self, z_e_x):
        z_e_x_ = z_e_x.contiguous()
        latents = vq(z_e_x_, self.embedding)
        return latents

    def visualize_usage(self):
        import matplotlib.pyplot as plt
        # plt.hist(self.ema_count.cpu().numpy(), bins=100)
        x_axis = np.arange(self.ema_count.shape[0])
        plt.bar(x_axis, self.ema_count.cpu().numpy())

        plt.savefig("vq_usage.png")
        plt.close()
    
    def visualiaze_codebook(self):
        import matplotlib.pyplot as plt
        data = self.embedding.cpu().numpy()
        from sklearn.manifold import TSNE
        data = TSNE(n_components=2).fit_transform(data)
        print(data.shape)
        colors = self.ema_count.cpu().numpy()/self.ema_count.sum().item()
        
        plt.scatter(data[:, 0], data[:, 1], c=colors, cmap='YlOrRd', s=5)
        plt.savefig("vq_codebook.png")
        plt.close()
        exit()
    
    def visualiaze_codebook_and_data(self, data2):
        import matplotlib.pyplot as plt
        data = self.embedding.cpu().detach().numpy()
        data2 = data2.view(-1, data.shape[-1])
        data_2 = data2.detach().cpu().numpy()
        total_data = np.concatenate([data, data_2], axis=0)
        from sklearn.manifold import TSNE
        data_ = TSNE(n_components=2).fit_transform(total_data)
        
        num = data.shape[0]
        plt.scatter(data_[num:, 0], data_[num:, 1], c='b', s=5)
        plt.scatter(data_[:num, 0], data_[:num, 1], c='r', s=5)
        
        plt.savefig("vq_codebook_embedding.png")
        plt.close()
        # exit()
    
    def entropy(self):
        prob = self.ema_count / self.ema_count.sum()
        entropy = -torch.sum(prob * torch.log(prob+1e-8))
        return entropy
    
    def straight_through(self, z_e_x):
        
        K, D = self.embedding.size()

        if not self.initialized and self.is_training:
            print('initializing codebook')
            kmeans = KMeans(self.embedding.shape[0])
            flatten_data = z_e_x.view(-1, D)
            data = kmeans.fit(flatten_data.detach().cpu().numpy()).cluster_centers_
            data = torch.from_numpy(data).to(flatten_data.device)
            self.embedding[:] = data.detach()
           
        # self.visualize_usage()
        # self.visualiaze_codebook()

        z_e_x_ = z_e_x.contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding)
        
        if not self.initialized and self.is_training:
            encodings = F.one_hot(indices, K).float()
            self.ema_count = torch.sum(encodings, dim=0)
            self.initialized[:] = 1
        # print(indices)
        
        z_q_x = z_q_x_.contiguous()


        if self.is_training:
            encodings = F.one_hot(indices, K).float()
            # print(encodings.shape, indices.shape, z_e_x_.shape)
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
            # self.visualize_usage()
            # print(self.ema_count.min(), self.ema_count.max(), self.ema_count.mean())
            dw = encodings.transpose(1, 0)@z_e_x_.reshape([-1, D])
            
            update_decay = 0.99
            self.ema_w = update_decay * self.ema_w + (1 - update_decay) * dw

            self.embedding = self.ema_w / (self.ema_count.unsqueeze(-1))

            threshold = 0.5
            unused_mask = self.ema_count < threshold
            
            if any(unused_mask):
                print("unused mask", torch.where(unused_mask))
                # kmeans = KMeans(self.embedding.shape[0])
                
                flatten_data = z_e_x.view(-1, D)
                num_unused = unused_mask.sum()
                index = torch.randint(0, flatten_data.shape[0], [num_unused])
                latent = flatten_data[index]
                # latent = find_max_min(flatten_data, self.embedding, num_unused)
                # latent = torch.permute(flatten_data,)
                self.embedding[unused_mask] = latent + torch.randn_like(latent) * 0.01
                self.ema_count[unused_mask] = 1
                
            self.embedding = self.embedding.detach()
            self.ema_w = self.ema_w.detach()
            self.ema_count = self.ema_count.detach()

        z_q_x_bar_flatten = torch.index_select(self.embedding, dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.contiguous()

        # 用z_q_x去predict和计算reconstruct loss, 用z_q_x_bar去计算commitment loss
        return z_q_x, z_q_x_bar, indices

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
        # print(x.shape)
        x = x + self.pe[:x.size(1), :].view(1, x.size(1), -1) # batch first
        return self.dropout(x)
    
    def get_encoding(self, batch, time):
        return self.pe[time,:].view(1,-1).tile(batch,1)

class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.Lin1 = nn.Linear(d_model, d_model)
    def forward(self, x):
        return self.Lin1(x)+x

class ResConv1DBolck(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(input_size, output_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(output_size, output_size, 1, 1, 0),
            )
    def forward(self, x):
        return self.model(x)+x

class ResNet1D(nn.Module):
    def __init__(self, input_size, output_size, num_layers=3):
        super().__init__()
        self.model = nn.Sequential(
            *[ResConv1DBolck(input_size, output_size) for _ in range(num_layers)]
            )
    def forward(self, x):
        return self.model(x)

class ConvEncoder(nn.Module):
    def __init__(self, input_size, feature_size, output_size):
        super().__init__()
        blocks = []
        for i in range(2):
            block = nn.Sequential(
                nn.Conv1d(input_size if i==0 else feature_size, feature_size, 4, padding=1, stride=2),
                ResNet1D(feature_size, feature_size),
            )
            blocks.append(block)
        
        blocks.append(nn.Conv1d(feature_size, output_size, 3, padding=1))
        self.model = nn.Sequential(*blocks)
        
    def encode_tranpose(self, x):
        x = x.transpose(1, 2) # (batch, input_size, seq_length)
        res = self(x)
        return res.transpose(1, 2) # (batch, seq_length, output_size)
    
    def encode_not_tranpose(self, x):
        x = x.transpose(1, 2) # (batch, input_size, seq_length)
        res = self(x)
        return res # (batch, output_size, seq_length)
    
    def forward(self, x):
        
        return self.model(x)        

class ConvDecoder(nn.Module):
    def __init__(self, input_size, feature_size, dynamic_size, kinematic_size ):
        super().__init__()
        blocks = []
        blocks.append(nn.Conv1d(input_size, feature_size, 3, padding=1))
        for i in range(2):
            block = nn.Sequential(
                ResNet1D(feature_size, feature_size),
                nn.ConvTranspose1d(feature_size, feature_size, 4, padding=1, stride=2),
            )
            blocks.append(block)
        
        blocks.append(nn.ConvTranspose1d(feature_size, feature_size, 3, padding=1, stride = 1))
        self.model = nn.Sequential(*blocks)
        self.kinematic_head = nn.Sequential(
            nn.Conv1d(feature_size, feature_size, 3, padding=1),
            ResNet1D(feature_size, feature_size),
            nn.Conv1d(feature_size, kinematic_size, 3, padding=1),
        )
        self.dynamic_head = nn.Sequential(
            nn.Conv1d(feature_size, feature_size, 3, padding=1),
            ResNet1D(feature_size, feature_size),
            nn.Conv1d(feature_size, dynamic_size, 3, padding=1),
        )
    
    def decode_not_tranpose(self, x):
        x = x.transpose(1, 2) 
        res = self.model(x)
        return res
    
    def decode_dynamic(self, x):
        res = self.decode_not_tranpose(x)
        res = self.dynamic_head(res)
        return res.transpose(1, 2)
    def decode_kinematic(self, x):
        res = self.decode_not_tranpose(x)
        res = self.kinematic_head(res)
        return res.transpose(1, 2)
    def decode(self, x):
        res = self.decode_not_tranpose(x)
        res1 = self.kinematic_head(res)
        res2 = self.dynamic_head(res)
        return res2.transpose(1, 2), res1.transpose(1, 2)
    

 
import einops
class VQSeqEncoder(nn.Module):
    def __init__(self, obs_size, feature_size, seq_length, training, **kargs):
        super().__init__()
        
        self.int_num = kargs['int_num']
        self.encoder = ConvEncoder(obs_size, 512, feature_size*self.int_num)
        self.bottle_neck = VQEmbeddingMovingAverage(512, feature_size, training)
        self.decoder = ConvDecoder(feature_size*self.int_num, 512, feature_size, obs_size)
        
    def build_mask(self, seq_length):
        return torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    
    def quant(self, latent_seq):
        
        latent_seq = latent_seq.view(latent_seq.shape[0], -1, latent_seq.shape[-1]//self.int_num)
        latent_vq, latent_vq_bar, indices = self.bottle_neck.straight_through(latent_seq)
        commit_loss = (torch.norm(latent_seq - latent_vq_bar.detach() ) **2) * 0.05 / np.prod(latent_seq.shape[:-1])
        latent_vq = latent_vq.view(latent_seq.shape[0], -1, latent_seq.shape[-1]*self.int_num)
        return latent_vq, commit_loss
    
    def forward(self, begin_obs, target):
        
        mu_seq = self.encoder.encode_tranpose(target)
        # assert mu_seq.shape[1:] == ( 6, 768), mu_seq.shape
        latent_seq = mu_seq.contiguous()
        latent_vq, commit_loss = self.quant(latent_seq)
        
        latent_dynamic, kinematic = self.decoder.decode(latent_vq)
        # assert kinematic.shape[1:] == (24, 323), kinematic.shape
        # latent_vq = latent_vq + torch.randn_like(latent_vq) * 0.3
        return latent_dynamic, {
            'commit_loss': commit_loss,
            'mu_seq': mu_seq,
            'latent_vq': latent_vq,
            'latent_dynamic': latent_dynamic,
            'kinematic': kinematic,
        }
            
class RVQSeqEncoder(nn.Module):
    def __init__(self, obs_size, feature_size, seq_length, training, **kargs):
        super().__init__()
        
        self.int_num = kargs['int_num']
        self.encoder = ConvEncoder(obs_size, 512, feature_size*self.int_num)
        self.bottle_neck_list = nn.ModuleList()
        self.num = 8
        for i in range(self.num):
            self.bottle_neck_list.append(VQEmbeddingMovingAverage(512, feature_size*self.int_num, training))
        # self.bottle_neck = VQEmbeddingMovingAverage(512, feature_size*self.int_num, training)
        self.decoder = ConvDecoder(feature_size*self.int_num, 512, feature_size, obs_size)
        self.limit = training
        print('limit', self.limit)
        
    def build_mask(self, seq_length):
        return torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    
    def quant(self, latent_seq, limit = -1):
        
        latents = []
        indexs = []
        latent = latent_seq
        commit_loss = 0
        if self.limit:
            limit_number = np.random.randint(1, self.num)
        for i, bottle_neck in enumerate(self.bottle_neck_list):
            latent_vq, latent_vq_bar, indice = bottle_neck.straight_through(latent)
            commit_loss += (torch.norm(latent - latent_vq_bar.detach() ) **2) * 0.05 / np.prod(latent.shape[:-1])
            latents.append(latent_vq)
            indexs.append(indice[None, ...])
            latent = latent - latent_vq_bar.detach()
            # print(bottle_neck.embedding.abs().mean())
            # if i==1:
            #     break
            if self.limit and i == limit_number:
                break
            if i==limit:
                break
        latent_vq = sum(latents)
        no_vq = False
        if no_vq:
            latent_vq = latent_seq
        return latent_vq, commit_loss, torch.cat(indexs, dim=0)
    
    def forward(self, begin_obs, target):
        
        mu_seq = self.encoder.encode_tranpose(target)
        # assert mu_seq.shape[1:] == ( 6, 768), mu_seq.shape
        latent_seq = mu_seq.contiguous()
        latent_vq, commit_loss, indexs = self.quant(latent_seq)
        
        latent_dynamic, kinematic = self.decoder.decode(latent_vq)
        # assert kinematic.shape[1:] == (24, 323), kinematic.shape
        # latent_vq = latent_vq + torch.randn_like(latent_vq) * 0.3
        return latent_dynamic, {
            'commit_loss': commit_loss,
            'mu_seq': mu_seq,
            'latent_vq': latent_vq,
            'latent_seq': latent_seq,
            'latent_dynamic': latent_dynamic,
            'kinematic': kinematic,
            'indexs': indexs,
        }



if __name__ == '__main__':
    model = VQSeqEncoder(323, 4, 24, False)
    begin_obs = torch.randn(1, 323)
    target = torch.randn(1, 24, 323)
    latent_vq, info = model(begin_obs, target)
    
    print(latent_vq)
    
    print(latent_vq.shape)
    assert torch.all( latent_vq[:,0] == latent_vq[:,1]) and torch.all(latent_vq[:,1] == latent_vq[:,2])
    loss = latent_vq.mean()
    loss.backward()
    print(target.grad)