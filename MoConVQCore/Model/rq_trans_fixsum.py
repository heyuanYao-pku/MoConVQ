import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical

def PE1d_sincos(seq_length, dim):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if dim % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(dim))
    pe = torch.zeros(seq_length, dim)
    position = torch.arange(0, seq_length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                         -(math.log(10000.0) / dim)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe.unsqueeze(1)

class PositionEmbedding(nn.Module):
    """
    Absolute pos embedding (standard), learned.
    """
    def __init__(self, seq_length, dim, dropout, grad=False):
        super().__init__()
        self.embed = nn.Parameter(data=PE1d_sincos(seq_length, dim), requires_grad=grad)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # x.shape: bs, seq_len, feat_dim
        l = x.shape[1]
        x = x.permute(1, 0, 2) + self.embed[:l].expand(x.permute(1, 0, 2).shape)
        x = self.dropout(x.permute(1, 0, 2))
        return x


class Text2Motion_Transformer(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                embeddings = None
                ):
        super().__init__()
        # 8, 4, 3
        # -> 8, 2, 2
        self.trans_temporal = CrossCondTransFeature(num_vq, embed_dim, clip_dim, block_size, 8, n_head, drop_out_rate, fc_rate)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head, dim_feedforward=embed_dim*fc_rate, dropout=drop_out_rate, batch_first=True)
        # self.continuous_header = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.linear = nn.Linear(embed_dim, embed_dim)
        
        self.trans_base = CrossCondTransBase(num_vq, embed_dim, embed_dim, block_size, 2, n_head, drop_out_rate, fc_rate)
        self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, 2, n_head, drop_out_rate, fc_rate)
        self.block_size = block_size
        self.num_vq = num_vq
        self.embedding = embeddings
        self.max_depth = 4
        for i in range(self.max_depth):
            self.trans_base.tok_emb[i].weight.data[:] = self.embedding[i]
            self.trans_base.tok_emb[i].weight.requires_grad = False
        
    def get_block_size(self):
        return self.block_size

    def forward(self, latents, idxs):
        b, t, c = latents.shape
        b, t, d= idxs.shape
        feature = self.trans_temporal(latents) # b, t, c
        feat = self.trans_base(idxs.view(b*t, d), feature.view(b*t, c)) # b*d, t, c
        logits = self.trans_head(feat) # b*d, t, num_vq
        # logits = logits[:, :-1, :]
        # print(logits.shape)
        return logits.view(b, t, d+1, self.num_vq), self.linear( feature )

    def sample(self, label, if_categorial=True):
        for k in range(self.block_size):
            if k ==0:
                latent = label
                idxs = []
                ls = label
            else:
                latent = ls
                
            fs = self.trans_temporal(latent)
            cur_fs = fs[:, -1, :]
            
            cur_latent = torch.zeros((1, 768), device=cur_fs.device)
            end_flag = False
            
            for i in range(self.max_depth):
                # print(i)
                if i ==0:
                    x = []
                else:
                    x = xs
                nfeature = self.trans_base(x, cur_fs)
                logits = self.trans_head(nfeature)[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                # print(probs.shape)
                # if k ==0 and i ==0:
                #     from matplotlib import pyplot as plt
                #     plt.plot(probs[0].detach().cpu().numpy())
                #     plt.savefig('vis_probs.png')
                #     plt.close()
                # exit()
                if if_categorial:
                    prob, idx = torch.topk(probs, k=5, dim=-1)
                    probs = prob / prob.sum(dim=-1, keepdim=True)
                    dist = Categorical(probs)
                    idx_ = dist.sample()
                    idx = idx.gather(-1, idx_.unsqueeze(-1)).squeeze(-1)
                    if torch.any(idx >= self.num_vq):
                        end_flag = True
                        # break
                    idx = idx.unsqueeze(-1)
                else:
                    _, idx = torch.topk(probs, k=1, dim=-1)
                # print(idx.max())
                if torch.any(idx >= self.num_vq):
                    end_flag = True
                    # break
                if i == 0:
                    xs = idx
                    # print(torch.topk(probs, k=3, dim=-1))
                else:
                    xs = torch.cat((xs, idx), dim=1)
                idxs.append(idx)
                # print(cur_latent.shape, self.embedding[i].shape, idx.shape)
                cur_latent += self.embedding[i][idx[0]]
            
            if k == 0 and label is None:
                ls = cur_latent.unsqueeze(1)
            else:
                ls = torch.cat((ls, cur_latent.unsqueeze(1)), dim=1) 
            if end_flag and k > 4:
                pass
            if ls.shape[1]>=50:
                break
                # break
            # print(ls.shape)
        return ls[:, :-1, :], torch.cat(idxs, dim=0)

class CausalCrossConditionalSelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    
class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalCrossConditionalSelfAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CrossBlock(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4, bert_dim=1024, bert_max_len=77):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalCrossConditionalSelfAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )
        self.ln3 = nn.LayerNorm(embed_dim)
        self.ln4 = nn.LayerNorm(embed_dim)
        self.ln5 = nn.LayerNorm(embed_dim)
        # self.ln6 = nn.LayerNorm(embed_dim)
        self.attn2 = torch.nn.MultiheadAttention(embed_dim, n_head, drop_out_rate, batch_first=True)
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )
        
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(bert_dim, embed_dim)
        self.linear3 = nn.Linear(bert_dim, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        self.drop2 = nn.Dropout(drop_out_rate)
    def forward(self, x, y, mask):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.mlp(x))
        
        q = self.linear1(x)
        k = self.linear2(y)
        v = self.linear3(y)
        r = self.attn2( q, k, v, key_padding_mask=mask)[0]
        # print(r.abs().mean(), x.abs().mean())
        x = self.ln4(x + self.drop(r))
        x = self.ln5(x + self.drop2(self.mlp2((x))))
        return x

class CrossCondTransFeature(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                bert_dim=1024,
                bert_max_len = 512
                ):
        super().__init__()
        
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate))
        self.pos_embed = PositionEmbedding(block_size, embed_dim, 0.0, False)
        self.bert_pos_embed = PositionEmbedding(bert_max_len, bert_dim, 0.0, False)
        self.block_size = block_size
        self.apply(self._init_weights)
        self.dropout = nn.Dropout(drop_out_rate)
        encoder_layer = nn.TransformerEncoderLayer(d_model=bert_dim, nhead=n_head, dim_feedforward=embed_dim*fc_rate, dropout=drop_out_rate, batch_first=True)
        self.bert_header = nn.TransformerEncoder(encoder_layer, num_layers=3)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, latents):

        # if len(latents) == 0:
            # token_embeddings = self.cond_emb(clip_feature).unsqueeze(1) # b, 1, c
        # else:
            # b, t, c = latents.shape
            # token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), latents], dim=1)
        token_embeddings = latents

            
        x = self.pos_embed(token_embeddings)
        # bert_feature = self.bert_header(bert_feature, src_key_padding_mask = bert_mask)
        # bert_feature = self.bert_pos_embed(bert_feature)
        # bert_feature = self.dropout(bert_feature)
        for blk in self.blocks:
            x = blk(x)

        return x # b,t,c


class CrossCondTransBase(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        self.tok_emb = nn.ModuleList( [nn.Embedding(num_vq + 2, embed_dim) for i in range(8)]  )
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.pos_embed = PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, clip_feature):

        if len(idx) == 0:
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        else:
            b, d = idx.shape
            embed_list = []
            for i in range(d):
                token_embeddings = self.tok_emb[i](idx[:, i]) # b * t * c
                if i!=0:
                    token_embeddings = token_embeddings.view(b, 1, -1) + embed_list[-1]
                embed_list.append(token_embeddings.view(b, 1, -1) )
            embed = torch.cat(embed_list, dim=1) # b*t, 8, c
            token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), embed], dim=1)
            
        x = self.pos_embed(token_embeddings)
        x = self.blocks(x)

        return x


class CrossCondTransHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        
        self.head = nn.ModuleList( [nn.Linear(embed_dim, num_vq, bias=False) for i in range(8)]  )
        # self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
        
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.blocks(x)
        x = self.ln_f(x)
        res = []
        for i in range(x.shape[1]):
            res.append(self.head[i](x[:, i:i+1, :]))
        logits = torch.cat(res, dim=1)
        # logits = self.head(x)
        return logits

    


        

