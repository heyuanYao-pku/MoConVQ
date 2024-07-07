from MoConVQCore.Model.rq_trans_fixsum import Text2Motion_Transformer
import torch.nn as nn
from torch.utils.data import Dataset
import h5py
from torch.utils.data import DataLoader
import numpy as np
import torch
import MoConVQCore.Utils.pytorch_utils as ptu
import os
from torch.nn import functional as F
from torch.distributions import Categorical

@torch.no_grad()
def build_raw(dataset, dataset_output, agent):
    dataset = h5py.File(dataset, 'r')
    dataset_output = h5py.File(dataset_output, 'w')
    for i, k in enumerate(dataset.keys()):
        try:
            if 'observation' not in dataset[k].keys():
                continue
            obs = dataset[k]['observation'][:]
            dataset_output.create_group(k)
            dataset_output[k]['observation'] = obs
            info = agent.encode_seq_all(obs[0], obs)
            dataset_output[k]['latent'] = info['latent_vq'].detach().cpu().numpy()
            dataset_output[k]['index'] = info['indexs'].detach().cpu().numpy()
        except Exception as e:
            print(e)
    dataset_output.close()
    dataset.close()


class gpt_config():
    def __init__(self):
        self.num_vq = 512
        self.embed_dim = 768
        self.clip_dim = 512
        self.block_size = 52
        self.num_layers = 9
        self.n_head = 8
        self.drop_out_rate = 0.1
        self.fc_rate = 2

class GPTDataset(Dataset):
    def __init__(self, dataset, agent):
        dataset = r'data_uncond.h5'
        self.dataset = h5py.File(dataset, 'r')
        self.lens = []
        self.indices = []
        self.embedding = []
        self.data = []
        for bottleneck in agent.posterior.bottle_neck_list:
            embed = bottleneck.embedding.detach().cpu().numpy()
            embed = np.concatenate([embed, np.zeros((2, embed.shape[1]))], axis=0)
            self.embedding.append(embed)
        
        for k in self.dataset.keys():
            l = len(self.dataset[k]['observation']) - 50
            l = 1 if l < 1 else l
            self.lens.append(l)
            self.indices.append( np.array(self.dataset[k]['index']).transpose())
            self.data.append(self.dataset[k]['latent'][0])
            
        self.lens = np.array(self.lens)
        self.max_depth = 4
        
    def __len__(self):
        # return sum(self.lens)
        return 128 * 50
    
    def __getitem__(self, index):
        # motion_index = np.random.randint(0, len(self.lens) ,self.lens/sum(self.lens))
        motion_index = np.random.choice(len(self.lens), p=self.lens/sum(self.lens))
        frame_index = np.random.randint(0, self.lens[motion_index])
        indices = self.indices[motion_index][ frame_index:frame_index+50]
        depth = np.random.randint(1, self.max_depth+1)
        data = self.data[motion_index][frame_index:frame_index+50]
        data = np.zeros_like(data)
        # print(data.shape)
        for i in range(depth):
            data += self.embedding[i][indices[:,i]]    
        if len(data) < 50:
            data = np.concatenate([data, np.zeros((50-len(data), data.shape[1]))], axis=0)
            indices = np.concatenate([indices, np.ones((50-len(indices), indices.shape[1])) * 513], axis=0)
        return data.astype(np.float32), indices.astype(np.int64) 
    
    def sampe_an_embedding(self):
        motion_index = np.random.choice(len(self.lens), p=self.lens/sum(self.lens))
        frame_index = np.random.randint(0, self.lens[motion_index])
        indices = self.indices[motion_index][ frame_index:frame_index+50]
        depth = np.random.randint(1, self.max_depth+1)
        data = self.data[motion_index][frame_index:frame_index+50]
        data = np.zeros_like(data)
        # print(data.shape)
        for i in range(depth):
            data += self.embedding[i][indices[:,i]]    
        if len(data) < 50:
            data = np.concatenate([data, np.zeros((50-len(data), data.shape[1]))], axis=0)
            indices = np.concatenate([indices, np.ones((50-len(indices), indices.shape[1])) * 513], axis=0)
        return data, indices.astype(np.int64) 
            


def length2mask(lens, max_len=50):
    res = torch.arange(max_len).to(ptu.device).expand(len(lens), max_len) >= lens.unsqueeze(1)
    return res

class Trainer():
    def __init__(self, agent_ = None, device_list = None, **kargs) -> None:
        self.name = kargs['name']
        tau = [
            0.30649805068969727 ,
            0.1750335693359375 ,
            0.14291979372501373 ,
            0.12483198940753937 ,
            0.11151637136936188 ,
            0.10116757452487946 ,
            0.09271004050970078 ,
            0.08552911877632141 ,
        ]
        if agent_ is None:
            global agent
            agent = agent
        else:
            agent = agent_

        embed_torch = [
            torch.cat( [bottle_neck.embedding, torch.zeros_like(bottle_neck.embedding[:2])], dim = 0 ) for bottle_neck in agent.posterior.bottle_neck_list
        ]
        self.embedding = embed_torch
        gpt = Text2Motion_Transformer(**vars(gpt_config()), embeddings = embed_torch).to(ptu.device)
        self.gpt = gpt

    
    @torch.no_grad()
    def evaluate_long(self, idx, dir, agent_=None, env_ = None, clip_feature = None, **kargs):
        if agent_ is None:
            global agent, env
            agent_ = agent
            env_ = env
        try:
            gpt = self.gpt.module
        except:
            gpt = self.gpt
        
        indices = torch.randint(0, 512, (1,)).to(ptu.device)
        data = torch.zeros((1, 1, 768)).to(ptu.device)
        for i in range(4):
            embedding = self.embedding[i][indices]
            data[0] += embedding
        print('sampling....')
        import os, sys
        sys.stdout.flush()
        
        
        cur_embedding, indices = gpt.sample(data)
        for i in range(1): # 4->0
            embedding_next, indices = gpt.sample(cur_embedding[:,-5:] )
            if i ==0:
                cur_embedding = embedding_next
            else:
                cur_embedding = torch.cat([cur_embedding[:,:-5], embedding_next], dim = 1)
                

        dconv = agent_.posterior.decoder.decode_dynamic(cur_embedding)
        
        import VclSimuBackend
        CharacterToBVH = VclSimuBackend.ODESim.CharacterTOBVH
        saver = CharacterToBVH(agent.env.sim_character, 120)
        saver.bvh_hierarchy_no_root()
        
        observation, info = agent_.env.reset(0)
        
        for i in range(dconv.shape[1]):
            obs = observation['observation']
            action, info = agent_.act_tracking(
                obs_history = [obs.reshape(1,323)],
                target_latent = dconv[:,i],
            )
            action = ptu.to_numpy(action).flatten()
            for i in range(6):
                saver.append_no_root_to_buffer()
                if i == 0:
                    step_generator = agent_.env.step_core(action, using_yield = True)
                info = next(step_generator)
                
            try:
                info_ = next(step_generator)
            except StopIteration as e:
                info_ = e.value
            new_observation, rwd, done, info = info_
            observation = new_observation
            
        saver.to_file(os.path.join(dir,f'evaluate_gpt{idx}.bvh'))

if __name__ == '__main__':
    from Script.moconvq_builder import build_agent
    import psutil
    p = psutil.Process()
    cpu_lst = p.cpu_affinity()
    device = 0
    agent, env = build_agent(gpu = device)
    ptu.init_gpu(True, device)
    agent.simple_load('moconvq_base.data', strict=True)
    agent.eval()
    
    trainer = Trainer(agent_ = agent, device_list = [device], name = 'Test', build=False)
    data = torch.load(r'unconditional_GPT.pth', map_location=ptu.device)
    data_ = {   }
    for key in data.keys():
        if key.startswith('module'):
            data_[key[7:]] = data[key]
        else:
            data_[key] = data[key]
    trainer.gpt.load_state_dict(data_)
    trainer.gpt = trainer.gpt.eval()
    trainer.evaluate_long(0, 'out/uncondition', agent, env)