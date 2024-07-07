# from ControlVAECore.Model.t2m_trans import Text2Motion_Transformer
from MoConVQCore.Model.cross_trans_ori_fixsum import Text2Motion_Transformer
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


def text2bert(text):
    global bert, bert_tokenizer
    encoded_input = bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
    encoded_input = {key: value.to(ptu.device) for key, value in encoded_input.items()}
    with torch.no_grad():
        output = bert(**encoded_input)
    return output.last_hidden_state, ~encoded_input['attention_mask'].bool()

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

def length2mask(lens, max_len=50):
    res = torch.arange(max_len).to(ptu.device).expand(len(lens), max_len) >= lens.unsqueeze(1)
    return res

class Trainer():
    def __init__(self, device_list = None, **kargs) -> None:
        self.name = kargs['name']
        embed_torch = [
            torch.cat( [bottle_neck.embedding, torch.zeros_like(bottle_neck.embedding[:2])], dim = 0 ) for bottle_neck in agent.posterior.bottle_neck_list
        ]
        
        gpt = Text2Motion_Transformer(**vars(gpt_config()), embeddings = embed_torch).to(ptu.device)
        if device_list is None:
            device_list = [4,5]
        self.gpt = nn.DataParallel(gpt, device_ids=device_list)

    def evaluate_generate(self, text, idx, dir, agent_=None, env_ = None, clip_feature = None, **kargs):
        
        if agent_ is None:
            global agent, env
            agent_ = agent
        
        if 'bert_feature' not in kargs:
            bert_feature, bert_mask = text2bert(text)
        else:
            bert_feature = kargs['bert_feature']
            bert_mask = kargs['bert_mask']
            
        try:
            gpt = self.gpt.module
        except:
            gpt = self.gpt
        # clip_feature = torch.zeros_like(clip_feature)
        clip_feature = torch.zeros((1, 512)).to(ptu.device)
        cur_embedding, _ = gpt.sample(clip_feature, bert_feature, bert_mask)
        # print(_.view(-1, 4))
        dconv = agent_.posterior.decoder.decode_dynamic(cur_embedding)
        
        import VclSimuBackend
        CharacterToBVH = VclSimuBackend.ODESim.CharacterTOBVH
        saver = CharacterToBVH(agent_.env.sim_character, 120)
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
    from moconvq_builder import build_agent
    device = 0
    agent, env = build_agent(gpu = device)
    ptu.init_gpu(True, gpu_id=device)
    
    from transformers import T5Tokenizer, T5EncoderModel
    bert_tokenizer = T5Tokenizer.from_pretrained('t5-large', resume_download=True)
    
    bert = T5EncoderModel.from_pretrained('t5-large', resume_download=True).to(ptu.device)
    bert.eval()
    
    agent.simple_load(r'moconvq_base.data', strict=True)
    agent.eval()
    
    trainer = Trainer(device_list = [device], name = 'test', build = False)
    trainer.gpt.load_state_dict(torch.load('text_generation_GPT.pth', map_location=ptu.device))
    trainer.gpt = trainer.gpt.eval()
    
    text = 'A person walking while talking on the phone.'
    
    bert_feature, bert_mask = text2bert(text)
    trainer.evaluate_generate(text, 0, 'out/conditional', agent_=agent, env_=env, clip_feature=None, bert_feature=bert_feature, bert_mask=bert_mask)