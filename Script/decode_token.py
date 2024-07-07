import argparse
from enum import Enum
from MoConVQCore.Env.vclode_track_env import VCLODETrackEnv
from MoConVQCore.Model.MoConVQ import MoConVQ
# from DiffusionMore.diffusion import Diffusion as MoConVQ
# from MoConVQCore.Model.valina_MoConVQ import MoConVQ
from MoConVQCore.Utils.misc import *
import psutil
import MoConVQCore.Utils.pytorch_utils as ptu
from MoConVQCore.Utils.motion_dataset import MotionDataSet

import os 

def flatten_dict(dd, separator='_', prefix=''):
    return { prefix + separator + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in flatten_dict(vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }
        
class Generator:
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        self.value = yield from self.gen
        
def build_args(parser=None, args_in=None):
    if parser is None:
        parser = argparse.ArgumentParser()
        
    parser.add_argument('--config_file', default='Data/Parameters/bigdata.yml', help= 'a yaml file contains the training information')
    parser.add_argument('--seed', type = int, default=0, help='seed for root process')
    parser.add_argument('--experiment_name', type = str, default="debug", help="")
    parser.add_argument('--load', default=False, action='store_true')
    parser.add_argument('--gpu', type = int, default=0, help='gpu id')
    parser.add_argument('--cpu_b', type = int, default=0, help='cpu begin idx')
    parser.add_argument('--cpu_e', type = int, default=-1, help='cpu end idx')
    parser.add_argument('--using_vanilla', default=False, action='store_true')
    parser.add_argument('--no_train', default=False, action='store_true')
    parser.add_argument('--train_prior', default=False, action='store_true')
    parser.add_argument('indices', type=int, nargs='*', help="")
    parser.add_argument('-f', '--index-file', type = str, default="", help="")
    parser.add_argument('-l', '--index-level', type=int, default=1, choices=range(1, 9))
    
    # add args for each content 
    parser = VCLODETrackEnv.add_specific_args(parser)
    parser = MoConVQ.add_specific_args(parser)
    args = vars(parser.parse_args())
    # yaml
    config = load_yaml(args['config_file'])
    config = flatten_dict(config)
    args.update(config)
    
    if args['load']:
        import tkinter.filedialog as fd
        config_file = fd.askopenfilename(filetypes=[('YAML','*.yml')])
        data_file = fd.askopenfilename(filetypes=[('DATA','*.data')])
        config = load_yaml(config_file)
        config = flatten_dict(config)
        args.update(config)
        args['load'] = True
        args['data_file'] = data_file
        
    #! important!
    seed = args['seed']
    args['seed'] = seed
    VCLODETrackEnv.seed(seed)
    MoConVQ.set_seed(seed)
    
    return args

def get_model(args):    
    print(args['gpu'])
    ptu.init_gpu(True, gpu_id=args['gpu'])
    if args['cpu_e'] !=-1:
        p = psutil.Process()
        cpu_lst = p.cpu_affinity()
        try:
            p.cpu_affinity(range(args['cpu_b'], args['cpu_b']))   
        except:
            pass 
    
    
    #build each content
    env = VCLODETrackEnv(**args)
    agent = MoConVQ(323, 12, 57, 120, env, training=False, **args)
    
    # agent.try_load(r'24000_vq.data', strict=True)
    agent.try_load(r'36000_vq.data', strict=True)
    # env.stable_pd.tor_lim = env.stable_pd.tor_lim.clip(max=200)
    agent.eval()
    agent.posterior.limit = False
    
    torch.set_grad_enabled(False)
    
    return agent, env
    
def parse_indices(args):    
    indices = args['indices']
    
    if len(indices) == 0:
        indices = '184 463 501 188 471 220 184 463 501 188 471 220 184 463 501 188 471 220 184 463 501 188 471 220 184 463 501 188 471 220 110 273  67 178 299  56 419 504 386 110 437 335 248  28  28 437 368  94 227 337'
        # indices = '419 504 419 504 419 504 419 504 419 504 419 504'
        # index = sum(index, [])
        # index = index*2        
        indices = [int(x) for x in indices.split()]
        # index = np.random.randint(0, 512, size=(200,))
        # indices = np.arange(512).reshape(512,-1).repeat(10,1).flatten()
        
    if len(args['index_file']) > 0:
        with open(args['index_file'], 'r') as f:
            indices = [int(x) for x in f.read().split()]
        
    
    indices = np.asarray(indices)
    if args['index_level'] > 0:
        indices = indices.reshape(args['index_level'], -1)
    
    print(f'# indices: {indices.shape}')
    
    return indices
    
        
def decode(agent:MoConVQ, indices):
    if len(indices) == 0:
        return

    indices = np.asarray(indices)
    if indices.ndim == 1:
        seq_vq = agent.posterior.bottle_neck_list[0].embedding[indices]
        seq_vq = seq_vq.unsqueeze(0)
    elif indices.ndim == 2:
        seq_vq = 0
        for vocab, rq_indices in zip(agent.posterior.bottle_neck_list, indices):
            seq_vq += vocab.embedding[rq_indices]
        seq_vq = seq_vq.unsqueeze(0)
    else:
        print('wrong indices!')
        return
    
    seq_latent, _ = agent.posterior.decoder.decode(seq_vq)
    print(len(indices), seq_latent.shape)
        
    import VclSimuBackend
    CharacterToBVH = VclSimuBackend.ODESim.CharacterTOBVH
    saver = CharacterToBVH(agent.env.sim_character, 120)
    saver.bvh_hierarchy_no_root()
    
    observation, info = agent.env.reset(0)    
    
    # env.scene.contact_type = 1
    # env.scene.extract_contact = False
    num_steps = seq_latent.shape[1]
    for step_idx in range(num_steps):
        obs = observation['observation']
            
        if step_idx == seq_latent.shape[1]:
            break
        
        action, info = agent.act_tracking(
            obs_history = [obs.reshape(1,323)],
            target_latent = seq_latent[:, step_idx],
        )
        action = ptu.to_numpy(action).flatten()        
        step_generator = Generator(agent.env.step_core(action, using_yield = True))
        for i, info_ in enumerate(step_generator):
            saver.append_no_root_to_buffer()
            
        print(f'\r{step_idx+1:4d}/{num_steps}', end='')
            
        info_ = step_generator.value
            
        new_observation, rwd, done, info = info_
        observation = new_observation
        # print(character.body_info.avel)
        # print(observation)
        
    print()
    
    return saver
    
if __name__ == "__main__":    
    args = build_args()
    indices = parse_indices(args)
    agent, _ = get_model(args)
    saver = decode(agent, indices)   
    
    import time
    motion_name = os.path.join('out', f'track_{time.time()}.bvh')
    saver.to_file(motion_name)
    