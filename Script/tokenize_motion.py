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
    parser.add_argument('--train_prior', default=False, action='store_true')
    
    
    # add args for each content 
    parser = VCLODETrackEnv.add_specific_args(parser)
    parser = MoConVQ.add_specific_args(parser)
    
    args = vars(parser.parse_args(args=args_in))
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
    #build each content
    env = VCLODETrackEnv(**args)
    agent = MoConVQ(323, 12, 57, 120, env, training=False, **args)
    
    agent.simple_load(r'moconvq_base.data', strict=True)
    agent.eval()
    agent.posterior.limit = False
    
    torch.set_grad_enabled(False)
    
    print('torch.cuda.is_available:', torch.cuda.is_available())
    
    return agent, env
  
def encode(agent:MoConVQ, observation:np.ndarray):
    info = agent.encode_seq_all(None, observation)
    seq_latent = info['latent_dynamic']
    seq_vq = info['latent_vq']
    seq_indices = info['indexs']
    
    seq_indices = seq_indices.squeeze().detach().cpu().numpy()
    
    return seq_indices
          

if __name__ == "__main__":    
    model_args = build_args(args_in=[])
        
    parser = argparse.ArgumentParser()
    parser.add_argument('bvh-file', type=str, nargs='+')
    parser.add_argument('--is-bvh-folder', default=False, action='store_true')
    parser.add_argument('--flip-bvh', default=False, action='store_true')
    parser.add_argument('-o', '--output-file', type=str, default='', help='output file (*.txt, *.npz, *.npy, *.h5)')
    parser.add_argument('-l', '--output-level', type=int, default=8, choices=range(1, 9), help='number of RVQ layers')
    parser.add_argument('--use-dataset-observations-instead-input-file', default=False, action='store_true')
    parser.add_argument('--gpu', type = int, default=0, help='gpu id')
    parser.add_argument('--cpu_b', type = int, default=0, help='cpu begin idx')
    parser.add_argument('--cpu_e', type = int, default=-1, help='cpu end idx')
    
    # print(args)
    args = vars(parser.parse_args())
    model_args.update(args)
    
    ptu.init_gpu(True, gpu_id=args['gpu'])
    
    agent, env = get_model(model_args)
    
    if not args['use_dataset_observations_instead_input_file']:    
        motion_data = MotionDataSet(20)
        for fn in args['bvh-file']:
            if args['is_bvh_folder']:
                motion_data.add_folder_bvh(fn, env.sim_character)
            else:            
                flip = args['flip_bvh']
                motion_data.add_bvh_with_character(fn, env.sim_character, flip=flip)
                
        observation = motion_data.observation
    else:
        observation = env.motion_data.observation
        
    print('# motion_data', observation.shape, observation.size)
    
    seq_indices = encode(agent, np.asarray(observation))
    print('# seq_indices', seq_indices.shape)
    
    out_fn:str = args['output_file']
    if len(out_fn) > 0:
        if out_fn.endswith('.h5'):
            import h5py
            h5file = h5py.File(out_fn, 'w')
            h5file.create_dataset('tokens', data=seq_indices)
            h5file.close()
            
        elif out_fn.endswith('.npz'):
            np.savez_compressed(out_fn, tokens=seq_indices)
            
        elif out_fn.endswith('.npy'):
            np.save(out_fn, seq_indices)
            
        else:
            np.savetxt(out_fn, seq_indices[:args['output_level']], fmt='%3d')
            # with open(out_fn, 'w') as f:
            #     for l in range(args['output_level']):
            #         f.write(' '.join(f'{d}' for d in seq_indices[l]))
            #         f.write('\n')
   
            
        print('saved indices to', out_fn)
    
    else:
        print(seq_indices)
    