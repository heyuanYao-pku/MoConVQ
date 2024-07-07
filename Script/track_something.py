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

from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()

import os

import cProfile
def profile(filename=None, comm=MPI.COMM_WORLD):
  def prof_decorator(f):
    def wrap_f(*args, **kwargs):
      pr = cProfile.Profile()
      pr.enable()
      result = f(*args, **kwargs)
      pr.disable()

      if filename is None:
        pr.print_stats()
      else:
        filename_r = filename + ".{}".format(comm.rank)
        pr.dump_stats(filename_r)

      return result
    return wrap_f
  return prof_decorator

def flatten_dict(dd, separator='_', prefix=''):
    return { prefix + separator + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in flatten_dict(vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }

def build_args(parser):
    # add args for each content 
    parser = VCLODETrackEnv.add_specific_args(parser)
    parser = MoConVQ.add_specific_args(parser)
    args = vars(parser.parse_args())
    # yaml
    config = load_yaml(args['config_file'])
    config = flatten_dict(config)
    args.update(config)
    
    if args['load'] and mpi_rank ==0:
        import tkinter.filedialog as fd
        config_file = fd.askopenfilename(filetypes=[('YAML','*.yml')])
        data_file = fd.askopenfilename(filetypes=[('DATA','*.data')])
        config = load_yaml(config_file)
        config = flatten_dict(config)
        args.update(config)
        args['load'] = True
        args['data_file'] = data_file
        
    #! important!
    seed = args['seed'] + mpi_rank
    args['seed'] = seed
    VCLODETrackEnv.seed(seed)
    MoConVQ.set_seed(seed)
    return args
   
    return agent, args

# @profile(filename='profile')
def train(agent):
    agent.train_loop()

if __name__ == "__main__":
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
    
    parser.add_argument('bvh-file', type=str, nargs='+')
    parser.add_argument('--is-bvh-folder', default=False, action='store_true')
    parser.add_argument('--flip-bvh', default=False, action='store_true')
    parser.add_argument('-o', '--output-file', type=str, default='')
    
    args = build_args(parser)
    print(args['gpu'])
    ptu.init_gpu(True, gpu_id=args['gpu'])
    if args['cpu_e'] !=-1:
        p = psutil.Process()
        cpu_lst = p.cpu_affinity()
        try:
            p.cpu_affinity(range(args['cpu_b'],args['cpu_b'+mpi_rank]))   
        except:
            pass 
    
    
    #build each content
    env = VCLODETrackEnv(**args)
    agent = MoConVQ(323, 12, 57, 120,env, training=False, **args)
    
    motion_data = MotionDataSet(20)
    
    for fn in args['bvh-file']:
        if args['is_bvh_folder']:
            motion_data.add_folder_bvh(fn, env.sim_character)
        else:            
            flip = args['flip_bvh']
            motion_data.add_bvh_with_character(fn, env.sim_character, flip=flip)
            
    agent.simple_load(r'moconvq_base.data', strict=True)
    agent.eval()
    agent.posterior.limit = False
    import VclSimuBackend
    CharacterToBVH = VclSimuBackend.ODESim.CharacterTOBVH
    
    
    saver = CharacterToBVH(agent.env.sim_character, 120)
    saver.bvh_hierarchy_no_root()
    
    observation, info = agent.env.reset(0)
    
    period = 1000000
    
    momentum_list, net_force_list, total_momentum = [], [], []
    info = agent.encode_seq_all(motion_data.observation, motion_data.observation)
    
    '''
    info:
    'latent_seq': output of the encoder,
    'latent_vq': vector quantization of the latent_seq,
    'latent_dynamic': upsampling of the latent_vq, which is the control signal for the policy,
    'indexs': indexs of the latent_vq in the codebook,
    '''
    
    # decode the latent_vq with simulator, and save the motion
    seq_latent = info['latent_dynamic']
    for i in range(seq_latent.shape[1]):
        obs = observation['observation']
        
        action, info = agent.act_tracking(
            obs_history = [obs.reshape(1,323)],
            target_latent = seq_latent[:,i%period],
        )
        action = ptu.to_numpy(action).flatten()
        for i in range(6):
            saver.append_no_root_to_buffer()
            if i == 0:
                step_generator = agent.env.step_core(action, using_yield = True)
            info = next(step_generator)
            avel = env.sim_character.body_info.get_body_ang_velo()
        try:
            info_ = next(step_generator)
        except StopIteration as e:
            info_ = e.value
        new_observation, rwd, done, info = info_
        observation = new_observation
    import time
    motion_name = os.path.join('out', f'track_{time.time()}.bvh')
    saver.to_file(motion_name)