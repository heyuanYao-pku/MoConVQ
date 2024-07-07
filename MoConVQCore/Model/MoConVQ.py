import random
from typing import List,Dict
from numpy import dtype
import torch
from torch import nn
import torch.distributions as D
from MoConVQCore.Model.trajectory_collection import TrajectorCollector
from MoConVQCore.Utils.mpi_utils import gather_dict_ndarray
from MoConVQCore.Utils.replay_buffer import ReplayBuffer
from tensorboardX import SummaryWriter
from modules import *
from ..Utils.motion_utils import *
from ..Utils import pytorch_utils as ptu
from ..Utils.index_counter import index_counter
import time
import sys
from MoConVQCore.Utils.radam import RAdam
from mpi4py import MPI
from MoConVQCore.Model.causal_transformer import CausalEncoder
# from MoConVQCore.Utils.gaussian_diffusion import DiffusionHelper
from MoConVQCore.Model.world_model import SimpleWorldModel
from MoConVQCore.Model.convvq import *

mpi_comm = MPI.COMM_WORLD
mpi_world_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

# whether this process should do tasks such as trajectory collection....
# it's true when it's not root process or there is only root process (no subprocess)
should_do_subprocess_task = mpi_rank > 0 or mpi_world_size == 1

class MoConVQ(nn.Module):
    """
    A ContorlVAE agent which includes encoder, decoder and world model
    """
    def __init__(self, observation_size, future_size, action_size, delta_size, env, **kargs):
        super().__init__()
        
        # components of MoConVQ
        int_num = 3
        kargs['latent_size'] = 256
        self.posterior = RVQSeqEncoder(
            observation_size, kargs['latent_size'], 24, training = kargs.get('training', mpi_rank ==0) , int_num = int_num
        ).to(ptu.device)
        
        
        
        self.agent =  GatingMixedDecoder(
            latent_size= kargs['latent_size'],
            condition_size= observation_size,
            output_size=action_size,
            actor_hidden_layer_num= kargs['actor_hidden_layer_num'],
            actor_hidden_layer_size= kargs['actor_hidden_layer_size'],
            actor_num_experts= kargs['actor_num_experts'],
            actor_gate_hidden_layer_size= kargs['actor_gate_hidden_layer_size'],
        ).to(ptu.device)
        
        self.future_project = nn.Linear(future_size, future_size).to(ptu.device)
        
        # statistics, will be used to normalize each dimention of observation
        statistics = env.stastics
        self.obs_mean = nn.Parameter(ptu.from_numpy(statistics['obs_mean']), requires_grad = False).to(ptu.device)
        self.obs_std = nn.Parameter(ptu.from_numpy(statistics['obs_std']), requires_grad= False).to(ptu.device)
        self.future_mean = nn.Parameter(ptu.from_numpy(statistics['future_mean']), requires_grad = False).to(ptu.device)
        self.future_std = nn.Parameter(ptu.from_numpy(statistics['future_std']), requires_grad= False).to(ptu.device)
        
        # world model
        self.world_model = SimpleWorldModel(observation_size, action_size, delta_size, env.dt, statistics, **kargs).to(ptu.device)
        
        # optimizer
        self.wm_optimizer = RAdam(self.world_model.parameters(), kargs['world_model_lr'], weight_decay=1e-3)
        self.vae_optimizer = RAdam( list(self.posterior.parameters()) + 
                                    list(self.future_project.parameters()) +
                                   list(self.agent.parameters()) 
                                #    +
                                #    list(self.kinematic_world_model.parameters()) + 
                                #    list(self.kinematic_agent.parameters())
                                , kargs['MoConVQ_lr']
                                   )
        self.beta_scheduler = ptu.scheduler(0,8,0.009,0.2,1000*8)
        
        #hyperparameters....
        self.action_sigma = 0.05
        self.max_iteration = kargs['max_iteration']
        self.collect_size = kargs['collect_size']
        self.sub_iter = kargs['sub_iter']
        self.save_period = kargs['save_period']
        self.evaluate_period = kargs['evaluate_period']
        self.world_model_rollout_length = kargs['world_model_rollout_length']
        self.MoConVQ_rollout_length = kargs['MoConVQ_rollout_length']
        self.world_model_batch_size = kargs['world_model_batch_size']
        self.MoConVQ_batch_size = kargs['MoConVQ_batch_size']
        
        # policy training weights                                    
        self.weight = {}
        for key,value in kargs.items():
            if 'MoConVQ_weight' in key:
                self.weight[key.replace('MoConVQ_weight_','')] = value
        
        self.rwd_weight = self.weight.copy()
        self.rwd_weight['vel'] = 0.5
        self.rwd_weight['avel'] = 0.5
        
        # clip_features = torch.load('sad_left_both_happy.pt', map_location = ptu.device)
        # clip_feature = clip_features[mpi_rank - 1].to(ptu.device) #if mpi_rank != 1 else None
        # if mpi_rank == 4:
        #     clip_feature += clip_features[1]
        # if mpi_rank != 0:
        #     import h5py
        #     features = h5py.File('lafan_big_fixed.h5', 'r')
        #     if mpi_rank == 1:
        #         clip_feature = features['jumps1_subject5.bvh'][1000]
        #     elif mpi_rank == 2:
        #         clip_feature = features['0008_Skipping001-mocap-100.bvh'][20]
        #     elif mpi_rank == 3:
        #         clip_feature = features['0005_JumpRope001-mocap-100.bvh'][30]
        #     elif mpi_rank == 4:
        #         clip_feature = features['0005_Jogging001-mocap-100.bvh'][60]
        #     clip_feature = ptu.from_numpy(clip_feature)
        # self.clip_feature = clip_feature
        # for real trajectory collection
        self.runner = TrajectorCollector(venv = env, actor = self, runner_with_noise = True, clip_feature = None)
        self.runner.track = not kargs['train_prior']
        self.env = env    
        self.replay_buffer = ReplayBuffer(self.replay_buffer_keys, kargs['replay_buffer_size']) if mpi_rank ==0 else None
        self.kargs = kargs
        
        self.wm_weight = {}
        for key,value in kargs.items():
            if 'world_model_weight' in key:
                self.wm_weight[key.replace('world_model_weight_','')] = value
        if mpi_rank == 0:
            pass
        else:
            self.eval()

        self.observation_dim = observation_size
        self.latent_dim = kargs['latent_size']
        self.future_dim = future_size
        
    #--------------------------------for MPI sync------------------------------------#
    def parameters_for_sample(self):
        '''
        this part will be synced using mpi for sampling, world model is not necessary
        '''
        return {
            'encoder': self.posterior.state_dict(),
            'future_project': self.future_project.state_dict(),
            'agent': self.agent.state_dict()
        }
    def load_parameters_for_sample(self, dict):
        self.posterior.load_state_dict(dict['encoder'])
        self.agent.load_state_dict(dict['agent'])
        self.future_project.load_state_dict(dict['future_project'])
    
    #-----------------------------for replay buffer-----------------------------------#
    @property
    def world_model_data_name(self):
        return ['state', 'action']
    
    @property
    def policy_data_name(self):
        return ['state', 'future', 'target', 'clip_feature']
    
    @property
    def replay_buffer_keys(self):
        return ['state', 'action', 'future', 'target', 'clip_feature']

    @property
    def transition_data_name(self):
        return ['future', 'target', 'clip_feature']
    
    #----------------------------for training-----------------------------------------#
    def train_one_step(self):
        
        time1 = time.perf_counter()
        
        # data used for training world model
        name_list = self.world_model_data_name
        rollout_length = self.world_model_rollout_length
        data_loader = self.replay_buffer.generate_data_loader(name_list, 
                            rollout_length+1, # needs additional one state...
                            self.world_model_batch_size, 
                            self.sub_iter)
        for batch in  data_loader:
            world_model_log = self.train_world_model(*batch)
        # world_model_log = {}
        
        time2 = time.perf_counter()
        
        # data used for training policy
        name_list = self.policy_data_name
        rollout_length = self.MoConVQ_rollout_length
        data_loader = self.replay_buffer.generate_data_loader(name_list, 
                            rollout_length, 
                            self.MoConVQ_batch_size, 
                            self.sub_iter)
        for batch in data_loader:
            policy_log = self.train_policy(*batch)
        # policy_log = {}
        
        # log training time...
        time3 = time.perf_counter()      
        
        
        # train_transition
        name_list = self.transition_data_name
        rollout_length = self.MoConVQ_rollout_length
        data_loader = self.replay_buffer.generate_transition_data_loader(name_list, 
                            rollout_length, 
                            self.MoConVQ_batch_size, 
                            self.sub_iter//2)
        for batch in data_loader:
            transition_log = self.train_transition(*batch)
        time4 = time.perf_counter()
        
        world_model_log['training_time'] = (time2-time1)
        policy_log['training_time'] = (time3-time2)
        transition_log['training_time'] = (time4-time3)
        
        # merge the training log...
        return self.merge_dict([world_model_log, policy_log, transition_log], ['WM','Policy', 'Transition'])
    
    def mpi_sync(self):
        
        # sample trajectories
        if should_do_subprocess_task:
            with torch.no_grad():
                path : dict = self.runner.trajectory_sampling( math.floor(self.collect_size/max(1, mpi_world_size -1)), self )
                self.env.update_val(path['done'], path['rwd'], path['frame_num'])
        else:
            path = {}

        tmp = np.zeros_like(self.env.val)
        mpi_comm.Allreduce(self.env.val, tmp)        
        self.env.val = tmp / mpi_world_size
        self.env.update_p()
        
        res = gather_dict_ndarray(path)
        if mpi_rank == 0:
            paramter = self.parameters_for_sample()
            mpi_comm.bcast(paramter, root = 0)
            self.replay_buffer.add_trajectory(res)
            info = {
                'rwd_mean': np.mean(res['rwd']),
                'rwd_std': np.std(res['rwd']),
                'episode_length': len(res['rwd'])/(res['done']!=0).sum()
            }
        else:
            paramter = mpi_comm.bcast(None, root = 0)
            self.load_parameters_for_sample(paramter)    
            info = None
        return info
    
    
    def train_loop(self):
        """training loop, MPI included
        """
        for i in range(self.max_iteration):
            # if i ==0:
            info = self.mpi_sync() # communication, collect samples and broadcast policy
            
            if mpi_rank == 0:
                print(f"----------training {i} step--------")
                if self.kargs['no_train']:
                    continue
                sys.stdout.flush()
                log = self.train_one_step()   
                log.update(info)       
                self.try_save(i)
                self.try_log(log, i)

            with torch.no_grad():
                self.try_evaluate(i)
                
    # -----------------------------------for logging----------------------------------#
    @property
    def dir_prefix(self):
        return 'Experiment'
    
    @property
    def copy_folder_list(self):
        return ["MoConVQCore"]
    
    def save_before_train(self, args):
        """build directories for log and save
        """
        import os, time, yaml
        time_now = time.strftime("%Y%m%d %H-%M-%S", time.localtime())
        dir_name = args['experiment_name']+'_'+time_now
        dir_name = mpi_comm.bcast(dir_name, root = 0)
        
        self.log_dir_name = os.path.join(self.dir_prefix,'log',dir_name)
        self.data_dir_name = os.path.join(self.dir_prefix,'checkpoint',dir_name)
        if mpi_rank == 0:
            os.makedirs(self.log_dir_name)
            os.makedirs(self.data_dir_name)

        mpi_comm.barrier()
        if mpi_rank > 0:
            f = open(os.path.join(self.log_dir_name,f'mpi_log_{mpi_rank}.txt'),'w')
            sys.stdout = f
            return
        else:
            yaml.safe_dump(args, open(os.path.join(self.data_dir_name,'config.yml'),'w'))
            self.logger = SummaryWriter(self.log_dir_name)
            for folder in self.copy_folder_list:
                os.system(f'cp -r {folder} "{os.path.join(self.data_dir_name,folder)}" ')
            # os.system(f'cp -r MoConVQCore "{os.path.join(self.data_dir_name,"MoConVQCore")}" ')
            
    def try_evaluate(self, iteration):
        if mpi_rank == 0:
            # if iteration % self.evaluate_period == 0 or self.kargs['no_train']:
            #     bvh_saver = self.runner.kinematic_eval_one_trajectory(self)
            #     bvh_saver.to_file(os.path.join(self.data_dir_name,f'{iteration}_{mpi_rank}.bvh'))
            return
     
        if iteration % self.evaluate_period == 0 or self.kargs['no_train']:
            bvh_saver = self.runner.eval_one_trajectory(self)
            bvh_saver.to_file(os.path.join(self.data_dir_name,f'{iteration}_{mpi_rank}.bvh'))
        pass    
    
    def try_save(self, iteration):
        if iteration % self.save_period ==0:
            check_point = {
                'self': self.state_dict(),
                'wm_optim': self.wm_optimizer.state_dict(),
                'vae_optim': self.vae_optimizer.state_dict(),
                'balance': self.env.val
            }
            torch.save(check_point, os.path.join(self.data_dir_name,f'{iteration}.data'))
    
    def try_load(self, data_file, strict = False):
        data = torch.load(data_file, map_location=ptu.device)
        self.load_state_dict(data['self'], strict = strict)
        self.wm_optimizer.load_state_dict(data['wm_optim'])
        self.vae_optimizer.load_state_dict(data['vae_optim'])
        # self.posterior.bottle_neck.is_training = False
        if 'balance' in data:
            self.env.val = data['balance']
            self.env.update_p()
        return data
    
    def simple_load(self, data_file, strict = False):
        data = torch.load(data_file, map_location=ptu.device)
        self.load_state_dict(data, strict = strict)
        
    def try_log(self, log, iteration):
        for key, value in log.items():
            self.logger.add_scalar(key, value, iteration)
        self.logger.flush()
    
    def cal_rwd(self, **obs_info):
        observation = obs_info['observation']
        target = obs_info['target']
        error = pose_err(torch.from_numpy(observation), torch.from_numpy(target), self.rwd_weight, dt = self.env.dt)
        error = sum(error).item()
        return np.exp(-error/20)
    
    
    @staticmethod
    def add_specific_args(arg_parser):
        arg_parser.add_argument("--latent_size", type = int, default = 64, help = "dim of latent space")
        arg_parser.add_argument("--max_iteration", type = int, default = 40001, help = "iteration for MoConVQ training")
        arg_parser.add_argument("--collect_size", type = int, default = 2048, help = "number of transition collect for each iteration")
        arg_parser.add_argument("--sub_iter", type = int, default = 8, help = "num of batch in each iteration")
        arg_parser.add_argument("--save_period", type = int, default = 500, help = "save checkpoints for every * iterations")
        arg_parser.add_argument("--evaluate_period", type = int, default = 200, help = "save checkpoints for every * iterations")
        arg_parser.add_argument("--replay_buffer_size", type = int, default = 50000, help = "buffer size of replay buffer")

        return arg_parser
    
    #--------------------------API for encode and decode------------------------------#
    
    def decode(self, normalized_obs, latent, **kargs):
        """decode latent code into action space

        Args:
            normalized_obs (tensor): normalized current observation
            latent (tensor): latent code

        Returns:
            tensor: action
        """
        action = self.agent(latent, normalized_obs)        
        return action
    

    
    def normalize_obs(self, observation):
        if isinstance(observation, np.ndarray):
            observation = ptu.from_numpy(observation)
        if len(observation.shape) == 1:
            observation = observation[None,...]
        return ptu.normalize(observation, self.obs_mean, self.obs_std)
    
    def unnormalize_obs(self, n_obs):
        if isinstance(n_obs, np.ndarray):
            n_obs = ptu.from_numpy(n_obs)
        if len(n_obs.shape) == 1:
            n_obs = n_obs[None,...]
        return ptu.unnormalize(n_obs, self.obs_mean, self.obs_std)
    
    def normalize_future(self, future, with_noise = False):
        if isinstance(future, np.ndarray):
            future = ptu.from_numpy(future)
        if len(future.shape) == 1:
            future = future[None,...]
        n_future =  ptu.normalize(future, self.future_mean, self.future_std)
        n_future = n_future + 0.3 * torch.randn_like(n_future)
        return n_future
    
    def obsinfo2n_obs(self, obs_info):
        if 'n_observation' in obs_info:
            n_observation = obs_info['n_observation']
        else:
            if 'observation' in obs_info:
                observation = obs_info['observation']
            else:
                observation = state2ob(obs_info['state'])
            n_observation = self.normalize_obs(observation)
        return n_observation
    
    
    def list2tensor(self, list, max_length, size, random = False):
        if not len(list) == max_length:
            if not random:
                pad = torch.zeros(1, max_length - len(list), size)
            else:
                pad = torch.randn(1, max_length - len(list), size)
        tensor = torch.cat(list + [pad], dim = 1)
        return tensor
    
    def obsinfo2tensor(self, obs_info):
        cur_frame = len(obs_info['observation'])
        obs_tensor = self.list2tensor(obs_info['observation'], cur_frame, self.observation_dim)
        latent_tensor = self.list2tensor(obs_info['latent'], cur_frame, self.latent_dim)
        future_tensor = self.list2tensor(obs_info['future'], cur_frame, self.future_dim)
        return {
            'observation': obs_tensor,
            'latent': latent_tensor,
            'future': future_tensor,
            'cur_frame': cur_frame
        }
    
    
    def act_tracking(self, **obs_info):
        """
        obs_info{
            obs_history
            latent_history
            future_history
            target
        }
        """
        
        observation = obs_info['obs_history'][-1]
        n_observation = self.normalize_obs(observation)
        
        
        latent = obs_info['target_latent']
        
        # print(n_observation.shape, latent.shape)
        action = self.decode(n_observation.squeeze(1), latent)
        info = {
            "mu_prior": torch.zeros_like(latent),
            "mu_post": torch.zeros_like(latent),
            "latent_code": latent
        }
        return action, info
    
    
    def get_prior(self, obs_info, noise_latent = None, **diffusion_info):
        """
        try to track reference motion
        diffusion_info{
            cur_steps:
            dt:
            recursive
            }
        """
        cur_frame = diffusion_info['cur_frame']
        observation = np.concatenate(obs_info['obs_history'][-(cur_frame+1):], axis = 1)
        observation = ptu.from_numpy(observation)
        n_observation = self.normalize_obs(observation)
        
        future = np.concatenate(obs_info['future_history'][-(cur_frame+1):], axis = 1)
        future = ptu.from_numpy(future)
        n_future = self.normalize_future(future)
        
        condition = torch.cat([n_observation, n_future], dim = -1)

        latent = ptu.from_numpy(np.concatenate(obs_info['latent_history'][-(cur_frame):], axis = 1)) if cur_frame !=0 else None

        if not diffusion_info['recursive']: # one step reconstruction
            assert False
            if noise_latent is None:
                noise_latent = torch.randn_like(latent[:,:-1,:])
            x_0 = self.prior.pred_x0(condition, latent, noise_latent, diffusion_info['cur_frame'])
        else: # gaussian to x0
            
            real_condition = condition[:,-1,:]
            _, new_mu = self.prior_mean(real_condition[:,None,:], None)
            clip_feature = diffusion_info['clip_feature'] 
            # if clip_feature is not None:
            #     _, new_mu2 = self.prior_mean(real_condition[:,None,:], clip_feature)
            # else:
            #     new_mu2 = None
            
            mu_history = obs_info['mu_history'][-(cur_frame):] if cur_frame !=0 else None
            mu_history = ptu.from_numpy(np.concatenate(mu_history, axis = 1)) if cur_frame !=0 else None
            mu = torch.cat([mu_history, new_mu], axis = 1) if cur_frame !=0 else new_mu
            noise_latent = torch.randn( (1,1,self.latent_dim), device = condition.device)
            # noise_latent = new_mu
            x_0 = self.prior.denoise_to_x0(condition, latent, noise_latent, diffusion_info['dt'], mu, clip_feature)
            noise_latent = new_mu
            
        return x_0[:, cur_frame, :], noise_latent
    
    def act_prior(self, obs_info, noise_latent = None, **diffusion_info):
        prior, mu = self.get_prior(obs_info, noise_latent, **diffusion_info)
        observation = obs_info['obs_history'][-1]
        n_observation = self.normalize_obs(observation).squeeze(1)
        # print(n_observation.shape, prior.shape, mu.shape)
        action = self.decode(n_observation, prior+mu[:,0,:])
        info = {
            "latent_code": ptu.to_numpy(prior),
            'mu_prior': ptu.to_numpy(mu)
        }
        return action, info
        
    
    #----------------------------------API imitate PPO--------------------------------#
    def act_determinastic(self, obs_info):
        action, info = self.act_tracking(**obs_info)
        return action, info
                
    def act_distribution(self, obs_info, p_prior = None, **diffusion_info):
        """
        Add noise to the output action
        """
        #TODO: prior
        if p_prior is not None and np.random.rand() < p_prior:
            action, info = self.act_prior(obs_info, **diffusion_info)
        else:
            action, info = self.act_determinastic(obs_info)
               
        action_distribution = D.Independent(D.Normal(action, self.action_sigma), -1)
        return action_distribution, info
    
    #--------------------------------------Utils--------------------------------------#
    @staticmethod
    def merge_dict(dict_list: List[dict], prefix: List[str]):
        """Merge dict with prefix, used in merge logs from different model

        Args:
            dict_list (List[dict]): different logs
            prefix (List[str]): prefix you hope to add before keys
        """
        res = {}
        for dic, prefix in zip(dict_list, prefix):
            for key, value in dic.items():
                res[prefix+'_'+key] = value
        return res
    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
    def try_load_world_model(self, data_file):
        data = torch.load(data_file, map_location=ptu.device)
        wm_dict = data['self']
        wm_dict = {k.replace('world_model.',''):v for k,v in wm_dict.items() if 'world_model' in k}
        self.world_model.load_state_dict(wm_dict)
        return data
    #--------------------------------Training submodule-------------------------------#
    
    def sample_data_for_transition(self, batch, rollout_length):
        if not hasattr(self, 'target'):
            self.target = np.array(self.env.motion_data.observation)
            self.clip_features = np.array(self.env.motion_data.clip_feature)
            self.states = np.array(self.env.motion_data.state)
            self.futures = np.array(self.env.motion_data.future)
            self.init_index = index_counter.calculate_feasible_index(self.env.motion_data.done,rollout_length) 
        idx = index_counter.sample_rollout(self.init_index, batch, rollout_length)
        targets = self.target[idx]
        targets = ptu.from_numpy(targets)
        clip_features = self.clip_features[idx]
        clip_features = ptu.from_numpy(clip_features)
        states = self.states[idx -1]
        states = ptu.from_numpy(states)
        futures = self.futures[idx]
        futures = ptu.from_numpy(futures)
        return states, targets, clip_features, futures
    
    
    
    def train_transition(self, states, futures, targets, clip_features ):
        
        _, targets, clip_features, _ = self.sample_data_for_transition(targets.shape[0], targets.shape[1])
        
        res = self.train_policy(states, futures, targets, clip_features, using_first = True)

        return res
    
    def encode_seq(self, obs, target, **kargs):
        if len(target.shape) == 2:
            target = target[None, ...]
        batch, length, _ = target.shape
        
        n_observation = self.normalize_obs(obs)
        n_target = self.normalize_obs(target).view(-1, length, 323)
        return self.posterior(n_observation, n_target)[0]
    
    def encode_seq_all(self, obs, target, **kargs):
        n_observation = self.normalize_obs(obs) if obs is not None else None
        if len(target.shape) == 2:
            target = target[None, ...]
        batch, length, _ = target.shape
        
        n_target = self.normalize_obs(target).view(-1, length, 323)
        return self.posterior(n_observation, n_target)[1]
    
    def project_and_encode(self, obs, target, **kargs):
        n_observation = self.normalize_obs(obs)
        n_target = self.normalize_obs(target).view(-1, 24, 323)
        _, info = self.posterior(n_observation, n_target)
        kinematic = info['kinematic']
        return self.posterior(n_observation, kinematic)[0]
    
    def train_policy(self, states, futures, targets, clip_features, using_first = False):
        
        rollout_length = states.shape[1]
        loss_name = ['pos', 'rot', 'vel', 'avel', 'height', 'up_dir', 'acs', 'kl']
        
        
        loss_num = len(loss_name)
        loss = list( ([] for _ in range(loss_num)) )
        k_loss = list( ([] for _ in range(loss_num)) )
        
        states = states.transpose(0,1).contiguous().to(ptu.device)
        targets = targets.transpose(0,1).contiguous().to(ptu.device)

        
        cur_state = states[0]
        cur_observation = state2ob(cur_state)
        n_observation = self.normalize_obs(cur_observation)
        n_target = self.normalize_obs(targets.transpose(0,1))
        latent_vq, info = self.posterior(n_observation, n_target)
        latent_vq = latent_vq + torch.randn_like(latent_vq)*0.05
        
        for i in range(rollout_length):
            
            target = targets[i]
            action, _ = self.act_tracking(
                obs_history = [cur_observation], 
                future_history = [futures[i]], 
                target_latent = latent_vq[:,i],
                )
            
            action = action + torch.randn_like(action)*0.05
            cur_state = self.world_model(cur_state, action, n_observation = n_observation)
            cur_observation = state2ob(cur_state)
            n_observation = self.normalize_obs(cur_observation)
            loss_tmp = pose_err(cur_observation, target, self.weight, dt = self.env.dt)
            for j, value in enumerate(loss_tmp):
                loss[j].append(value)        
            acs_loss = self.weight['l2'] * torch.mean(torch.sum(action**2,dim = -1)) \
                + self.weight['l1'] * torch.mean(torch.norm(action, p=1, dim=-1))
            loss[-2].append(acs_loss)
        
        kinematic = info['kinematic']
        # kinematic = self.unnormalize_obs(kinematic)
        k_loss_tmp = pose_err(kinematic, n_target, self.weight, dt = self.env.dt)
        k_loss = sum(k_loss_tmp) * 0.05
        
        
        
        commit_loss = info['commit_loss']
        kl_loss = commit_loss.clamp(max = 10)
        
        loss_value = [  sum( (0.95**i)*l[i] for i in range(rollout_length) )/rollout_length for l in loss[:-1]]
        loss_value.append(kl_loss)
        loss = sum(loss_value)
        
        loss = loss + k_loss
        
        self.vae_optimizer.zero_grad()
            
        loss.backward()
        for parameter in self.posterior.parameters():
            torch.nn.utils.clip_grad_norm_(parameter, 1)
        self.vae_optimizer.step()
        self.beta_scheduler.step()
        
        res = {loss_name[i]: loss_value[i] for i in range(loss_num)}
        res['loss'] = loss
        res['beta'] = self.beta_scheduler.value
        
        res['entropy'] = self.posterior.bottle_neck.entropy()
        
        res['k_loss'] = k_loss
        
        return res
        
    
    

    def train_world_model(self, states, actions):
        rollout_length = states.shape[1] -1
        loss_name = ['pos', 'rot', 'vel', 'avel']
        loss_num = len(loss_name)
        loss = list( ([] for _ in range(loss_num)) )
        states = states.transpose(0,1).contiguous().to(ptu.device)
        actions = actions.transpose(0,1).contiguous().to(ptu.device)
        cur_state = states[0]
        for i in range(rollout_length):
            next_state = states[i+1]
            n_observation = self.normalize_obs(state2ob(cur_state))
            pred_next_state = self.world_model(cur_state, actions[i+1], n_observation)
            loss_tmp = self.world_model.loss(pred_next_state, next_state)
            cur_state = pred_next_state
            for j in range(loss_num):
                loss[j].append(loss_tmp[j])
        
        loss_value = [sum(i) for i in loss]
        loss = sum(loss_value)
        
        self.wm_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1, error_if_nonfinite=True)
        self.wm_optimizer.step()
        res= {loss_name[i]: loss_value[i] for i in range(loss_num)}
        res['loss'] = loss
        return res
    