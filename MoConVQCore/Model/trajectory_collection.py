# import numpy as np
# import torch
# import operator
# from ..Utils import pytorch_utils as ptu

# class TrajectorCollector():
#     def __init__(self, **kargs) -> None:
#         self.reset(**kargs)
    
#     def reset(self, venv, **kargs):
#         # set environment and actor
#         self.env = venv
        
#         # set property
#         self.with_noise = kargs['runner_with_noise']
        
        
#     # @torch.no_grad
#     def trajectory_sampling(self, sample_size, actor):
#         cnt = 0
#         res = []
#         while cnt < sample_size:
#             trajectory =self.sample_one_trajectory(actor) 
#             res.append(trajectory)
#             cnt+= len(trajectory['done'])
#             # print(cnt)
#         # res = functools.reduce(operator.add, map(collections.Counter, res))
#         res_dict = {}
#         for key in res[0].keys():
#             res_dict[key] = np.concatenate( list(map(operator.itemgetter(key), res)) , axis = 0)
#         return res_dict
    
#     def eval_one_trajectory(self, actor):
#         saver = self.env.get_bvh_saver()
#         begin_frame = 0
#         observation, info = self.env.reset()
#         transformer_info = None
#         observation_history = []
#         future_history = []
#         latent_history = []
#         mu_history = []
        
#         cur_step = 0
#         while True: 
            
#             saver.append_no_root_to_buffer()
#             observation_history.append(observation['observation'].reshape(1,1,323))
#             # observation['future'][...,0] = 0.2
#             # observation['future'][...,2] = 0.4
#             # observation['future'][...,4] = 0.6
#             future_history.append(observation['future'].reshape(1,1,-1))
#             # future_history.append(np.zeros_like(observation['future'].reshape(1,1,-1)))
            
#             observation_info = {
#                 'obs_history':  observation_history[-24:],
#                 'future_history': future_history[-24:],
#                 'latent_history': latent_history[-24:],
#                 'mu_history': mu_history[-24:],
#                 'target': observation['target'],
#             }
#             # when eval, we do not hope noise...
#             action, info = actor.act_prior(observation_info, 
#                                            dt = 20,
#                                            cur_frame = (cur_step % 12) + (12 if cur_step > 12 else 0),
#                                            recursive = True
#                                            )
#             latent_history.append(info['latent_code'].reshape(1,1,-1))
#             mu_history.append(info['mu_prior'].reshape(1,1,-1))
#             action = ptu.to_numpy(action).flatten()
#             new_observation, rwd, done, info = self.env.step(action, count_func= lambda: cur_step+1)
#             observation = new_observation
#             cur_step += 1
#             if done or cur_step > 300:
#                 break
#         return saver
    
#     # @torch.no_grad
#     def sample_one_trajectory(self, actor):
#         observation, info = self.env.reset() 
        
#         states, targets, actions, rwds, dones, frame_nums = [[] for i in range(6)]
#         futures= []
        
#         observation_history = []
#         future_history = []
#         latent_history = []
#         mu_history = []
        
#         cur_step = 0
#         while True: 
#             observation_history.append(observation['observation'].reshape(1,1,323))
#             future_history.append(np.zeros_like(observation['future']).reshape(1,1,-1))
            
#             observation_info = {
#                 'obs_history':  observation_history[-24:],
#                 'future_history': future_history[-24:],
#                 'latent_history': latent_history[-24:],
#                 'mu_history': mu_history[-24:],
#                 'target': observation['target'],
#             }
            
#             action_distribution, info = actor.act_distribution(observation_info, 0.1,
#                                                             dt = 50,
#                                                             cur_frame = cur_step % 24,
#                                                             recursive = True
#                                                             )
#             action = action_distribution.sample()
            
            
#             action = ptu.to_numpy(action).flatten()
            
#             states.append(observation['state']) 
#             actions.append(action.flatten())
#             targets.append(observation['target']) 
#             futures.append(observation['future'])
            
#             # for seq input
#             latent_code = info['latent_code']
#             mu = info['mu_prior']
#             if isinstance(mu, torch.Tensor):
#                 mu = ptu.to_numpy(mu)
#             mu_history.append(mu.reshape(1,1,-1))
#             if isinstance(latent_code, torch.Tensor):
#                 latent_code = ptu.to_numpy(latent_code)
#             latent_history.append(latent_code.reshape(1,1,-1))
            
#             new_observation, rwd, done, info = self.env.step(action)
            
#             rwd = actor.cal_rwd(observation = new_observation['observation'], target = observation['target'])
#             rwds.append(rwd)
#             dones.append(done)
#             frame_nums.append(info['frame_num'])
            
#             observation = new_observation
#             cur_step += 1
#             if done:
#                 break
#         return {
#             'state': states,
#             'action': actions,
#             'future': futures,
#             'target': targets,
#             'done': dones,
#             'rwd': rwds,
#             'frame_num': frame_nums
#         }
            
            
            
import numpy as np
import torch
import operator
from ..Utils import pytorch_utils as ptu
from MoConVQCore.Utils.motion_utils import state2ob

class TrajectorCollector():
    def __init__(self, **kargs) -> None:
        self.reset(**kargs)
    
    def reset(self, venv, **kargs):
        # set environment and actor
        self.env = venv
        
        # set property
        self.with_noise = kargs['runner_with_noise']
        self.clip_feature = kargs['clip_feature']
        
        
    # @torch.no_grad
    def trajectory_sampling(self, sample_size, actor):
        cnt = 0
        res = []
        while cnt < sample_size:
            trajectory =self.sample_one_trajectory(actor) 
            res.append(trajectory)
            cnt+= len(trajectory['done'])
            # print(cnt)
        # res = functools.reduce(operator.add, map(collections.Counter, res))
        res_dict = {}
        for key in res[0].keys():
            res_dict[key] = np.concatenate( list(map(operator.itemgetter(key), res)) , axis = 0)
        return res_dict
    
    def generate_one_trajectory(self, actor):
        saver = self.env.get_bvh_saver()
        observation, info = self.env.reset()
        transformer_info = None
        observation_history = []
        future_history = []
        latent_history = []
        mu_history = []
        
        cur_step = 0
        
        future_length = 12
        
        begin_obs = observation['observation']
        while True: 
            observation_history.append(observation['observation'].reshape(1,1,-1))
            if cur_step % future_length == 0:
                target = self.env.motion_data.observation[self.env.counter: self.env.counter + 24]
                obs = observation['observation']
                seq_latent = actor.encode_seq(obs, target, obs_history = observation_history[-(24-future_length):],

                                              )
                # print(seq_latent.shape)
            saver.append_no_root_to_buffer()
            
            future_history.append(observation['future'].reshape(1,1,-1))
            
            observation_info = {
                'obs_history':  observation_history[-24:],
                'future_history': future_history[-24:],
                'latent_history': latent_history[-24:],
                'mu_history': mu_history[-24:],
                'target': observation['target'],
                'target_latent': seq_latent[:,cur_step % future_length]
            }
            action, info = actor.act_tracking(**observation_info)
            latent_history.append(info['latent_code'].reshape(1,1,-1))
            mu_history.append(info['mu_prior'].reshape(1,1,-1))
            action = ptu.to_numpy(action).flatten()
            new_observation, rwd, done, info = self.env.step(action)
            observation = new_observation
            cur_step += 1
            if cur_step > 300:
                break
        return saver
    
    def eval_one_trajectory(self, actor):
        saver = self.env.get_bvh_saver()
        observation, info = self.env.reset()
        transformer_info = None
        observation_history = []
        future_history = []
        latent_history = []
        mu_history = []
        
        cur_step = 0
        
        test_case = 24
        begin_obs = observation['observation']
        while True: 
            if cur_step % test_case == 0:
                target = self.env.motion_data.observation[self.env.counter: self.env.counter + 24]
                obs = observation['observation']
                # obs = begin_obs
                seq_latent = actor.encode_seq(obs, target)
                # print(seq_latent.shape)
            saver.append_no_root_to_buffer()
            observation_history.append(observation['observation'].reshape(1,1,-1))
            
            future_history.append(observation['future'].reshape(1,1,-1))
            
            observation_info = {
                'obs_history':  observation_history[-24:],
                'future_history': future_history[-24:],
                'latent_history': latent_history[-24:],
                'mu_history': mu_history[-24:],
                'target': observation['target'],
                'target_latent': seq_latent[:,cur_step % test_case]
            }
            action, info = actor.act_tracking(**observation_info)
            latent_history.append(info['latent_code'].reshape(1,1,-1))
            mu_history.append(info['mu_prior'].reshape(1,1,-1))
            action = ptu.to_numpy(action).flatten()
            new_observation, rwd, done, info = self.env.step(action)
            observation = new_observation
            cur_step += 1
            if done:
                break
        return saver
    
    def kinematic_eval_one_trajectory(self, actor):
        saver = self.env.get_bvh_saver()
        observation, info = self.env.reset()
        transformer_info = None
        observation_history = []
        latent_history = []
        mu_history = []
        
        cur_step = 0
        
        test_case = 24
        state = ptu.from_numpy(observation['state'])[None,...]
        
        actor.posterior.bottle_neck.is_training = False
        
        while True: 
            if cur_step % test_case == 0:
                target = self.env.motion_data.observation[self.env.counter: self.env.counter + 24]
                obs = observation['observation']
                # obs = begin_obs
                seq_latent = actor.encode_seq(obs, target)
                # print(seq_latent.shape)
            self.env.load_character_state(self.env.sim_character, ptu.to_numpy(state[0]))
            saver.append_no_root_to_buffer()
            observation_history.append(observation['observation'].reshape(1,1,-1))
            
            
            observation_info = {
                'obs_history':  observation_history[-24:],
                'target_latent': seq_latent[:,cur_step % test_case]
            }
            action, info = actor.act_tracking_kinematic(**observation_info)
            
            state = actor.kinematic_world_model(state, action, n_observation = actor.normalize_obs(observation['observation']))
            observation['observation'] = state2ob(state)
            cur_step += 1
            if cur_step >=96:
                break
        actor.posterior.bottle_neck.is_training = True
        return saver
    
    # @torch.no_grad
    def sample_one_trajectory(self, actor):
        observation, info = self.env.reset() 
        
        states, targets, actions, rwds, dones, frame_nums = [[] for i in range(6)]
        futures= []
        
        observation_history = []
        future_history = []
        latent_history = []
        mu_history = []
        clip_features = []
        
        cur_step = 0
        
        
        while True: 
            if cur_step % 24 == 0:
                
                target = self.env.motion_data.observation[self.env.counter: self.env.counter + 24]
                obs = observation['observation']
                seq_latent = actor.encode_seq(obs, target, using_post_p = 1)
                seq_latent = seq_latent + torch.randn_like(seq_latent) * 0.05
            observation_history.append(observation['observation'].reshape(1,1,-1))
            future_history.append(observation['future'].reshape(1,1,-1))
            
            observation_info = {
                'obs_history':  observation_history[-24:],
                'future_history': future_history[-24:],
                'latent_history': latent_history[-24:],
                'mu_history': mu_history[-24:],
                'target': observation['target'],
                'target_latent': seq_latent[:, cur_step % 24]
            }
            clip_feature = ptu.from_numpy(observation['clip_feature']) if np.random.rand() < 0.5 else None
                
            action_distribution, info = actor.act_distribution(observation_info, 0,
                                                            dt = 100,
                                                            cur_frame = cur_step % 24,
                                                            recursive = True,
                                                            clip_feature= clip_feature
                                                            )
            action = action_distribution.sample()
            
            
            action = ptu.to_numpy(action).flatten()
            
            states.append(observation['state']) 
            actions.append(action.flatten())
            targets.append(observation['target']) 
            futures.append(observation['future'])
            clip_features.append(observation['clip_feature'])
            
            # for seq input
            latent_code = info['latent_code']
            mu = info['mu_prior']
            if isinstance(mu, torch.Tensor):
                mu = ptu.to_numpy(mu)
            mu_history.append(mu.reshape(1,1,-1))
            if isinstance(latent_code, torch.Tensor):
                latent_code = ptu.to_numpy(latent_code)
            latent_history.append(latent_code.reshape(1,1,-1))
            
            new_observation, rwd, done, info = self.env.step(action)
            
            rwd = actor.cal_rwd(observation = new_observation['observation'], target = observation['target'])
            rwds.append(rwd)
            dones.append(done)
            frame_nums.append(info['frame_num'])
            
            observation = new_observation
            cur_step += 1
            if done:
                break
        return {
            'state': states,
            'action': actions,
            'future': futures,
            'target': targets,
            'done': dones,
            'rwd': rwds,
            'clip_feature': clip_features,
            'frame_num': frame_nums
        }
            
            
            
            