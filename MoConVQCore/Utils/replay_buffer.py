import numpy as np
import torch
import MoConVQCore.Utils.pytorch_utils as ptu



from MoConVQCore.Utils.index_counter import index_counter

class cycle_nd_queue():
    def __init__(self, max_size) -> None:
        self.max_size = max_size
        self.content = None

    def append(self, item: np.ndarray, b_idx, e_idx) -> None:
        
        # item is tooooooo big
        if e_idx - b_idx > self.max_size:
            print(e_idx, b_idx, self.max_size)
            raise ValueError

        if self.content is None:
            shape = list(item.shape)
            shape[0] = self.max_size
            self.content = np.ones(shape, dtype= item.dtype) * -1


        b_idx = b_idx % self.max_size
        e_idx = e_idx % self.max_size
        if b_idx > e_idx:
            l = self.max_size - b_idx
            self.content[b_idx:] = item[:l]
            self.content[:e_idx] = item[l:]
        else:
            self.content[b_idx:e_idx] = item
    
    def __getitem__(self, idx):
        
        return self.content.__getitem__(idx%self.max_size)

class ReplayBuffer(object):

    def __init__(self, keys, max_size=50000):

        self.max_size = max_size
        self.content = {}
        for key in keys:
            self.content[key] = cycle_nd_queue(max_size)
        self.content['done'] = cycle_nd_queue(max_size)
        
        self.end_idx = 0
    
    def reset_max_size(self, max_size):
        #! this function can only be called before adding data
        self.max_size = max_size
    
    def clear(self):
        self.end_idx = 0
        self.content['done'].content = np.ones_like(self.content['done'].content) * -1

    def add_trajectory(self, trajectory):
        num = trajectory['done'].shape[0]
        e_idx = self.end_idx + num
        for key,value in trajectory.items():
            if key in self.content:
                self.content[key].append(value, self.end_idx, e_idx)
        self.end_idx = e_idx % self.max_size
        
    
    def feasible_index(self, rollout_length):
        return index_counter.calculate_feasible_index(self.content['done'].content, rollout_length)
    
    def generate_data_loader(self, name_list, rollout_length, mini_batch_size, mini_batch_num):
        
        index = index_counter.sample_rollout(
                self.feasible_index(rollout_length), # ensure [i,i+rollout_length) is feasible 
                mini_batch_size* mini_batch_num, # total num of rollouts
                rollout_length 
                )
        
        res = []
        for name in name_list:
            res.append(  torch.Tensor(self.content[name][index])  )
        dataset = torch.utils.data.TensorDataset(*res)
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            mini_batch_size, 
            shuffle = False
            )
        # list(data_loader)
        return data_loader
        
    
    def generate_transition_data_loader(self, name_list, rollout_length, mini_batch_size, mini_batch_num):
        
        index = index_counter.sample_rollout(
                self.feasible_index(rollout_length), # ensure [i,i+rollout_length) is feasible 
                mini_batch_size* mini_batch_num, # total num of rollouts
                rollout_length 
                )
        
        state_idx = index_counter.sample_rollout(
                self.feasible_index(rollout_length), # ensure [i,i+rollout_length) is feasible 
                mini_batch_size* mini_batch_num, # total num of rollouts
                rollout_length 
                )
        
        res = [ torch.Tensor(self.content['state'][state_idx]) ]
        for name in name_list:
            res.append(  torch.Tensor(self.content[name][index])  )
        dataset = torch.utils.data.TensorDataset(*res)
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            mini_batch_size, 
            shuffle = False
            )
        # list(data_loader)
        return data_loader