import numpy as np
class index_counter():
    def __init__(self, done_flag) -> None:
        self.done_flag = done_flag
        self.cur_frame = 0
    
    @staticmethod
    def sample_rollout(feasible_index, batch_size, rollout_length):
        """generate index for rollout sampling

        Args:
            feasible_index (np.ndarray): please make sure [i,i+rollout_length) is useful
            batch_size (int): nop
            rollout_length (int): nop
        """
        begin_idx = np.random.choice(feasible_index.flatten(), [batch_size,1])
        bias = np.arange(rollout_length).reshape(1,-1)
        res_idx = begin_idx + bias
        return res_idx
    
    @staticmethod
    def calculate_feasible_index(done_flag, rollout_length):
        res_flag = np.ones_like(done_flag).astype(int)
        terminate_idx = np.where(done_flag!=0)[0].reshape(-1,1)
        bias = np.arange(rollout_length).reshape(1,-1)
        terminate_idx = terminate_idx - bias
        res_flag[terminate_idx.flatten()] = 0
        return np.where(res_flag)[0]
    
    @staticmethod
    def random_select(feasible_index, p = None):
        return np.random.choice(feasible_index, p = p)