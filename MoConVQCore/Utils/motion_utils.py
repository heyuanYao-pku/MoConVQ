import numpy as np
import torch
import typing
import VclSimuBackend
try:
    from VclSimuBackend.pymotionlib.Utils import quat_product
    from VclSimuBackend.Common.MathHelper import MathHelper
    from VclSimuBackend.ODESim import BodyInfoState
except ImportError:
    quat_product = VclSimuBackend.pymotionlib.Utils.quat_product
    BodyInfoState = VclSimuBackend.ODESim.BodyInfoState
    
from diff_quat import *
import diff_quat as DiffRotation
from scipy.spatial.transform import Rotation as R

import RotationLibTorch as special_rot

# import DiffRotation as DiffRotation # a cuda speed up, but not significant...
#----------------------------------State Utils----------------------------------------------#

@torch.jit.script
def broadcast_quat_apply(q: torch.Tensor, vec3: torch.Tensor):
    t = 2 * torch.linalg.cross(q[..., :3], vec3, dim=-1)
    xyz: torch.Tensor = vec3 + q[..., 3, None] * t + torch.linalg.cross(q[..., :3], t, dim=-1)
    return xyz
@torch.jit.script
def broadcast_quat_multiply(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    multiply 2 quaternions. p.shape == q.shape
    """
    
    w: torch.Tensor = p[..., 3:4] * q[..., 3:4] - torch.sum(p[..., :3] * q[..., :3], dim=-1, keepdim=True)
    xyz: torch.Tensor = (
                p[...,3,None] * q[..., :3] + q[..., 3, None] * p[..., :3] + torch.linalg.cross(p[..., :3], q[..., :3], dim=-1))

    return torch.cat([xyz, w], dim=-1)

def resample(old_pos, old_rot, cur_pos, cur_rot, dt):
    '''
        resample velocity and angular velocity to dt using backward finite difference
    '''
    vel = (cur_pos - old_pos)/dt
    sign = np.sign((cur_rot * old_rot).sum(axis=-1, keepdims=True))
    qd = (cur_rot * sign - old_rot)/dt
    q_conj = old_rot.copy()
    q_conj[..., :3] *= -1
    avel = 2* quat_product(qd.reshape(-1,4), q_conj)
    return vel, avel[...,:3]   

def character_pos_rot(character):
    pos = character.body_info.get_body_pos()
    rot = character.body_info.get_body_quat()
    mask = rot[...,3] < 0
    mask[mask == False] = -1
    rot = rot * mask[...,None]
    return pos, rot

def character_state(character, old_state = None, dt = None):
    '''
        return full state: {pos, rot, vel, avel}
        :param old_state: if old state is not None, it will try to recompute avel 
        and vel according to dt, otherwise it will just return current vel and avel 
    '''
    pos, rot = character_pos_rot(character)
    if old_state is not None:
        # recompute vel
        assert dt is not None
        old_pos, old_rot = old_state[:,:3], old_state[:,3:7]
        vel, avel = resample(old_pos, old_rot, pos, rot, dt)
    else:
        vel = character.body_info.get_body_velo()
        avel = character.body_info.get_body_ang_velo()
    state = np.concatenate([pos, rot, vel, avel], axis=-1, dtype=np.float32)
    return state

def state_to_BodyInfoState(state):
    res = BodyInfoState.BodyInfoState()
    res.pos = np.ascontiguousarray(state[..., 0:3].flatten(), dtype=np.float64)
    res.quat = np.ascontiguousarray(state[..., 3:7].flatten(), dtype=np.float64)
    res.linear_vel = np.ascontiguousarray(state[..., 7:10].flatten(), dtype=np.float64)
    res.angular_vel = np.ascontiguousarray(state[..., 10:13].flatten(), dtype=np.float64)
    res.rot = np.ascontiguousarray(Rotation.from_quat(state[...,3:7].reshape(-1,4)).as_matrix().flatten(), dtype=np.float64)
    return res

def decompose_state(state):
    assert state.shape[-1] ==13
    
    return state[...,0:3], state[...,3:7], state[...,7:10], state[...,10:13]

#---------------------------Observation Utils--------------------------------------------#
def state2ob_old(states):
    '''
    :param states: full state
    :return: observation { local{pos, rot, vel, avel}, height, up_dir}
    '''

    if len(states.shape) == 2:
        states = states[None]
    assert len(states.shape) == 3, "state shape error"
    batch_size = states.shape[0]
    # assert states.shape == (batch_size, 20, 13)

    batch_size, num_body, _ = states.shape

    pos = states[..., 0:3].view(-1, 3)
    rot = states[..., 3:7].view(-1, 4)
    vel = states[..., 7:10].view(-1, 3)
    avel = states[..., 10:13].view(-1, 3)

    root_pos = torch.tile(states[:, 0, 0:3], [1, 1, num_body]).view(-1, 3)
    root_rot = torch.tile(states[:, 0, 3:7], [1, 1, num_body]).view(-1, 4)
    root_rot_inv = quat_inv(root_rot)

    local_pos = quat_apply(root_rot_inv, pos - root_pos).view(batch_size, -1)
    local_vel = quat_apply(root_rot_inv, vel).view(batch_size, -1)
    local_rot = quat_to_matrix(flip_quat_by_w(quat_multiply(root_rot_inv, rot
                                             )))
    # Why here must be reshape ????
    local_rot = torch.transpose(local_rot, -2, -1)[:,:2,:].reshape(batch_size,-1)
    
    local_avel = quat_apply(root_rot_inv, avel).view(batch_size, -1)
    height = pos[..., 1].view(batch_size, -1)
    up_dir = torch.as_tensor([0, 1, 0]).view(-1, 3).tile([batch_size, 1]).float().to(states.device)
    local_up_dir = quat_apply(root_rot_inv[::num_body,:], up_dir).view(batch_size, -1)

    if batch_size == 1:
        local_pos = local_pos.flatten()
        local_rot = local_rot.flatten()
        local_vel = local_vel.flatten()
        local_avel = local_avel.flatten()
        height = height.flatten()
        local_up_dir = local_up_dir.flatten()
    return torch.cat([local_pos, local_rot, local_vel, local_avel, height, local_up_dir], dim=-1)

# @torch.jit.script
def state2ob(states):
    # needs pytorch >= 11.0
    if len(states.shape) == 2:
        states = states[None]
    batch_size, num_body, _ = states.shape

    pos = states[..., 0:3]
    rot = states[..., 3:7]
    vel = states[..., 7:10]
    avel = states[..., 10:13]

    root_pos = pos[:,0,:].view(-1,1,3)
    root_rot_inv = quat_inv( special_rot.y_decompose(rot[:,0,:]).view(-1,1,4)  )
    # root_rot_inv = quat_inv( rot[:,0,:].view(-1,1,4)  )
    local_pos = broadcast_quat_apply(root_rot_inv, pos - root_pos ).view(batch_size, -1)
    local_vel = broadcast_quat_apply(root_rot_inv, vel).view(batch_size, -1)
    local_avel = broadcast_quat_apply(root_rot_inv, avel).view(batch_size, -1)
    local_rot_ = flip_quat_by_w(broadcast_quat_multiply(root_rot_inv, rot
                                             )).view(-1,4)
    local_rot = quat_to_vec6d(local_rot_).view(batch_size,-1)
    # Why here must be reshape ????
    
    height = pos[..., 1].view(batch_size, -1)
    up_dir = torch.as_tensor([0, 1, 0]).view(1,1, 3).float().to(states.device)
    
    local_up_dir = broadcast_quat_apply(
        local_rot_.view(-1,num_body, 4)[:,0,:].view(-1,1,4), up_dir.view(1,1,3).float().to(root_rot_inv.device)).view(-1,3)

    if batch_size == 1:
        local_pos = local_pos.flatten()
        local_rot = local_rot.flatten()
        local_vel = local_vel.flatten()
        local_avel = local_avel.flatten()
        height = height.flatten()
        local_up_dir = local_up_dir.flatten()
    return torch.cat([local_pos, local_rot, local_vel, local_avel, height, local_up_dir], dim=-1)
    
# add jit will be slower... why...?
# @torch.jit.script
def decompose_obs(obs):
    num_dim = obs.shape[-1]
    assert (num_dim - 3) % 16 == 0, "dim error"
    num_body = (num_dim - 3)//16
    pos = obs[...,0:3*num_body]
    rot = obs[...,3*num_body:9*num_body]
    vel = obs[...,9*num_body:12*num_body]
    avel = obs[...,12*num_body:15*num_body]
    height = obs[...,15*num_body:16*num_body]
    up_dir = obs[...,16*num_body:]
    return pos, rot, vel, avel, height, up_dir
    
@torch.jit.script
def pose_err(obs, target, weight:typing.Dict[str, float], dt:float = 1/20):
    
    target = target.view(obs.shape)
    assert obs.shape == target.shape

    delta_pos, delta_rot, delta_vel, delta_avel, delta_height, delta_up_dir = decompose_obs(obs - target)
    
    weight_pos, weight_vel, weight_rot, weight_avel = weight[
        "pos"], weight["vel"], weight["rot"], weight["avel"]

    weight_height, weight_up_dir, weight_l2, weight_l1 = weight[
        "height"], weight["up_dir"], weight["l2"], weight["l1"]

    pos_loss = weight_pos * \
        torch.mean(torch.norm(delta_pos, p=1, dim=-1))
    rot_loss = weight_rot * \
        torch.mean(torch.norm(delta_rot, p=1, dim=-1))
    vel_loss = weight_vel * \
        torch.mean(torch.norm(delta_vel, p=1, dim=-1))
    avel_loss = weight_avel * \
        torch.mean(torch.norm(delta_avel, p=1, dim=-1))
    height_loss = weight_height * \
        torch.mean(torch.norm(delta_height, p=1, dim=-1))
    up_dir_loss = weight_up_dir * \
        torch.mean(torch.norm(delta_up_dir, p=1, dim=-1))
    
    return pos_loss, rot_loss, dt*vel_loss, \
        dt*avel_loss, height_loss, up_dir_loss

def extract_future_pos(root_pos, root_rot, bone_pos, frames = [6, 12, 18]):
    rot = R.from_quat(root_rot).inv()
    res = []
    for t in frames:
        idx = np.arange(bone_pos.shape[0]) + t
        idx[idx >= bone_pos.shape[0]] = bone_pos.shape[0] - 1
        pos = rot.apply(bone_pos[idx] - root_pos)
        res.append(pos[:,[0,2]])
    return np.concatenate(res, axis = 1)

def extract_future_rot(root_rot, bone_rot, frames = [6, 12, 18]):
    rot = R.from_quat(root_rot).inv()
    res = []
    for t in frames:
        idx = np.arange(bone_rot.shape[0]) + t
        idx[idx >= bone_rot.shape[0]] = bone_rot.shape[0] - 1
        direction = (rot*R.from_quat(bone_rot[idx])).apply(np.array([0,0,1]))
        res.append(direction[:,[0,2]])
    return np.concatenate(res, axis = 1)


def states2future(states):
    root_pos = states[:,0,0:3]
    root_rot = states[:,0,3:7]
    
    future_pos = extract_future_pos(root_pos, root_rot, root_pos)
    future_rot = extract_future_rot(root_rot, root_rot)

    return np.concatenate([future_pos, future_rot], axis = -1)
    
