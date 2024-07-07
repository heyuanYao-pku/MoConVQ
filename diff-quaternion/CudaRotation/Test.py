import copy
import numpy as np
import time
import torch
from torch import nn
from scipy.spatial.transform import Rotation
import RotationCuda

from VclSimuBackend.DiffODE import DiffQuat

def test_quat_multiply():  # check OK
    batch = int(1e7)
    dtype = torch.float64
    a = torch.randn(batch, 4, device="cuda", dtype=dtype)
    a /= torch.linalg.norm(a, dim=-1, keepdim=True)
    b = torch.randn(batch, 4, device="cuda", dtype=dtype)
    b /= torch.linalg.norm(b, dim=-1, keepdim=True)
    # gt_res = (Rotation(a.cpu().numpy()) * Rotation(b.cpu().numpy())).as_quat()
    param_a, param_b = nn.Parameter(a.clone(), requires_grad=True), nn.Parameter(b.clone(), requires_grad=True)
    
    start  = time.time()
    res_cuda = RotationCuda.quat_multiply(param_a, param_b)
    loss_cuda = (1 + res_cuda).cos().sum()
    loss_cuda.backward()
    end = time.time()
    print(end - start)

    param_c, param_d = nn.Parameter(a.clone(), requires_grad=True), nn.Parameter(b.clone(), requires_grad=True)
    start = time.time()
    res_torch = DiffQuat.quat_multiply(param_c, param_d)
    loss_torch = (1 + res_torch).cos().sum()
    loss_torch.backward()
    end = time.time()
    print(end - start)

    with torch.no_grad():
        print(torch.max(torch.abs(param_c.grad - param_a.grad)))
        print(torch.max(torch.abs(param_d.grad - param_b.grad)))

def test_quat_from_matrix():
    for i in range(10):
        batch = int(1)
        dtype = torch.float64
        a = torch.randn(batch, 4, device="cpu", dtype=dtype)
        a /= torch.linalg.norm(a, dim=-1, keepdim=True)
        mat = Rotation(a.cpu().numpy()).as_matrix()

        mat_0 = nn.Parameter(torch.from_numpy(mat.copy()), True)
        qa = RotationCuda.quat_from_matrix(mat_0)
        la = (qa ** 4).sum()
        la.backward()

        mat_1 = nn.Parameter(torch.from_numpy(mat.copy()), True)
        qb = DiffQuat.quat_from_matrix(mat_1)
        lb = (qb ** 4).sum()
        lb.backward()

        with torch.no_grad():
            print(la - lb, "\n", torch.max(torch.abs(mat_0.grad - mat_1.grad)))
    

if __name__ == "__main__":
    test_quat_multiply()