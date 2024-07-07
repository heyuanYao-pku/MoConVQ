import numpy as np
import torch
from scipy.spatial.transform import Rotation
import DiffRotation
import time


@torch.jit.script
def torch_quat_multiply(p, q):
    w: torch.Tensor = p[:, 3:4] * q[:, 3:4] - torch.sum(p[:, :3] * q[:, :3], dim=1, keepdim=True)
    xyz: torch.Tensor = (
                p[:, None, 3] * q[:, :3] + q[:, None, 3] * p[:, :3] + torch.cross(p[:, :3], q[:, :3], dim=1))

    return torch.cat([xyz, w], dim=-1)

def test_func():
    num_quat = 512*20
    num_loop = 5000
    qa = Rotation.random(num_quat)
    qb = Rotation.random(num_quat)
    device = torch.device("cuda")
    dtype = torch.float64
    qa_t = torch.nn.Parameter(torch.as_tensor(qa.as_quat(), dtype=dtype, device=device))
    qb_t = torch.nn.Parameter(torch.as_tensor(qb.as_quat(), dtype=dtype, device=device))
    v_t = torch.nn.Parameter(torch.rand((num_quat, 3), dtype=dtype, device=device))

    qa_tt = torch.nn.Parameter(torch.as_tensor(qa.as_quat(), dtype=dtype, device=device))
    qb_tt = torch.nn.Parameter(torch.as_tensor(qb.as_quat(), dtype=dtype, device=device))
    v_tt = torch.nn.Parameter(v_t.detach().clone())

    def zero_grad1():
        qa_t.grad = None
        qb_t.grad = None
        v_t.grad = None

    def zero_grad2():
        qa_tt.grad = None
        qb_tt.grad = None
        v_tt.grad = None

    def test_quat_multiply():
        start_time = time.time()
        for i in range(num_loop):
            zero_grad1()
            ret_t = DiffRotation.quat_multiply(qa_t, qb_t)
            # loss_t = torch.sum(torch.cos(ret_t + 1))
            loss_t = torch.sum(ret_t)
            loss_t.backward()

        end_time = time.time()
        print(end_time - start_time)

        start_time = time.time()
        for i in range(num_loop):
            zero_grad2()
            # ret_tt = DiffRotation.libtorch.quat_multiply(qa_tt, qb_tt)
            ret_tt = torch_quat_multiply(qa_tt, qb_tt)
            # loss_tt = torch.sum(torch.cos(ret_tt + 1))
            loss_tt = torch.sum(ret_tt)
            loss_tt.backward()

        end_time = time.time()
        print(end_time - start_time)

        print(torch.max(ret_t - ret_tt))
        print(torch.max(qa_t.grad - qa_tt.grad))
        print(torch.max(qb_t.grad - qb_tt.grad))

        ret_np = (qa * qb).as_quat()
        print(np.max(np.abs(ret_t.detach().cpu().numpy() - ret_np)), "\n\n")

    def test_quat_apply():
        start_time = time.time()
        for i in range(num_loop):
            zero_grad1()
            ret_t = DiffRotation.quat_apply(qa_t, v_t)
            # loss_t = torch.sum(torch.cos(ret_t + 1))
            loss_t = torch.sum(ret_t)
            loss_t.backward()

        end_time = time.time()
        print(end_time - start_time)

        start_time = time.time()
        for i in range(num_loop):
            zero_grad2()
            ret_tt = DiffRotation.libtorch.quat_apply(qa_tt, v_tt)
            # loss_t = torch.sum(torch.cos(ret_t + 1))
            loss_tt = torch.sum(ret_tt)
            loss_tt.backward()

        end_time = time.time()
        print(end_time - start_time)
        print(torch.max(ret_t - ret_tt))
        print(torch.max(qa_tt.grad - qa_t.grad))
        print(torch.max(v_tt.grad - v_t.grad), "\n\n")

    def test_quat_integrate():
        start_time = time.time()
        for i in range(num_loop):
            zero_grad1()
            ret_t = DiffRotation.quat_integrate(qa_t, v_t, 0.01)
            # loss_t = torch.sum(torch.cos(ret_t + 1))
            loss_t = torch.sum(ret_t)
            loss_t.backward()

        end_time = time.time()
        print(end_time - start_time)

        start_time = time.time()
        for i in range(num_loop):
            zero_grad2()
            ret_tt = DiffRotation.libtorch.quat_integrate(qa_tt, v_tt, 0.01)
            # loss_t = torch.sum(torch.cos(ret_t + 1))
            loss_tt = torch.sum(ret_tt)
            loss_tt.backward()

        end_time = time.time()
        print(end_time - start_time)
        print(torch.max(ret_t - ret_tt))
        print(torch.max(qa_tt.grad - qa_t.grad))
        print(torch.max(v_tt.grad - v_t.grad), "\n\n")

    test_quat_multiply()
    # test_quat_apply()
    # test_quat_integrate()


if __name__ == "__main__":
    test_func()
