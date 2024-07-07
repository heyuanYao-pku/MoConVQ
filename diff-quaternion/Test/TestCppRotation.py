import time
import numpy as np
from scipy.spatial.transform import Rotation
import RotationCpp

def test_quat_multiply():
    batch_size = int(1e8)
    # a = np.random.randn(batch_size, 4)
    # b = np.random.randn(batch_size, 4)

    a = np.empty((batch_size, 4), dtype=np.float32)
    b = np.empty((batch_size, 4), dtype=np.float32)

    # start = time.time()
    # res_scipy = (Rotation(a, False, False) * Rotation(b, False, False)).as_quat()
    # end = time.time()
    # print(end - start)

    # It seems using single thread is better than multi thread..why..
    RotationCpp.set_use_parallel_flag(1)
    for i in range(6):
        start = time.time()
        res_our = RotationCpp.quat_multiply(a, b)
        end = time.time()
        print(end - start)
        time.sleep(0.1)
    print("\n")

    RotationCpp.set_use_parallel_flag(0)
    for i in range(6):
        start = time.time()
        res_our = RotationCpp.quat_multiply(a, b)
        end = time.time()
        print(end - start)
    print("\n")

if __name__ == "__main__":
    test_quat_multiply()