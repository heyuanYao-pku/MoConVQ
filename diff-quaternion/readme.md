Author: Zhenhua Song(zhenhuasong@stu.pku.edu.cn)
# Rotation operation in cuda and cpu

开发环境:
NVIDIA 1060显卡,
Windows 10,
Visual Studio 2019,
Python 3.8 64位 (python.org下载的),
Pytorch 1.9.0+cu111,
cuda 11.1,
cudnn 8.0.5.39 (cudnn是否用得到我忘记了...好像没有cudnn会报warning来着? 从NVIDIA官网下载, 跟cuda版本保持一致就好)

其实系统cuda版本, 和pytorch对应的cuda版本, 相同就行

还没有试过多显卡的情况

用Linux+gcc+nvcc我还没试过, 应该没有问题

安装方式: pip install -e .
使用方式: 参考Test.py.
import torch # 这个一定要写在前面
import DiffRotation
会自动判断Tensor所在的device(cuda或cpu)

只有 quat_multiply, quat_apply, quat_integrate 写完了, 其他的部分下一次摸鱼的时候实现

编译时会报一堆warning, 我并不知道怎么去掉

TODO:
1. 能够支持对输入tensor做broadcast
2. cpu实现版本加上openmp并行
