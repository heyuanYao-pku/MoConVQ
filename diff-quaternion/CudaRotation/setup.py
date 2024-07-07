"""
rotation forward and backward operation on cuda
"""
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

fnames = [node for node in os.listdir(os.path.dirname(__file__)) if node.endswith(".cpp") or node.endswith(".cu")]
# fnames = ["DiffRotationWrapper.cpp", "CudaConfig.cu"]
fnames.extend(["../CppRotation/Config.cpp", ])
# fnames = [""]
setup(
    name="RotationCuda",
    ext_modules=[CUDAExtension("RotationCuda", fnames)],
    cmdclass={
        "build_ext": BuildExtension
    }
)
