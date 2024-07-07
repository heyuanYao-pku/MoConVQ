"""
rotation forward and backward operation on libtorch
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

fnames = ["../Common/Common.cpp", "RotationLibTorch.cpp"]
setup(
    name="RotationLibTorch",
    version="0.1",
    description="An implementation of rotation operation on LibTorch",
    author="Zhenhua Song",
    author_email="szh2016sdu@qq.com",
    ext_modules=[CUDAExtension("RotationLibTorch", fnames)],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Environment :: Console",
        "Topic :: Scientific/Engineering :: Physics",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Operating System :: Linux",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10"
    ],
    cmdclass={"build_ext": BuildExtension}
)
