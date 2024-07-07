import os
import numpy as np
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
import platform
is_windows = "Windows" in platform.platform()
if is_windows:
    extra_args = ["/openmp", "/MP"]
    extra_link_args = []
else:
    extra_args = ["-fopenmp"]
    extra_link_args = ["-fopenmp"]

# The performance of /Ox and /O2 are similar on Visual Studio 2022.
setup(
    name="RotationCpp",
    ext_modules=[Pybind11Extension(
        "RotationCpp", ["Config.cpp", "RotationCpp.cpp"],
        language="c++",
        extra_compile_args=extra_args,
        extra_link_args=extra_link_args,
        include_dirs=[np.get_include()]
    )]
)