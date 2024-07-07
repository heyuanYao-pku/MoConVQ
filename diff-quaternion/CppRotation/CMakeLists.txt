cmake_minimum_required(VERSION 3.16)
project(CppRotation LANGUAGES CXX CUDA)

set(CUDNN_INCLUDE_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/include")
# CUDNN_LIBRARY:FILEPATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/lib/cudnn.lib"

find_package(Python REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories("D:/Program Files/Python310/Lib/site-packages/torch/include")
include_directories("D:/Program Files/Python310/Lib/site-packages/torch/include/torch/csrc/api/include")
# include_directories("D:/downloads/libtorch-win-shared-with-deps-debug-1.13.0+cu117/libtorch/include")

execute_process(
    COMMAND
        ${Python_EXECUTABLE} -c
            "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE DPYTHON_INCLUDE_DIRS)
set(PYTHON_INCLUDE_DIRS "${DPYTHON_INCLUDE_DIRS}")
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
execute_process(
    COMMAND
        ${Python_EXECUTABLE} -c
            "import torch.utils; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE DCMAKE_PREFIX_PATH)

set(CMAKE_PREFIX_PATH "${DCMAKE_PREFIX_PATH}")
message(${CMAKE_PREFIX_PATH})

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# aux_source_directory("." SRCS)
add_executable(main RotationCpp.cpp)
# add_library(add2 SHARED ${SRCS})
target_link_libraries(main ${PYTHON_LIBRARY} "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")
