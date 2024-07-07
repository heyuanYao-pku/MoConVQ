#include <algorithm>
#include <exception>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "CppCudaWrapper.cuh"
#include "CudaConfig.h"
#include "DiffRotationInterface.h"


PYBIND11_MODULE(RotationCuda, m)
{
    CudaConfig::init();

    pybind11::module_ m_cuda = m.def_submodule("cuda", "config cuda setting");
    m_cuda.def("print_info", &CudaConfig::print_info);

    m.def(
        "quat_multiply",
        &DiffRotationInterface::quat_multiply,
        py::arg("q1"),
        py::arg("q2")
    );

    m.def(
        "quat_apply",
        &DiffRotationInterface::quat_apply,
        py::arg("q"),
        py::arg("v")
    );

    m.def(
        "quat_integrate",
        &DiffRotationInterface::quat_integrate,
        py::arg("q"),
        py::arg("omega"),
        py::arg("dt")
    );

    m.def(
        "quat_to_angle",
        &DiffRotationInterface::quat_to_angle,
        py::arg("q")
    );

    m.def(
        "quat_to_rotvec",
        &DiffRotationInterface::quat_to_rotvec,
        py::arg("q")
    );

    m.def(
        "quat_from_rotvec",
        &DiffRotationInterface::quat_from_rotvec,
        py::arg("x")
    );

    m.def(
        "quat_from_matrix",
        &DiffRotationInterface::quat_from_matrix,
        py::arg("x")
    );


    m.def(
        "quat_from_vec6d",
        &DiffRotationInterface::quat_from_vec6d,
        "Input: 6d representation in shape (*, 3, 2),"
        "Output: quaternion in shape (*, 4)",
        py::arg("x")
    );

    m.def(
        "quat_to_vec6d",
        &DiffRotationInterface::quat_to_vec6d,
        "Input: quaternion in shape (*, 4),"
        "Output: 6d representation in shape (*, 3, 2),",
        py::arg("q")
    );

    m.def(
        "normalize_vec6d",
        &DiffRotationInterface::normalize_vec6d,
        "Input: 6d representation in shape (*, 3, 2),\n"
        "Output: 6d representation in shape (*, 3, 2)",
        py::arg("x")
    );

    m.def(
        "mat33_det",
        &DiffRotationInterface::mat33_det,
        "Input: rotation matrix in shape (*, 3, 3),\n"
        "Output: det of matrix in shape (*)",
        py::arg("x")
    );

    m.def(
        "mat33_svd",
        &DiffRotationInterface::mat33_svd,
        "Input: rotation matrix in shape (*, 3, 3),\n"
        "Output:",
        py::arg("x")
    );

    // =================================
    /* m.def(
        "enable_debug_print",
        &Config::enable_debug_print
    );

    m.def(
        "disable_debug_print",
        &Config::disable_debug_print
    ); */
}

