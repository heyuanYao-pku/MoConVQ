#include "RotationCppSingle.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace RotationCpp
{
    namespace py = pybind11;
    /*
    torch::Tensor quat_multiply(torch::Tensor q1, torch::Tensor q2);
    torch::Tensor quat_apply(torch::Tensor q, torch::Tensor v);

    torch::Tensor quat_inv(torch::Tensor q);
    torch::Tensor flip_quat_by_w(torch::Tensor q);
    torch::Tensor quat_normalize(torch::Tensor q);
    torch::Tensor quat_integrate(torch::Tensor q, torch::Tensor omega, double dt);
    
    torch::Tensor quat_to_angle(torch::Tensor q);

    torch::Tensor quat_to_rotvec(torch::Tensor q);
    torch::Tensor quat_from_rotvec(torch::Tensor x);

    torch::Tensor quat_from_matrix(torch::Tensor matrix);
    torch::Tensor quat_to_matrix(torch::Tensor q);

    torch::Tensor quat_from_vec6d(torch::Tensor x);
    torch::Tensor quat_to_vec6d(torch::Tensor x);

    std::vector<torch::Tensor> normalize_vec6d(torch::Tensor x);
    torch::Tensor normalize_vec6d_cat(torch::Tensor x);
    torch::Tensor vec6d_to_matrix(torch::Tensor x);
    
    torch::Tensor vector_to_cross_matrix(torch::Tensor x);
    torch::Tensor matrix_to_angle(torch::Tensor x);
    torch::Tensor rotation_matrix_inv(torch::Tensor x);
    torch::Tensor mat22_det(torch::Tensor x);
    torch::Tensor mat33_det(torch::Tensor x);
    torch::Tensor mat33_svd(torch::Tensor x);
    torch::Tensor mat44_det(torch::Tensor x); */
}

PYBIND11_MODULE(RotationNumpy, m) {
    m.doc() = "";
    namespace py = pybind11;
    m.def("quat_multiply", nullptr, "", py::arg("q1"), py::arg("q2"));
    m.def("quat_apply", nullptr, "", py::arg("q"), py::arg("v"));
    m.def("quat_inv", nullptr, "", py::arg("q"));

    m.def("flip_quat_by_w", nullptr);
    m.def("quat_integrate", nullptr, "", py::arg("q"), py::arg("omega"), py::arg("dt"));
    m.def("quat_to_angle", nullptr, "", py::arg("q"));
    m.def("quat_to_rotvec", nullptr, "", py::arg("q"));
    m.def("quat_from_rotvec", nullptr, "", py::arg("rotvec"));

    m.def("quat_from_matrix", nullptr);
    m.def("quat_to_matrix", nullptr);
    m.def("quat_from_vec6d", nullptr);
    m.def("quat_to_vec6d", nullptr);

    m.def("normalize_vec6d", nullptr);
    m.def("normalize_vec6d_cat", nullptr);
    m.def("vec6d_to_matrix", nullptr);
    m.def("mat22_det", nullptr);
    m.def("mat33_det", nullptr);
    m.def("mat33_svd", nullptr);
    m.def("mat44_det", nullptr);
}
