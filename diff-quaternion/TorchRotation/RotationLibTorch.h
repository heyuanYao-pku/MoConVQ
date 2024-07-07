// write the rotation operation with libtorch directly.
#pragma once
#include <torch/torch.h>
#include "../Common/Common.h"
#define BUILD_ROTATION_AS_PYBIND11 1

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS 1
#endif

namespace RotationLibTorch
{
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
    torch::Tensor mat44_det(torch::Tensor x);

    class NormalizeVec6dLayer: public torch::nn::Module
    {
        public:
            torch::Tensor forward(torch::Tensor x);
    };

}
