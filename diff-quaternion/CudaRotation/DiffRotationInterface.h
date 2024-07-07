#pragma once
#include <torch/extension.h>

namespace DiffRotationInterface
{
    torch::Tensor quat_multiply(torch::Tensor q1, torch::Tensor q2);
    torch::Tensor quat_apply(torch::Tensor q, torch::Tensor v);
    torch::Tensor quat_inv(torch::Tensor q);

    torch::Tensor quat_integrate(torch::Tensor q, torch::Tensor omega, double dt);

    torch::Tensor quat_to_angle(torch::Tensor q);
    torch::Tensor quat_to_rotvec(torch::Tensor q);
    torch::Tensor quat_from_rotvec(torch::Tensor x);
    torch::Tensor quat_from_matrix(torch::Tensor x);
    torch::Tensor quat_from_vec6d(torch::Tensor x);
    torch::Tensor quat_to_vec6d(torch::Tensor x);
    torch::Tensor normalize_vec6d(torch::Tensor x);
    torch::Tensor mat22_det(torch::Tensor x);
    torch::Tensor mat33_det(torch::Tensor x);
    torch::Tensor mat33_svd(torch::Tensor x); 
};