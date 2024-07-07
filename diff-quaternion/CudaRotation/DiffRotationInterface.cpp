#include "DiffRotationInterface.h"
#include "CppCudaWrapper.cuh"

namespace DiffRotationInterface
{
    torch::Tensor quat_multiply(torch::Tensor q1, torch::Tensor q2)
    {
        return DiffRotationImpl::QuatMultiply::apply(q1, q2);
    }

    torch::Tensor quat_apply(torch::Tensor q, torch::Tensor v)
    {
        return DiffRotationImpl::QuatApply::apply(q, v);
    }

    torch::Tensor quat_inv(torch::Tensor q)
    {
        return DiffRotationImpl::QuatInv::apply(q);
    }

    torch::Tensor quat_integrate(torch::Tensor q, torch::Tensor omega, double dt)
    {
        return DiffRotationImpl::QuatIntegrate::apply(q, omega, dt);
    }

    torch::Tensor quat_to_angle(torch::Tensor q)
    {
        return DiffRotationImpl::QuatToAngle::apply(q);
    }

    torch::Tensor quat_to_rotvec(torch::Tensor q)
    {
        return DiffRotationImpl::QuatToRotVec::apply(q);
    }

    torch::Tensor quat_from_rotvec(torch::Tensor x)
    {
        return DiffRotationImpl::QuatFromRotVec::apply(x);
    }

    torch::Tensor quat_from_matrix(torch::Tensor x)
    {
        return DiffRotationImpl::QuatFromMatrix::apply(x);
    }

    torch::Tensor quat_from_vec6d(torch::Tensor x)
    {
        return DiffRotationImpl::QuatFromVec6d::apply(x);
    }

    torch::Tensor quat_to_vec6d(torch::Tensor x)
    {
        return DiffRotationImpl::QuatToVec6d::apply(x);
    }

    torch::Tensor normalize_vec6d(torch::Tensor x)
    {
        return DiffRotationImpl::NormalizeVec6d::apply(x);
    }

    torch::Tensor mat22_det(torch::Tensor x)
    {
        return DiffRotationImpl::Mat22Det::apply(x);
    }

    torch::Tensor mat33_det(torch::Tensor x)
    {
        return DiffRotationImpl::Mat33Det::apply(x);
    }

    torch::Tensor mat33_svd(torch::Tensor x)
    {
        return DiffRotationImpl::Mat33SVD::apply(x);
    }

    torch::Tensor mat44_det(torch::Tensor x)
    {
        torch::Tensor result;
        return result;
    }
};