#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "RotationCudaSingle.h"

namespace DiffRotationKernel
{
    template<typename data_type>
    __global__ void mat3_set_as_eye_forward(data_type * x, size_t num_mat)
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_mat; i += gridDim.x * blockDim.x)
        {
            DiffRotationSingleKernel::mat3_set_as_eye(x + 9 * i);
        }
    }

    template<typename data_type>
    __global__ void mat3_vec3_multiply_forward(const data_type * a, const data_type * x, data_type * b, size_t num)
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += gridDim.x * blockDim.x)
        {
            DiffRotationSingleKernel::mat3_vec3_multiply(a + 9 * i, x + 3 * i, b + 3 * i);
        }
    }

    template<typename data_type>
    __global__ void mat3_vec3_multiply_backward(const data_type * a,
        const data_type * x,
        const data_type * grad_in,
        data_type * grad_a,
        data_type * grad_x
    )
    {

    }

    template<typename data_type>
    __global__ void quat_multiply_forward(const data_type * q1, const data_type * q2, data_type * q, size_t num_quat)
    {
        // #pragma unroll
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_quat; i += gridDim.x * blockDim.x)
        {
            DiffRotationSingleKernel::quat_multiply<data_type>(q1 + 4 * i, q2 + 4 * i, q + 4 * i);
        }
    }

    template<typename data_type>
    __global__ void quat_multiply_backward(
        const data_type * q1,
        const data_type * q2,
        const data_type * grad_q,
        data_type * grad_q1,
        data_type * grad_q2,
        size_t num_quat
    )
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_quat; i += gridDim.x * blockDim.x)
        {
            DiffRotationSingleKernel::quat_multiply_backward<data_type>(
                q1 + 4 * i,
                q2 + 4 * i,
                grad_q + 4 * i,
                grad_q1 + 4 * i,
                grad_q2 + 4 * i
            );
        }
    }

    template<typename data_type>
    __global__ void quat_apply_forward(const data_type * q, const data_type * v, data_type * o, size_t num_quat)
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_quat; i += gridDim.x * blockDim.x)
        {
            DiffRotationSingleKernel::quat_apply<data_type>(q + 4 * i, v + 3 * i, o + 3 * i);
        }
    }

    template<typename data_type>
    __global__ void quat_apply_backward(
        const data_type * q,
        const data_type * v,
        const data_type * o_grad,
        data_type * q_grad,
        data_type * v_grad,
        size_t num_quat
    )
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_quat; i += gridDim.x * blockDim.x)
        {
            DiffRotationSingleKernel::quat_apply_backward<data_type>(
                q + 4 * i,
                v + 3 * i,
                o_grad + 3 * i,
                q_grad + 4 * i,
                v_grad + 3 * i
            );
        }
    }

    template<typename data_type>
    __global__ void vector_to_cross_matrix_forward(const data_type * vec, data_type * mat, size_t num_vec)
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_vec; i += gridDim.x * blockDim.x)
        {
            DiffRotationSingleKernel::vector_to_cross_matrix<data_type>(vec + 3 * i, mat + 9 * i);
        }
    }

    template<typename data_type>
    __global__ void vector_to_cross_matrix_backward(
        const data_type * vec,
        const data_type * grad_in,
        data_type * grad_out,
        size_t num_vec
    )
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_vec; i += gridDim.x * blockDim.x)
        {
            DiffRotationSingleKernel::vector_to_cross_matrix_backward<data_type>(
                vec + 3 * i, grad_in + 9 * i, grad_out + 3 * i);
        }
    }

    template<typename data_type>
    __global__ void quat_to_rotvec_forward(const data_type * q, data_type * angle, data_type * rotvec, size_t num_quat)
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_quat; i += gridDim.x * blockDim.x)
        {
            DiffRotationSingleKernel::quat_to_rotvec<data_type>(q + 4 * i, angle[i], rotvec + 3 * i);
        }
    }

    template<typename data_type>
    __global__ void quat_to_rotvec_backward(
        const data_type * q,
        data_type * angle,
        const data_type * grad_in,
        data_type * grad_out,
        size_t num_quat
    )
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_quat; i += gridDim.x * blockDim.x)
        {
            DiffRotationSingleKernel::quat_to_rotvec_backward<data_type>(q + 4 * i, angle[i], grad_in + 3 * i, grad_out + 4 * i);
        }
    }

    template<typename data_type>
    __global__ void quat_from_rotvec_forward(const data_type * rotvec, data_type * q, size_t num_quat)
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_quat; i += gridDim.x * blockDim.x)
        {
            DiffRotationSingleKernel::quat_from_rotvec<data_type>(rotvec + 3 * i, q + 4 * i);
        }
    }

    template<typename data_type>
    __global__ void quat_from_rotvec_backward(
        const data_type * rotvec,
        const data_type * grad_in,
        data_type * grad_out,
        size_t num_quat
    )
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_quat; i += gridDim.x * blockDim.x)
        {
            DiffRotationSingleKernel::quat_from_rotvec_backward<data_type>(rotvec + 3 * i, grad_in + 4 * i, grad_out + 3 * i);
        }
    }

    template<typename data_type>
    __global__ void quat_inv_forward(const data_type * q, data_type * out_q, size_t num_quat)
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_quat; i += gridDim.x * blockDim.x)
        {
            DiffRotationSingleKernel::quat_inv<data_type>(q + 4 * i, out_q + 4 * i);
        }
    }

    template<typename data_type>
    __global__ void quat_inv_backward(const data_type * q, const data_type * grad_in, data_type * grad_out, size_t num_quat)
    {
        for(size_t i = 0; i < num_quat; i++)
        {
            DiffRotationSingleKernel::quat_inv_backward<data_type>(q + 4 * i, grad_in + 4 * i, grad_out + 4 * i);
        }
    }

    template<typename data_type>
    __global__ void quat_integrate_forward(const data_type * q, const data_type * omega, data_type dt, data_type * result, size_t num_quat)
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_quat; i += gridDim.x * blockDim.x)
        {
            DiffRotationSingleKernel::quat_integrate<data_type>(q + 4 * i, omega + 3 * i, dt, result + 4 * i);
        }
    }

    template<typename data_type>
    __global__ void quat_integrate_backward(
        const data_type * q,
        const data_type * omega,
        data_type dt,
        const data_type * grad_in,
        data_type * q_grad,
        data_type * omega_grad,
        size_t num_quat
    )
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_quat; i += gridDim.x * blockDim.x)
        {
            DiffRotationSingleKernel::quat_integrate_backward<data_type>(q + 4 * i, omega + 3 * i, dt, grad_in + 4 * i, q_grad + 4 * i, omega_grad + 3 * i);
        }
    }

    /* template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    __global__ void quat_to_vec6d(const data_type* q, data_type* vec6d, size_t num_quat)
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_quat; i += gridDim.x * blockDim.x)
        {
            DiffRotationSingleKernel::quat_to_vec6d<data_type>(q + 4 * i, vec6d + 6 * i);
        }
    } */
}

namespace DiffRotationImplVoid
{
    template<typename data_type>
    void quat_multiply_forward(const void* q1, const void* q2, void* q, size_t num_quat, int grid, int block)
    {
        DiffRotationKernel::quat_multiply_forward<data_type> <<<grid, block>>> (static_cast<const data_type* >(q1), static_cast<const data_type* >(q2), static_cast<data_type*>(q), num_quat);
    }

    template<typename data_type>
    void quat_multiply_backward(const void* q1, const void* q2, const void* grad_q, void* grad_q1, void* grad_q2, size_t num_quat, int grid, int block)
    {
        DiffRotationKernel::quat_multiply_backward<data_type> <<<grid, block>>> (static_cast<const data_type*>(q1), static_cast<const data_type*>(q2), static_cast<const data_type*>(grad_q), static_cast<data_type*>(grad_q1), static_cast<data_type*>(grad_q2), num_quat);
    }

    template<typename data_type>
    void quat_apply_forward(const void * q, const void * v, void * o, size_t num_quat, int grid, int block)
    {
        DiffRotationKernel::quat_apply_forward<data_type> <<<grid, block>>> (static_cast<const data_type*>(q), static_cast<const data_type*>(v), static_cast<data_type*>(o), num_quat);
    }

    template<typename data_type>
    void quat_apply_backward(const void * q, const void * v, const void * o_grad, void * q_grad, void * v_grad, size_t num_quat, int grid, int block)
    {
        DiffRotationKernel::quat_apply_backward<data_type> <<<grid, block>>>(static_cast<const data_type*>(q), static_cast<const data_type*> (v), static_cast<const data_type*> (o_grad), static_cast<data_type*> (q_grad), static_cast<data_type*> (v_grad), num_quat);
    }

    template<typename data_type>
    void vector_to_cross_matrix_forward(const void * vec, void * mat, size_t num_vec, int grid, int block)
    {
        DiffRotationKernel::vector_to_cross_matrix_forward<data_type> <<<grid, block>>>(static_cast<const data_type*> (vec), static_cast<data_type*> (mat), num_vec);
    }

    template<typename data_type>
    void vector_to_cross_matrix_backward(const void * vec, const void * grad_in, void * grad_out, size_t num_vec, int grid, int block)
    {
        DiffRotationKernel::vector_to_cross_matrix_backward<data_type> <<<grid, block>>>(static_cast<const data_type*> (vec), static_cast<const data_type*> (grad_in), static_cast<data_type*> (grad_out), num_vec);
    }

    template<typename data_type>
    void quat_to_rotvec_forward(const void * q, void * angle, void * rotvec, size_t num_quat, int grid, int block)
    {
        data_type* angle_ptr = (angle == nullptr) ? nullptr : static_cast<data_type*> (angle);
        DiffRotationKernel::quat_to_rotvec_forward<data_type> <<<grid, block>>>(static_cast<const data_type*> (q), static_cast<data_type*> (angle), static_cast<data_type*> (rotvec), num_quat);
    }

    template<typename data_type>
    void quat_to_rotvec_backward(const void * q, void * angle, const void * grad_in, void * grad_out, size_t num_quat, int grid, int block)
    {
        DiffRotationKernel::quat_to_rotvec_backward<data_type> <<<grid, block>>>(static_cast<const data_type*> (q), static_cast<data_type*> (angle), static_cast<const data_type*> (grad_in), static_cast<data_type*> (grad_out), num_quat);
    }

    template<typename data_type>
    void quat_from_rotvec_forward(const void * rotvec, void * q, size_t num_quat, int grid, int block)
    {
        DiffRotationKernel::quat_from_rotvec_forward<data_type> <<<grid, block>>> (static_cast<const data_type*> (rotvec), static_cast<data_type*> (q), num_quat);
    }

    template<typename data_type>
    void quat_from_rotvec_backward(const void * rotvec, const void * grad_in, void * grad_out, size_t num_quat, int grid, int block)
    {
        DiffRotationKernel::quat_from_rotvec_backward<data_type> <<<grid, block>>>(static_cast<const data_type*> (rotvec), static_cast<const data_type*> (grad_in), static_cast<data_type*> (grad_out), num_quat);
    }

    template<typename data_type>
    void quat_inv_forward(const void * q, void * out_q, size_t num_quat, int grid, int block)
    {
        DiffRotationKernel::quat_inv_forward<data_type> <<<grid, block>>>(static_cast<const data_type*> (q), static_cast<data_type*> (out_q), num_quat);
    }

    template<typename data_type>
    void quat_inv_backward(const void * q, const void * grad_in, void * grad_out, size_t num_quat, int grid, int block)
    {
        DiffRotationKernel::quat_inv_backward<data_type> <<<grid, block>>>(static_cast<const data_type*> (q), static_cast<const data_type*> (grad_in), static_cast<data_type*> (grad_out), num_quat);
    }

    template<typename data_type>
    void quat_integrate_forward(const void * q, const void * omega, data_type dt, void * result, size_t num_quat, int grid, int block)
    {
        DiffRotationKernel::quat_integrate_forward<data_type> <<<grid, block>>>(static_cast<const data_type*> (q), static_cast<const data_type*> (omega), dt, static_cast<data_type*> (result), num_quat);
    }

    template<typename data_type>
    void quat_integrate_backward(const void * q, const void * omega, data_type dt, const void * grad_in, void * q_grad, void * omega_grad, size_t num_quat, int grid, int block)
    {
        DiffRotationKernel::quat_integrate_backward<data_type> <<<grid, block>>>(static_cast<const data_type*> (q), static_cast<const data_type*> (omega), dt, static_cast<const data_type*> (grad_in), static_cast<data_type*> (q_grad), static_cast<data_type*> (omega_grad), num_quat);
    }
}