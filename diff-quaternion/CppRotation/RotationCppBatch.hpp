#pragma once
#include "Config.h"
#if ROTATION_WITH_OPENMP
#include <omp.h>
#endif
#include <cmath>
#include <random>
#include <cstring>
#include "RotationCppSingle.hpp"

namespace RotationCppBatch
{
    using namespace RotationCppConfig;

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void minus_vector(const data_type * x_in, data_type * x_out, size_t dim)
    {
        for (long long i = 0; i < dim; i++)
        {
            x_out[i] = -x_in[i];
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void mat3_vec3_multiply(
        const data_type * a,
        const data_type * x,
        data_type * b,
        size_t num_mat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if (use_parallel_flag)
        #endif
        for (long long i = 0; i < num_mat; i++)
        {
            RotationCppSingle::mat3_vec3_multiply(a + 9 * i, x + 3 * i, b + 3 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void mat3_vec3_multiply_backward(
        const data_type * a,
        const data_type * x,
        const data_type * grad_in,
        data_type * grad_a,
        data_type * grad_x,
        size_t num_mat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_mat; i++)
        {
            RotationCppSingle::mat3_vec3_multiply_backward(
                a + 9 * i,
                x + 3 * i,
                grad_in + 3 * i,
                grad_a + 9 * i,
                grad_x + 3 * i
            );
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_multiply(
        const data_type * q1,
        const data_type * q2,
        data_type * q,
        size_t num_quat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i=0; i<num_quat; i++)
        {
            RotationCppSingle::quat_multiply(q1 + 4 * i, q2 + 4 * i, q + 4 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_multiply_backward(
        const data_type * q1,
        const data_type * q2,
        const data_type * grad_q,
        data_type * grad_q1,
        data_type * grad_q2,
        size_t num_quat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i=0; i<num_quat; i++)
        {
            RotationCppSingle::quat_multiply_backward(
                q1 + 4 * i, q2 + 4 * i, grad_q + 4 * i, grad_q1 + 4 * i, grad_q2 + 4 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_apply(
        const data_type * q,
        const data_type * v,
        data_type * o,
        size_t num_quat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_quat; i++)
        {
            RotationCppSingle::quat_apply(q + i * 4, v + i * 3, o + i * 3);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_apply_backward(
        const data_type * q,
        const data_type * v,
        const data_type * o_grad,
        data_type * q_grad,
        data_type * v_grad,
        size_t num_quat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_quat; i++)
        {
            RotationCppSingle::quat_apply_backward(q + i * 4, v + i * 3, o_grad + i * 3, q_grad + i * 4, v_grad + i * 3);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_inv(const data_type* q, data_type* out_q, size_t num_quat)
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for (long long i = 0; i < num_quat; i++)
        {
            RotationCppSingle::quat_inv(q + 4 * i, out_q + 4 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_inv_backward(const data_type* q, const data_type* grad_in, data_type * grad_out, size_t num_quat)
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for (long long i = 0; i < num_quat; i++)
        {
            RotationCppSingle::quat_inv_backward(q + 4 * i, grad_in + 4 * i, grad_out + 4 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void flip_quat_by_w(const data_type * q, data_type * q_out, size_t num_quat)
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_quat; i++)
        {
            data_type flag = 1;
            if (q[i * 4 + 3] < 0)
            {
                flag = -1;
            }
            for(size_t j = 0; j < 4; j++)
            {
                q_out[i * 4 + j] = flag * q[i * 4 + j];
            }
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void flip_quat_by_w_backward(
        const data_type * q,
        const data_type * grad_in,
        data_type * grad_out,
        size_t num_quat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_quat; i++)
        {
            data_type flag = 1;
            if (q[i * 4 + 3] < 0)
            {
                flag = -1;
            }
            for(size_t j = 0; j < 4; j++)
            {
                grad_out[i * 4 + j] = flag * grad_in[i * 4 + j];
            }
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_to_vec6d(const data_type * q, data_type * vec6d, size_t num_quat)
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_quat; i++)
        {
            RotationCppSingle::quat_to_vec6d(q + 4 * i, vec6d + 6 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_to_vec6d_backward(const data_type* q, const data_type* grad_in, data_type* grad_out, size_t num_quat) // TODO: check gradient
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for (long long i = 0; i < num_quat; i++)
        {
            RotationCppSingle::quat_to_vec6d_backward(q + 4 * i, grad_in + 6 * i, grad_out + 4 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_to_matrix(const data_type * q, data_type * mat, size_t num_quat)
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_quat; i++)
        {
            RotationCppSingle::quat_to_matrix(q + 4 * i, mat + 9 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_to_matrix_backward(
        const data_type * q,
        const data_type * grad_in,
        data_type * grad_out,
        size_t num_quat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_quat; i++)
        {
            RotationCppSingle::quat_to_matrix_backward(q + 4 * i, grad_in + 9 * i, grad_out + 4 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_from_matrix(const data_type * mat, data_type * q, size_t num_quat)
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_quat; i++)
        {
            RotationCppSingle::quat_from_matrix<data_type>(mat + 9 * i, q + 4 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_from_matrix_backward(
        const data_type * mat,
        const data_type * grad_in,
        data_type * grad_out,
        size_t num_quat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_quat; i++)
        {
            RotationCppSingle::quat_from_matrix_backward(mat + 9 * i, grad_in + 4 * i, grad_out + 9 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void cross_product(const data_type * a, const data_type * b, data_type * c, size_t num_vector)
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_vector; i++)
        {
            RotationCppSingle::cross_product(a + 3 * i, b + 3 * i, c + 3 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void cross_product_backward(
        const data_type * a, const data_type * b, const data_type * grad_in,
        data_type * grad_a, data_type * grad_b, size_t num_quat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_quat; i++)
        {
            RotationCppSingle::cross_product_backward(a + 3 * i, b + 3 * i, grad_in + 3 * i, grad_a + 3 * i, grad_b + 3 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void vector_to_cross_matrix(
        const data_type * vec,
        data_type * mat,
        size_t num_vec
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_vec; i++)
        {
            RotationCppSingle::vector_to_cross_matrix(vec + 3 * i, mat + 9 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void vector_to_cross_matrix_backward(
        const data_type * vec,
        const data_type * grad_in,
        data_type * grad_out,
        size_t num_vec
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_vec; i++)
        {
            RotationCppSingle::vector_to_cross_matrix_backward(vec + 3 * i, grad_in + 9 * i, grad_out + 3 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void vec6d_to_matrix_single(const data_type * vec6d, data_type * mat)
    {
        // cross product
        data_type a[3] = {vec6d[0], vec6d[2], vec6d[4]};
        data_type b[3] = {vec6d[1], vec6d[3], vec6d[5]};
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_from_vec6d_single(
        const data_type * vec6d,
        data_type * q
    )
    {
        data_type mat[9];

    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_from_vec6d(
        const data_type * mat,
        data_type * q,
        size_t num_quat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_quat; i++)
        {
            quat_from_vec6d_single(mat + 6 * i, q + 4 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_from_vec6d_backward_single(
        const data_type * vec6d,
        const data_type * grad_in,
        data_type * grad_out
    )
    {

    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_from_vec6d_backward(
        const data_type * vec6d,
        const data_type * grad_in,
        data_type * grad_out,
        size_t num_quat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_quat; i++)
        {
            quat_from_vec6d_backward_single(vec6d + 6 * i, grad_in + 4 * i, grad_out);
        }
    }

    

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_to_rotvec(const data_type * q, data_type * angle, data_type * rotvec, size_t num_quat)
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_quat; i++)
        {
            RotationCppSingle::quat_to_rotvec(q + 4 * i, angle[i], rotvec + 3 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_to_rotvec_backward(
        const data_type * q,
        const data_type * angle,
        const data_type * grad_in,
        data_type * grad_out,
        size_t num_quat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_quat; i++)
        {
            RotationCppSingle::quat_to_rotvec_backward(q + 4 * i, angle[i], grad_in + 3 * i, grad_out + 4 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_from_rotvec(
        const data_type * rotvec, data_type * q, size_t num_quat)
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_quat; i++)
        {
            RotationCppSingle::quat_from_rotvec(rotvec + 3 * i, q + 4 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_from_rotvec_backward(
        const data_type * rotvec,
        const data_type * grad_in,
        data_type * grad_out,
        size_t num_quat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_quat; i++)
        {
            RotationCppSingle::quat_from_rotvec_backward(rotvec + 3 * i, grad_in + 4 * i, grad_out + 3 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void normalize_quaternion(
        const data_type * q_in,
        data_type * q_out,
        size_t num_quat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_quat; i++)
        {
            data_type sum_value = 0.0;
            const data_type * q = q_in + 4 * i;
            data_type * o = q_out + 4 * i;
            for(int j = 0; j < 4; j++)
            {
                sum_value += q[j] * q[j];
            }
            sum_value = 1.0 / std::sqrt(sum_value);
            for(int j = 0; j < 4; j++)
            {
                o[j] = q[j] * sum_value;
            }
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void normalize_quaternion_backward(
        const data_type * q_in,
        const data_type * grad_in,
        data_type * grad_out,
        size_t num_quat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_quat; i++)
        {
            RotationCppSingle::vector_normalize_backward(q_in + 4 * i, 4, grad_in + 4 * i, grad_out + 4 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_integrate(
        const data_type * q,
        const data_type * omega,
        data_type dt,
        data_type * result,
        size_t num_quat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_quat; i++)
        {
            RotationCppSingle::quat_integrate(q + 4 * i, omega + 3 * i, dt, result + 4 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_integrate_backward(
        const data_type * q,
        const data_type * omega,
        data_type dt,
        const data_type * grad_in,
        data_type * q_grad,
        data_type * omega_grad,
        size_t num_quat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_quat; i++)
        {
            RotationCppSingle::quat_integrate_backward(q + 4 * i, omega + 3 * i, dt, grad_in + 4 * i, q_grad + 4 * i, omega_grad + 3 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void clip_vec3_arr_by_length(
        const data_type * x,
        const data_type * max_len,
        data_type * result,
        size_t num_vecs
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_vecs; i++)
        {
            RotationCppSingle::clip_vec_by_length(x + 3 * i, max_len[i], result + 3 * i, 3);
        }
    }

    

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void clip_vec3_arr_by_length_backward(
        const data_type * x,
        const data_type * max_len,
        const data_type * grad_in,
        data_type * grad_out,
        size_t num_vecs
    )
    {
        #pragma omp parallel for if(use_parallel_flag)
        for(long long i = 0; i < num_vecs; i++)
        {
            RotationCppSingle::clip_vec_by_length_backward(x + 3 * i, max_len[i], grad_in + 3 * i, grad_out + 3 * i, 3);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void mat22_det(
        const data_type * x,
        data_type * result,
        size_t num_mat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_mat; i++)
        {
            RotationCppSingle::mat22_det(x + 4 * i, result[i]);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void mat22_det_backward(
        const data_type * x,
        data_type * grad_in,
        data_type * grad_out,
        size_t num_mat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_mat; i++)
        {
            RotationCppSingle::mat22_det_backward(x + 4 * i, grad_in[i], grad_out + 4 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void mat33_det(
        const data_type * x,
        data_type * result,
        size_t num_mat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_mat; i++)
        {
            RotationCppSingle::mat33_det(x + 9 * i, result[i]);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void mat33_det_backward(
        const data_type * x,
        const data_type * grad_in,
        data_type * grad_out,
        size_t num_mat
    )
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for(long long i = 0; i < num_mat; i++)
        {
            RotationCppSingle::mat33_det_backward(x + 9 * i, grad_in[i], grad_out + 9 * i);
        }
    }


    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void mat33_svd_backward_single()
    {

    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void mat33_svd_forward()
    {

    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void mat33_svd_backward()
    {

    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_between(const data_type* a, const data_type* b, data_type* result, size_t num_vector)
    {
        #if ROTATION_WITH_OPENMP
        #pragma omp parallel for if(use_parallel_flag)
        #endif
        for (size_t i = 0; i < num_vector; i++)
        {
            RotationCppSingle::quat_between(a + 3 * i, b + 3 * i, result + 3 * i);
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_between_backward()
    {

    }
}

namespace RotationCppBatchVoid
{
    using namespace RotationCppConfig;

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void minus_vector(const void* x_in, void* x_out, size_t dim)
    {
        RotationCppBatch::minus_vector(static_cast<const data_type*>(x_in), static_cast<data_type*>(x_out), dim);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void mat3_vec3_multiply(const void* a, const void* x, void* b, size_t num_mat)
    {
        RotationCppBatch::mat3_vec3_multiply(static_cast<const data_type*>(a), static_cast<const data_type*>(x), static_cast<data_type*>(b), num_mat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void mat3_vec3_multiply_backward(const void* a, const void* x, const void* grad_in, void* grad_a, void* grad_x, size_t num_mat)
    {
        RotationCppBatch::mat3_vec3_multiply_backward(static_cast<const data_type*>(a), static_cast<const data_type*>(x), static_cast<const data_type*>(grad_in), static_cast<data_type*>(grad_a), static_cast<data_type*>(grad_x), num_mat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_multiply(const void* q1, const void* q2, void* q, size_t num_quat)
    {
        RotationCppBatch::quat_multiply(static_cast<const data_type*>(q1), static_cast<const data_type*>(q2), static_cast<data_type*>(q), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_multiply_backward(const void* q1, const void* q2, const void* grad_q, void* grad_q1, void* grad_q2, size_t num_quat)
    {
        RotationCppBatch::quat_multiply_backward(static_cast<const data_type *>(q1), static_cast<const data_type *>(q2), static_cast<const data_type *>(grad_q), static_cast<data_type *>(grad_q1), static_cast<data_type *>(grad_q2), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_apply(const void* q, const void* v, void* o, size_t num_quat)
    {
        RotationCppBatch::quat_apply(static_cast<const data_type*>(q), static_cast<const data_type*>(v), static_cast<data_type*>(o), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_apply_backward(const void* q, const void* v, const void* o_grad, void* q_grad, void* v_grad, size_t num_quat)
    {
        RotationCppBatch::quat_apply_backward(static_cast<const data_type*>(q), static_cast<const data_type*>(v), static_cast<const data_type*>(o_grad), static_cast<data_type*>(q_grad), static_cast<data_type*>(v_grad), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_inv(const void* q, void* out_q, size_t num_quat)
    {
        RotationCppBatch::quat_inv(static_cast<const data_type *> (q), static_cast<data_type *> (out_q), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_inv_backward(const void* q, const void* grad_in, void* grad_out, size_t num_quat)
    {
        RotationCppBatch::quat_inv_backward(static_cast<const data_type*> (q), static_cast<const data_type*> (grad_in), static_cast<data_type*> (grad_out), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void flip_quat_by_w(const void* q, void* q_out, size_t num_quat)
    {
        RotationCppBatch::flip_quat_by_w(static_cast<const data_type*>(q), static_cast<data_type*> (q_out), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void flip_quat_by_w_backward(const void* q, const void* grad_in, void* grad_out, size_t num_quat)
    {
        RotationCppBatch::flip_quat_by_w_backward(static_cast<const data_type *>(q), static_cast<const data_type *>(grad_in), static_cast<data_type *>(grad_out), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_to_vec6d(const void* q, void* vec6d, size_t num_quat)
    {
        RotationCppBatch::quat_to_vec6d(static_cast<const data_type *> (q), static_cast<data_type *> (vec6d), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_to_matrix(const void* q, void* mat, size_t num_quat)
    {
        RotationCppBatch::quat_to_matrix(static_cast<const data_type *> (q), static_cast<data_type *> (mat), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_to_matrix_backward(const void* q, const void* grad_in, void* grad_out, size_t num_quat)
    {
        RotationCppBatch::quat_to_matrix_backward(static_cast<const data_type *> (q), static_cast<const data_type *> (grad_in), static_cast<data_type *> (grad_out), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_from_matrix(const void* mat, void* q, size_t num_quat)
    {
        RotationCppBatch::quat_from_matrix(static_cast<const data_type *> (mat), static_cast<data_type *> (q), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_from_matrix_backward(const void* mat, const void* grad_in, void* grad_out, size_t num_quat)
    {
        RotationCppBatch::quat_from_matrix_backward(static_cast<const data_type *> (mat), static_cast<const data_type *> (grad_in), static_cast<data_type *> (grad_out), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void cross_product(const void* a, const void* b, void* c, size_t num_vector)
    {
        RotationCppBatch::cross_product(static_cast<const data_type*> (a), static_cast<const data_type*> (b), static_cast<data_type*> (c), num_vector);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void cross_product_backward(const void* a, const void* b, const void* grad_in, void* grad_a, void* grad_b, size_t num_quat)
    {
        RotationCppBatch::cross_product_backward(static_cast<const data_type *> (a), static_cast<const data_type *> (b), static_cast<const data_type *> (grad_in), static_cast<data_type *> (grad_a), static_cast<data_type *> (grad_b), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void vector_to_cross_matrix(const void* vec, void* mat, size_t num_vec)
    {
        RotationCppBatch::vector_to_cross_matrix(static_cast<const data_type *> (vec), static_cast<data_type *> (mat), num_vec);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void vector_to_cross_matrix_backward(const void* vec, const void* grad_in, void* grad_out, size_t num_vec)
    {
        RotationCppBatch::vector_to_cross_matrix_backward(static_cast<const data_type *> (vec), static_cast<const data_type *> (grad_in), static_cast<data_type *> (grad_out), num_vec);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void vec6d_to_matrix_single(const data_type* vec6d, data_type* mat)
    {
        // cross product
        data_type a[3] = { vec6d[0], vec6d[2], vec6d[4] };
        data_type b[3] = { vec6d[1], vec6d[3], vec6d[5] };
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_from_vec6d(const void* mat, void* q, size_t num_quat)
    {
        RotationCppBatch::quat_from_vec6d(static_cast<const data_type *> (mat), static_cast<data_type *>(q), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_from_vec6d_backward(const void* vec6d, const void* grad_in, void* grad_out, size_t num_quat)
    {
        RotationCppBatch::quat_from_vec6d_backward(static_cast<const data_type *> (vec6d), static_cast<const data_type *> (grad_in), static_cast<data_type *> (grad_out), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_to_rotvec(const void* q, void* angle, void* rotvec, size_t num_quat)
    {
        RotationCppBatch::quat_to_rotvec(static_cast<const data_type *> (q), static_cast<data_type *> (angle), static_cast<data_type *> (rotvec), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_to_rotvec_backward(const void* q, const void* angle, const void* grad_in, void* grad_out, size_t num_quat)
    {
        RotationCppBatch::quat_to_rotvec_backward(static_cast<const data_type *> (q), static_cast<const data_type *>(angle), static_cast<const data_type *> (grad_in), static_cast<data_type *> (grad_out), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_from_rotvec(const void* rotvec, void* q, size_t num_quat)
    {
        RotationCppBatch::quat_from_rotvec(static_cast<const data_type *> (rotvec), static_cast<data_type *> (q), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_from_rotvec_backward(const void* rotvec, const void* grad_in, void* grad_out, size_t num_quat)
    {
        RotationCppBatch::quat_from_rotvec_backward(static_cast<const data_type *> (rotvec), static_cast<const data_type *> (grad_in), static_cast<data_type *> (grad_out), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void normalize_quaternion(const void* q_in, void* q_out, size_t num_quat)
    {
        RotationCppBatch::normalize_quaternion(static_cast<const data_type *> (q_in), static_cast<data_type *> (q_out), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void normalize_quaternion_backward(const data_type* q_in, const data_type* grad_in, data_type* grad_out, size_t num_quat)
    {
        RotationCppBatch::normalize_quaternion_backward(static_cast<const data_type *> (q_in), static_cast<const data_type *> (grad_in), static_cast<data_type *> (grad_out), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_integrate(const void* q, const void* omega, data_type dt, void* result, size_t num_quat)
    {
        RotationCppBatch::quat_integrate(static_cast<const data_type*> (q), static_cast<const data_type*> (omega), dt, static_cast<data_type*> (result), num_quat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_integrate_backward(const void* q, const void* omega, data_type dt, const void* grad_in, void* q_grad, void* omega_grad, size_t num_quat)
    {
        RotationCppBatch::quat_integrate_backward(
            static_cast<const data_type *> (q),
            static_cast<const data_type *> (omega),
            dt,
            static_cast<const data_type *> (grad_in),
            static_cast<data_type *> (q_grad),
            static_cast<data_type *> (omega_grad),
            num_quat
        );
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void clip_vec3_arr_by_length(const void* x, const void* max_len, void* result, size_t num_vecs)
    {
        RotationCppBatch::clip_vec3_arr_by_length(static_cast<const data_type *> (x), static_cast<const data_type *> (max_len), static_cast<data_type *> (result), num_vecs);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void clip_vec3_arr_by_length_backward(const void* x, const void* max_len, const void* grad_in, void* grad_out, size_t num_vecs)
    {
        RotationCppBatch::clip_vec3_arr_by_length_backward(static_cast<const data_type *> (x), static_cast<const data_type *> (max_len), static_cast<const data_type *> (grad_in), static_cast<data_type *> (grad_out), num_vecs);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void mat22_det(const void* x, void* result, size_t num_mat)
    {
        RotationCppBatch::mat22_det(static_cast<const data_type*> (x), static_cast<data_type*> (result), num_mat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void mat22_det_backward(const void* x, void* grad_in, void* grad_out, size_t num_mat)
    {
        RotationCppBatch::mat22_det_backward(static_cast<const data_type *> (x), static_cast<data_type *> (grad_in), static_cast<data_type *> (grad_out), num_mat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void mat33_det(const void* x, void* result, size_t num_mat)
    {
        RotationCppBatch::mat33_det(static_cast<const data_type *> (x), static_cast<data_type *> (result), num_mat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void mat33_det_backward(const void* x, const void* grad_in, void* grad_out, size_t num_mat)
    {
        RotationCppBatch::mat33_det_backward(static_cast<const data_type *> (x), static_cast<const data_type *> (grad_in), static_cast<data_type *> (grad_out), num_mat);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void mat33_svd_backward_single()
    {

    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void mat33_svd_forward()
    {

    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void mat33_svd_backward()
    {

    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_between_single(const void* a, const void* b, void* result)
    {

    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void quat_between(const void* a, const void* b, void* result, size_t num_vector)
    {

    }
}