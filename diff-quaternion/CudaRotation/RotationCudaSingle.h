/*!
* @file   RotationCudaSingle.h
* @brief  Forward and backward for rotation operation.
* @author szh
* @date   2023-01-28
*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

namespace DiffRotationSingleKernel
{
    /*!
     * @param  x in shape (9,)
     * @note   set as identity matrix
     */
    template<typename data_type>
    __device__ void mat3_set_as_eye(data_type * x)
    {
        x[0] = 1; x[1] = 0; x[2] = 0;
        x[3] = 0; x[4] = 1; x[5] = 0;
        x[6] = 0; x[7] = 0; x[8] = 1;
    }

    /*!
     * @param  a in shape (9,)
     * @param  x in shape (3,)
     * @param  b in shape (3,)
     * @note   compute b = Ax
     */
    template<typename data_type>
    __device__ void mat3_vec3_multiply(const data_type * a, const data_type * x, data_type * b)
    {
        for (int i = 0; i < 3; i++)
        {
            b[i] = 0;
            for(int j = 0; j < 3; j++)
            {
                b[i] += a[3 * i + j] * b[j];
            }
        }
    }

    /*!
     * @param  a in shape (9,)
     * @param  x in shape (3,)
     * @param  grad_in in shape (3,), for \nabla b.
     * @param  grad_a in shape (9,), for \nabla A = \nabla b * (\partial b / \partial A)
     * @param  grad_x in shape (3,), for \nabla x = \nabla x * (\partial b / \partial x)
     * @note   backward path for b = Ax.
     */
    template<typename data_type>
    __device__ void mat3_vec3_multiply_backward(const data_type * a, const data_type * x, const data_type * grad_in, data_type * grad_a, data_type * grad_x)
    {
        // b[0] = a[0] * x[0] + a[1] * x[1] + a[2] * x[2]
        // b[1] = a[3] * x[0] + a[4] * x[1] + a[5] * x[2]
        // b[2] = a[6] * x[0] + a[7] * x[2] + a[8] * x[2]
        grad_a[0] = grad_in[0] * x[0];
        grad_a[1] = grad_in[0] * x[1];
        grad_a[2] = grad_in[0] * x[2];
        grad_a[3] = grad_in[1] * x[0];
        grad_a[4] = grad_in[1] * x[1];
        grad_a[5] = grad_in[1] * x[2];
        grad_a[6] = grad_in[2] * x[0];
        grad_a[7] = grad_in[2] * x[1];
        grad_a[8] = grad_in[2] * x[2];

        grad_x[0] = grad_in[0] * a[0] + grad_in[1] * a[3] + grad_in[2] * a[6];
        grad_x[1] = grad_in[0] * a[1] + grad_in[1] * a[4] + grad_in[2] * a[7];
        grad_x[2] = grad_in[0] * a[2] + grad_in[1] * a[5] + grad_in[2] * a[8];
    }

    /*!
     * @param  q1, input quaternion in shape (4,), in format (x, y, z, w)
     * @param  q2, input quaternion in shape (4,), in format (x, y, z, w)
     * @param  q,  result in shape (4,), in format (x, y, z, w)
     * @note   quaternion multiply,  q = q1 * q2
     */
    template<typename data_type>
    __device__ void quat_multiply(const data_type * q1, const data_type * q2, data_type * q)
    {
        data_type x1 = q1[0], y1 = q1[1], z1 = q1[2], w1 = q1[3];
        data_type x2 = q2[0], y2 = q2[1], z2 = q2[2], w2 = q2[3];
        q[0] = + w1 * x2 - z1 * y2 + y1 * z2 + x1 * w2;
        q[1] = + z1 * x2 + w1 * y2 - x1 * z2 + y1 * w2;
        q[2] = - y1 * x2 + x1 * y2 + w1 * z2 + z1 * w2;
        q[3] = - x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2;
    }

    /*!
     * @param  q1, input quaternion in shape (4,), in format (x, y, z, w)
     * @param  q2, input quaternion in shape (4,), in format (x, y, z, w)
     * @param  grad_q, input \nabla q in shape (4,)
     * @param  grad_q1, output \nabla q1 in shape (4,). \nabla q1 = \nabla q * (\partial q / \partial q1)
     * @param  grad_q2, output \nabla q2 in shape (4,). \nabla q2 = \nabla q * (\partial q / \partial q2)
     * @note   backward path for quaternion multiply
     */
    template<typename data_type>
    __device__ void quat_multiply_backward(
        const data_type * q1,
        const data_type * q2,
        const data_type * grad_q, // \frac{\partial L}{\partial q_x, q_y, q_z, q_w}
        data_type * grad_q1,
        data_type * grad_q2
    )
    {
        data_type x1 = q1[0], y1 = q1[1], z1 = q1[2], w1 = q1[3];
        data_type x2 = q2[0], y2 = q2[1], z2 = q2[2], w2 = q2[3];

        data_type gx = grad_q[0], gy = grad_q[1], gz = grad_q[2], gw = grad_q[3];

        grad_q1[0] = + gx * w2 - gy * z2 + gz * y2 - gw * x2;
        grad_q1[1] = + gx * z2 + gy * w2 - gz * x2 - gw * y2;
        grad_q1[2] = - gx * y2 + gy * x2 + gz * w2 - gw * z2;
        grad_q1[3] = + gx * x2 + gy * y2 + gz * z2 + gw * w2;

        grad_q2[0] = + gx * w1 + gy * z1 - gz * y1 - gw * x1;
        grad_q2[1] = - gx * z1 + gy * w1 + gz * x1 - gw * y1;
        grad_q2[2] = + gx * y1 - gy * x1 + gz * w1 - gw * z1;
        grad_q2[3] = + gx * x1 + gy * y1 + gz * z1 + gw * w1;
    }

    /*!
     * @param  q, input quaternion in shape (4,), in format (x, y, z, w)
     * @param  v, input vector in shape (3,)
     * @param  o, output, rotated vector in shape (3,)
     * @note   rotate the vector through quaternion, v' = qvq^{-1}
     */
    template<typename data_type>
    __device__ void quat_apply(const data_type * q, const data_type * v, data_type * o)
    {
        data_type qx = q[0], qy = q[1], qz = q[2], qw = q[3];
        data_type vx = v[0], vy = v[1], vz = v[2];
        o[0] = qw*(2*qy*vz - 2*qz*vy) + qy*(2*qx*vy - 2*qy*vx) - qz*(-2*qx*vz + 2*qz*vx) + vx;
        o[1] = qw*(-2*qx*vz + 2*qz*vx) - qx*(2*qx*vy - 2*qy*vx) + qz*(2*qy*vz - 2*qz*vy) + vy;
        o[2] = qw*(2*qx*vy - 2*qy*vx) + qx*(-2*qx*vz + 2*qz*vx) - qy*(2*qy*vz - 2*qz*vy) + vz;
    }

    /*!
     * @param  q, input quaternion in shape (4,), in format (x, y, z, w)
     * @param  v, input vector in shape (3,)
     * @param  o_grad, input grad for \nabla o
     * @param  q_grad, output grad for \nabla q in shape (4,). \nabla q = \nabla o * (\partial o / \partial q)
     * @note   the backward path to rotate the vector through quaternion, v' = qvq^{-1}
     */
    template<typename data_type>
    __device__ void quat_apply_backward(
        const data_type * q,
        const data_type * v,
        const data_type * o_grad,
        data_type * q_grad,
        data_type * v_grad
    )
    {
        data_type qx = q[0], qy = q[1], qz = q[2], qw = q[3];
        data_type vx = v[0], vy = v[1], vz = v[2];
        q_grad[0] = o_grad[0] * (2*qy*vy + 2*qz*vz)              + o_grad[1] * (-2*qw*vz - 4*qx*vy + 2*qy*vx)    + o_grad[2] * (2*qw*vy - 4*qx*vz + 2*qz*vx);
        q_grad[1] = o_grad[0] * (2*qw*vz + 2*qx*vy - 4*qy*vx)    + o_grad[1] * (2*qx*vx + 2*qz*vz)               + o_grad[2] * (-2*qw*vx - 4*qy*vz + 2*qz*vy);
        q_grad[2] = o_grad[0] * (-2*qw*vy + 2*qx*vz - 4*qz*vx)   + o_grad[1] * (2*qw*vx + 2*qy*vz - 4*qz*vy)     + o_grad[2] * (2*qx*vx + 2*qy*vy);
        q_grad[3] = o_grad[0] * (2*qy*vz - 2*qz*vy)              + o_grad[1] * (-2*qx*vz + 2*qz*vx)              + o_grad[2] * (2*qx*vy - 2*qy*vx);
        v_grad[0] = o_grad[0] * (-2*qy*qy - 2*qz*qz + 1)         + o_grad[1] * (2*qw*qz + 2*qx*qy)               + o_grad[2] * (-2*qw*qy + 2*qx*qz);
        v_grad[1] = o_grad[0] * (-2*qw*qz + 2*qx*qy)             + o_grad[1] * (-2*qx*qx - 2*qz*qz + 1)          + o_grad[2] * (2*qw*qx + 2*qy*qz);
        v_grad[2] = o_grad[0] * (2*qw*qy + 2*qx*qz)              + o_grad[1] * (-2*qw*qx + 2*qy*qz)              + o_grad[2] * (-2*qx*qx - 2*qy*qy + 1);
    }

    /*!
     * @param  vec, input vector in shape (3,)
     * @param  mat: output matrix in shape (9,)
     * @note   vec \times x == mat @ x
     */
    template<typename data_type>
    __device__ void vector_to_cross_matrix(const data_type* vec, data_type* mat)
    {
        data_type x0 = vec[0], x1 = vec[1], x2 = vec[2];
        mat[0] = 0;   mat[1] = -x2; mat[2] = x1;
        mat[3] = x2;  mat[4] = 0;   mat[5] = -x0;
        mat[6] = -x1; mat[7] = x0;  mat[8] = 0;
    }

    template<typename data_type>
    __device__ void vector_to_cross_matrix_backward(const data_type * vec, const data_type * grad_in, data_type * grad_out)
    {
        grad_out[0] = -grad_in[5] + grad_in[7];
        grad_out[1] = grad_in[2] - grad_in[6];
        grad_out[2] = -grad_in[1] + grad_in[3];
    }

    template<typename data_type>
    __device__ void quat_to_rotvec(const data_type * q, data_type & angle, data_type * rotvec)
    {
        // first, flip the quaternion by w.
        data_type ratio = 1;
        if (q[3] < 0)
        {
            ratio = -1;
        }
        data_type qx = ratio * q[0], qy = ratio * q[1], qz = ratio * q[2], qw = ratio * q[3];
        data_type ulen = std::sqrt(qx * qx + qy * qy + qz * qz);
        angle = 2 * std::atan2(ulen, qw);
        data_type scale = 1;
        if (std::abs(angle) < static_cast<data_type>(1e-3))
        {
            data_type angle_2 = angle * angle;
            data_type angle_4 = angle_2 * angle_2;
            scale = 2 + static_cast<data_type>(1.0/12.0) * angle_2 + static_cast<data_type>(7.0 / 2880.0) * angle_4;
        }
        else
        {
            scale = angle / std::sin(0.5 * angle);
        }
        rotvec[0] = scale * qx;
        rotvec[1] = scale * qy;
        rotvec[2] = scale * qz;
    }

    template<typename data_type>
    __device__ void quat_to_rotvec_backward(const data_type * q, data_type angle, const data_type * grad_in, data_type * grad_out)
    {
        data_type ratio = data_type(1);
        if (q[3] < 0)
        {
            ratio = -data_type(1);
        }
        data_type x = ratio * q[0], y = ratio * q[1], z = ratio * q[2], w = ratio * q[3];
        data_type ulen = std::sqrt(x * x + y * y + z * z);
        data_type atan_val = data_type(0.5) * angle;
        data_type atan_val2 = atan_val * atan_val;
        data_type atan_val3 = atan_val2 * atan_val;
        data_type atan_val4 = atan_val3 * atan_val;
        data_type ulen2 = ulen * ulen;
        data_type ulen3 = ulen2 * ulen;
        if (std::abs(angle) < data_type(1e-3)) // This branch checks OK.
        {
            data_type basic = 7*atan_val4/180 + atan_val2/3 + 2;
            data_type basic0 = data_type(0.0);
            if (ulen > data_type(1e-10)) // avoid divide by zero..Note, when ulen is close to 0, we should use equivalent infinitesimal
            {
                basic0 = (7*atan_val3/15 + 2*atan_val)/(ulen*3);
            }
            else
            {
                basic0 = data_type(2.0 / 3.0);
            }
            data_type basic1 = w*basic0;

            // partial L / partial x = (partial L / partial ox) * (partial ox / partial x)
            data_type basic_xyzw[4][3] = {{
                x*x*basic1 + basic,
                y*x*basic1,
                z*x*basic1
            },
            {
                x*y*basic1,
                y*y*basic1 + basic,
                z*y*basic1
            },
            {
                x*z*basic1,
                y*z*basic1,
                z*z*basic1 + basic
            },
            {
                -x*basic0*ulen2,
                -y*basic0*ulen2,
                -z*basic0*ulen2
            }};

            for(int i=0; i<4; i++)
            {
                data_type tmp = 0;
                for(int j=0; j<3; j++)
                {
                    tmp += grad_in[j] * basic_xyzw[i][j];
                }
                grad_out[i] = tmp;
            }
        }
        else
        {
            data_type basic1 = 2*atan_val/ulen;
            data_type basic = 2*w/ulen2 + basic1 - 2*atan_val/ulen3;
            data_type basic2 = w*basic1 - 2;
            data_type basic_xyzw[4][3] = {{
                x*x*basic + basic1,
                x*y*basic,
                x*z*basic
            },
            {
                y*x*basic,
                y*y*basic + basic1,
                y*z*basic
            },
            {
                z*x*basic,
                z*y*basic,
                z*z*basic + basic1
            },
            {
                x*basic2,
                y*basic2,
                z*basic2
            }};
            for(int i=0; i<4; i++)
            {
                data_type tmp = 0;
                for(int j=0; j<3; j++)
                {
                    tmp += grad_in[j] * basic_xyzw[i][j];
                }
                grad_out[i] = tmp;
            }
        }
        if (q[3] < 0)
        {
            for(int i=0; i<4; i++)
            {
                grad_out[i] = -grad_out[i];
            }
        }
    }

    template<typename data_type>
    __device__ void quat_from_rotvec(const data_type * rotvec, data_type * q)
    {
        // q: qx, qy, qz, qw
        data_type angle = std::sqrt(rotvec[0] * rotvec[0] + rotvec[1] * rotvec[1] + rotvec[2] * rotvec[2]);
        data_type half_angle = data_type(0.5) * angle;
        data_type ratio = data_type(0.0);
        if (angle < 1e-3)
        {
            data_type angle2 = angle * angle;
            data_type angle4 = angle2 * angle2;
            ratio = 0.5 - angle2 / 48 + angle4 / 3840;
        }
        else
        {
            ratio = std::sin(half_angle) / angle;
        }
        q[0] = ratio * rotvec[0];
        q[1] = ratio * rotvec[1];
        q[2] = ratio * rotvec[2];
        q[3] = std::cos(half_angle);
    }

    template<typename data_type>
    __device__ void quat_from_rotvec_backward(
        const data_type * rotvec,
        const data_type * grad_in,
        data_type * grad_out
    )
    {
        data_type sqr_angle = rotvec[0] * rotvec[0] + rotvec[1] * rotvec[1] + rotvec[2] * rotvec[2];
        data_type angle = std::sqrt(sqr_angle);
        data_type half_angle = 0.5 * angle;
        data_type x = rotvec[0], y = rotvec[1], z = rotvec[2];
        data_type sin_half = std::sin(half_angle);
        data_type cos_half = std::cos(half_angle);
        grad_out[0] = grad_out[1] = grad_out[2] = 0;
        if (angle < 1e-3)
        {
            data_type sin_div = 0.5;
            data_type ratio_w = -0.5 * sin_div;
            data_type angle4 = angle * angle;
            data_type basic0 = sqr_angle / 960 - 24;
            data_type basic1 = 0.5 - sqr_angle / 48 + angle4 / 3840;
            //x*y*(sqr_angle/960 - 24)
            //-sqr_angle/48 + x*(x*sqr_angle/960 - x/24) + angle4/3840 + 0.5
            data_type grad_table[4][3] = {
                {x * x * basic0 + basic1, x * y * basic0         , x * z * basic0         },
                {x * y * basic0         , y * y * basic0 + basic1, y * z * basic0         },
                {x * z * basic0         , y * z * basic0         , z * z * basic0 + basic1},
                {x * ratio_w            , y * ratio_w            , z * ratio_w            }
            };
            for(int rj = 0; rj < 3; rj++)
            {
                for(int qi = 0; qi < 4; qi++)
                {
                    grad_out[rj] += grad_table[qi][rj] * grad_in[qi];
                }
            }
        }
        else
        {
            data_type angle_inv = 1.0 / angle;
            data_type sqr_angle_inv = angle_inv * angle_inv;
            data_type sin_div = sin_half * angle_inv;
            data_type ratio_base = 0.5 * sqr_angle_inv * cos_half - sqr_angle_inv * sin_div;
            data_type ratio_x = x * ratio_base;
            data_type ratio_y = y * ratio_base;
            data_type ratio_z = z * ratio_base;
            data_type ratio_w = -0.5 * sin_div;
            data_type grad_table[4][3] = {
                {x * ratio_x + sin_div, y * ratio_x, z * ratio_x}, //  partial qx / partial x, partial qx / partial y, partial qx / partial z
                {x * ratio_y, y * ratio_y + sin_div, z * ratio_y}, // partial qy / partial x, partial qy / partial y, partial qy / partial z
                {x * ratio_z, y * ratio_z, z * ratio_z + sin_div}, // partial qz / partial x, partial qz / partial y, partial qz / partial z
                {x * ratio_w, y * ratio_w, z * ratio_w} // partial qw / partial x, partial qw / partial y, partial qw / partial z
            };

            for(int rj = 0; rj < 3; rj++)
            {
                for(int qi = 0; qi < 4; qi++)
                {
                    grad_out[rj] += grad_table[qi][rj] * grad_in[qi];
                }
            }
        }
    }

    template<typename data_type>
    __device__ void quat_inv(const data_type * q, data_type * out_q)
    {
        out_q[0] = q[0];
        out_q[1] = q[1];
        out_q[2] = q[2];
        out_q[3] = -q[3];
    }

    template<typename data_type>
    __device__ void quat_inv_backward(
        const data_type * q,
        const data_type * grad_in,
        data_type * grad_out
    )
    {
        grad_out[0] = grad_in[0];
        grad_out[1] = grad_in[1];
        grad_out[2] = grad_in[2];
        grad_out[3] = -grad_in[3];
    }

    template<typename data_type>
    __device__ void quat_integrate(
        const data_type * q,
        const data_type * omega,
        data_type dt,
        data_type * result
    )
    {
        data_type omega_q[4] = {omega[0], omega[1], omega[2], 0.0}, delta_q[4], res_len = 0.0;
        quat_multiply(omega_q, q, delta_q);
        for(int i = 0; i < 4; i++)
        {
            result[i] = q[i] + 0.5 * dt * delta_q[i];
            res_len += result[i] * result[i];
        }
        res_len = 1.0 / std::sqrt(res_len);
        for(int i = 0; i < 4; i++)
        {
            result[i] = result[i] * res_len;
        }
    }

    template <typename data_type>
    __device__ void vector_normalize(
        const data_type * x,
        size_t ndim,
        data_type * result
    )
    {
        data_type tot_len = 0.0;
        for(size_t i = 0; i < ndim; i++)
        {
            tot_len += x[i] * x[i];
        }
        if (tot_len < 1e-13)
        {
            for(size_t i = 0; i < ndim; i++)
            {
                result[i] = x[i];
            }
        }
        else
        {
            tot_len = 1.0 / std::sqrt(tot_len);
            for(size_t i = 0; i < ndim; i++)
            {
                result[i] = tot_len * x[i];
            }
        }
    }

    template <typename data_type>
    __device__ void _vector_normalize_backward_with_known_tot_len(
        const data_type * x,
        size_t ndim,
        const data_type * grad_in,
        data_type * grad_out,
        data_type tot_len
    )
    {
        tot_len = std::sqrt(tot_len);
        data_type tot_len2 = tot_len * tot_len;
        data_type tot_len3 = tot_len2 * tot_len;
        data_type tot_len_inv = 1.0 / tot_len;
        data_type tot_len3_inv = 1.0 / tot_len3;

        data_type sum_value = 0.0;
        for(size_t i = 0; i < ndim; i++)
        {
            sum_value += x[i] * grad_in[i];
        }
        sum_value = -tot_len3_inv * sum_value;
        for(size_t i = 0; i < ndim; i++)
        {
            grad_out[i] = grad_in[i] * tot_len_inv + sum_value * x[i];
        }
    }

    template<typename data_type>
    __device__ void vector_normalize_backward(
        const data_type * x,
        size_t ndim,
        const data_type * grad_in,
        data_type * grad_out
    )
    {
        data_type tot_len = 0.0;
        for(size_t i = 0; i < ndim; i++)
        {
            tot_len += x[i] * x[i];
        }
        if (tot_len < 1e-13)
        {
            for(size_t i = 0; i < ndim; i++)
            {
                grad_out[i] = grad_in[i];
            }
        }
        else
        {
            _vector_normalize_backward_with_known_tot_len(x, ndim, grad_in, grad_out, tot_len);
        }
    }

    template<typename data_type>
    __device__ void quat_integrate_backward(
        const data_type * q,
        const data_type * omega,
        data_type dt,
        const data_type * grad_in,
        data_type * q_grad,
        data_type * omega_grad
    )
    {
        data_type omega_q[4] = {omega[0], omega[1], omega[2], 0.0}, delta_q[4], result[4];
        quat_multiply(omega_q, q, delta_q);
        for(int i = 0; i < 4; i++)
        {
            result[i] = q[i] + 0.5 * dt * delta_q[i];
        }

        // 1. compute normalize gradient, that is, (partial L / partial result) = (partial L / partial final_result) * (partial final_result / partial )
        data_type result_grad[4], delta_q_grad[4], delta_q_grad_ratio = 0.5 * dt;
        vector_normalize_backward(result, 4, grad_in, result_grad);

        // 2. compute add gradient  result[i] = q[i] + delta_q[i];
        for(int i = 0; i < 4; i++) q_grad[i] = result_grad[i];
        for(int i = 0; i < 4; i++) delta_q_grad[i] = delta_q_grad_ratio * result_grad[i];

        // 3. compute quaternion multiply gradient
        data_type omega_q_grad[4], q_tmp_grad[4];
        quat_multiply_backward(omega_q, q, delta_q_grad, omega_q_grad, q_tmp_grad);
        for(int i = 0; i < 4; i++) q_grad[i] += q_tmp_grad[i];

        // 4. compute omega gradient
        for(int i = 0; i < 3; i++) omega_grad[i] = omega_q_grad[i];
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    __device__ void clip_vec(const data_type* x,
        data_type min_val,
        data_type max_val,
        data_type* result,
        size_t ndim
    )
    {
        for (size_t i = 0; i < ndim; i++)
        {
            if (x[i] < min_val) result[i] = min_val;
            else if (x[i] > max_val) result[i] = max_val;
            else result[i] = x[i];
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    __device__ void clip_vec_by_length(
        const data_type* x,
        data_type max_len,
        data_type* result,
        size_t ndim
    )
    {
        data_type tot_len = 0;
        for (size_t i = 0; i < ndim; i++)
        {
            tot_len += x[i] * x[i];
        }
        tot_len = std::sqrt(tot_len);
        if (tot_len <= max_len)
        {
            for (size_t i = 0; i < ndim; i++)
            {
                result[i] = x[i];
            }
        }
        else
        {
            // clip the result..
            tot_len = max_len / tot_len;
            for (size_t i = 0; i < ndim; i++)
            {
                result[i] = tot_len * x[i];
            }
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    __device__ void clip_vec_by_length_backward(
        const data_type* x,
        data_type max_len,
        const data_type* grad_in,
        data_type* grad_out,
        size_t ndim
    )
    {
        data_type tot_len = 0;
        for (size_t i = 0; i < ndim; i++)
        {
            tot_len += x[i] * x[i];
        }
        if (tot_len <= max_len * max_len)
        {
            for (size_t i = 0; i < ndim; i++)
            {
                grad_out[i] = grad_in[i];
            }
        }
        else
        {
            _vector_normalize_backward_with_known_tot_len(x, ndim, grad_in, grad_out, tot_len);
            for (size_t i = 0; i < ndim; i++)
            {
                grad_out[i] *= max_len;
            }
        }
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    __device__ void mat22_det(const data_type* x, data_type& res)
    {
        res = x[0] * x[3] - x[1] * x[2];
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    __device__ void mat22_det_backward(
        const data_type* x,
        data_type grad_in,
        data_type* grad_out
    )
    {
        grad_out[0] = grad_in * x[3];
        grad_out[1] = grad_in * -x[2];
        grad_out[2] = grad_in * -x[1];
        grad_out[3] = grad_in * x[0];
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    __device__ void mat33_det(const data_type* x, data_type& result)
    {
        data_type a11 = x[0], a12 = x[1], a13 = x[2];
        data_type a21 = x[3], a22 = x[4], a23 = x[5];
        data_type a31 = x[6], a32 = x[7], a33 = x[8];
        result = +a11 * a22 * a33\
            + a12 * a23 * a31\
            + a13 * a21 * a32\
            - a31 * a22 * a13\
            - a32 * a23 * a11\
            - a21 * a12 * a33;
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    __device__ void mat33_det_backward(
        const data_type* x,
        data_type grad_in,
        data_type* grad
    )
    {
        data_type a11 = x[0], a12 = x[1], a13 = x[2];
        data_type a21 = x[3], a22 = x[4], a23 = x[5];
        data_type a31 = x[6], a32 = x[7], a33 = x[8];
        grad[0] = a22 * a33 - a23 * a32;
        grad[1] = -a21 * a33 + a23 * a31;
        grad[2] = a21 * a32 - a22 * a31;
        grad[3] = -a12 * a33 + a13 * a32;
        grad[4] = a11 * a33 - a13 * a31;
        grad[5] = -a11 * a32 + a12 * a31;
        grad[6] = a12 * a23 - a13 * a22;
        grad[7] = -a11 * a23 + a13 * a21;
        grad[8] = a11 * a22 - a12 * a21;
        for (int i = 0; i < 9; i++)
        {
            grad[i] *= grad_in;
        }
    }

    // SVD decompose
    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    __device__ void mat33_svd(data_type* x)
    {

    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    __device__ void mat33_inverse(const data_type* x, data_type* res)
    {
        data_type a11 = x[0], a12 = x[1], a13 = x[2];
        data_type a21 = x[3], a22 = x[4], a23 = x[5];
        data_type a31 = x[6], a32 = x[7], a33 = x[8];

        data_type det = mat33_det(x);
        if (std::abs(det) < data_type(1e-8))
        {
            throw std::logic_error("The input matrix is singular.");
        }
        data_type det_inv = 1 / det;

        res[0] = det_inv * (a22 * a33 - a23 * a32);
        res[1] = det_inv * (-a12 * a33 + a13 * a32);
        res[2] = det_inv * (a12 * a23 - a13 * a22);
        res[3] = det_inv * (-a21 * a33 + a23 * a31);
        res[4] = det_inv * (a11 * a33 - a13 * a31);
        res[5] = det_inv * (-a11 * a23 + a13 * a21);
        res[6] = det_inv * (a21 * a32 - a22 * a31);
        res[7] = det_inv * (-a11 * a32 + a12 * a31);
        res[8] = det_inv * (a11 * a22 - a12 * a21);
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    __device__ void mat33_inverse_backward(const data_type* x, const data_type* grad_in, data_type* grad_out)
    {

    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    __device__ data_type mat44_det(const data_type* x)
    {
        data_type a00 = x[0], a01 = x[1], a02 = x[2], a03 = x[3];
        data_type a10 = x[4], a11 = x[5], a12 = x[6], a13 = x[7];
        data_type a20 = x[8], a21 = x[9], a22 = x[10], a23 = x[11];
        data_type a30 = x[12], a31 = x[13], a32 = x[14], a33 = x[15];
        data_type det = a00 * a11 * a22 * a33\
            - a00 * a11 * a23 * a32\
            - a00 * a12 * a21 * a33\
            + a00 * a12 * a23 * a31\
            + a00 * a13 * a21 * a32\
            - a00 * a13 * a22 * a31\
            - a01 * a10 * a22 * a33\
            + a01 * a10 * a23 * a32\
            + a01 * a12 * a20 * a33\
            - a01 * a12 * a23 * a30\
            - a01 * a13 * a20 * a32\
            + a01 * a13 * a22 * a30\
            + a02 * a10 * a21 * a33\
            - a02 * a10 * a23 * a31\
            - a02 * a11 * a20 * a33\
            + a02 * a11 * a23 * a30\
            + a02 * a13 * a20 * a31\
            - a02 * a13 * a21 * a30\
            - a03 * a10 * a21 * a32\
            + a03 * a10 * a22 * a31\
            + a03 * a11 * a20 * a32\
            - a03 * a11 * a22 * a30\
            - a03 * a12 * a20 * a31\
            + a03 * a12 * a21 * a30;
        return det;
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    __device__ void mat44_det_backward(const data_type* x, const data_type grad_in, data_type* grad)
    {
        data_type a00 = x[0], a01 = x[1], a02 = x[2], a03 = x[3];
        data_type a10 = x[4], a11 = x[5], a12 = x[6], a13 = x[7];
        data_type a20 = x[8], a21 = x[9], a22 = x[10], a23 = x[11];
        data_type a30 = x[12], a31 = x[13], a32 = x[14], a33 = x[15];
        grad[0] = a11 * a22 * a33 - a11 * a23 * a32 - a12 * a21 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a13 * a22 * a31;
        grad[1] = -a10 * a22 * a33 + a10 * a23 * a32 + a12 * a20 * a33 - a12 * a23 * a30 - a13 * a20 * a32 + a13 * a22 * a30;
        grad[2] = a10 * a21 * a33 - a10 * a23 * a31 - a11 * a20 * a33 + a11 * a23 * a30 + a13 * a20 * a31 - a13 * a21 * a30;
        grad[3] = -a10 * a21 * a32 + a10 * a22 * a31 + a11 * a20 * a32 - a11 * a22 * a30 - a12 * a20 * a31 + a12 * a21 * a30;
        grad[4] = -a01 * a22 * a33 + a01 * a23 * a32 + a02 * a21 * a33 - a02 * a23 * a31 - a03 * a21 * a32 + a03 * a22 * a31;
        grad[5] = a00 * a22 * a33 - a00 * a23 * a32 - a02 * a20 * a33 + a02 * a23 * a30 + a03 * a20 * a32 - a03 * a22 * a30;
        grad[6] = -a00 * a21 * a33 + a00 * a23 * a31 + a01 * a20 * a33 - a01 * a23 * a30 - a03 * a20 * a31 + a03 * a21 * a30;
        grad[7] = a00 * a21 * a32 - a00 * a22 * a31 - a01 * a20 * a32 + a01 * a22 * a30 + a02 * a20 * a31 - a02 * a21 * a30;
        grad[8] = a01 * a12 * a33 - a01 * a13 * a32 - a02 * a11 * a33 + a02 * a13 * a31 + a03 * a11 * a32 - a03 * a12 * a31;
        grad[9] = -a00 * a12 * a33 + a00 * a13 * a32 + a02 * a10 * a33 - a02 * a13 * a30 - a03 * a10 * a32 + a03 * a12 * a30;
        grad[10] = a00 * a11 * a33 - a00 * a13 * a31 - a01 * a10 * a33 + a01 * a13 * a30 + a03 * a10 * a31 - a03 * a11 * a30;
        grad[11] = -a00 * a11 * a32 + a00 * a12 * a31 + a01 * a10 * a32 - a01 * a12 * a30 - a02 * a10 * a31 + a02 * a11 * a30;
        grad[12] = -a01 * a12 * a23 + a01 * a13 * a22 + a02 * a11 * a23 - a02 * a13 * a21 - a03 * a11 * a22 + a03 * a12 * a21;
        grad[13] = a00 * a12 * a23 - a00 * a13 * a22 - a02 * a10 * a23 + a02 * a13 * a20 + a03 * a10 * a22 - a03 * a12 * a20;
        grad[14] = -a00 * a11 * a23 + a00 * a13 * a21 + a01 * a10 * a23 - a01 * a13 * a20 - a03 * a10 * a21 + a03 * a11 * a20;
        grad[15] = a00 * a11 * a22 - a00 * a12 * a21 - a01 * a10 * a22 + a01 * a12 * a20 + a02 * a10 * a21 - a02 * a11 * a20;
        for (int i = 0; i < 16; i++)
        {
            grad[i] *= grad_in;
        }
    }
}
