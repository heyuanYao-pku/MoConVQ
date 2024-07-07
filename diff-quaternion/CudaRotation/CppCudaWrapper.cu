#pragma once
#include <algorithm>
#include <exception>
#include <vector>
#include <torch/extension.h>
#include <torch/autograd.h>
#include "DiffRotation.cuh"
#include "CppCudaWrapper.cuh"
#include "CudaConfig.h"
#include "../CppRotation/RotationCppBatch.hpp"


namespace DiffRotationImpl
{
    using torch::autograd::Function;
    using torch::autograd::variable_list;
    using torch::autograd::AutogradContext;

    const char * DTYPE_NOT_MATCH = "dtype doesn't match";
    const char * DTYPE_ONLY_FLOAT = "only support dtype == torch.float32 or dtype == torch.float64";
    const char * NOT_CONTIGUOUS = "The Tensor is not contiguous";
    const char * NOT_SAME_CUDA_DEVICE = "The Tensor should on the same cuda device";
    inline void check_condition(bool condition, const char * err_info)
    {
        if (!condition)
        {
            throw std::logic_error(err_info);
        }
    }

    inline void check_input_pair(const torch::Tensor a, const torch::Tensor b)
    {
        const auto dtype = a.dtype();
        check_condition(torch::kFloat32 == dtype || torch::kFloat64 == dtype, DTYPE_ONLY_FLOAT);
        check_condition(b.dtype() == dtype, DTYPE_NOT_MATCH);
        const auto a_dev = a.device(), b_dev = b.device(); // here the device should match..
        check_condition(a_dev == b_dev, NOT_SAME_CUDA_DEVICE);
    }

    // 
    bool is_requires_grad(const variable_list & inputs)
    {
        for(size_t i = 0; i < inputs.size(); i++)
        {
            if (inputs[i].requires_grad())
            {
                return true;
            }
        }
        return false;
    }

    size_t get_batch_size(const torch::Tensor & x, int offset = 1)
    {
        auto sizes = x.sizes();
        auto ret = std::accumulate(sizes.begin(), sizes.end() - offset, 1, std::multiplies<size_t>());
        return ret;
    }

    torch::Tensor QuatMultiply::forward(
        AutogradContext * ctx,
        torch::Tensor q1,
        torch::Tensor q2
    )
    {
        {
            check_input_pair(q1, q2);
            auto broadcast = torch::broadcast_tensors({ q1, q2 });
            q1 = broadcast[0];
            q2 = broadcast[1];
            if (!q1.is_contiguous()) q1 = q1.contiguous();
            if (!q2.is_contiguous()) q2 = q2.contiguous();
        }
        
        variable_list input_list = {q1, q2};
        bool requires_grad = is_requires_grad(input_list);
        if (requires_grad) ctx->save_for_backward(input_list);
        torch::Tensor result = torch::empty_like(q1); // alloc memory for the result
        result.set_requires_grad(requires_grad);

        // compute forward on devices.
        auto device = result.device();
        auto dtype = result.dtype();
        int batch_size = get_batch_size(result);
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());
            if (dtype == torch::kFloat32) DiffRotationImplVoid::quat_multiply_forward<float>(q1.data_ptr(), q2.data_ptr(), result.data_ptr(), batch_size, grid, block);
            else if (dtype == torch::kFloat64) DiffRotationImplVoid::quat_multiply_forward<double>(q1.data_ptr(), q2.data_ptr(), result.data_ptr(), batch_size, grid, block);
            else throw std::runtime_error("dtype not supported."); // this case will not occur.
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32) RotationCppBatchVoid::quat_multiply<float>(q1.data_ptr(), q2.data_ptr(), result.data_ptr(), batch_size);
            else if (dtype == torch::kFloat64) RotationCppBatchVoid::quat_multiply<double>(q1.data_ptr(), q2.data_ptr(), result.data_ptr(), batch_size);
            else throw std::runtime_error("dtype not supported."); // this case will not occur.
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }

        return result;
    }

    variable_list QuatMultiply::backward(AutogradContext * ctx, variable_list grad_output)
    {
        variable_list saved_input = ctx->get_saved_variables(); // TODO: do we need to clean the saved variables manully..?
        variable_list ret_grad;  // allocate for memory
        for(int i = 0; i < 2; i++)
        {
            ret_grad.emplace_back(torch::empty_like(saved_input[i]));
        }

        // compute backward on devices.
        auto device = saved_input[0].device();
        torch::Tensor grad_q = grad_output[0].contiguous();
        auto dtype = grad_q.dtype();
        int batch_size = get_batch_size(grad_q);
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());
            if (dtype == torch::kFloat32)DiffRotationImplVoid::quat_multiply_backward<float>(saved_input[0].data_ptr(), saved_input[1].data_ptr(), grad_q.data_ptr(), ret_grad[0].data_ptr(), ret_grad[1].data_ptr(), batch_size, grid, block);
            else if (dtype == torch::kFloat64) DiffRotationImplVoid::quat_multiply_backward<double>(saved_input[0].data_ptr(), saved_input[1].data_ptr(), grad_q.data_ptr(), ret_grad[0].data_ptr(), ret_grad[1].data_ptr(), batch_size, grid, block);
            else throw std::runtime_error("dtype not supported.");
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32) RotationCppBatchVoid::quat_multiply_backward<float>(saved_input[0].data_ptr(), saved_input[1].data_ptr(), grad_q.data_ptr(), ret_grad[0].data_ptr(), ret_grad[1].data_ptr(), batch_size);
            else if (dtype == torch::kFloat64) RotationCppBatchVoid::quat_multiply_backward<double>(saved_input[0].data_ptr(), saved_input[1].data_ptr(), grad_q.data_ptr(), ret_grad[0].data_ptr(), ret_grad[1].data_ptr(), batch_size);
            else throw std::runtime_error("dtype not supported.");
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return ret_grad;
    }

    torch::Tensor QuatApply::forward(
        AutogradContext * ctx,
        torch::Tensor q,
        torch::Tensor v
    )
    {
        if(!q.is_contiguous()) q = q.contiguous();
        if (!v.is_contiguous()) v = v.contiguous();

        variable_list input_list = {q, v};
        bool requires_grad = is_requires_grad(input_list);
        if (requires_grad) ctx->save_for_backward(input_list);
        torch::Tensor result = torch::empty_like(v);
        result.set_requires_grad(requires_grad);

        int batch_size = get_batch_size(q);
        auto device = q.device();
        auto dtype = q.dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());
            if (dtype == torch::kFloat32) DiffRotationImplVoid::quat_apply_forward<float>(q.data_ptr(), v.data_ptr(), result.data_ptr(), batch_size, grid, block);
            else if (dtype == torch::kFloat64)  DiffRotationImplVoid::quat_apply_forward<double>(q.data_ptr(), v.data_ptr(), result.data_ptr(), batch_size, grid, block);
            else throw std::runtime_error("dtype not supported.");
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32) RotationCppBatchVoid::quat_apply<float>(q.data_ptr(), v.data_ptr(), result.data_ptr(), batch_size);
            else if (dtype == torch::kFloat64) RotationCppBatchVoid::quat_apply<double>(q.data_ptr(), v.data_ptr(), result.data_ptr(), batch_size);
            else throw std::runtime_error("dtype not supported.");
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return result;
    }

    variable_list QuatApply::backward(AutogradContext * ctx, variable_list grad_output)
    {
        variable_list saved_input = ctx->get_saved_variables();
        variable_list ret_grad;  // allocate for memory
        for(int i = 0; i < 2; i++)
        {
            ret_grad.emplace_back(torch::empty_like(saved_input[i]));
        }

        torch::Tensor grad_o = grad_output[0];
        if (!grad_o.is_contiguous()) grad_o = grad_o.contiguous();
        auto device = grad_o.device();
        auto dtype = grad_o.dtype();
        int batch_size = get_batch_size(grad_o);

        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());
            if (dtype == torch::kFloat32) DiffRotationImplVoid::quat_apply_backward<float>(saved_input[0].data_ptr(), saved_input[1].data_ptr(), grad_o.data_ptr(), ret_grad[0].data_ptr(), ret_grad[1].data_ptr(), batch_size, grid, block);
            else if (dtype == torch::kFloat64) DiffRotationImplVoid::quat_apply_backward<double>(saved_input[0].data_ptr(), saved_input[1].data_ptr(), grad_o.data_ptr(), ret_grad[0].data_ptr(), ret_grad[1].data_ptr(), batch_size, grid, block);
            else throw std::runtime_error("dtype not supported.");
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32) RotationCppBatchVoid::quat_apply_backward<float>(saved_input[0].data_ptr(), saved_input[1].data_ptr(), grad_o.data_ptr(), ret_grad[0].data_ptr(), ret_grad[1].data_ptr(), batch_size);
            else if (dtype == torch::kFloat64) RotationCppBatchVoid::quat_apply_backward<double>(saved_input[0].data_ptr(), saved_input[1].data_ptr(), grad_o.data_ptr(), ret_grad[0].data_ptr(), ret_grad[1].data_ptr(), batch_size);
            else throw std::runtime_error("dtype not supported.");
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return ret_grad;
    }

    torch::Tensor QuatInv::forward(AutogradContext * ctx, torch::Tensor q)
    {
        torch::Tensor result = torch::empty_like(q);
        bool requires_grad = q.requires_grad();
        result.set_requires_grad(requires_grad);
        if (requires_grad) ctx->save_for_backward({ q });
        if (! q.is_contiguous()) q = q.contiguous();
        int batch_size = get_batch_size(q);
        auto device = q.device();
        auto dtype = q.dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());
            if (dtype == torch::kFloat32) DiffRotationImplVoid::quat_inv_forward<float>(q.data_ptr(), result.data_ptr(), batch_size, grid, block);
            else if (dtype == torch::kFloat64) DiffRotationImplVoid::quat_inv_forward<double>(q.data_ptr(), result.data_ptr(), batch_size, grid, block);
            else throw std::runtime_error("dtype not supported.");
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32) RotationCppBatchVoid::quat_inv<float>(q.data_ptr(), result.data_ptr(), batch_size);
            else if (dtype == torch::kFloat64) RotationCppBatchVoid::quat_inv<double>(q.data_ptr(), result.data_ptr(), batch_size);
            else throw std::runtime_error("dtype not supported.");
        }
        return result;
    }

    variable_list QuatInv::backward(AutogradContext * ctx, variable_list grad_output)
    {
        torch::Tensor grad = grad_output[0];
        if (!grad.is_contiguous()) grad = grad.contiguous();
        int batch_size = get_batch_size(grad);
        auto device = grad.device();
        auto dtype = grad.dtype();
        torch::Tensor q = ctx->get_saved_variables()[0];
        torch::Tensor result = torch::empty_like(q);
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());
            if (dtype == torch::kFloat32) DiffRotationImplVoid::quat_inv_backward<float>(q.data_ptr(), grad.data_ptr(), result.data_ptr(), batch_size, grid, block);
            else if (dtype == torch::kFloat64) DiffRotationImplVoid::quat_inv_backward<double>(q.data_ptr(), grad.data_ptr(), result.data_ptr(), batch_size, grid, block);
            else throw std::runtime_error("dtype not supported.");
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32) RotationCppBatchVoid::quat_inv_backward<float>(q.data_ptr(), grad.data_ptr(), result.data_ptr(), batch_size);
            else if (dtype == torch::kFloat64) RotationCppBatchVoid::quat_inv_backward<double>(q.data_ptr(), grad.data_ptr(), result.data_ptr(), batch_size);
            else throw std::runtime_error("dtype not supported.");
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return {result};
    }

    torch::Tensor QuatIntegrate::forward(AutogradContext* ctx, torch::Tensor q, torch::Tensor omega, double dt)
    {
        torch::Tensor result = torch::empty_like(q);
        bool requires_grad = q.requires_grad();
        result.set_requires_grad(q.requires_grad());
        if (!q.is_contiguous()) q = q.contiguous();
        if (!omega.is_contiguous()) omega = omega.contiguous();
        if (requires_grad)
        {
            variable_list input_list = { q, omega, torch::tensor(dt) };
            ctx->save_for_backward(input_list);
        }
        int batch_size = get_batch_size(q);
        auto device = q.device();
        auto dtype = q.dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());
            if (dtype == torch::kFloat32) DiffRotationImplVoid::quat_integrate_forward<float>(q.data_ptr(), omega.data_ptr(), dt, result.data_ptr(), batch_size, grid, block);
            else if (dtype == torch::kFloat64) DiffRotationImplVoid::quat_integrate_forward<double>(q.data_ptr(), omega.data_ptr(), dt, result.data_ptr(), batch_size, grid, block);
            else throw std::runtime_error("dtype not supported.");
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32) RotationCppBatchVoid::quat_integrate<float>(q.data_ptr(), omega.data_ptr(), dt, result.data_ptr(), batch_size);
            else if (dtype == torch::kFloat64) RotationCppBatchVoid::quat_integrate<double>(q.data_ptr(), omega.data_ptr(), dt, result.data_ptr(), batch_size);
            else throw std::runtime_error("dtype not supported.");
        }
        return result;
    }

    variable_list QuatIntegrate::backward(AutogradContext * ctx, variable_list grad_output)
    {
        torch::Tensor grad_o = grad_output[0];
        if (!grad_o.is_contiguous()) grad_o = grad_o.contiguous();

        int batch_size = get_batch_size(grad_o);
        auto device = grad_o.device();
        auto dtype = grad_o.dtype();
        variable_list saved_input = ctx->get_saved_variables();
        double dt = saved_input[2].item().toDouble();
        variable_list ret_grad(3);
        for (int i = 0; i < 2; i++) ret_grad[i] = torch::empty_like(saved_input[i]);
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            if (dtype == torch::kFloat32) DiffRotationImplVoid::quat_integrate_backward<float>(saved_input[0].data_ptr(), saved_input[1].data_ptr(), dt, grad_o.data_ptr(), ret_grad[0].data_ptr(), ret_grad[1].data_ptr(), batch_size, grid, block);
            else if (dtype == torch::kFloat64) DiffRotationImplVoid::quat_integrate_backward<double>(saved_input[0].data_ptr(), saved_input[1].data_ptr(), dt, grad_o.data_ptr(), ret_grad[0].data_ptr(), ret_grad[1].data_ptr(), batch_size, grid, block);
            else throw std::runtime_error("dtype not supported.");
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32) RotationCppBatchVoid::quat_integrate_backward<float>(saved_input[0].data_ptr(), saved_input[1].data_ptr(), dt, grad_o.data_ptr(), ret_grad[0].data_ptr(), ret_grad[1].data_ptr(), batch_size);
            else if (dtype == torch::kFloat64) RotationCppBatchVoid::quat_integrate_backward<float>(saved_input[0].data_ptr(), saved_input[1].data_ptr(), dt, grad_o.data_ptr(), ret_grad[0].data_ptr(), ret_grad[1].data_ptr(), batch_size);
            else throw std::runtime_error("dtype not supported.");
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return ret_grad;
    }

    torch::Tensor QuatToAngle::forward(AutogradContext * ctx, torch::Tensor q)
    {
        throw std::runtime_error("Not implement ");
        if (!q.is_contiguous()) q = q.contiguous();
        auto q_shape = q.sizes().vec();
        q_shape.pop_back();
        auto option = q.options();
        torch::Tensor result = torch::empty(q_shape, q.options());
        
        int batch_size = get_batch_size(q);
        auto device = q.device();
        auto dtype = q.dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            if (dtype == torch::kFloat32) 
            {
                
            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32); // RotationCppBatchVoid::quat_to_angle()
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        return result;
    }

    variable_list QuatToAngle::backward(AutogradContext * ctx, variable_list grad_output)
    {
        torch::Tensor result;
        torch::Tensor grad_o = grad_output[0].contiguous();
        int batch_size = get_batch_size(grad_o);
        auto device = grad_o.device();
        auto dtype = grad_o.dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return {result};
    }

    torch::Tensor QuatToRotVec::forward(AutogradContext * ctx, torch::Tensor q)
    {
        bool requires_grad = q.requires_grad();
        if (!q.is_contiguous()) q = q.contiguous();
        auto q_shape = q.sizes().vec();
        q_shape[q_shape.size() - 1] = 3;
        auto option = q.options();
        torch::Tensor result = torch::empty(q_shape, option);
        result.set_requires_grad(requires_grad);
        
        int batch_size = get_batch_size(q);
        auto device = q.device();
        auto dtype = q.dtype();
        q_shape.pop_back();
        torch::Tensor angle = torch::empty(q_shape, option); // we can save the angle for backward..
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());
            if (dtype == torch::kFloat32) DiffRotationImplVoid::quat_to_rotvec_forward<float>(q.data_ptr(), angle.data_ptr(), result.data_ptr(), batch_size, grid, block);
            else if (dtype == torch::kFloat64) DiffRotationImplVoid::quat_to_rotvec_forward<double>(q.data_ptr(), angle.data_ptr(), result.data_ptr(), batch_size, grid, block);
            else throw std::runtime_error("dtype not supported.");
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32) RotationCppBatchVoid::quat_to_rotvec<float>(q.data_ptr(), angle.data_ptr(), result.data_ptr(), batch_size);
            else if (dtype == torch::kFloat64) RotationCppBatchVoid::quat_to_rotvec<double>(q.data_ptr(), angle.data_ptr(), result.data_ptr(), batch_size);
            else throw std::runtime_error("dtype not supported.");
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }

        if (requires_grad)
        {
            ctx->save_for_backward({ q, angle });
        }
        return result;
    }

    variable_list QuatToRotVec::backward(AutogradContext * ctx, variable_list grad_output)
    {
        torch::Tensor grad = grad_output[0];
        if (!grad.is_contiguous()) grad = grad.contiguous();
        auto ctx_saved = ctx->get_saved_variables();
        torch::Tensor quat = ctx_saved[0];
        torch::Tensor angle = ctx_saved[1];

        torch::Tensor result = torch::empty_like(quat);
        result.set_requires_grad(false);
        int batch_size = get_batch_size(quat);

        auto device = grad_output[0].device();
        auto dtype = grad_output[0].dtype();
        
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            if (dtype == torch::kFloat32) DiffRotationImplVoid::quat_to_rotvec_backward<float>(quat.data_ptr(), angle.data_ptr(), grad.data_ptr(), result.data_ptr(), batch_size, grid, block);
            else if (dtype == torch::kFloat64) DiffRotationImplVoid::quat_to_rotvec_backward<double>(quat.data_ptr(), angle.data_ptr(), grad.data_ptr(), result.data_ptr(), batch_size, grid, block);
            else throw std::runtime_error("dtype not supported.");
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32) RotationCppBatchVoid::quat_to_rotvec_backward<float>(quat.data_ptr(), angle.data_ptr(), grad.data_ptr(), result.data_ptr(), batch_size);
            else if (dtype == torch::kFloat64) RotationCppBatchVoid::quat_to_rotvec_backward<double>(quat.data_ptr(), angle.data_ptr(), grad.data_ptr(), result.data_ptr(), batch_size);
            else throw std::runtime_error("dtype not supported.");
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return {result};
    }

    torch::Tensor QuatFromRotVec::forward(AutogradContext * ctx, torch::Tensor x)
    {
        if (!x.is_contiguous()) x = x.contiguous();
        auto shape = x.sizes().vec();
        shape[shape.size() - 1] = 4;
        auto option = x.options();
        torch::Tensor result = torch::empty(shape, option);
        bool requires_grad = x.requires_grad();
        if (requires_grad)
        {
            result.set_requires_grad(requires_grad);
            ctx->save_for_backward({ x });
        }
        
        int batch_size = get_batch_size(x);
        auto device = x.device();
        auto dtype = x.dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());
            if (dtype == torch::kFloat32) DiffRotationImplVoid::quat_from_rotvec_forward<float>(x.data_ptr(), result.data_ptr(), batch_size, grid, block);
            else if (dtype == torch::kFloat64) DiffRotationImplVoid::quat_from_rotvec_forward<double>(x.data_ptr(), result.data_ptr(), batch_size, grid, block);
            else throw std::runtime_error("dtype not supported.");
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32) RotationCppBatchVoid::quat_from_rotvec<float>(x.data_ptr(), result.data_ptr(), batch_size);
            else if (dtype == torch::kFloat64) RotationCppBatchVoid::quat_from_rotvec<double>(x.data_ptr(), result.data_ptr(), batch_size);
            else throw std::runtime_error("dtype not supported.");
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return result;
    }

    variable_list QuatFromRotVec::backward(AutogradContext * ctx, variable_list grad_output)
    {
        torch::Tensor rotvec = ctx->get_saved_variables()[0];
        torch::Tensor grad_o = grad_output[0];
        if (!grad_o.is_contiguous()) grad_o = grad_o.contiguous();
        torch::Tensor result = torch::empty_like(rotvec);
        int batch_size = get_batch_size(grad_o);
        auto device = grad_o.device();
        auto dtype = grad_o.dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());
            if (dtype == torch::kFloat32) DiffRotationImplVoid::quat_from_rotvec_backward<float>(rotvec.data_ptr(), grad_o.data_ptr(), result.data_ptr(), batch_size, grid, block);
            else if (dtype == torch::kFloat64) DiffRotationImplVoid::quat_from_rotvec_backward<double>(rotvec.data_ptr(), grad_o.data_ptr(), result.data_ptr(), batch_size, grid, block);
            else throw std::runtime_error("dtype not supported.");
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32) RotationCppBatchVoid::quat_from_rotvec_backward<float>(rotvec.data_ptr(), grad_o.data_ptr(), result.data_ptr(), batch_size);
            else if (dtype == torch::kFloat64) RotationCppBatchVoid::quat_from_rotvec_backward<double>(rotvec.data_ptr(), grad_o.data_ptr(), result.data_ptr(), batch_size);
            else throw std::runtime_error("dtype not supported.");
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return {result};
    }

    torch::Tensor QuatFromMatrix::forward(AutogradContext * ctx, torch::Tensor x)
    {
        auto shape = x.sizes().vec();
        shape.pop_back();
        shape[shape.size() - 1] = 4;
        torch::Tensor result = torch::empty(shape, x.options());
        bool requires_grad = x.requires_grad();
        result.set_requires_grad(requires_grad);
        if (!x.is_contiguous()) x = x.contiguous();
        if (requires_grad) ctx->save_for_backward({ x });
        int batch_size = get_batch_size(x, 2);
        auto device = x.device();
        auto dtype = x.dtype();
        if (device.is_cuda())
        {
            throw std::runtime_error("Not implemented");
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            if (dtype == torch::kFloat32)
            {
                
            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32) RotationCppBatchVoid::quat_from_matrix<float>(x.data_ptr(), result.data_ptr(), batch_size);
            else if (dtype == torch::kFloat64) RotationCppBatchVoid::quat_from_matrix<double>(x.data_ptr(), result.data_ptr(), batch_size);
            else throw std::runtime_error("dtype not supported.");
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return result;
    }

    variable_list QuatFromMatrix::backward(AutogradContext * ctx, variable_list grad_output)
    {
        torch::Tensor grad_o = grad_output[0];
        if (!grad_o.is_contiguous()) grad_o = grad_o.contiguous();
        torch::Tensor mat = ctx->get_saved_variables()[0];
        torch::Tensor result = torch::empty_like(mat);
        int batch_size = get_batch_size(result, 2);
        auto device = grad_output[0].device();
        auto dtype = grad_output[0].dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32) RotationCppBatchVoid::quat_from_matrix_backward<float>(mat.data_ptr(), grad_o.data_ptr(), result.data_ptr(), batch_size);
            else if (dtype == torch::kFloat64) RotationCppBatchVoid::quat_from_matrix_backward<double>(mat.data_ptr(), grad_o.data_ptr(), result.data_ptr(), batch_size);
            else throw std::runtime_error("dtype not supported.");
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return {result};
    }

    torch::Tensor QuatFromVec6d::forward(AutogradContext * ctx, torch::Tensor x)
    {
        throw std::runtime_error("Not implemented");
        torch::Tensor result;
        result.set_requires_grad(x.requires_grad());
        if (!x.is_contiguous()) x = x.contiguous();
        int batch_size = get_batch_size(x, 2);
        auto device = x.device();
        auto dtype = x.dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return result;
    }

    variable_list QuatFromVec6d::backward(AutogradContext * ctx, variable_list grad_output)
    {
        throw std::runtime_error("Not implemented");
        torch::Tensor result;
        torch::Tensor grad_o = grad_output[0];
        if (!grad_o.is_contiguous()) grad_o = grad_o.contiguous();
        int batch_size = get_batch_size(grad_o);
        auto device = grad_output[0].device();
        auto dtype = grad_output[0].dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return {result};
    }

    torch::Tensor QuatToVec6d::forward(AutogradContext * ctx, torch::Tensor x)
    {
        throw std::runtime_error("Not Implemented Error");
        // we should check the shape here..
        auto shape = x.sizes().vec();
        shape[shape.size() - 1] = 3;
        shape.push_back(2);
        auto option = x.options();
        torch::Tensor result = torch::empty(shape, option);
        bool requires_grad = x.requires_grad();
        result.set_requires_grad(requires_grad);
        if (!x.is_contiguous()) x = x.contiguous();
        if (requires_grad) ctx->save_for_backward({ x });
        int batch_size = get_batch_size(x);
        auto device = x.device();
        auto dtype = x.dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            //if (dtype == torch::kFloat32) DiffRotationImplVoid::quat_to_vec6d<float>(x.data_ptr(), result.data_ptr(), batch_size, grid, block);
            //else if (dtype == torch::kFloat64)
           // {

            //}
            //else throw std::runtime_error("dtype not supported.");
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32) RotationCppBatchVoid::quat_to_vec6d<float>(x.data_ptr(), result.data_ptr(), batch_size);
            else if (dtype == torch::kFloat64)  RotationCppBatchVoid::quat_to_vec6d<double>(x.data_ptr(), result.data_ptr(), batch_size);
            else throw std::runtime_error("dtype not supported.");
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return result;
    }

    variable_list QuatToVec6d::backward(AutogradContext * ctx, variable_list grad_output)
    {
        torch::Tensor result;
        torch::Tensor grad_o = grad_output[0];
        if (!grad_o.is_contiguous()) grad_o = grad_o.contiguous();
        int batch_size = get_batch_size(grad_o);
        auto device = grad_output[0].device();
        auto dtype = grad_output[0].dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return {result};
    }

    torch::Tensor NormalizeVec6d::forward(AutogradContext * ctx, torch::Tensor x)
    {
        throw std::runtime_error("Not Implemented Error");
        torch::Tensor result;
        result.set_requires_grad(result.requires_grad());
        int batch_size = 0;
        auto device = x.device();
        auto dtype = x.dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            if (dtype == torch::kFloat32) 
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return result;
    }

    variable_list NormalizeVec6d::backward(AutogradContext * ctx, variable_list grad_output)
    {
        torch::Tensor result;
        int batch_size = 0;
        auto device = grad_output[0].device();
        auto dtype = grad_output[0].dtype();

        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return {result};
    }

    torch::Tensor Mat22Det::forward(AutogradContext * ctx, torch::Tensor x)
    {
        throw std::runtime_error("Not Implemented Error");
        torch::Tensor result;
        result.set_requires_grad(result.requires_grad());
        int batch_size = 0;
        auto device = x.device();
        auto dtype = x.dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            if (dtype == torch::kFloat32) 
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return result;
    }

    variable_list Mat22Det::backward(AutogradContext * ctx, variable_list grad_output)
    {
        throw std::runtime_error("Not Implemented Error");
        torch::Tensor result;
        int batch_size = 0;
        auto device = grad_output[0].device();
        auto dtype = grad_output[0].dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return {result};
    }

    torch::Tensor Mat22Inv::forward(AutogradContext * ctx, torch::Tensor x)
    {
        throw std::runtime_error("Not Implemented Error");
        torch::Tensor result;
        result.set_requires_grad(result.requires_grad());
        int batch_size = 0;
        auto device = x.device();
        auto dtype = x.dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return result;
    }

    variable_list Mat22Inv::backward(AutogradContext * ctx, variable_list grad_output)
    {
        throw std::runtime_error("Not Implemented Error");
        torch::Tensor result;
        int batch_size = 0;
        auto device = grad_output[0].device();
        auto dtype = grad_output[0].dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return {result};
    }

    torch::Tensor Mat33Det::forward(AutogradContext * ctx, torch::Tensor x)
    {
        // if (x.ndimension() == 2) x = x.view({ 1, 3, 3 });
        bool requires_grad = x.requires_grad();
        auto shape = x.sizes().vec();
        shape.pop_back();
        shape.pop_back();
        torch::Tensor result = torch::empty(shape, x.options());
        result.set_requires_grad(requires_grad);
        if (requires_grad) ctx->save_for_backward({ x });
        int batch_size = get_batch_size(result, 0);
        auto device = x.device();
        auto dtype = x.dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32) RotationCppBatchVoid::mat33_det<float>(x.data_ptr(), result.data_ptr(), batch_size);
            else if (dtype == torch::kFloat64) RotationCppBatchVoid::mat33_det<double>(x.data_ptr(), result.data_ptr(), batch_size);
            else throw std::runtime_error("dtype not supported.");
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return result;
    }

    variable_list Mat33Det::backward(AutogradContext * ctx, variable_list grad_output)
    {   
        torch::Tensor mat = ctx->get_saved_variables()[0];
        torch::Tensor result = torch::empty_like(mat);
        int batch_size = get_batch_size(mat, 2);
        torch::Tensor grad = grad_output[0];
        auto device = grad.device();
        auto dtype = grad.dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            if (dtype == torch::kFloat32) 
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32) RotationCppBatchVoid::mat33_det_backward<float>(mat.data_ptr(), grad.data_ptr(), result.data_ptr(), batch_size);
            else if (dtype == torch::kFloat64) RotationCppBatchVoid::mat33_det_backward<double>(mat.data_ptr(), grad.data_ptr(), result.data_ptr(), batch_size);
            else throw std::runtime_error("dtype not supported.");
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return {result};
    }

    torch::Tensor Mat33Inv::forward(AutogradContext * ctx, torch::Tensor x)
    {
        throw std::runtime_error("Not Implemented Error");
        torch::Tensor result;
        result.set_requires_grad(result.requires_grad());
        int batch_size = 0;
        auto device = x.device();
        auto dtype = x.dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return result;
    }

    variable_list Mat33Inv::backward(AutogradContext * ctx, variable_list grad_output)
    {
        torch::Tensor result;
        int batch_size = 0;
        auto device = grad_output[0].device();
        auto dtype = grad_output[0].dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return {result};
    }

    torch::Tensor Mat33SVD::forward(AutogradContext * ctx, torch::Tensor x)
    {
        if (!x.is_contiguous()) x = x.contiguous();
        torch::Tensor result;
        result.set_requires_grad(result.requires_grad());
        int batch_size = 0;
        auto device = x.device();
        auto dtype = x.dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index());

            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return result;
    }

    variable_list Mat33SVD::backward(AutogradContext * ctx, variable_list grad_output)
    {
        torch::Tensor result;
        int batch_size = 0;
        auto device = grad_output[0].device();
        auto dtype = grad_output[0].dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index()); // TODO: get device ID

            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return {result};
    }

    torch::Tensor Mat44Det::forward(AutogradContext * ctx, torch::Tensor x)
    {
        throw std::runtime_error("Not Implemented Error");
        torch::Tensor result;
        result.set_requires_grad(result.requires_grad());
        int batch_size = 0;
        auto device = x.device();
        auto dtype = x.dtype();
        if (device.is_cuda())
        {
            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return result;
    }

    variable_list Mat44Det::backward(AutogradContext * ctx, variable_list grad_output)
    {
        torch::Tensor result;
        int batch_size = 0;
        auto device = grad_output[0].device();
        auto dtype = grad_output[0].dtype();
        if (device.is_cuda())
        {
            int grid, block;
            CudaConfig::get_thread_block(batch_size, grid, block, device.index()); // TODO: get device ID

            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else if (device.is_cpu())
        {
            if (dtype == torch::kFloat32)
            {

            }
            else if (dtype == torch::kFloat64)
            {

            }
            else
            {
                throw std::runtime_error("dtype not supported.");
            }
        }
        else
        {
            throw std::runtime_error("Only support cuda and cpu device");
        }
        return {result};
    }
};
