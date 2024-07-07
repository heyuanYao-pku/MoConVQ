#pragma once
#include <torch/torch.h>

namespace DiffRotationImpl
{
    using torch::autograd::Function;
    using torch::autograd::variable_list;
    using torch::autograd::AutogradContext;

    extern const char * DTYPE_NOT_MATCH;
    extern const char * DTYPE_ONLY_FLOAT;
    extern const char * NOT_CONTIGUOUS;
    extern const char * NOT_SAME_CUDA_DEVICE;
    inline void check_condition(bool condition, const char * err_info);

    inline void check_input_pair(const torch::Tensor a, const torch::Tensor b);
    bool is_requires_grad(const variable_list & inputs);

    class QuatMultiply: public Function<QuatMultiply>
    {
        public:
            static torch::Tensor forward(AutogradContext * ctx, torch::Tensor q1, torch::Tensor q2);
            static variable_list backward(AutogradContext * ctx, variable_list grad_output);
    };

    class QuatApply: public Function<QuatApply>
    {
        public:
            static torch::Tensor forward(AutogradContext * ctx, torch::Tensor q, torch::Tensor v);
            static variable_list backward(AutogradContext * ctx, variable_list grad_output);
    };

    class QuatInv : public Function<QuatInv>
    {
        public:
            static torch::Tensor forward(AutogradContext * ctx, torch::Tensor q);
            static variable_list backward(AutogradContext * ctx, variable_list grad_output);
    };

    class QuatIntegrate : public Function<QuatIntegrate>
    {
        public:
            static torch::Tensor forward(AutogradContext * ctx, torch::Tensor q, torch::Tensor v, double dt);
            static variable_list backward(AutogradContext * ctx, variable_list grad_output);
    };

    class QuatToAngle: public Function<QuatToAngle>
    {
        public:
            static torch::Tensor forward(AutogradContext * ctx, torch::Tensor q);
            static variable_list backward(AutogradContext * ctx, variable_list grad_output);
    };

    class QuatToRotVec: public Function<QuatToRotVec>
    {
        public:
            static torch::Tensor forward(AutogradContext * ctx, torch::Tensor q);
            static variable_list backward(AutogradContext * ctx, variable_list grad_output);
    };

    class QuatFromRotVec: public Function<QuatFromRotVec>
    {
        public:
            static torch::Tensor forward(AutogradContext * ctx, torch::Tensor x);
            static variable_list backward(AutogradContext * ctx, variable_list grad_output);
    };

    class QuatFromMatrix: public Function<QuatFromMatrix>
    {
        public:
            static torch::Tensor forward(AutogradContext * ctx, torch::Tensor x);
            static variable_list backward(AutogradContext * ctx, variable_list grad_output);
    };

    class QuatFromVec6d: public Function<QuatFromVec6d>
    {
        public:
            static torch::Tensor forward(AutogradContext * ctx, torch::Tensor x);
            static variable_list backward(AutogradContext * ctx, variable_list grad_output);
    };

    class QuatToVec6d: public Function<QuatToVec6d>
    {
        public:
            static torch::Tensor forward(AutogradContext * ctx, torch::Tensor x);
            static variable_list backward(AutogradContext * ctx, variable_list grad_output);
    };

    class NormalizeVec6d: public Function<NormalizeVec6d>
    {
        public:
            static torch::Tensor forward(AutogradContext * ctx, torch::Tensor x);
            static variable_list backward(AutogradContext * ctx, variable_list grad_output);
    };

    class Mat22Det: public Function<Mat22Det>
    {
        public:
            static torch::Tensor forward(AutogradContext * ctx, torch::Tensor x);
            static variable_list backward(AutogradContext * ctx, variable_list grad_output);
    };

    class Mat22Inv: public Function<Mat22Inv>
    {
        public:
            static torch::Tensor forward(AutogradContext * ctx, torch::Tensor x);
            static variable_list backward(AutogradContext * ctx, variable_list grad_output);
    };

    class Mat33Det: public Function<Mat33Det>
    {
        public:
            static torch::Tensor forward(AutogradContext * ctx, torch::Tensor x);
            static variable_list backward(AutogradContext * ctx, variable_list grad_output);
    };

    class Mat33Inv: public Function<Mat33Inv>
    {
        public:
            static torch::Tensor forward(AutogradContext * ctx, torch::Tensor x);
            static variable_list backward(AutogradContext * ctx, variable_list grad_output);
    };

    class Mat33SVD: public Function<Mat33SVD>
    {
        public:
            static torch::Tensor forward(AutogradContext * ctx, torch::Tensor x);
            static variable_list backward(AutogradContext * ctx, variable_list grad_output);
    };

    class Mat44Det: public Function<Mat33Det>
    {
        public:
            static torch::Tensor forward(AutogradContext * ctx, torch::Tensor x);
            static variable_list backward(AutogradContext * ctx, variable_list grad_output);
    };
}