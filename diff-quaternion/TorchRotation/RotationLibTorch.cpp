#include "RotationLibTorch.h"
#if BUILD_ROTATION_AS_PYBIND11
#include <pybind11/pybind11.h>
#endif

namespace RotationLibTorch
{
    using namespace torch::indexing;

    // quaternion multiply
    torch::Tensor quat_multiply(torch::Tensor p, torch::Tensor q)
    {
        if (q.size(-1) != 4 || p.size(-1) != 4)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input quaternion should be (..., 4)");
        }
        torch::Tensor pw = p.index({"...", Slice(3, 4)});
        torch::Tensor qw = q.index({"...", Slice(3, 4)});
        torch::Tensor pxyz = p.index({"...", Slice(0, 3)});
        torch::Tensor qxyz = q.index({"...", Slice(0, 3)});
        torch::Tensor w = pw * qw - torch::sum(pxyz * qxyz, -1, true);
        torch::Tensor xyz = pw * qxyz + qw * pxyz + torch::cross(pxyz, qxyz, -1);
        torch::Tensor result = torch::cat({xyz, w}, -1);
        return result;
    }

    // apply quaternion rotation to vectors
    torch::Tensor quat_apply(torch::Tensor q, torch::Tensor v)
    {
        if (q.size(-1) != 4 || v.size(-1) != 3)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input quaternion should be (..., 4) ");
        }
        torch::Tensor qxyz = q.index({"...", Slice(0, 3)});
        torch::Tensor t = 2 * torch::cross(qxyz, v, -1);
        torch::Tensor xyz = v + q.index({"...", Slice(3, 4)}) * t + torch::cross(qxyz, t, -1);
        return xyz;
    }

    // The inverse of quaternion
    torch::Tensor quat_inv(torch::Tensor q)
    {
        if (q.size(-1) != 4)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input quaternion should be (..., 4)");
        }
        torch::Tensor w = q.index({"...", Slice(3, 4)});
        torch::Tensor xyz = q.index({"...", Slice(0, 3)});
        torch::Tensor result = torch::cat({xyz, -w}, -1);
        return result;
    }

    torch::Tensor flip_quat_by_w(torch::Tensor q)
    {
        if (q.size(-1) != 4)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input quaternion should be (..., 4)");
        }
        torch::Tensor w = q.index({"...", torch::indexing::Slice(3, 4)});
        torch::Tensor mask = (w < 0).to(torch::kInt32);
        mask.masked_fill_(mask == 1, -1);
        mask.masked_fill_(mask == 0, 1);
        torch::Tensor result = q * mask;
        return result;
    }

    torch::Tensor vec_normalize(torch::Tensor q)
    {
        torch::Tensor length = torch::linalg::norm(q, 2, -1, true, c10::nullopt);
        torch::Tensor result = q / length;
        return result;
    }

    // normalize quaternion
    torch::Tensor quat_normalize(torch::Tensor q)
    {
        if (q.size(-1) != 4)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input quaternion should be (..., 4)");
        }
        torch::Tensor length = torch::linalg::norm(q, 2, -1, true, c10::nullopt);
        torch::Tensor result = q / length;
        return result;
    }

    // quaternion integrate
    torch::Tensor quat_integrate(torch::Tensor q, torch::Tensor omega, double dt)
    {
        if (q.size(-1) != 4 || omega.size(-1) !=3)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input quaternion should be (..., 4), omega should be (..., 3)");
        }
        auto option = q.options();
        auto sizes = omega.sizes().vec();
        sizes[sizes.size() - 1] = 1;
        torch::Tensor zeros = torch::zeros(sizes, option);
        omega = torch::cat({omega, zeros}, -1);
        torch::Tensor dq = quat_multiply(omega, q);
        torch::Tensor result = quat_normalize(q + (0.5 * dt) * dq);
        return result;
    }

    // Rotation from vector a to vector b
    torch::Tensor quat_between(torch::Tensor a, torch::Tensor b)
    {
        if (a.size(-1) != 3 || b.size(-1) != 3)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input vector should be (..., 3)");
        }
        torch::Tensor cross_res = torch::cross(a, b, -1);
        torch::Tensor w_ = torch::sqrt((a * a).sum(-1) * (b * b).sum(-1)) + (a * b).sum(-1);
        torch::Tensor res_ = torch::cat({ cross_res, w_.index({Ellipsis, None})}, -1);
        return vec_normalize(res_);
    }

    torch::Tensor decompose_rotation(torch::Tensor q, torch::Tensor vb)
    {
        bool is_flatten = q.ndimension() == 1;
        if (is_flatten)
        {
            q = q.index({ None });
        }
        auto q_shape = q.sizes().vec();
        if (*q_shape.rbegin() != 4)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input quaternion should be (..., 4)");
        }
        *q_shape.rbegin() = 3;
        if (!q.is_contiguous()) q = q.contiguous();
        vb = torch::broadcast_to(vb, q_shape);
        torch::Tensor va = vec_normalize(quat_apply(q, vb));
        // emm.. we should not use arccos here.. use quat_between directly.
        // torch::Tensor rot_axis = vec_normalize(torch::cross(va, vb, -1));
        // torch::Tensor rot_angle = -torch::arccos(torch::clip(torch::sum(va * vb, -1), -1 + 1e-10f, 1 - 1e-10f)).index({"...", None});
        // torch::Tensor tmp = quat_from_rotvec(rot_angle * (-rot_axis));
        torch::Tensor tmp = quat_between(va, vb);
        torch::Tensor result = quat_normalize(quat_multiply(tmp, q));
        if (is_flatten)
        {
            result = result.view({ 4 });
        }
        return result;
    }

    torch::Tensor x_decompose(torch::Tensor q)
    {
        torch::Tensor vb = torch::zeros({ 1, 3 }, q.options());
        vb.set_requires_grad(false);
        vb.index({0, 0}) = 1;
        return decompose_rotation(q, vb);
    }

    torch::Tensor y_decompose(torch::Tensor q)
    {
        torch::Tensor vb = torch::zeros({ 1, 3 }, q.options());
        vb.set_requires_grad(false);
        vb.index({ 0, 1 }) = 1;
        return decompose_rotation(q, vb);
    }

    torch::Tensor z_decompose(torch::Tensor q)
    {
        torch::Tensor vb = torch::zeros({ 1, 3 }, q.options());
        vb.set_requires_grad(false);
        vb.index({ 0, 2 }) = 1;
        return decompose_rotation(q, vb);
    }

    // convert quaternion rotation to angle
    torch::Tensor quat_to_angle(torch::Tensor q)
    {
        torch::Tensor result;
        return result;
    }

    // convert quaternion to axis angle format
    torch::Tensor quat_to_rotvec(torch::Tensor q)
    {
        if (q.size(-1) != 4)
        {
            throw std::length_error("size of q should be 4");
        }
        auto option = q.options();
        const float eps = 1e-3f;
        q = flip_quat_by_w(q);
        torch::Tensor xyz = q.index({"...", Slice(0, 3)});
        torch::Tensor w = q.index({"...", 3});
        torch::Tensor xyz_norm = torch::linalg::norm(xyz, 2, -1, false, c10::nullopt);
        torch::Tensor angle = 2 * torch::atan2(xyz_norm, w);
        torch::Tensor small_angle = angle <= eps;
        torch::Tensor scale_small = 2 + (1.0 / 12) * torch::pow(angle, 2) + (7.0 / 2880) * torch::pow(angle, 4);
        torch::Tensor scale_large = angle / torch::sin(0.5 * angle);
        torch::Tensor scale = torch::where(small_angle, scale_small, scale_large);
        torch::Tensor result = scale.index({"...", None}) * xyz;
        return result;
    }

    // build quaternion from axis angle format
    torch::Tensor quat_from_rotvec(torch::Tensor x)
    {
        if (x.size(-1) != 3)
        {
            throw std::length_error("shape should be (*, 3)");
        }
        torch::Tensor norms = torch::linalg::norm(x, 2, -1, false, c10::nullopt);
        torch::Tensor small_angle = norms <= 1e-3;
        torch::Tensor scale_small = 0.5 - (1.0 / 48) * torch::square(norms) + torch::pow(norms, 4) / 3840;
        torch::Tensor scale_large = torch::sin(0.5 * norms) / norms;
        torch::Tensor scale = torch::where(small_angle, scale_small, scale_large);
        torch::Tensor quat_xyz = scale.index({Ellipsis, None}) * x;
        torch::Tensor quat_w = torch::cos(0.5 * norms).index({Ellipsis, None});
        torch::Tensor quat = torch::cat({quat_xyz, quat_w}, -1);

        return quat;
    }

    class _QuatCatFunc: public torch::autograd::Function<_QuatCatFunc>
    {
    public:
        static torch::Tensor forward(torch::autograd::AutogradContext * ctx, std::vector<torch::Tensor> args);
        static std::vector<torch::Tensor> backward(
            torch::autograd::AutogradContext * ctx,
            std::vector<torch::Tensor> grad_outputs
        );
    };

    // for quat from matrix
    torch::Tensor _QuatCatFunc::forward(
        torch::autograd::AutogradContext * ctx,
        std::vector<torch::Tensor> args)
    {
        bool need_grad = false;
        torch::Tensor qi = args[0], qj = args[1], qk = args[2], qw = args[3];
        torch::Tensor i = args[4], j = args[5], k = args[6];
        auto option = qi.options();
        auto size = qi.size(0);
        for (size_t i = 0; i < args.size(); i++)
        {
            need_grad |= args[i].requires_grad();
        }
        auto idx_option = option.dtype(torch::kLong);
        torch::Tensor idx = torch::arange(size, idx_option);
        torch::Tensor xyz = torch::zeros({size, 4}, option);
        xyz.index_put_({ idx, i }, qi);
        xyz.index_put_({ idx, j }, qj);
        xyz.index_put_({ idx, k }, qk);
        xyz.index_put_({ idx, 3 }, qw);
        if (need_grad)
        {
            xyz.requires_grad_(need_grad);
            if (ctx != nullptr)
            {
                ctx->save_for_backward({i, j, k, idx});
            }
        }
        return xyz;
    }

    std::vector<torch::Tensor> _QuatCatFunc::backward(
        torch::autograd::AutogradContext * ctx,
        std::vector<torch::Tensor> grad_outputs
    )
    {
        torch::Tensor grad_xyz = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        torch::Tensor i = saved[0], j = saved[1], k = saved[2], idx = saved[3];
        std::vector<torch::Tensor> result(7);
        result[0] = grad_xyz.index({idx, i});
        result[1] = grad_xyz.index({idx, j});
        result[2] = grad_xyz.index({idx, k});
        result[3] = grad_xyz.index({ Ellipsis, 3 });
        return result;
    }

    // build quaternion from matrix format.
    torch::Tensor quat_from_matrix(torch::Tensor matrix)
    {
        auto ori_shape = matrix.sizes().vec();
        int ndim = ori_shape.size();
        if (ndim < 2 || ori_shape[ndim - 1] != 3 || ori_shape[ndim - 2] != 3)
        {
            throw std::length_error("size not match.");
        }
        matrix = matrix.view({-1, 3, 3});  // (batch_size, 3, 3)
        auto batch_size = matrix.size(0);
        auto option = matrix.options();
        torch::Tensor decision_xyz = torch::diagonal(matrix, 0, 1, 2); // (batch_size, 3)
        torch::Tensor decision_w = decision_xyz.sum(1, true); // (batch_size, 1)
        torch::Tensor decision_matrix = torch::cat({decision_xyz, decision_w}, -1); // (batch_size, 4)
        torch::Tensor choices = decision_matrix.argmax(1); // (batch_size,)
        torch::Tensor flg = choices != 3;
        torch::Tensor neq_ind = torch::nonzero(flg).flatten(); // dtype == int64
        torch::Tensor quat_neq3, quat_eq3;
        if (neq_ind.numel() > 0)
        {
            torch::Tensor i = choices.index({ neq_ind }); // choices[ind]
            torch::Tensor j = (i + 1) % 3;
            torch::Tensor k = (j + 1) % 3;
            torch::Tensor sub_decision = decision_matrix.index({ neq_ind });
            torch::Tensor quat_neq3_i = 0.5 * torch::sqrt(1 - sub_decision.select(1, 3) + 2 * matrix.index({neq_ind, i, i}));
            torch::Tensor ratio = 0.25 / quat_neq3_i;
            torch::Tensor quat_neq3_j = ratio * (matrix.index({neq_ind, j, i}) + matrix.index({neq_ind, i, j}));
            torch::Tensor quat_neq3_k = ratio * (matrix.index({neq_ind, k, i}) + matrix.index({neq_ind, i, k}));
            torch::Tensor quat_neq3_3 = ratio * (matrix.index({neq_ind, k, j}) - matrix.index({neq_ind, j, k}));
            auto neq_in = std::vector<torch::Tensor>({quat_neq3_i, quat_neq3_j, quat_neq3_k, quat_neq3_3, i, j, k});
            quat_neq3 = _QuatCatFunc::apply(neq_in);
        }

        torch::Tensor eq_ind = torch::nonzero(~flg).flatten();
        if (eq_ind.numel() > 0)
        {
            torch::Tensor quat_eq3_3 = 0.5 * torch::sqrt(1 + decision_matrix.index({ eq_ind, 3 }));
            torch::Tensor ratio = 0.25 / quat_eq3_3;
            torch::Tensor quat_eq3_0 = ratio * (matrix.index({eq_ind, 2, 1}) - matrix.index({eq_ind, 1, 2}));
            torch::Tensor quat_eq3_1 = ratio * (matrix.index({eq_ind, 0, 2}) - matrix.index({eq_ind, 2, 0}));
            torch::Tensor quat_eq3_2 = ratio * (matrix.index({eq_ind, 1, 0}) - matrix.index({eq_ind, 0, 1}));
            
            auto eq_in = std::vector<torch::Tensor>({quat_eq3_0, quat_eq3_1, quat_eq3_2, quat_eq3_3});
            for(int t = 0; t < 4; t++) eq_in[t] = eq_in[t].index({Ellipsis, None});
            quat_eq3 = torch::cat(eq_in, -1);
        }

        torch::Tensor quat;
        if(!quat_eq3.defined())
        {
            quat = quat_neq3;
        }
        else if (!quat_neq3.defined())
        {
            quat = quat_eq3;
        }
        else
        {
            auto new_opt = option.requires_grad(false);
            torch::Tensor quat_neq3_m = torch::zeros({batch_size, 4}, new_opt);
            quat_neq3_m.index_put_({neq_ind}, quat_neq3);
            torch::Tensor quat_eq3_m = torch::zeros({batch_size, 4}, new_opt);
            quat_eq3_m.index_put_({eq_ind}, quat_eq3);
            quat = quat_neq3_m + quat_eq3_m;
        }
        ori_shape.pop_back();
        *ori_shape.rbegin() = 4;
        return quat_normalize(quat.view(ori_shape));
    }

    // convert quaternion to rotation matrix
    torch::Tensor quat_to_matrix(torch::Tensor q)
    {
        if (q.size(-1) != 4)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input quaternion should be (..., 4)");
        }
        torch::Tensor x = q.index({Ellipsis, Slice(0, 1)});
        torch::Tensor y = q.index({Ellipsis, Slice(1, 2)});
        torch::Tensor z = q.index({Ellipsis, Slice(2, 3)});
        torch::Tensor w = q.index({Ellipsis, Slice(3, 4)});

        torch::Tensor x2 = torch::square(x);
        torch::Tensor y2 = torch::square(y);
        torch::Tensor z2 = torch::square(z);
        torch::Tensor w2 = torch::square(w);

        torch::Tensor xy = x * y;
        torch::Tensor zw = z * w;
        torch::Tensor xz = x * z;
        torch::Tensor yw = y * w;
        torch::Tensor yz = y * z;
        torch::Tensor xw = x * w;

        torch::Tensor res00 = x2 - y2 - z2 + w2;
        torch::Tensor res10 = 2 * (xy + zw);
        torch::Tensor res20 = 2 * (xz - yw);

        torch::Tensor res01 = 2 * (xy - zw);
        torch::Tensor res11 = - x2 + y2 - z2 + w2;
        torch::Tensor res21 = 2 * (yz + xw);

        torch::Tensor res02 = 2 * (xz + yw);
        torch::Tensor res12 = 2 * (yz - xw);
        torch::Tensor res22 = - x2 - y2 + z2 + w2;

        // TODO: check the output and dimension, and reshape
        torch::Tensor result = torch::cat({
            res00, res01, res02,
            res10, res11, res12,
            res20, res21, res22}, -1);

        auto shape = q.sizes().vec();
        shape[shape.size() - 1] = 3;
        shape.push_back(3);
        result = result.view(shape);
        return result;
    }

    // build quaternion from 6d vector
    torch::Tensor quat_from_vec6d(torch::Tensor x)
    {
        torch::Tensor mat = vec6d_to_matrix(x);
        torch::Tensor quat = quat_from_matrix(mat);
        return quat;
    }

    // convert quaternion to 6d representation
    torch::Tensor quat_to_vec6d(torch::Tensor x)
    {
        torch::Tensor mat = quat_to_matrix(x);
        return mat.index({"...", Slice(0, 2)}).contiguous();
    }

    // normalize 6d rotation representation
    std::vector<torch::Tensor> normalize_vec6d(torch::Tensor x)
    {
        auto ori_shape = x.sizes().vec();
        int ndim = ori_shape.size();
        if(ndim < 2 || ori_shape[ndim - 2] != 3 || ori_shape[ndim-1] != 2)
        {
            throw std::length_error("size not match.");
        }
        x = x / torch::linalg::norm(x, 2, -2, true, c10::nullopt);
        torch::Tensor first_col = x.index({"...", 0});
        torch::Tensor second_col = x.index({"...", 1});
        torch::Tensor last_col = torch::cross(first_col, second_col, -1);
        last_col = last_col / torch::linalg::norm(last_col, 2, -1, true, c10::nullopt);
        second_col = torch::cross(-first_col, last_col, -1);
        second_col = second_col / torch::linalg::norm(second_col, 2, -1, true, c10::nullopt);
        auto result_list = std::vector<torch::Tensor>({first_col, second_col, last_col});
        return result_list;
    }

    torch::Tensor normalize_vec6d_cat(torch::Tensor x)
    {
        std::vector<torch::Tensor> res = normalize_vec6d(x);
        auto cat_result = torch::cat({res[0].index({"...", None}), res[1].index({"...", None})}, -1);
        return cat_result;
    }

    torch::Tensor vec6d_to_matrix(torch::Tensor x)
    {
        std::vector<torch::Tensor> res = normalize_vec6d(x);
        for (int i = 0; i < 3; i++)
        {
            res[i] = res[i].index({"...", None});
        }
        auto cat_result = torch::cat(res, -1);
        return cat_result;
    }

    torch::Tensor vector_to_cross_matrix(torch::Tensor x)
    {
        if (x.size(-1) != 3)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input vector should be (..., 3)");
        }
        torch::Tensor x0 = x.index({"...", 0});
        torch::Tensor x1 = x.index({"...", 1});
        torch::Tensor x2 = x.index({"...", 2});
        torch::Tensor zero00 = torch::zeros_like(x0);
        torch::Tensor mat = torch::cat({
            zero00, -x2, x1,
            x2, zero00, -x0,
            -x1, x0, zero00}, -1);
        auto shape = x.sizes().vec();
        shape.push_back(3);
        mat = mat.view(shape);
        return mat;
    }

    torch::Tensor matrix_to_angle(torch::Tensor x)
    {
        // acos((tr(R)-1)/2)
        if (x.size(-1) !=3 || x.size(-2) !=3)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input matrix should be (..., 3, 3)");
        }
        float eps = 1e-7;
        torch::Tensor diag = torch::diagonal(x, 0, -1, -2);
        torch::Tensor trace = torch::sum(diag, -1);
        torch::Tensor trace_inside = torch::clamp(0.5 * (trace - 1), -1.0+eps, 1.0-eps);  // avoid NaN in acos function
        torch::Tensor angle = torch::acos(trace_inside);
        return angle;
    }

    torch::Tensor rotation_matrix_inv(torch::Tensor x)
    {
        if (x.ndimension() < 2 || x.size(-1) !=3 || x.size(-2) !=3)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input matrix should be (..., 3, 3)");
        }
        return torch::transpose(x, -1, -2);
    }

    // compute det of 2x2 matrix. input: (*, 2, 2)
    torch::Tensor mat22_det(torch::Tensor x)
    {
        if (x.ndimension() < 2 || x.size(-1) != 2 || x.size(-2) != 2)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input matrix should be (..., 2, 2)");
        }
        torch::Tensor x00 = x.index({"...", 0, 0});
        torch::Tensor x01 = x.index({"...", 0, 1});
        torch::Tensor x10 = x.index({"...", 1, 0});
        torch::Tensor x11 = x.index({"...", 1, 1});
        torch::Tensor result = x00 * x11 - x01 * x10;
        return result;
    }

    torch::Tensor mat33_det(torch::Tensor x)
    {
        if (x.ndimension() < 2 || x.size(-1) !=3 || x.size(-2) !=3)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input matrix should be (..., 3, 3)");
        }
        torch::Tensor a1 = x.index({"...", 0, 0});
        torch::Tensor b1 = x.index({"...", 0, 1});
        torch::Tensor c1 = x.index({"...", 0, 2});
        torch::Tensor a2 = x.index({"...", 1, 0});
        torch::Tensor b2 = x.index({"...", 1, 1});
        torch::Tensor c2 = x.index({"...", 1, 2});
        torch::Tensor a3 = x.index({"...", 2, 0});
        torch::Tensor b3 = x.index({"...", 2, 1});
        torch::Tensor c3 = x.index({"...", 2, 2});
        torch::Tensor result =
            a1 * b2 * c3 +
            b1 * c2 * a3 +
            c1 * a2 * b3 -
            a3 * b2 * c1 -
            b3 * c2 * a1 -
            c3 * a2 * b1;

        return result;
    }

    // svd decomposion of 3x3 matrix
    torch::Tensor mat33_svd(torch::Tensor x)
    {
        if (x.ndimension() < 2 || x.size(-1) !=3 || x.size(-2) !=3)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input matrix should be (..., 3, 3)");
        }
        torch::Tensor result;
        return result;
    }

    torch::Tensor mat44_det(torch::Tensor x)
    {
        if (x.ndimension() < 2 || x.size(-1) !=4 || x.size(-2) !=4)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input quaternion should be (..., 4, 4)");
        }
        throw std::logic_error("not implemented");
        torch::Tensor result;
        return result;
    }

    torch::Tensor flip_vector(torch::Tensor vt, torch::Tensor normal)
    {
        auto shape = vt.sizes().vec();
        vt = vt.reshape({ -1, 3 });
        normal = normal.reshape({ -1, 3 });
        torch::Tensor res = vt - (2 * torch::sum(vt * normal, -1, true)) * normal;
        *shape.rbegin() = 3;
        return res.view(shape);
    }

    torch::Tensor flip_quaternion(torch::Tensor qt, torch::Tensor normal)
    {
        torch::Tensor vec_flip = flip_vector(qt.index({ Ellipsis, Slice(0, 3) }), normal);
        torch::Tensor res = torch::cat({ vec_flip, -1 * qt.index({Ellipsis, Slice(3, 4)}) }, -1);
        return res;
    }


#if BUILD_ROTATION_AS_PYBIND11
    void add_pybind11_wrapper(pybind11::module & m)
    {
        using namespace DiffRotation;
        namespace py = pybind11;
        m.doc() = "Fast implementation of rotation operation with PyTorch. Cuda and cudnn are required.";
        m.def("quat_multiply", &quat_multiply, quat_multiply_help, py::arg("q1"), py::arg("q2"));
        m.def("quat_apply", &quat_apply, quat_apply_help, py::arg("q"), py::arg("v"));
        m.def("quat_inv", &quat_inv, quat_inv_help, py::arg("q"));

        m.def("flip_quat_by_w", &flip_quat_by_w);
        m.def("quat_normalize", &quat_normalize, "Normalize the quaternion.", py::arg("q"));
        m.def("quat_integrate", &quat_integrate, quat_integrate_help, py::arg("q"), py::arg("omega"), py::arg("dt"));
        m.def("quat_between", &quat_between, py::arg("a"), py::arg("b"));
        m.def("decompose_rotation", &decompose_rotation, py::arg("q"), py::arg("vb"));
        m.def("x_decompose", &x_decompose, py::arg("q"));
        m.def("y_decompose", &y_decompose, py::arg("q"));
        m.def("z_decompose", &z_decompose, py::arg("q"));

        m.def("quat_to_angle", &quat_to_angle, quat_to_angle_help, py::arg("q"));
        m.def("quat_to_rotvec", &quat_to_rotvec, quat_to_axis_angle_help, py::arg("q"));
        m.def("quat_from_rotvec", &quat_from_rotvec, quat_from_rotvec_help, py::arg("rotvec"));

        m.def("quat_from_matrix", &quat_from_matrix);
        m.def("quat_to_matrix", &quat_to_matrix);
        m.def("quat_from_vec6d", &quat_from_vec6d);
        m.def("quat_to_vec6d", &quat_to_vec6d);

        m.def("normalize_vec6d", &normalize_vec6d);
        m.def("normalize_vec6d_cat", &normalize_vec6d_cat);
        m.def("vec6d_to_matrix", &vec6d_to_matrix);

        m.def("matrix_to_angle", &matrix_to_angle, py::arg("x"));
        m.def("mat22_det",&mat22_det);
        m.def("mat33_det", &mat33_det);
        m.def("mat33_svd", &mat33_svd);
        m.def("mat44_det", &mat33_det);

        m.def("flip_vector", &flip_vector, py::arg("qt"), py::arg("normal"));
        m.def("flip_quaternion", &flip_quaternion, py::arg("qt"), py::arg("normal"));

        // TODO: support euler angle.
    }
#endif

};

#if BUILD_ROTATION_AS_PYBIND11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    RotationLibTorch::add_pybind11_wrapper(m);
}
#endif

