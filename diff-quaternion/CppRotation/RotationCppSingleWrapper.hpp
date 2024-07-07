#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "RotationCppSingle.hpp"

namespace RotationCppSingleWrapper
{
    namespace py = pybind11;
    using namespace RotationCppSingle;

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    py::array_t<data_type> quat_multiply_np(
        py::array_t<data_type, py::array::c_style> q1,
        py::array_t<data_type, py::array::c_style> q2)
    {
        #if ROTATION_CHECK_INPUT_SIZE
        if (q1.size() < 4 || q2.size() < 4)
        {
            throw py::index_error("shape should be larger than 4");
        }
        #endif
        auto result = py::array_t<data_type>(4);
        quat_multiply(
            static_cast<data_type *>(q1.request().ptr),
            static_cast<data_type *>(q2.request().ptr),
            static_cast<data_type *>(result.request().ptr)
        );
        return result;
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    py::array_t<data_type> quat_apply_np(
        py::array_t<data_type, py::array::c_style> q,
        py::array_t<data_type, py::array::c_style> v)
    {
        #if ROTATION_CHECK_INPUT_SIZE
        if (q.size() < 4 || v.size() < 3)
        {
            throw py::index_error("shape should be larger than 4");
        }
        #endif
        auto result = py::array_t<data_type>(3);
        quat_apply(
            static_cast<data_type *>(q.request().ptr),
            static_cast<data_type *>(v.request().ptr),
            static_cast<data_type *>(result.request().ptr)
        );
        return result;
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    py::array_t<data_type> quat_to_matrix_np(py::array_t<data_type, py::array::c_style> q)
    {
        #if ROTATION_CHECK_INPUT_SIZE
        if (q.size() < 4)
        {
            throw py::index_error("shape should be larger than 4");
        }
        #endif
        auto result = py::array_t<data_type>({3, 3});
        quat_to_matrix(
            static_cast<data_type *>(q.request().ptr),
            static_cast<data_type *>(result.request().ptr)
        );
        return result;
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    py::array_t<data_type> quat_from_matrix_np(py::array_t<data_type, py::array::c_style> m)
    {
        #if ROTATION_CHECK_INPUT_SIZE
            if (m.size() < 9)
            {
                throw py::index_error("shape should be (3, 3)");
            }
        #endif
        auto result = py::array_t<data_type>(4);
        quat_from_matrix(
            static_cast<data_type *>(m.request().ptr),
            static_cast<data_type *>(result.request().ptr)
        );
        return result;
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    py::array_t<data_type> cross_product_np(
         py::array_t<data_type, py::array::c_style> a,
         py::array_t<data_type, py::array::c_style> b)
    {
        #if ROTATION_CHECK_INPUT_SIZE
            if (a.size() < 3 || b.size() < 3)
            {
                throw py::index_error("shape should be (3,)");
            }
        #endif
        auto result = py::array_t<data_type>(3);
        cross_product(
            static_cast<data_type *>(a.request().ptr),
            static_cast<data_type *>(b.request().ptr),
            static_cast<data_type *>(result.request().ptr)
        );
        return result;
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    py::array_t<data_type> vector_to_cross_matrix_np(
         py::array_t<data_type, py::array::c_style> a)
    {
        #if ROTATION_CHECK_INPUT_SIZE
            if (a.size() < 3)
            {
                throw py::index_error("shape should be (3,)");
            }
        #endif
        auto result = py::array_t<data_type>({3, 3});
        vector_to_cross_matrix(
            static_cast<data_type *>(a.request().ptr),
            static_cast<data_type *>(result.request().ptr)
        );
        return result;
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    py::array_t<data_type> quat_to_rotvec_np(
         py::array_t<data_type, py::array::c_style> q)
    {
        #if ROTATION_CHECK_INPUT_SIZE
            if (q.size() < 4)
            {
                throw py::index_error("shape should be (4,)");
            }
        #endif
        auto result = py::array_t<data_type>(3);
        data_type angle;
        quat_to_rotvec(
            static_cast<data_type *>(q.request().ptr), angle,
            static_cast<data_type *>(result.request().ptr)
        );
        return result;
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    py::array_t<data_type> quat_from_rotvec_np(py::array_t<data_type, py::array::c_style> x)
    {
        #if ROTATION_CHECK_INPUT_SIZE
        if (x.size() < 3)
        {
            throw py::index_error("shape should be (3,)");
        }
        #endif
        auto result = py::array_t<data_type>(4);
        quat_from_rotvec(
            static_cast<data_type *>(x.request().ptr),
            static_cast<data_type *>(result.request().ptr)
        );
        return result;
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    py::array_t<data_type> vector_normalize_np(py::array_t<data_type, py::array::c_style> x)
    {
        auto size = x.size();
        auto result = py::array_t<data_type>(size);
        vector_normalize(
            static_cast<data_type *>(x.request().ptr), size,
            static_cast<data_type *>(result.request().ptr)
        );
        return result;
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    py::array_t<data_type> quat_integrate_np(
        py::array_t<data_type, py::array::c_style> q,
        py::array_t<data_type, py::array::c_style> omega,
        data_type dt
    )
    {
        #if ROTATION_CHECK_INPUT_SIZE
        if (q.size() < 4 || omega.size() < 3)
        {
            throw py::index_error("shape should be (3,)");
        }
        #endif
        auto result = py::array_t<data_type>(4);
        quat_integrate(
            static_cast<data_type *>(q.request().ptr),
            static_cast<data_type *>(omega.request().ptr),
            dt,
            static_cast<data_type *>(result.request().ptr)
        );
        return result;
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void build_pybind11_wrapper(pybind11::module_ m)
    {
        m.def("quat_multiply", &quat_multiply_np<data_type>);
        m.def("quat_apply", &quat_apply_np<data_type>);
        m.def("quat_to_matrix", &quat_to_matrix_np<data_type>);
        m.def("quat_from_matrix", &quat_from_matrix_np<data_type>);
        m.def("cross_product", &cross_product_np<data_type>);
        m.def("vector_to_cross_matrix", &vector_to_cross_matrix_np<data_type>);
        m.def("quat_to_rotvec", &quat_to_rotvec_np<data_type>);
        m.def("quat_from_rotvec", &quat_from_rotvec_np<data_type>);
        m.def("vector_normalize", &vector_normalize_np<data_type>);
        m.def("quat_integrate", &quat_integrate_np<data_type>);
    }
    /* void build_pybind11_wrapper(pybind11::module m)
    {
        m.doc() = "";

        m.def("quat_to_vec6d", nullptr);
        m.def("clip_vec", nullptr);
        m.def("clip_vec_by_length", nullptr);
        m.def("mat22_det", nullptr);
        m.def("mat33_det", nullptr);
        m.def("mat33_svd", nullptr);
    } */

    void build_wrapper(pybind11::module_ m)
    {
        auto impl32 = m.def_submodule("impl32", "implementation in float32");
        auto impl64 = m.def_submodule("impl64", "implementation in float64");
        build_pybind11_wrapper<float>(impl32);
        build_pybind11_wrapper<double>(impl64);
        build_pybind11_wrapper<double>(m);
    }
}


