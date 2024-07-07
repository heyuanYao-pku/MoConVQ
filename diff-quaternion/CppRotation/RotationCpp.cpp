#include <pybind11/pybind11.h>
#include "RotationCppBatchWrapper.hpp"
#include "RotationCppSingleWrapper.hpp"

PYBIND11_MODULE(RotationCpp, m) {
//    auto single_wrap = m.def_submodule("single", "For single input");
//    RotationCppSingleWrapper::build_wrapper(single_wrap);
//    auto multi_wrap = m.def_submodule("batch", "For single input");
//    RotationCppBatchWrapper::build_wrapper(multi_wrap);
    RotationCppBatchWrapper::build_wrapper(m);
}
