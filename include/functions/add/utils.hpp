#ifndef ADD_UTILS_HPP
#define ADD_UTILS_HPP

#include <pybind11/numpy.h>
#include <vector>
#include <stdexcept>
#include "include/fastpy/array.hpp"

namespace py = pybind11;

namespace add_utils {

fastArray* create_result_array(
    float* result,
    const std::vector<size_t>& a_shape,
    const std::vector<size_t>& b_shape
);

}

#endif