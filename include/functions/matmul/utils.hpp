#ifndef MATMUL_UTILS_HPP
#define MATMUL_UTILS_HPP

#include <pybind11/numpy.h>
#include <vector>
#include <stdexcept>
#include "include/fastpy/array.hpp"

namespace py = pybind11;

namespace matmul_utils {

struct MatmulInputs {
    float* a_ptr;
    float* b_ptr;
    size_t M, N, K, D;
};

MatmulInputs prepare_matmul_inputs(
    fastArray& A,
    fastArray& B
);

fastArray* create_result_array(
    float* result,
    const std::vector<size_t>& a_shape,
    const std::vector<size_t>& b_shape,
    size_t M, size_t N, size_t D
);

}

#endif