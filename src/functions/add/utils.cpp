#include "include/functions/add/utils.hpp"
#include "include/fastpy/array.hpp"
#include <algorithm>
#include <cstring>

namespace add_utils {

fastArray* create_result_array(
    float* result,
    const std::vector<size_t>& a_shape,
    const std::vector<size_t>& b_shape
) {
    size_t result_ndim = std::max(a_shape.size(), b_shape.size());
    
    // Create numpy array from result
    std::vector<size_t> shape;
    
    // Add higher dimensions first
    for (size_t i = 0; i < result_ndim; i++) {
        size_t a_dim = (i < a_shape.size()) ? a_shape[i] : 1;
        size_t b_dim = (i < b_shape.size()) ? b_shape[i] : 1;
        shape.push_back(std::max(a_dim, b_dim));
    }
    
    return new fastArray(result, shape, CUDA);
}
}

