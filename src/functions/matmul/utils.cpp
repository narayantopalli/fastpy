#include "include/functions/matmul/utils.hpp"
#include "include/fastpy/array.hpp"
#include <algorithm>
#include <cstring>

namespace matmul_utils {

MatmulInputs prepare_matmul_inputs(
    fastArray& A,
    fastArray& B
) {
    size_t result_ndim = std::max(A.ndim(), B.ndim());

    if (A.ndim() < 2 || B.ndim() < 2) {
        throw std::runtime_error("Matrix A and B must be at least 2D arrays");
    }

    if (A.get_device_type() == CPU || B.get_device_type() == CPU) {
        throw std::runtime_error("Arguments must be on GPU!");
    }

    // Extract matrix dimensions from the last two dimensions
    size_t M = A.get_shape()[A.ndim() - 2];
    size_t K_A = A.get_shape()[A.ndim() - 1];
    
    size_t K_B = B.get_shape()[B.ndim() - 2];
    size_t N = B.get_shape()[B.ndim() - 1];

    // Calculate D (product of higher dimensions - all except last 2)
    size_t D = 1;
    for (size_t i = 0; i < result_ndim - 2; i++) {
        // For the matrix with fewer dimensions, treat missing dimensions as 1
        size_t a_dim = (i < A.ndim() - 2) ? A.get_shape()[i] : 1;
        size_t b_dim = (i < B.ndim() - 2) ? B.get_shape()[i] : 1;
        
        // Both matrices should have the same size in higher dimensions
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            throw std::runtime_error("Matrix dimensions mismatch at dimension " + std::to_string(i) + 
                                   ": A.shape[" + std::to_string(i) + "] = " + std::to_string(a_dim) + 
                                   ", B.shape[" + std::to_string(i) + "] = " + std::to_string(b_dim));
        }
        
        D *= std::max(a_dim, b_dim);
    }

    // Validate that K dimensions match
    if (K_A != K_B) {
        throw std::runtime_error("Matrix multiplication dimension mismatch: A.shape[-1] (" + 
                               std::to_string(K_A) + ") != B.shape[-2] (" + std::to_string(K_B) + ")");
    }
    
    return {A.ptr, B.ptr, M, N, K_A, D};
}

fastArray* create_result_array(
    float* result,
    const std::vector<size_t>& a_shape,
    const std::vector<size_t>& b_shape,
    size_t M, size_t N, size_t D
) {
    size_t result_ndim = std::max(a_shape.size(), b_shape.size());
    
    // Create numpy array from result
    std::vector<size_t> shape;
    
    // Add higher dimensions first
    for (size_t i = 0; i < result_ndim - 2; i++) {
        size_t a_dim = (i < a_shape.size() - 2) ? a_shape[i] : 1;
        size_t b_dim = (i < b_shape.size() - 2) ? b_shape[i] : 1;
        shape.push_back(std::max(a_dim, b_dim));
    }
    
    // Add matrix dimensions last
    shape.push_back(M);
    shape.push_back(N);
    
    return new fastArray(result, shape, CUDA);
}

}
