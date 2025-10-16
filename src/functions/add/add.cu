#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include "include/functions/add/add.hpp"

#define THREADS_PER_BLOCK 256

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
        } \
    } while(0)

// Helper function to compute output shape and strides for broadcasting
static void compute_batch_shapes_and_offsets(
    const std::vector<size_t>& a_shape,
    const std::vector<size_t>& b_shape,
    std::vector<size_t>& out_shape,
    std::vector<size_t>& Aestr,
    std::vector<size_t>& Bestr
) {
    const size_t a_nd = a_shape.size();
    const size_t b_nd = b_shape.size();
    const size_t ndim = std::max(a_nd, b_nd);
    out_shape.assign(ndim, 1);
    for (size_t i = 0; i < ndim; ++i) {
        const size_t a_dim = (i < ndim - a_nd) ? 1 : a_shape[i - (ndim - a_nd)];
        const size_t b_dim = (i < ndim - b_nd) ? 1 : b_shape[i - (ndim - b_nd)];
        if (!(a_dim == b_dim || a_dim == 1 || b_dim == 1)) throw std::runtime_error("Incompatible dimensions for broadcasting.");
        out_shape[i] = std::max(a_dim, b_dim);
    }
    Aestr.assign(ndim, 0); Bestr.assign(ndim, 0);
    size_t sA = 1, sB = 1;
    for (size_t i = ndim; i-- > 0;) {
        const size_t a_dim = (i < ndim - a_nd) ? 1 : a_shape[i - (ndim - a_nd)];
        const size_t b_dim = (i < ndim - b_nd) ? 1 : b_shape[i - (ndim - b_nd)];
        Aestr[i] = (a_dim == 1) ? 0 : sA;
        Bestr[i] = (b_dim == 1) ? 0 : sB;
        sA *= a_dim; sB *= b_dim;
    }
}

__global__ void kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const size_t ndim,
    const size_t* __restrict__ out_strides,
    const size_t* __restrict__ Aestr,
    const size_t* __restrict__ Bestr,
    const size_t total
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    size_t tmp = idx; size_t offA = 0, offB = 0;
    for (size_t i = 0; i < ndim; ++i) { size_t idim = tmp / out_strides[i]; tmp %= out_strides[i]; offA += idim * Aestr[i]; offB += idim * Bestr[i]; }
    C[idx] = A[offA] + B[offB];
}

// Add operation with broadcasting support
float* add(
    const float* d_A,
    const float* d_B,
    const std::vector<size_t>& a_shape,
    const std::vector<size_t>& b_shape
) {
    std::vector<size_t> out_shape, Aestr_h, Bestr_h;
    compute_batch_shapes_and_offsets(a_shape, b_shape, out_shape, Aestr_h, Bestr_h);
    
    // Total batches D is the product of out_shape
    size_t D = 1;
    for (size_t d : out_shape) D *= d;
    
    const size_t ndim = out_shape.size();

    // Compute output strides
    std::vector<size_t> out_strides(ndim, 1);
    for (size_t i = 0; i < ndim; ++i) {
        size_t s = 1;
        for (size_t j = i + 1; j < ndim; ++j) {
            s *= out_shape[j];
            out_strides[i] = s;
        }
    }
    CUDA_CHECK(cudaSetDevice(0));
    float* d_C = nullptr;
    size_t *d_out_strides = nullptr;
    size_t *d_Aestr = nullptr;
    size_t *d_Bestr = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_C, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_strides, ndim * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_Aestr, ndim * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_Bestr, ndim * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_out_strides, out_strides.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Aestr, Aestr_h.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bestr, Bestr_h.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice));

    // Launch kernel
    const size_t blocks = (D + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    kernel<<<(unsigned int)blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, ndim, d_out_strides, d_Aestr, d_Bestr, D);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free temporary device memory
    CUDA_CHECK(cudaFree(d_out_strides));
    CUDA_CHECK(cudaFree(d_Aestr));
    CUDA_CHECK(cudaFree(d_Bestr));
    
    return d_C;
}