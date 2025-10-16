#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <algorithm>
#include "include/functions/matmul/matmul.hpp"

#define TILE_SIZE 32

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
        } \
    } while(0)

__global__ void kernel(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    const size_t M, const size_t K, const size_t N,
    const size_t D,
    const size_t * __restrict__ Aoffs,  // length D (elements, not bytes)
    const size_t * __restrict__ Boffs   // length D
) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    const size_t bx = blockIdx.x, by = blockIdx.y;
    const size_t depth = blockIdx.z;
    const size_t tx = threadIdx.x, ty = threadIdx.y;

    const size_t row = by * TILE_SIZE + ty;
    const size_t col = bx * TILE_SIZE + tx;

    // Per-batch base pointers adjusted for broadcasting
    const float* Abase = A + Aoffs[depth];
    const float* Bbase = B + Boffs[depth];

    float sum = 0.0f;

    const size_t tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (size_t t = 0; t < tiles; ++t) {
        const size_t kA = t * TILE_SIZE + tx;
        const size_t kB = t * TILE_SIZE + ty;

        if (row < M && kA < K) {
            // A slice layout: [M, K] row-major
            sharedA[ty][tx] = Abase[row * K + kA];
        } else {
            sharedA[ty][tx] = 0.0f;
        }

        if (col < N && kB < K) {
            // B slice layout: [K, N] row-major
            sharedB[ty][tx] = Bbase[kB * N + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (size_t k = 0; k < TILE_SIZE; ++k) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N && depth < D) {
        // C is laid out as [D, M, N] row-major (flattened D)
        C[depth * M * N + row * N + col] = sum;
    }
}

// Helper to compute broadcasted batch shape and offsets for A and B
static void compute_batch_shapes_and_offsets(
    const std::vector<size_t>& a_shape,  // includes M,K at end
    const std::vector<size_t>& b_shape,  // includes K,N at end
    size_t M, size_t K, size_t N,
    std::vector<size_t>& c_batch_shape,  // OUT: broadcasted batch shape
    std::vector<size_t>& Aoffs,          // OUT: length D (elements)
    std::vector<size_t>& Boffs           // OUT: length D (elements)
) {
    const size_t a_nd = a_shape.size();
    const size_t b_nd = b_shape.size();

    const size_t a_batch_nd = (a_nd >= 2) ? (a_nd - 2) : 0;
    const size_t b_batch_nd = (b_nd >= 2) ? (b_nd - 2) : 0;
    const size_t batch_nd   = std::max(a_batch_nd, b_batch_nd);

    // Build c_batch_shape by broadcasting each batch dim (higher dims first)
    c_batch_shape.clear();
    c_batch_shape.resize(batch_nd, 1);

    for (size_t i = 0; i < batch_nd; ++i) {
        const bool inA = (i < a_batch_nd);
        const bool inB = (i < b_batch_nd);
        const size_t a_dim = inA ? a_shape[i] : 1;
        const size_t b_dim = inB ? b_shape[i] : 1;

        if (!(a_dim == b_dim || a_dim == 1 || b_dim == 1)) {
            throw std::runtime_error("Incompatible batch dimensions for broadcasting.");
        }
        c_batch_shape[i] = std::max(a_dim, b_dim);
    }

    // Compute batch strides for contiguous row-major
    // For A: slice_size = M*K, for B: slice_size = K*N
    std::vector<size_t> Astride(batch_nd, 0), Bstride(batch_nd, 0);
    {
        size_t cur = M * K;
        for (size_t i = batch_nd; i-- > 0;) {
            const size_t a_dim = (i < a_batch_nd) ? a_shape[i] : 1;
            Astride[i] = cur;   // stride in elements for this batch dim
            cur *= a_dim;
        }
    }
    {
        size_t cur = K * N;
        for (size_t i = batch_nd; i-- > 0;) {
            const size_t b_dim = (i < b_batch_nd) ? b_shape[i] : 1;
            Bstride[i] = cur;
            cur *= b_dim;
        }
    }

    // Effective strides: zero out where operand is broadcast (dim==1)
    std::vector<size_t> Aestr(batch_nd, 0), Bestr(batch_nd, 0);
    for (size_t i = 0; i < batch_nd; ++i) {
        const size_t a_dim = (i < a_batch_nd) ? a_shape[i] : 1;
        const size_t b_dim = (i < b_batch_nd) ? b_shape[i] : 1;
        Aestr[i] = (a_dim == 1) ? 0 : Astride[i];
        Bestr[i] = (b_dim == 1) ? 0 : Bstride[i];
    }

    // Total batches D is the product of c_batch_shape
    size_t D = 1;
    for (size_t d : c_batch_shape) D *= d;

    Aoffs.resize(D);
    Boffs.resize(D);

    // For each flattened batch index, decode into multi-index and compute offsets
    for (size_t b = 0; b < D; ++b) {
        size_t tmp = b;
        size_t offA = 0, offB = 0;
        for (size_t i = batch_nd; i-- > 0;) {
            const size_t dim = c_batch_shape[i];
            const size_t idx = (dim == 0) ? 0 : (tmp % dim);
            tmp /= (dim == 0 ? 1 : dim);

            offA += idx * Aestr[i];
            offB += idx * Bestr[i];
        }
        Aoffs[b] = offA;
        Boffs[b] = offB;
    }
}

// Matmul operation with broadcasting support
float* matmul(
    const float* d_A,
    const float* d_B,
    const std::vector<size_t>& a_shape,  // includes M,K
    const std::vector<size_t>& b_shape,  // includes K,N
    size_t M, size_t N, size_t K
) {
    // Compute per-batch offsets on host
    std::vector<size_t> c_batch_shape, Aoffs_h, Boffs_h;
    compute_batch_shapes_and_offsets(a_shape, b_shape, M, K, N,
                                     c_batch_shape, Aoffs_h, Boffs_h);

    // Flattened D
    size_t D = 1;
    for (size_t d : c_batch_shape) D *= d;

    // Allocate device memory for C and for offsets
    float *d_C = nullptr;
    size_t *d_Aoffs = nullptr, *d_Boffs = nullptr;

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&d_C, D * M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Aoffs, D * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_Boffs, D * sizeof(size_t)));

    CUDA_CHECK(cudaMemcpy(d_Aoffs, Aoffs_h.data(), D * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Boffs, Boffs_h.data(), D * sizeof(size_t), cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE,
              D);

    // Launch
    kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N, D, d_Aoffs, d_Boffs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free offsets device memory
    CUDA_CHECK(cudaFree(d_Aoffs));
    CUDA_CHECK(cudaFree(d_Boffs));

    return d_C;
}