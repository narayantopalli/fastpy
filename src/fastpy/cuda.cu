#include "include/fastpy/cuda.hpp"

// Allocate CUDA memory
void* cuda_malloc(size_t size) {
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

// Free CUDA memory
void cuda_free(void* ptr) {
    if (ptr != nullptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

// Copy data from host to device
void cuda_memcpy_host_to_device(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

// Copy data from device to host
void cuda_memcpy_device_to_host(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

// Copy data from device to device
void cuda_memcpy_device_to_device(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}
