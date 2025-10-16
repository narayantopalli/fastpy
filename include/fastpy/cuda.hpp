#ifndef CUDA_HPP
#define CUDA_HPP

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
        } \
    } while(0)

// CUDA memory management functions
void* cuda_malloc(size_t size);
void cuda_free(void* ptr);
void cuda_memcpy_host_to_device(void* dst, const void* src, size_t size);
void cuda_memcpy_device_to_host(void* dst, const void* src, size_t size);
void cuda_memcpy_device_to_device(void* dst, const void* src, size_t size);

#endif