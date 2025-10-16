#include "include/fastpy/array.hpp"
#include "include/fastpy/cuda.hpp"
#include <iostream>
#include <cstring>
#include <stdexcept>

// Constructor
fastArray::fastArray(py::array_t<float, py::array::c_style | py::array::forcecast> A, int device_type) 
    : device_type(device_type) {
    
    auto buffer_info = A.request();
    size = buffer_info.size;
    
    // Copy shape information to the shape vector
    shape.resize(buffer_info.ndim);
    for (size_t i = 0; i < buffer_info.ndim; ++i) {
        shape[i] = static_cast<size_t>(buffer_info.shape[i]);
    }
    
    size_t size_bytes = size * sizeof(float);
    
    if (device_type == CPU) {
        // Allocate CPU memory and copy data
        ptr = new float[size];
        std::memcpy(ptr, buffer_info.ptr, size_bytes);
    }
    else if (device_type == CUDA) {
        // Allocate GPU memory
        ptr = static_cast<float*>(cuda_malloc(size_bytes));
        
        // Copy data from host to device
        cuda_memcpy_host_to_device(ptr, buffer_info.ptr, size_bytes);
    }
    else {
        throw std::invalid_argument("Invalid device type: " + std::to_string(device_type));
    }
}

fastArray::fastArray(float* ptr, std::vector<size_t> shape, int device_type) 
    : device_type(device_type), ptr(ptr), shape(shape) {

    size = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        size *= shape[i];
    }
}

// Destructor
fastArray::~fastArray() {
    if (ptr != nullptr) {
        if (device_type == CPU) {
            delete[] ptr;
        }
        else if (device_type == CUDA) {
            cuda_free(ptr);
        }
        ptr = nullptr;
    }
}

// Move data to specified device
void fastArray::to_device(int device_type) {
    if (this->device_type == device_type) {
        return; // Already on the target device
    }
    
    size_t size_bytes = size * sizeof(float);
    
    if (device_type == CPU) {
        // Moving to CPU
        if (this->device_type == CUDA) {
            // Currently on GPU, copy to CPU
            float* cpu_ptr = new float[size];
            cuda_memcpy_device_to_host(cpu_ptr, ptr, size_bytes);
            
            // Free GPU memory
            cuda_free(ptr);
            
            // Update pointer and device type
            ptr = cpu_ptr;
            this->device_type = CPU;
        }
    }
    else if (device_type == CUDA) {
        // Moving to GPU
        if (this->device_type == CPU) {
            // Currently on CPU, copy to GPU
            float* gpu_ptr = static_cast<float*>(cuda_malloc(size_bytes));
            cuda_memcpy_host_to_device(gpu_ptr, ptr, size_bytes);
            
            // Free CPU memory
            delete[] ptr;
            
            // Update pointer and device type
            ptr = gpu_ptr;
            this->device_type = CUDA;
        }
    }
    else {
        throw std::invalid_argument("Invalid device type: " + std::to_string(device_type));
    }
}

// Convert back to NumPy array
py::array_t<float> fastArray::to_numpy() {
    if (device_type == CPU) {
        py::array_t<float> result(this->shape);
        auto result_buffer = result.request();
        float* result_ptr = static_cast<float*>(result_buffer.ptr);
        
        // Copy data
        size_t size_bytes = this->size * sizeof(float);
        std::memcpy(result_ptr, this->ptr, size_bytes);
        return result;
    }
    else if (device_type == CUDA) {
        // Data is on GPU, need to copy back to CPU
        // Create a new NumPy array with the stored shape
        py::array_t<float> result(shape);
        auto result_buffer = result.request();
        float* result_ptr = static_cast<float*>(result_buffer.ptr);
        
        // Copy data from GPU to CPU
        size_t size_bytes = size * sizeof(float);
        cuda_memcpy_device_to_host(result_ptr, ptr, size_bytes);
        
        return result;
    }
    else {
        throw std::runtime_error("Invalid device type in to_numpy()");
    }
}
