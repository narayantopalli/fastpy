#ifndef ARRAY_HPP
#define ARRAY_HPP

#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

#define CPU 0
#define CUDA 1

class fastArray {
    public:
        fastArray(py::array_t<float, py::array::c_style | py::array::forcecast> A, int device_type = CPU);
        fastArray(float* ptr, std::vector<size_t> shape, int device_type = CPU);
        ~fastArray();
        void to_device(int device_type);
        py::array_t<float> to_numpy();
        size_t ndim() const { return shape.size(); }
        const std::vector<size_t>& get_shape() const { return shape; }
        int get_device_type() const { return device_type; }
        float* ptr;
        size_t size;
    private:
        int device_type;
        std::vector<size_t> shape;
};

#endif