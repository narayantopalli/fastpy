#include "include/functions/matmul/matmul.hpp"
#include "include/functions/matmul/utils.hpp"
#include "include/functions/add/add.hpp"
#include "include/functions/add/utils.hpp"
#include "include/fastpy/array.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_array(py::module& m) {
    py::class_<fastArray>(m, "fastArray")
        .def(py::init<py::array_t<float, py::array::c_style | py::array::forcecast>, int>(), 
             py::arg("A"), py::arg("device_type") = 0)
        .def("to_device", &fastArray::to_device, py::arg("device_type"))
        .def("to_numpy", &fastArray::to_numpy)
        .def_property_readonly("ndim", &fastArray::ndim)
        .def_property_readonly("shape", &fastArray::get_shape)
        .def_property_readonly("device_type", &fastArray::get_device_type);
    
    // Add constants for device types
    m.attr("CPU") = py::int_(0);
    m.attr("CUDA") = py::int_(1);
}

void bind_matmul(py::module& m) {    
    m.def(
        "matmul"
        , [](fastArray& A, fastArray& B) {
            try {
                // Prepare inputs and handle broadcasting
                auto inputs = matmul_utils::prepare_matmul_inputs(A, B);
                
                // Call the matmul function
                float* result = matmul(inputs.a_ptr, inputs.b_ptr, A.get_shape(), B.get_shape(), inputs.M, inputs.N, inputs.K);
                
                // Create result array
                auto result_array = matmul_utils::create_result_array(result, A.get_shape(), B.get_shape(), inputs.M, inputs.N, inputs.D);
                
                return result_array;
            } catch (const std::exception& e) {
                throw std::runtime_error(e.what());
            }
        }
        , py::arg("A"), py::arg("B")
        , "Perform matrix multiplication C = A @ B on GPU. Automatically infers dimensions from input shapes."
    );
}

void bind_add(py::module& m) {
    m.def(
        "add"
        , [](fastArray& A, fastArray& B) {
            try {
                // Prepare inputs and handle broadcasting
                float* result = add(A.ptr, B.ptr, A.get_shape(), B.get_shape());
                
                // Create result array
                auto result_array = add_utils::create_result_array(result, A.get_shape(), B.get_shape());
                
                return result_array;
            } catch (const std::exception& e) {
                throw std::runtime_error(e.what());
            }
        }
        , py::arg("A"), py::arg("B")
        , "Perform element-wise addition C = A + B on GPU. Automatically infers dimensions from input shapes."
    );
}

PYBIND11_MODULE(fastpy, m) {
    m.doc() = "Fast matrix multiplication using CUDA";
    bind_array(m);
    bind_matmul(m);
    bind_add(m);
}