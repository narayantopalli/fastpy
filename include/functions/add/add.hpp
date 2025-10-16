#ifndef ADD_HPP
#define ADD_HPP

#include <vector>
#include <cstddef>

float* add(const float* d_A, const float* d_B, const std::vector<size_t>& a_shape, const std::vector<size_t>& b_shape);

#endif
