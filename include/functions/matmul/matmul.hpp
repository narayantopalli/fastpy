#ifndef MATMUL_HPP
#define MATMUL_HPP

#include <vector>
#include <cstddef>

float* matmul(const float* d_A, const float* d_B, const std::vector<size_t>& a_shape, const std::vector<size_t>& b_shape, size_t M, size_t N, size_t K);

#endif