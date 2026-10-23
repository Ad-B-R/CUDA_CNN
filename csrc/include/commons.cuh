#pragma once
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

constexpr int THREADS = 256;
inline int ceil_div(int a, int b) { return (a + b - 1) / b; }