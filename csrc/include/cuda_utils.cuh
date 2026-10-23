#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x); \
    CHECK_FLOAT(x)

constexpr int THREADS_PER_BLOCK = 256;
constexpr int TILE_DIM = 16;

inline int num_blocks(int64_t n, int threads = THREADS_PER_BLOCK) {
    return static_cast<int>((n + threads - 1) / threads);
}

inline cudaStream_t current_stream() {
    return at::cuda::getCurrentCUDAStream();
}
