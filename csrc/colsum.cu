#include "commons.cuh"
#include "ops.h"

// out[j] = sum of column j of X (M x N). One block per column.
__global__ void colsum_kernel(const float* X, float* out, int M, int N){
    __shared__ float warp_sums[THREADS / 32];
    int col = blockIdx.x;
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    // each thread adds up every 256th row of the column
    float sum = 0.0f;
    for (int row = threadIdx.x; row < M; row += blockDim.x){
        sum += X[row * N + col];
    }

    // each warp combines its 32 sums by reading other threads' registers; lane 0 ends up with the total
    for (int offset = 16; offset > 0; offset /= 2){
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    // the first warp combines the per-warp results
    if (warp == 0){
        sum = (lane < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2){
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) out[col] = sum;
    }
}

// out = X.sum(0)
torch::Tensor colsum_cuda(torch::Tensor X){
    const int M = X.size(0), N = X.size(1);
    auto out = torch::zeros({N}, X.options());
    if (M == 0 || N == 0) return out;

    auto stream = c10::cuda::getCurrentCUDAStream();
    colsum_kernel<<<N, THREADS, 0, stream>>>(X.data_ptr<float>(), out.data_ptr<float>(), M, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
