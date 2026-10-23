#include "commons.cuh"
#include "ops.h"

// Y is the (M x N) matmul output, row-major; b has N entries, one per output column.
// One thread per element of Y; adds the bias in place.
__global__ void biasadd_kernel(float* __restrict__ Y, const float* __restrict__ b, int M, int N){
    // TODO: write the kernel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if(idx<total){
        int j = idx%N;
        Y[idx] += b[j];
    }

}

// Y = X @ W + b    X: (M x K), W: (K x N), b: (N)
torch::Tensor biasadd_cuda(torch::Tensor X, torch::Tensor W, torch::Tensor b){
    auto Y = matmul_cuda(X, W);
    const int M = Y.size(0), N = Y.size(1);
    if (M==0 || N==0) return Y;

    auto stream = c10::cuda::getCurrentCUDAStream();
    biasadd_kernel<<<ceil_div(M*N, THREADS), THREADS, 0, stream>>>(
        Y.data_ptr<float>(), b.data_ptr<float>(), M, N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}
