#include "commons.cuh"
#include "ops.h"

__global__ void relu_kernel(const float *A, float *out, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n) out[i] = A[i] > 0.f ? A[i] : 0.f;
}

torch::Tensor relu_cuda(torch::Tensor a){
    auto out = torch::empty_like(a);
    int n = a.numel();
    if (n==0) return out;
    auto stream = c10::cuda::getCurrentCUDAStream();
    relu_kernel<<<ceil_div(n, THREADS), THREADS, 0, stream>>>(
        a.data_ptr<float>(), out.data_ptr<float>(), n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}