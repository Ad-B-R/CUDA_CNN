#include "commons.cuh"
#include "ops.h"

// One SGD step, in place: param[i] -= lr * grad[i]. One thread per element.
__global__ void sgd_step_kernel(float* param, const float* grad, float lr, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        param[i] -= lr * grad[i];
    }
}

// updates param in place (works for weights of any shape and for biases)
void sgd_step_cuda(torch::Tensor param, torch::Tensor grad, float lr){
    int n = param.numel();
    if (n == 0) return;

    auto stream = c10::cuda::getCurrentCUDAStream();
    sgd_step_kernel<<<ceil_div(n, THREADS), THREADS, 0, stream>>>(
        param.data_ptr<float>(), grad.data_ptr<float>(), lr, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
