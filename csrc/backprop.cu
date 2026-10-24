#include "commons.cuh"
#include "ops.h"

// One training step for a 2-layer MLP: X (B x D), W1 (D x H), b1 (H), W2 (H x C), b2 (C), y (B)

__global__ void relu_bwd_kernel(const float* dA, const float* Z, float* dZ, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        dZ[i] = (Z[i] > 0.0f) ? dA[i] : 0.0f;
    }
}

static torch::Tensor relu_bwd_cuda(torch::Tensor dA, torch::Tensor Z){
    auto dZ = torch::empty_like(Z);
    int n = Z.numel();
    if (n == 0) return dZ;

    auto stream = c10::cuda::getCurrentCUDAStream();
    relu_bwd_kernel<<<ceil_div(n, THREADS), THREADS, 0, stream>>>(
        dA.data_ptr<float>(), Z.data_ptr<float>(), dZ.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dZ;
}

// W1, b1, W2, b2 are updated in place; returns the mean loss of the batch (before the update)
torch::Tensor train_step_cuda(torch::Tensor X, torch::Tensor y,
                              torch::Tensor W1, torch::Tensor b1,
                              torch::Tensor W2, torch::Tensor b2, float lr){
    auto Z1 = biasadd_cuda(X, W1, b1);
    auto A1 = relu_cuda(Z1);
    auto logits = biasadd_cuda(A1, W2, b2);
    auto [probs, losses] = softmax_fwd_ce_cuda(logits, y);

    auto dlogits = softmax_bwd_ce_cuda(probs, y);
    auto dW2 = matmul_cuda(A1.t().contiguous(), dlogits);
    auto db2 = colsum_cuda(dlogits);
    auto dA1 = matmul_cuda(dlogits, W2.t().contiguous());
    auto dZ1 = relu_bwd_cuda(dA1, Z1);
    auto dW1 = matmul_cuda(X.t().contiguous(), dZ1);
    auto db1 = colsum_cuda(dZ1);

    // update only after every gradient is computed (dA1 needs the old W2)
    sgd_step_cuda(W1, dW1, lr);
    sgd_step_cuda(b1, db1, lr);
    sgd_step_cuda(W2, dW2, lr);
    sgd_step_cuda(b2, db2, lr);

    return losses.mean();
}
