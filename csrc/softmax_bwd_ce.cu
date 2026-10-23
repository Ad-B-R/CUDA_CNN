#include "commons.cuh"
#include "ops.h"

// Gradient of the mean cross-entropy loss with respect to the logits:
//   dlogits[r][j] = (probs[r][j] - (j == labels[r] ? 1 : 0)) / M
// probs is the (M x C) output of softmax_fwd_ce. One thread per element.
__global__ void softmax_bwd_ce_kernel(const float* probs, const int64_t* labels,
                                      float* dlogits, int M, int C){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * C) return;

    int row = idx / C;
    int col = idx % C;
    int64_t label = labels[row];

    // an out-of-range label gives NaN, same as the forward loss, so the problem shows up instead of training on it
    if (label < 0 || label >= C){
        dlogits[idx] = NAN;
        return;
    }

    float target = (col == label) ? 1.0f : 0.0f;
    dlogits[idx] = (probs[idx] - target) / M;
}

// returns dlogits (M x C)
torch::Tensor softmax_bwd_ce_cuda(torch::Tensor probs, torch::Tensor labels){
    const int M = probs.size(0), C = probs.size(1);
    auto dlogits = torch::empty_like(probs);
    if (M == 0) return dlogits;

    auto stream = c10::cuda::getCurrentCUDAStream();
    softmax_bwd_ce_kernel<<<ceil_div(M * C, THREADS), THREADS, 0, stream>>>(
        probs.data_ptr<float>(), labels.data_ptr<int64_t>(), dlogits.data_ptr<float>(), M, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dlogits;
}
