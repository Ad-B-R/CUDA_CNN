#include "commons.cuh"
#include "ops.h"

// Softmax + cross-entropy, one block per row of logits (M x C).
//   probs[r][j] = exp(logits[r][j] - max) / sum
//   losses[r]   = -log(probs[r][label]) = log(sum) + max - logits[r][label]
__global__ void softmax_fwd_ce_kernel(const float* logits, const int64_t* labels,
                                      float* probs, float* losses, int M, int C){
    __shared__ float s_max[THREADS];
    __shared__ float s_sum[THREADS];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    const float* row_logits = logits + (int64_t)row * C;
    float* row_probs = probs + (int64_t)row * C;

    // each thread walks its share of the row, keeping a running max and a running sum of exp(x - max).
    // when a bigger max shows up, the sum so far gets scaled down to match it.
    float my_max = -INFINITY;
    float my_sum = 0.0f;
    for (int j = tid; j < C; j += blockDim.x){
        float x = row_logits[j];
        float new_max = fmaxf(my_max, x);
        my_sum = my_sum * expf(my_max - new_max) + expf(x - new_max);
        my_max = new_max;
    }
    s_max[tid] = my_max;
    s_sum[tid] = my_sum;
    __syncthreads();

    // combine pairs of threads until thread 0 holds the max and sum for the whole row.
    // threads that had no columns (when C < 256) still hold -inf, so they add nothing
    // (computing exp(-inf - -inf) would give NaN).
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if (tid < stride){
            float max_a = s_max[tid];
            float max_b = s_max[tid + stride];
            float new_max = fmaxf(max_a, max_b);

            float sum_a = (max_a == -INFINITY) ? 0.0f : s_sum[tid] * expf(max_a - new_max);
            float sum_b = (max_b == -INFINITY) ? 0.0f : s_sum[tid + stride] * expf(max_b - new_max);

            s_max[tid] = new_max;
            s_sum[tid] = sum_a + sum_b;
        }
        __syncthreads();
    }
    float row_max = s_max[0];
    float row_sum = s_sum[0];

    for (int j = tid; j < C; j += blockDim.x){
        row_probs[j] = expf(row_logits[j] - row_max) / row_sum;
    }

    // loss from the logits directly, so a tiny probability can't turn into log(0).
    // an out-of-range label gives NaN instead of reading outside the row.
    if (tid == 0){
        int64_t label = labels[row];
        bool valid = (label >= 0 && label < C);
        losses[row] = valid ? logf(row_sum) + row_max - row_logits[label] : NAN;
    }
}

// returns (probs, losses): probs is (M x C), losses is (M); the batch loss is losses.mean()
std::tuple<torch::Tensor, torch::Tensor> softmax_fwd_ce_cuda(torch::Tensor logits, torch::Tensor labels){
    const int M = logits.size(0), C = logits.size(1);
    auto probs = torch::empty_like(logits);
    auto losses = torch::empty({M}, logits.options());
    if (M == 0) return {probs, losses};

    auto stream = c10::cuda::getCurrentCUDAStream();
    softmax_fwd_ce_kernel<<<M, THREADS, 0, stream>>>(
        logits.data_ptr<float>(), labels.data_ptr<int64_t>(),
        probs.data_ptr<float>(), losses.data_ptr<float>(), M, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {probs, losses};
}
