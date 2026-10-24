#include "commons.cuh"
#include "ops.h"

struct ConvShape {
    int N, C, H, W;
    int OC, KH, KW, OH, OW;
    int sh, sw, ph, pw, dh, dw;
};

// cols[k][p] = x[n][c][h][w] for the input pixel that weight k touches when producing output pixel p
// (0 where that pixel falls in the padding).
// TRANSPOSED writes cols^T (P x K) instead, which is what the weight gradient needs.
template <bool TRANSPOSED>
__global__ void im2col_kernel(const float* __restrict__ x, float* __restrict__ cols, ConvShape s){
    const int64_t K = (int64_t)s.C * s.KH * s.KW;
    const int64_t P = (int64_t)s.N * s.OH * s.OW;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * P) return;

    // consecutive threads take consecutive output entries, so the writes are coalesced in either layout
    int64_t k = TRANSPOSED ? idx % K : idx / P;
    int64_t p = TRANSPOSED ? idx / K : idx % P;

    int kw = k % s.KW;
    int kh = (k / s.KW) % s.KH;
    int c  = k / (s.KW * s.KH);
    int ow = p % s.OW;
    int oh = (p / s.OW) % s.OH;
    int n  = p / (s.OW * s.OH);

    int h = oh * s.sh - s.ph + kh * s.dh;
    int w = ow * s.sw - s.pw + kw * s.dw;
    bool inside = (h >= 0 && h < s.H && w >= 0 && w < s.W);
    cols[idx] = inside ? x[(((int64_t)n * s.C + c) * s.H + h) * s.W + w] : 0.0f;
}

__global__ void col2im_kernel(const float* __restrict__ dcols, float* __restrict__ dx, ConvShape s){
    const int64_t P = (int64_t)s.N * s.OH * s.OW;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)s.N * s.C * s.H * s.W) return;

    int w = idx % s.W;
    int h = (idx / s.W) % s.H;
    int c = (idx / ((int64_t)s.W * s.H)) % s.C;
    int n = idx / ((int64_t)s.W * s.H * s.C);

    float sum = 0.0f;
    for (int kh = 0; kh < s.KH; kh++){
        int h_off = h + s.ph - kh * s.dh;
        if (h_off < 0 || h_off % s.sh != 0) continue;
        int oh = h_off / s.sh;
        if (oh >= s.OH) continue;

        for (int kw = 0; kw < s.KW; kw++){
            int w_off = w + s.pw - kw * s.dw;
            if (w_off < 0 || w_off % s.sw != 0) continue;
            int ow = w_off / s.sw;
            if (ow >= s.OW) continue;

            int64_t k = ((int64_t)c * s.KH + kh) * s.KW + kw;
            int64_t p = ((int64_t)n * s.OH + oh) * s.OW + ow;
            sum += dcols[k * P + p];
        }
    }
    dx[idx] = sum;
}

// y[n][oc][s] = y2[oc][n][s] + b[oc]   (b may be null). One thread per element of y.
__global__ void cnhw_to_nchw_bias_kernel(const float* __restrict__ y2, const float* __restrict__ b,
                                         float* __restrict__ y, int N, int OC, int S){
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)N * OC * S) return;

    int sp = idx % S;
    int oc = (idx / S) % OC;
    int n  = idx / ((int64_t)S * OC);
    float bias = b ? b[oc] : 0.0f;
    y[idx] = y2[((int64_t)oc * N + n) * S + sp] + bias;
}

// dy2[oc][n][s] = dy[n][oc][s]: puts the output gradient back into the (OC x P) matmul layout.
__global__ void nchw_to_cnhw_kernel(const float* __restrict__ dy, float* __restrict__ dy2, int N, int OC, int S){
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)N * OC * S) return;

    int sp = idx % S;
    int n  = (idx / S) % N;
    int oc = idx / ((int64_t)S * N);
    dy2[idx] = dy[((int64_t)n * OC + oc) * S + sp];
}

// out[r] = sum of row r of X (M x L). One block per row; same reduction as colsum_kernel.
// Used for the bias gradient: db[oc] = sum of dy2[oc][:] over every sample and pixel.
__global__ void rowsum_kernel(const float* X, float* out, int64_t L){
    __shared__ float warp_sums[THREADS / 32];
    const float* row = X + (int64_t)blockIdx.x * L;
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    // the row is contiguous, so neighbouring threads read neighbouring floats
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < L; i += blockDim.x){
        sum += row[i];
    }

    for (int offset = 16; offset > 0; offset /= 2){
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0){
        sum = (lane < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2){
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) out[blockIdx.x] = sum;
    }
}

static int64_t conv_out_size(int64_t in, int64_t k, int64_t stride, int64_t pad, int64_t dil){
    return (in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
}

static ConvShape make_shape(const torch::Tensor& x, const torch::Tensor& W, const Conv2dParams& prm){
    ConvShape s;
    s.N = x.size(0);  s.C = x.size(1);  s.H = x.size(2);  s.W = x.size(3);
    s.OC = W.size(0); s.KH = W.size(2); s.KW = W.size(3);
    s.sh = prm.stride_h; s.sw = prm.stride_w;
    s.ph = prm.pad_h;    s.pw = prm.pad_w;
    s.dh = prm.dil_h;    s.dw = prm.dil_w;
    s.OH = conv_out_size(s.H, s.KH, s.sh, s.ph, s.dh);
    s.OW = conv_out_size(s.W, s.KW, s.sw, s.pw, s.dw);
    return s;
}

static int64_t blocks_for(int64_t n){
    return (n + THREADS - 1) / THREADS;
}

// returns cols (K x P), or cols^T (P x K) when transposed
static torch::Tensor im2col_cuda(const torch::Tensor& x, const ConvShape& s, bool transposed){
    const int64_t K = (int64_t)s.C * s.KH * s.KW;
    const int64_t P = (int64_t)s.N * s.OH * s.OW;
    auto cols = transposed ? torch::empty({P, K}, x.options()) : torch::empty({K, P}, x.options());

    auto stream = c10::cuda::getCurrentCUDAStream();
    if (transposed){
        im2col_kernel<true><<<blocks_for(K * P), THREADS, 0, stream>>>(x.data_ptr<float>(), cols.data_ptr<float>(), s);
    } else {
        im2col_kernel<false><<<blocks_for(K * P), THREADS, 0, stream>>>(x.data_ptr<float>(), cols.data_ptr<float>(), s);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return cols;
}

torch::Tensor conv2d_fwd_cuda(torch::Tensor x, torch::Tensor W, std::optional<torch::Tensor> b, Conv2dParams prm){
    const ConvShape s = make_shape(x, W, prm);
    const int S = s.OH * s.OW;
    auto y = torch::empty({s.N, s.OC, s.OH, s.OW}, x.options());
    if (y.numel() == 0) return y;
    if (s.C == 0){
        // nothing to sum over: the output is just the bias
        y.zero_();
        if (b) y.add_(b->view({1, s.OC, 1, 1}));
        return y;
    }

    auto cols = im2col_cuda(x, s, false);
    auto y2 = matmul_cuda(W.view({s.OC, -1}), cols);

    auto stream = c10::cuda::getCurrentCUDAStream();
    const float* b_ptr = b ? b->data_ptr<float>() : nullptr;
    cnhw_to_nchw_bias_kernel<<<blocks_for(y.numel()), THREADS, 0, stream>>>(
        y2.data_ptr<float>(), b_ptr, y.data_ptr<float>(), s.N, s.OC, S);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// dy is the gradient of the loss w.r.t. y (N, OC, OH, OW); returns (dx, dW, db)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conv2d_bwd_cuda(torch::Tensor dy, torch::Tensor x,
                                                                        torch::Tensor W, Conv2dParams prm){
    const ConvShape s = make_shape(x, W, prm);
    const int S = s.OH * s.OW;
    const int64_t P = (int64_t)s.N * S;
    auto dx = torch::zeros_like(x);
    auto dW = torch::zeros_like(W);
    auto db = torch::zeros({s.OC}, W.options());
    if (P == 0 || s.OC == 0) return {dx, dW, db};

    auto stream = c10::cuda::getCurrentCUDAStream();

    // dy in the (OC x P) layout the forward matmul produced
    auto dy2 = torch::empty({s.OC, P}, dy.options());
    nchw_to_cnhw_kernel<<<blocks_for(dy2.numel()), THREADS, 0, stream>>>(
        dy.data_ptr<float>(), dy2.data_ptr<float>(), s.N, s.OC, S);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // db[oc] = sum of dy over the batch and every pixel
    rowsum_kernel<<<s.OC, THREADS, 0, stream>>>(dy2.data_ptr<float>(), db.data_ptr<float>(), P);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (s.C == 0) return {dx, dW, db};

    // dW (OC x K) = dy2 (OC x P) @ cols^T (P x K); cols^T is built directly instead of transposing cols.
    // scoped so the (large) unrolled input is freed before dcols is allocated
    {
        auto colsT = im2col_cuda(x, s, true);
        dW = matmul_cuda(dy2, colsT).view(W.sizes());
    }

    // dcols (K x P) = W^T (K x OC) @ dy2 (OC x P), then fold the columns back onto the input pixels
    auto Wt = W.view({s.OC, -1}).t().contiguous();
    auto dcols = matmul_cuda(Wt, dy2);
    // padding alone can produce outputs from an empty (H or W = 0) input; dx then has nothing to fill
    if (dx.numel() == 0) return {dx, dW, db};
    col2im_kernel<<<blocks_for(dx.numel()), THREADS, 0, stream>>>(
        dcols.data_ptr<float>(), dx.data_ptr<float>(), s);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {dx, dW, db};
}
