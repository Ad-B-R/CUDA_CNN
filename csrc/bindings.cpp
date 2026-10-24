#include <torch/extension.h>
#include "ops.h"

torch::Tensor matmul(torch::Tensor A, torch::Tensor B){
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "matmul: A and B must be CUDA tensors");
    TORCH_CHECK(A.device() == B.device(), "matmul: A and B must be on the same device");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32,
                "matmul: A and B must be float32");
    TORCH_CHECK(A.dim()==2 && B.dim()==2 && A.size(1)==B.size(0),
                "matmul: expected (M x K) @ (K x N), got ", A.sizes(), " @ ", B.sizes());
    return matmul_cuda(A.contiguous(), B.contiguous());
}

torch::Tensor biasadd(torch::Tensor X, torch::Tensor W, torch::Tensor b){
    TORCH_CHECK(X.is_cuda() && W.is_cuda() && b.is_cuda(), "biasadd: X, W and b must be CUDA tensors");
    TORCH_CHECK(X.device() == W.device() && X.device() == b.device(),
                "biasadd: X, W and b must be on the same device");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 && W.scalar_type() == torch::kFloat32 &&
                b.scalar_type() == torch::kFloat32, "biasadd: X, W and b must be float32");
    TORCH_CHECK(X.dim()==2 && W.dim()==2 && X.size(1)==W.size(0),
                "biasadd: expected X (M x K) @ W (K x N), got ", X.sizes(), " @ ", W.sizes());
    TORCH_CHECK(b.dim()==1 && b.size(0)==W.size(1),
                "biasadd: expected b of shape (", W.size(1), "), got ", b.sizes());
    return biasadd_cuda(X.contiguous(), W.contiguous(), b.contiguous());
}

torch::Tensor colsum(torch::Tensor X){
    TORCH_CHECK(X.is_cuda(), "colsum: X must be a CUDA tensor");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32, "colsum: X must be float32");
    TORCH_CHECK(X.dim()==2, "colsum: expected a 2-D (M x N) tensor, got ", X.sizes());
    return colsum_cuda(X.contiguous());
}

torch::Tensor relu(torch::Tensor a){
    TORCH_CHECK(a.is_cuda(), "relu: a must be a CUDA tensor");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "relu: a must be float32");
    return relu_cuda(a.contiguous());
}

// shared checks for the softmax/cross-entropy pair: S is the (M x C) logits or probs
static void check_softmax_inputs(const char* name, const torch::Tensor& S, const torch::Tensor& labels){
    TORCH_CHECK(S.is_cuda() && labels.is_cuda(), name, ": inputs must be CUDA tensors");
    TORCH_CHECK(S.device() == labels.device(), name, ": inputs must be on the same device");
    TORCH_CHECK(S.scalar_type() == torch::kFloat32, name, ": scores must be float32");
    TORCH_CHECK(labels.scalar_type() == torch::kLong, name, ": labels must be int64");
    TORCH_CHECK(S.dim()==2 && S.size(1) > 0, name, ": expected a 2-D (M x C) tensor with C > 0, got ", S.sizes());
    TORCH_CHECK(labels.dim()==1 && labels.size(0)==S.size(0),
                name, ": expected labels of shape (", S.size(0), "), got ", labels.sizes());
}

std::tuple<torch::Tensor, torch::Tensor> softmax_fwd_ce(torch::Tensor logits, torch::Tensor labels){
    check_softmax_inputs("softmax_fwd_ce", logits, labels);
    return softmax_fwd_ce_cuda(logits.contiguous(), labels.contiguous());
}

torch::Tensor softmax_bwd_ce(torch::Tensor probs, torch::Tensor labels){
    check_softmax_inputs("softmax_bwd_ce", probs, labels);
    return softmax_bwd_ce_cuda(probs.contiguous(), labels.contiguous());
}

void sgd_step(torch::Tensor param, torch::Tensor grad, double lr){
    TORCH_CHECK(param.is_cuda() && grad.is_cuda(), "sgd_step: param and grad must be CUDA tensors");
    TORCH_CHECK(param.device() == grad.device(), "sgd_step: param and grad must be on the same device");
    TORCH_CHECK(param.scalar_type() == torch::kFloat32 && grad.scalar_type() == torch::kFloat32,
                "sgd_step: param and grad must be float32");
    TORCH_CHECK(param.sizes() == grad.sizes(),
                "sgd_step: shape mismatch ", param.sizes(), " vs ", grad.sizes());
    // param is updated in place, so it can't be swapped for a contiguous copy like the other inputs
    TORCH_CHECK(param.is_contiguous(), "sgd_step: param must be contiguous");
    sgd_step_cuda(param, grad.contiguous(), static_cast<float>(lr));
}

// stride / padding / dilation come in from Python as (h, w) pairs
static Conv2dParams make_conv_params(const char* name, const std::vector<int64_t>& stride,
                                     const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation){
    TORCH_CHECK(stride.size()==2 && padding.size()==2 && dilation.size()==2,
                name, ": stride, padding and dilation must each be an (h, w) pair");
    TORCH_CHECK(stride[0] > 0 && stride[1] > 0, name, ": stride must be positive, got ", stride);
    TORCH_CHECK(padding[0] >= 0 && padding[1] >= 0, name, ": padding must be non-negative, got ", padding);
    TORCH_CHECK(dilation[0] > 0 && dilation[1] > 0, name, ": dilation must be positive, got ", dilation);
    return Conv2dParams{static_cast<int>(stride[0]), static_cast<int>(stride[1]),
                        static_cast<int>(padding[0]), static_cast<int>(padding[1]),
                        static_cast<int>(dilation[0]), static_cast<int>(dilation[1])};
}

// shared checks for the conv pair: x is (N, C, H, W), W is (OC, C, KH, KW); returns the output (OH, OW)
static std::pair<int64_t, int64_t> check_conv_inputs(const char* name, const torch::Tensor& x,
                                                     const torch::Tensor& W, const Conv2dParams& prm){
    TORCH_CHECK(x.is_cuda() && W.is_cuda(), name, ": x and W must be CUDA tensors");
    TORCH_CHECK(x.device() == W.device(), name, ": x and W must be on the same device");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && W.scalar_type() == torch::kFloat32,
                name, ": x and W must be float32");
    TORCH_CHECK(x.dim()==4 && W.dim()==4, name, ": expected x (N, C, H, W) and W (OC, C, KH, KW), got ",
                x.sizes(), " and ", W.sizes());
    TORCH_CHECK(x.size(1)==W.size(1), name, ": x has ", x.size(1), " channels but W expects ", W.size(1));
    TORCH_CHECK(W.size(2) > 0 && W.size(3) > 0, name, ": kernel must be at least 1x1, got ", W.sizes());
    const int64_t OH = (x.size(2) + 2*prm.pad_h - prm.dil_h*(W.size(2)-1) - 1) / prm.stride_h + 1;
    const int64_t OW = (x.size(3) + 2*prm.pad_w - prm.dil_w*(W.size(3)-1) - 1) / prm.stride_w + 1;
    TORCH_CHECK(x.size(2) + 2*prm.pad_h >= prm.dil_h*(W.size(2)-1) + 1 &&
                x.size(3) + 2*prm.pad_w >= prm.dil_w*(W.size(3)-1) + 1,
                name, ": kernel ", W.sizes(), " (with dilation) is larger than the padded input ", x.sizes());
    return {OH, OW};
}

torch::Tensor conv2d_fwd(torch::Tensor x, torch::Tensor W, std::optional<torch::Tensor> b,
                         std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation){
    auto prm = make_conv_params("conv2d_fwd", stride, padding, dilation);
    check_conv_inputs("conv2d_fwd", x, W, prm);
    if (b){
        TORCH_CHECK(b->is_cuda() && b->device() == x.device(), "conv2d_fwd: b must be a CUDA tensor on x's device");
        TORCH_CHECK(b->scalar_type() == torch::kFloat32, "conv2d_fwd: b must be float32");
        TORCH_CHECK(b->dim()==1 && b->size(0)==W.size(0),
                    "conv2d_fwd: expected b of shape (", W.size(0), "), got ", b->sizes());
        b = b->contiguous();
    }
    return conv2d_fwd_cuda(x.contiguous(), W.contiguous(), b, prm);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conv2d_bwd(torch::Tensor dy, torch::Tensor x, torch::Tensor W,
                                                                   std::vector<int64_t> stride, std::vector<int64_t> padding,
                                                                   std::vector<int64_t> dilation){
    auto prm = make_conv_params("conv2d_bwd", stride, padding, dilation);
    auto [OH, OW] = check_conv_inputs("conv2d_bwd", x, W, prm);
    TORCH_CHECK(dy.is_cuda() && dy.device() == x.device(), "conv2d_bwd: dy must be a CUDA tensor on x's device");
    TORCH_CHECK(dy.scalar_type() == torch::kFloat32, "conv2d_bwd: dy must be float32");
    TORCH_CHECK(dy.dim()==4 && dy.size(0)==x.size(0) && dy.size(1)==W.size(0) && dy.size(2)==OH && dy.size(3)==OW,
                "conv2d_bwd: expected dy of shape (", x.size(0), ", ", W.size(0), ", ", OH, ", ", OW, "), got ", dy.sizes());
    return conv2d_bwd_cuda(dy.contiguous(), x.contiguous(), W.contiguous(), prm);
}

torch::Tensor train_step(torch::Tensor X, torch::Tensor y,
                         torch::Tensor W1, torch::Tensor b1,
                         torch::Tensor W2, torch::Tensor b2, double lr){
    for (const auto& t : {X, y, W1, b1, W2, b2}){
        TORCH_CHECK(t.is_cuda(), "train_step: all inputs must be CUDA tensors");
        TORCH_CHECK(t.device() == X.device(), "train_step: all inputs must be on the same device");
    }
    for (const auto& t : {X, W1, b1, W2, b2}){
        TORCH_CHECK(t.scalar_type() == torch::kFloat32, "train_step: X and parameters must be float32");
    }
    TORCH_CHECK(y.scalar_type() == torch::kLong, "train_step: y must be int64");
    for (const auto& t : {W1, b1, W2, b2}){
        TORCH_CHECK(t.is_contiguous(), "train_step: parameters must be contiguous");
    }
    TORCH_CHECK(X.dim()==2 && W1.dim()==2 && W2.dim()==2 && b1.dim()==1 && b2.dim()==1 && y.dim()==1,
                "train_step: expected X, W1, W2 to be 2-D and b1, b2, y to be 1-D");
    TORCH_CHECK(X.size(1)==W1.size(0) && W1.size(1)==b1.size(0) &&
                W1.size(1)==W2.size(0) && W2.size(1)==b2.size(0) && y.size(0)==X.size(0),
                "train_step: shape mismatch, expected X (B x D), W1 (D x H), b1 (H), W2 (H x C), b2 (C), y (B)");
    return train_step_cuda(X.contiguous(), y.contiguous(), W1, b1, W2, b2, static_cast<float>(lr));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("matmul", &matmul, "C = A @ B for float32 CUDA matrices (tiled kernel)");
    m.def("biasadd", &biasadd, "Y = X @ W + b for float32 CUDA tensors");
    m.def("colsum", &colsum, "out = X.sum(0) for a float32 CUDA matrix");
    m.def("relu", &relu, "out = max(a, 0) elementwise for a float32 CUDA tensor");
    m.def("softmax_fwd_ce", &softmax_fwd_ce, "(probs, per-sample losses) = softmax + cross-entropy of logits");
    m.def("softmax_bwd_ce", &softmax_bwd_ce, "dlogits = (probs - one_hot(labels)) / M");
    m.def("sgd_step", &sgd_step, "param -= lr * grad, in place");
    m.def("train_step", &train_step, "One forward/backward/SGD step of a 2-layer MLP; returns the mean loss");
    m.def("conv2d_fwd", &conv2d_fwd, "y = conv2d(x, W, b) in NCHW, like F.conv2d (b may be None)",
          py::arg("x"), py::arg("W"), py::arg("b") = py::none(),
          py::arg("stride") = std::vector<int64_t>{1, 1}, py::arg("padding") = std::vector<int64_t>{0, 0},
          py::arg("dilation") = std::vector<int64_t>{1, 1});
    m.def("conv2d_bwd", &conv2d_bwd, "(dx, dW, db) for conv2d_fwd given the output gradient dy",
          py::arg("dy"), py::arg("x"), py::arg("W"),
          py::arg("stride") = std::vector<int64_t>{1, 1}, py::arg("padding") = std::vector<int64_t>{0, 0},
          py::arg("dilation") = std::vector<int64_t>{1, 1});
}
