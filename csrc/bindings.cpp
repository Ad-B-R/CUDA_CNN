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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("matmul", &matmul, "C = A @ B for float32 CUDA matrices (tiled kernel)");
    m.def("biasadd", &biasadd, "Y = X @ W + b for float32 CUDA tensors");
    m.def("colsum", &colsum, "out = X.sum(0) for a float32 CUDA matrix");
    m.def("relu", &relu, "out = max(a, 0) elementwise for a float32 CUDA tensor");
    m.def("softmax_fwd_ce", &softmax_fwd_ce, "(probs, per-sample losses) = softmax + cross-entropy of logits");
    m.def("softmax_bwd_ce", &softmax_bwd_ce, "dlogits = (probs - one_hot(labels)) / M");
    m.def("sgd_step", &sgd_step, "param -= lr * grad, in place");
}
