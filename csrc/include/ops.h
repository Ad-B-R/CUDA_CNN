#pragma once

#include <torch/extension.h>
#include <optional>
#include <vector>

// stride / zero-padding / dilation of a 2-D convolution, one value per spatial dim
struct Conv2dParams {
    int stride_h, stride_w;
    int pad_h, pad_w;
    int dil_h, dil_w;
};

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
torch::Tensor biasadd_cuda(torch::Tensor X, torch::Tensor W, torch::Tensor b);
torch::Tensor colsum_cuda(torch::Tensor X);
torch::Tensor relu_cuda(torch::Tensor a);
std::tuple<torch::Tensor, torch::Tensor> softmax_fwd_ce_cuda(torch::Tensor logits, torch::Tensor labels);
torch::Tensor softmax_bwd_ce_cuda(torch::Tensor probs, torch::Tensor labels);
void sgd_step_cuda(torch::Tensor param, torch::Tensor grad, float lr);
torch::Tensor conv2d_fwd_cuda(torch::Tensor x, torch::Tensor W, std::optional<torch::Tensor> b, Conv2dParams prm);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conv2d_bwd_cuda(torch::Tensor dy, torch::Tensor x,
                                                                        torch::Tensor W, Conv2dParams prm);
torch::Tensor train_step_cuda(torch::Tensor X, torch::Tensor y,
                              torch::Tensor W1, torch::Tensor b1,
                              torch::Tensor W2, torch::Tensor b2, float lr);
