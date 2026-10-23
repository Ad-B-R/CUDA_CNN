#pragma once

#include <torch/extension.h>
#include <vector>

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
torch::Tensor biasadd_cuda(torch::Tensor X, torch::Tensor W, torch::Tensor b);
torch::Tensor colsum_cuda(torch::Tensor X);
torch::Tensor relu_cuda(torch::Tensor a);
std::tuple<torch::Tensor, torch::Tensor> softmax_fwd_ce_cuda(torch::Tensor logits, torch::Tensor labels);
torch::Tensor softmax_bwd_ce_cuda(torch::Tensor probs, torch::Tensor labels);
void sgd_step_cuda(torch::Tensor param, torch::Tensor grad, float lr);
