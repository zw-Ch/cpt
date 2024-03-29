#pragma once

#include <torch/torch.h>

class RNNImpl : public torch::nn::Module
{
public:
    RNNImpl(int64_t input_size, int64_t hidden_size, int64_t num_layers, int64_t num_classes);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::LSTM lstm;
    torch::nn::Linear fc;
};

TORCH_MODULE(RNN);
