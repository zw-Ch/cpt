//
// Created by ASUS on 2024/1/28.
//

#ifndef PYTORCH_FUNC_H
#define PYTORCH_FUNC_H

#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <string>

void resnet18(bool use_cuda = true);

void pytorch_basics();

void linear_regression();

void logistic_regression();

void convolutional_neural_network();

void recurrent_neural_network();

void deep_residual_network();

void language_model();

void generative_adversarial_network();

void variational_autoencoder();

void neural_style_transfer();

#endif //PYTORCH_FUNC_H
