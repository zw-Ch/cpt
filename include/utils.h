#pragma once

#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <torch/torch.h>

void print(const std::vector<size_t> &values);
void print(const std::vector<float> &values);
void print(const std::vector<std::string> &values);

torch::Tensor getTensor(const std::vector<std::vector<float>> &ndarray);

std::vector<std::string> split(const std::string &line, char delimiter);
