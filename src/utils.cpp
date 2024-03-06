#include "utils.h"

void print(const std::vector<size_t> &values)
{
    std::cout << "[";
    for (const auto &value : values)
    {
        std::cout << value << ", ";
    }
    std::cout << "]" << std::endl;
}

void print(const std::vector<float> &values)
{
    std::cout << "[";
    for (const auto &value : values)
    {
        std::cout << value << ", ";
    }
    std::cout << "]" << std::endl;
}

void print(const std::vector<std::string> &values)
{
    std::cout << "[";
    for (const auto &value : values)
    {
        std::cout << value << ", ";
    }
    std::cout << "]" << std::endl;
}

std::vector<std::string> split(const std::string &line, char delimiter)
{
    std::istringstream ss(line);
    std::vector<std::string> words;
    std::string word;

    while (getline(ss, word, delimiter))
    {
        words.push_back(word);
    }
    return words;
}

torch::Tensor getTensor(const std::vector<std::vector<float>> &ndarray)
{
    const int n_row = ndarray.size();
    const int n_col = ndarray[0].size();
    float* value = new float[n_row * n_col];

    for (int i = 0; i < n_row; ++i) {
        for (int j = 0; j < n_col; ++j) {
            value[i * n_col + j] = ndarray[i][j];
        }
    }

    auto tensor = torch::from_blob(value, {n_row, n_col}, torch::kFloat);
    delete[] value;
    return tensor;
}
