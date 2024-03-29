//
// Created by ASUS on 2024/1/29.
//

#ifndef CPT_IMAGE_IO_H
#define CPT_IMAGE_IO_H

#include <torch/torch.h>
#include <string>

namespace image_io
{
    enum class ImageFormat
    {
        PNG,
        JPG,
        BMP
    };

    torch::Tensor load_image(
        const std::string &file_path,
        torch::IntArrayRef shape = {},
        std::function<torch::Tensor(torch::Tensor)> transform = [](torch::Tensor x)
        { return x; });

    void save_image(torch::Tensor tensor,
                    const std::string &file_path,
                    int64_t nrow = 10, int64_t padding = 2,
                    bool normalize = false,
                    const std::vector<double> &range = {},
                    bool scale_each = false,
                    torch::Scalar pad_value = 0,
                    ImageFormat format = ImageFormat::PNG);
} // namespace image_io

#endif // CPT_IMAGE_IO_H
