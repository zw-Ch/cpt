//
// Created by ASUS on 2024/1/28.
//
#include "func.h"
#include <iostream>
#include "convnet.h"
#include "imagefolder_dataset.h"
#include "rnn.h"
#include "cifar10.h"
#include "transform.h"
#include "resnet.h"
#include "rnn_lm.h"
#include "corpus.h"
#include "image_io.h"
#include "vae.h"
#include "vggnet.h"

using dataset::ImageFolderDataset;
using image_io::load_image;
using image_io::save_image;

torch::DeviceType get_device_type(bool use_cuda)
{
    if (use_cuda && torch::cuda::is_available())
    {
        return at::kCUDA;
    }
    else
    {
        return at::kCPU;
    }
}

torch::Device choose_device()
{
    bool cuda_available = torch::cuda::is_available();
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';
    return cuda_available ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
}

void print_script_module(const torch::jit::script::Module &module, size_t spaces)
{
    for (const auto &sub_module : module.named_children())
    {
        if (!sub_module.name.empty())
        {
            std::cout << std::string(spaces, ' ') << sub_module.value.type()->name().value().name()
                      << " " << sub_module.name << "\n";
        }

        print_script_module(sub_module.value, spaces + 2);
    }
}

void resnet18(bool use_cuda)
{
    std::string model_path = "/home/chenziwei2021/cpp/cpt/resnet18.pt";
    std::vector<int64_t> input_shape = {1, 3, 224, 224};

    torch::jit::script::Module module;
    try
    {
        module = torch::jit::load(model_path); // 加载模型
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
    }

    torch::DeviceType device_type = get_device_type(use_cuda);
    module.to(device_type);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones(input_shape).to(device_type)); // 输入数据

    at::Tensor output = module.forward(inputs).toTensor(); // 进�?�推�?

    if (output.defined())
    {
        std::cout << output << std::endl; // 打印结果
    }
    else
    {
        std::cerr << "Inference failed." << std::endl;
    }
}

void pytorch_basics()
{
    std::cout << "Pytorch Basics\n\n";
    std::cout << std::fixed << std::setprecision(4);

    //    std::cout << "---- Basic AutoGrad Example 1 ----\n";
    //
    //    torch::Tensor x = torch::tensor(1.0, torch::requires_grad());
    //    torch::Tensor w = torch::tensor(2.0, torch::requires_grad());
    //    torch::Tensor b = torch::tensor(3.0, torch::requires_grad());
    //    auto y = w * x + b;
    //
    //    std::cout << "---- BASIC AUTOGRAD EXAMPLE 2 ----\n";
    //    x = torch::randn({10, 3});
    //    y = torch::randn({10, 2});
    //
    //    // Build a fully connected layer
    //    torch::nn::Linear linear(3, 2);
    //    std::cout << "w:\n" << linear->weight << '\n';
    //    std::cout << "b:\n" << linear->bias << '\n';
    //
    //    // Create loss function and optimizer
    //    torch::nn::MSELoss criterion;
    //    torch::optim::SGD optimizer(linear->parameters(), torch::optim::SGDOptions(0.01));
    //
    //    // Forward pass
    //    auto pred = linear->forward(x);
    //
    //    // Compute loss
    //    auto loss = criterion(pred, y);
    //    std::cout << "Loss: " << loss.item<double>() << '\n';
    //
    //    // Backward pass
    //    loss.backward();
    //
    //    // Print out the gradients
    //    std::cout << "dL/dw:\n" << linear->weight.grad() << '\n';
    //    std::cout << "dL/db:\n" << linear->bias.grad() << '\n';
    //
    //    // 1 step gradient descent
    //    optimizer.step();
    //
    //    // Print out the loss after 1-step gradient descent
    //    pred = linear->forward(x);
    //    loss = criterion(pred, y);
    //    std::cout << "Loss after 1 optimization step: " << loss.item<double>() << "\n\n";

    //    std::cout << "---- CREATING TENSORS FROM EXISTING DATA ----\n";
    //
    //    // Tensor From C-style array
    //    float data_array[] = {1, 2, 3, 4};
    //    torch::Tensor t1 = torch::from_blob(data_array, {2, 2});
    //    std::cout << "Tensor from array:\n" << t1 << '\n';
    //
    //    TORCH_CHECK(data_array == t1.data_ptr<float>());
    //
    //    // Tensor from vector:
    //    std::vector<float> data_vector = {1, 2, 3, 4};
    //    torch::Tensor t2 = torch::from_blob(data_vector.data(), {2, 2});
    //    std::cout << "Tensor from vector:\n" << t2 << "\n\n";
    //
    //    TORCH_CHECK(data_vector.data() == t2.data_ptr<float>());

    //    std::cout << "---- SLICING AND EXTRACTING PARTS FROM TENSORS ----\n";
    //    std::vector<int64_t> test_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    //    torch::Tensor s = torch::from_blob(test_data.data(), {3, 3}, torch::kInt64);
    //    std::cout << "s:\n" << s << '\n';
    //
    //    using torch::indexing::Slice;
    //    using torch::indexing::None;
    //    using torch::indexing::Ellipsis;
    //
    //    // Extract a single element tensor:
    //    std::cout << "\"s[0,2]\" as tensor:\n" << s.index({0, 2}) << '\n';
    //    std::cout << "\"s[0,2]\" as value:\n" << s.index({0, 2}).item<int64_t>() << '\n';
    //
    //    // Slice a tensor along a dimension at a given index.
    //    std::cout << "\"s[:,2]\":\n" << s.index({Slice(), 2}) << '\n';
    //
    //    // Slice a tensor along a dimension at given indices from
    //    std::cout << "\"s[:2,:]\":\n" << s.index({Slice(None, 2), Slice()}) << '\n';
    //    std::cout << "\"s[:,1:]\":\n" << s.index({Slice(), Slice(1, None)}) << '\n';
    //    std::cout << "\"s[:,::2]\":\n" << s.index({Slice(), Slice(None, None, 2)}) << '\n';
    //    std::cout << "\"s[:2,1]\":\n" << s.index({Slice(None, 2), 1}) << '\n';
    //    std::cout << "\"s[..., :2]\":\n" << s.index({Ellipsis, Slice(None, 2)}) << "\n\n";

    //    std::cout << "---- INPUT PIPELINE ----\n";
    //    const std::string MNIST_data_path = "/home/chenziwei2021/cpp/cpt/data/mnist";
    //
    //    auto dataset = torch::data::datasets::MNIST(MNIST_data_path)
    //            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
    //            .map(torch::data::transforms::Stack<>());
    //
    //    // Fetch one data pair
    //    auto example = dataset.get_batch(0);
    //    std::cout << "Sample data size: ";
    //    std::cout << example.data.sizes() << "\n";
    //    std::cout << "Sample target: " << example.target.item<int>() << "\n";
    //
    //    // Construct data loader
    //    auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
    //            dataset, 64);
    //
    //    // Fetch a mini-batch
    //    auto example_batch = *dataloader->begin();
    //    std::cout << "Sample batch - data size: ";
    //    std::cout << example_batch.data.sizes() << "\n";
    //    std::cout << "Sample batch - target size: ";
    //    std::cout << example_batch.target.sizes() << "\n\n";
    //
    //    std::cout << "---- PRETRAINED MODEL ----\n";
    //    const std::string pretrained_model_path = "/home/chenziwei2021/cpp/cpt/resnet18.pt";
    //    torch::jit::script::Module resnet;
    //
    //    try {
    //        resnet = torch::jit::load(pretrained_model_path);
    //    }
    //    catch (const torch::Error& error) {
    //        std::cerr << "Could not load scriptmodule from file " << pretrained_model_path << ".\n"
    //                  << "You can create this file using the provided Python script 'create_resnet18_scriptmodule.py'";
    //    }
    //
    //    std::cout << "Resnet18 model:\n";
    //    print_script_module(resnet, 2);
    //    std::cout << "\n";
    //
    //    const auto fc_weight = resnet.attr("fc").toModule().attr("weight").toTensor();
    //
    //    auto in_features = fc_weight.size(1);
    //    auto out_features = fc_weight.size(0);
    //
    //    std::cout << "Fully connected layer: in_features=" << in_features << ", out_features=" << out_features << "\n";
    //
    //    // Input sample
    //    auto sample_input = torch::randn({1, 3, 224, 224});
    //    std::vector<torch::jit::IValue> inputs{sample_input};
    //
    //    // Forward pass
    //    std::cout << "Input size: ";
    //    std::cout << sample_input.sizes() << "\n";
    //    auto output = resnet.forward(inputs).toTensor();
    //    std::cout << "Output size: ";
    //    std::cout << output.sizes() << "\n\n";

    std::cout << "---- SAVE AND LOAD A MODEL ----\n";

    // Simple example model
    torch::nn::Sequential model{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3).stride(2).padding(1)),
        torch::nn::ReLU()};

    // Path to the model output file (all folders must exist!).
    const std::string model_save_path = "/home/chenziwei2021/cpp/cpt/output/model.pt";

    // Save the model
    torch::save(model, model_save_path);

    std::cout << "Saved model:\n"
              << model << "\n";

    // Load the model
    torch::load(model, model_save_path);

    std::cout << "Loaded model:\n"
              << model;
}

void linear_regression()
{
    std::cout << "Linear Regression\n\n";

    // Hyper parameters
    torch::Device device = choose_device();
    const int64_t input_size = 1;
    const int64_t output_size = 1;
    const size_t num_epochs = 60;
    const double learning_rate = 0.001;

    // Sample dataset
    auto x_train = torch::randint(0, 10, {15, 1},
                                  torch::TensorOptions(torch::kFloat).device(device));

    auto y_train = torch::randint(0, 10, {15, 1},
                                  torch::TensorOptions(torch::kFloat).device(device));

    // Linear regression model
    torch::nn::Linear model(input_size, output_size);
    model->to(device);

    // Optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch)
    {
        // Forward pass
        auto output = model->forward(x_train);
        auto loss = torch::nn::functional::mse_loss(output, y_train);

        // Backward pass and optimize
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if ((epoch + 1) % 5 == 0)
        {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Loss: " << loss.item<double>() << "\n";
        }
    }

    std::cout << "Training finished!\n";
}

void logistic_regression()
{
    std::cout << "Logistic Regression\n\n";

    // Hyper parameters
    torch::Device device = choose_device();
    const int64_t input_size = 784;
    const int64_t num_classes = 10;
    const int64_t batch_size = 100;
    const size_t num_epochs = 20;
    const double learning_rate = 0.001;

    const std::string MNIST_data_path = "/home/chenziwei2021/cpp/cpt/data/mnist";

    // MNIST Dataset (images and labels)
    auto train_dataset = torch::data::datasets::MNIST(MNIST_data_path)
                             .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                             .map(torch::data::transforms::Stack<>());

    // Number of samples in the training set
    auto num_train_samples = train_dataset.size().value();

    auto test_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTest)
                            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                            .map(torch::data::transforms::Stack<>());

    // Number of samples in the testset
    auto num_test_samples = test_dataset.size().value();

    // Data loaders
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);

    // Logistic regression model
    torch::nn::Linear model(input_size, num_classes);

    model->to(device);

    // Loss and optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch)
    {
        // Initialize running metrics
        double running_loss = 0.0;
        size_t num_correct = 0;

        for (auto &batch : *train_loader)
        {
            auto data = batch.data.view({batch_size, -1}).to(device);
            auto target = batch.target.to(device);

            // Forward pass
            auto output = model->forward(data);

            // Calculate loss
            auto loss = torch::nn::functional::cross_entropy(output, target);

            // Update running loss
            running_loss += loss.item<double>() * data.size(0);

            // Calculate prediction
            auto prediction = output.argmax(1);

            // Update number of correctly classified samples
            num_correct += prediction.eq(target).sum().item<int64_t>();

            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<double>(num_correct) / num_train_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
                  << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
    }

    std::cout << "Training finished!\n\n";
    std::cout << "Testing...\n";

    // Test the model
    model->eval();
    torch::NoGradGuard no_grad;

    double running_loss = 0.0;
    size_t num_correct = 0;

    for (const auto &batch : *test_loader)
    {
        auto data = batch.data.view({batch_size, -1}).to(device);
        auto target = batch.target.to(device);

        auto output = model->forward(data);

        auto loss = torch::nn::functional::cross_entropy(output, target);

        running_loss += loss.item<double>() * data.size(0);

        auto prediction = output.argmax(1);

        num_correct += prediction.eq(target).sum().item<int64_t>();
    }

    std::cout << "Testing finished!\n";

    auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
    auto test_sample_mean_loss = running_loss / num_test_samples;

    std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
}

void convolutional_neural_network()
{
    std::cout << "Convolutional Neural Network\n\n";

    // Hyper parameters
    torch::Device device = choose_device();
    const int64_t num_classes = 10;
    const int64_t batch_size = 8;
    const size_t num_epochs = 10;
    const double learning_rate = 1e-3;
    const double weight_decay = 1e-3;

    const std::string imagenette_data_path = "/home/chenziwei2021/cpp/cpt/data/imagenette2-160";

    // Imagenette dataset
    auto train_dataset = ImageFolderDataset(imagenette_data_path, ImageFolderDataset::Mode::TRAIN, {160, 160})
                             .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
                             .map(torch::data::transforms::Stack<>());

    // Number of samples in the training set
    auto num_train_samples = train_dataset.size().value();

    auto test_dataset = ImageFolderDataset(imagenette_data_path, ImageFolderDataset::Mode::VAL, {160, 160})
                            .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
                            .map(torch::data::transforms::Stack<>());

    // Number of samples in the testset
    auto num_test_samples = test_dataset.size().value();

    // Data loader
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);

    // Model
    ConvNet model(num_classes);
    model->to(device);

    // Optimizer
    torch::optim::Adam optimizer(
        model->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch)
    {
        // Initialize running metrics
        double running_loss = 0.0;
        size_t num_correct = 0;

        for (auto &batch : *train_loader)
        {
            // Transfer images and target labels to device
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            // Forward pass
            auto output = model->forward(data);

            // Calculate loss
            auto loss = torch::nn::functional::cross_entropy(output, target);

            // Update running loss
            running_loss += loss.item<double>() * data.size(0);

            // Calculate prediction
            auto prediction = output.argmax(1);

            // Update number of correctly classified samples
            num_correct += prediction.eq(target).sum().item<int64_t>();

            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<double>(num_correct) / num_train_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
                  << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
    }

    std::cout << "Training finished!\n\n";
    std::cout << "Testing...\n";

    // Test the model
    model->eval();

    double running_loss = 0.0;
    size_t num_correct = 0;

    for (const auto &batch : *test_loader)
    {
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);

        auto output = model->forward(data);

        auto loss = torch::nn::functional::cross_entropy(output, target);
        running_loss += loss.item<double>() * data.size(0);

        auto prediction = output.argmax(1);
        num_correct += prediction.eq(target).sum().item<int64_t>();
    }

    std::cout << "Testing finished!\n";

    auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
    auto test_sample_mean_loss = running_loss / num_test_samples;

    std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
}

void recurrent_neural_network()
{
    std::cout << "Recurrent Neural Network\n\n";

    // Hyper parameters
    torch::Device device = choose_device();
    const int64_t sequence_length = 28;
    const int64_t input_size = 28;
    const int64_t hidden_size = 128;
    const int64_t num_layers = 2;
    const int64_t num_classes = 10;
    const int64_t batch_size = 100;
    const size_t num_epochs = 2;
    const double learning_rate = 0.01;

    const std::string MNIST_data_path = "/home/chenziwei2021/cpp/cpt/data/mnist/";

    auto train_dataset = torch::data::datasets::MNIST(MNIST_data_path)
                             .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                             .map(torch::data::transforms::Stack<>());
    auto num_train_samples = train_dataset.size().value();

    auto test_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTest)
                            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                            .map(torch::data::transforms::Stack<>());
    auto num_test_samples = test_dataset.size().value();

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), batch_size);
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(test_dataset), batch_size);

    RNN model(input_size, hidden_size, num_layers, num_classes);
    model->to(device);

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";
    for (size_t epoch = 0; epoch != num_epochs; ++epoch)
    {
        double running_loss = 0.0;
        size_t num_correct = 0;

        for (auto &batch : *train_loader)
        {
            auto data = batch.data.view({-1, sequence_length, input_size}).to(device);
            auto target = batch.target.to(device);

            auto output = model->forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, target);
            running_loss += loss.item<double>() * data.size(0);

            auto prediction = output.argmax(1);
            num_correct += prediction.eq(target).sum().item<int64_t>();

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<float>(num_correct) / num_train_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
                  << sample_mean_loss << ", Accuracy: " << accuracy << "\n";
    }
    std::cout << "Training finished!\n\n";
    std::cout << "Testing...\n";

    // Test the model
    model->eval();
    double running_loss = 0.0;
    size_t num_correct = 0;

    for (const auto &batch : *test_loader)
    {
        auto data = batch.data.view({-1, sequence_length, input_size}).to(device);
        auto target = batch.target.to(device);

        auto output = model->forward(data);

        auto loss = torch::nn::functional::cross_entropy(output, target);
        running_loss += loss.item<double>() * data.size(0);

        auto prediction = output.argmax(1);
        num_correct += prediction.eq(target).sum().item<int64_t>();
    }

    std::cout << "Testing finished!\n";

    auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
    auto test_sample_mean_loss = running_loss / num_test_samples;

    std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
}

void deep_residual_network()
{
    using resnet::ResidualBlock;
    using resnet::ResNet;
    using transform::ConstantPad;
    using transform::RandomCrop;
    using transform::RandomHorizontalFlip;

    std::cout << "Deep Residual Network\n\n";

    // Hyper parameters
    torch::Device device = choose_device();
    const int64_t num_classes = 10;
    const int64_t batch_size = 100;
    const size_t num_epochs = 20;
    const double learning_rate = 0.001;
    const size_t learning_rate_decay_frequency = 8; // number of epochs after which to decay the learning rate
    const double learning_rate_decay_factor = 1.0 / 3.0;

    const std::string CIFAR_data_path = "/home/chenziwei2021/cpp/cpt/data/cifar10";

    // CIFAR10 custom dataset
    auto train_dataset = CIFAR10(CIFAR_data_path)
                             .map(ConstantPad(4))
                             .map(RandomHorizontalFlip())
                             .map(RandomCrop({32, 32}))
                             .map(torch::data::transforms::Stack<>());

    // Number of samples in the training set
    auto num_train_samples = train_dataset.size().value();

    auto test_dataset = CIFAR10(CIFAR_data_path, CIFAR10::Mode::kTest)
                            .map(torch::data::transforms::Stack<>());

    // Number of samples in the testset
    auto num_test_samples = test_dataset.size().value();

    // Data loader
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);

    // Model
    std::array<int64_t, 3> layers{2, 2, 2};
    ResNet<ResidualBlock> model(layers, num_classes);
    model->to(device);

    // Optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    auto current_learning_rate = learning_rate;

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch)
    {
        // Initialize running metrics
        double running_loss = 0.0;
        size_t num_correct = 0;

        for (auto &batch : *train_loader)
        {
            // Transfer images and target labels to device
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            // Forward pass
            auto output = model->forward(data);

            // Calculate loss
            auto loss = torch::nn::functional::cross_entropy(output, target);

            // Update running loss
            running_loss += loss.item<double>() * data.size(0);

            // Calculate prediction
            auto prediction = output.argmax(1);

            // Update number of correctly classified samples
            num_correct += prediction.eq(target).sum().item<int64_t>();

            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        // Decay learning rate
        if ((epoch + 1) % learning_rate_decay_frequency == 0)
        {
            current_learning_rate *= learning_rate_decay_factor;
            static_cast<torch::optim::AdamOptions &>(optimizer.param_groups().front().options()).lr(current_learning_rate);
        }

        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<double>(num_correct) / num_train_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
                  << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
    }

    std::cout << "Training finished!\n\n";
    std::cout << "Testing...\n";

    // Test the model
    model->eval();

    double running_loss = 0.0;
    size_t num_correct = 0;

    for (const auto &batch : *test_loader)
    {
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);

        auto output = model->forward(data);

        auto loss = torch::nn::functional::cross_entropy(output, target);
        running_loss += loss.item<double>() * data.size(0);

        auto prediction = output.argmax(1);
        num_correct += prediction.eq(target).sum().item<int64_t>();
    }

    std::cout << "Testing finished!\n";

    auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
    auto test_sample_mean_loss = running_loss / num_test_samples;

    std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
}

void language_model()
{
    using data_utils::Corpus;
    using torch::indexing::Slice;

    std::cout << "Language Model\n\n";

    // Hyper parameters
    torch::Device device = choose_device();
    const int64_t embed_size = 128;
    const int64_t hidden_size = 1024;
    const int64_t num_layers = 1;
    const int64_t num_samples = 1000; // the number of words to be sampled
    const int64_t batch_size = 20;
    const int64_t sequence_length = 30;
    const size_t num_epochs = 5;
    const double learning_rate = 0.002;

    const std::string penn_treebank_data_path = "/home/chenziwei2021/cpp/cpt/data/penntreebank/train.txt";
    Corpus corpus(penn_treebank_data_path);

    auto ids = corpus.get_data(batch_size);
    auto vocab_size = corpus.get_dictionary().size();

    // Path to the output file (All folders must exist!)
    const std::string sample_output_path = "/home/chenziwei2021/cpp/cpt/output/sample.txt";

    // Model
    RNNLM model(vocab_size, embed_size, hidden_size, num_layers);
    model->to(device);

    // Optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch)
    {
        // Initialize running metrics
        double running_loss = 0.0;
        double running_perplexity = 0.0;
        size_t running_num_samples = 0;

        // Initialize hidden- and cell-states.
        auto h = torch::zeros({num_layers, batch_size, hidden_size}).to(device).detach();
        auto c = torch::zeros({num_layers, batch_size, hidden_size}).to(device).detach();

        for (int64_t i = 0; i < ids.size(1) - sequence_length; i += sequence_length)
        {
            // Transfer data and target labels to device
            auto data = ids.index({Slice(), Slice(i, i + sequence_length)}).to(device);
            auto target = ids.index({Slice(), Slice(i + 1, i + 1 + sequence_length)}).reshape(-1).to(device);

            // Forward pass
            torch::Tensor output;
            std::forward_as_tuple(output, std::tie(h, c)) = model->forward(data, std::make_tuple(h, c));

            h.detach_();
            c.detach_();

            // Calculate loss
            auto loss = torch::nn::functional::nll_loss(output, target);

            // Update running metrics
            running_loss += loss.item<double>() * data.size(0);
            running_perplexity += torch::exp(loss).item<double>() * data.size(0);
            running_num_samples += data.size(0);

            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 0.5);
            optimizer.step();
        }

        auto sample_mean_loss = running_loss / running_num_samples;
        auto sample_mean_perplexity = running_perplexity / running_num_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
                  << sample_mean_loss << ", Perplexity: " << sample_mean_perplexity << '\n';
    }

    std::cout << "Training finished!\n\n";
    std::cout << "Generating samples...\n";

    // Generate samples
    model->eval();

    std::ofstream sample_output_file(sample_output_path);

    // Initialize hidden- and cell-states.
    auto h = torch::zeros({num_layers, 1, hidden_size}).to(device);
    auto c = torch::zeros({num_layers, 1, hidden_size}).to(device);

    // Select one word-id at random
    auto prob = torch::ones(vocab_size);
    auto data = prob.multinomial(1).unsqueeze(1).to(device);

    for (size_t i = 0; i != num_samples; ++i)
    {
        // Forward pass
        torch::Tensor output;
        std::forward_as_tuple(output, std::tie(h, c)) = model->forward(data, std::make_tuple(h, c));

        // Sample one word id
        prob = output.exp();
        auto word_id = prob.multinomial(1).item();

        // Fill input data with sampled word id for the next time step
        data.fill_(word_id);

        // Write the word corresponding to the id to the file
        auto word = corpus.get_dictionary().word_at_index(word_id.toLong());
        word = (word == "<eos>") ? "\n" : word + " ";
        sample_output_file << word;
    }
    std::cout << "Finished generating samples!\nSaved output to " << sample_output_path << "\n";
}

void generative_adversarial_network()
{

    std::cout << "Generative Adversarial Network\n\n";

    // Hyper parameters
    torch::Device device = choose_device();
    const int64_t latent_size = 64;
    const int64_t hidden_size = 256;
    const int64_t image_size = 28 * 28;
    const int64_t batch_size = 100;
    const size_t num_epochs = 100;
    const double learning_rate = 0.0002;

    const std::string MNIST_data_path = "/home/chenziwei2021/cpp/cpt/data/mnist";
    const std::string sample_output_dir_path = "/home/chenziwei2021/cpp/cpt/output/";

    // MNIST dataset
    auto dataset = torch::data::datasets::MNIST(MNIST_data_path)
                       .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                       .map(torch::data::transforms::Stack<>());

    // Number of samples in the dataset
    auto num_samples = dataset.size().value();

    // Data loader
    auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(dataset), batch_size);

    // Model
    // - Discriminator
    torch::nn::Sequential D{
        torch::nn::Linear(image_size, hidden_size),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        torch::nn::Linear(hidden_size, hidden_size),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        torch::nn::Linear(hidden_size, 1),
        torch::nn::Sigmoid()};

    // - Generator
    torch::nn::Sequential G{
        torch::nn::Linear(latent_size, hidden_size),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden_size, hidden_size),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden_size, image_size),
        torch::nn::Tanh()};

    D->to(device);
    G->to(device);

    // Optimizers
    torch::optim::Adam d_optimizer(D->parameters(), torch::optim::AdamOptions(learning_rate));
    torch::optim::Adam g_optimizer(G->parameters(), torch::optim::AdamOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    auto denorm = [](torch::Tensor tensor)
    { return tensor.add(1).div_(2).clamp_(0, 1); };

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch)
    {
        torch::Tensor images;
        torch::Tensor fake_images;
        size_t batch_index = 0;

        for (auto &batch : *dataloader)
        {
            // Transfer images to device
            images = batch.data.reshape({batch_size, -1}).to(device);

            // Create the labels which are later used as input for the loss
            auto real_labels = torch::ones({batch_size, 1}).to(device);
            auto fake_labels = torch::zeros({batch_size, 1}).to(device);

            // ================================================================== #
            //                      Train the discriminator                       #
            // ================================================================== #

            // Compute binary cross entropy loss using real images where
            // binary_cross_entropy(x, y) = -y * log(D(x)) - (1 - y) * log(1 - D(x))
            // Second term of the loss is always zero since real_labels == 1
            auto outputs = D->forward(images);
            auto d_loss_real = torch::nn::functional::binary_cross_entropy(outputs, real_labels);
            auto real_score = outputs.mean().item<double>();

            // Compute binary cross entropy loss using fake images
            // First term of the loss is always zero since fake_labels == 0
            auto z = torch::randn({batch_size, latent_size}).to(device);
            fake_images = G->forward(z);
            outputs = D->forward(fake_images);
            auto d_loss_fake = torch::nn::functional::binary_cross_entropy(outputs, fake_labels);
            auto fake_score = outputs.mean().item<double>();

            auto d_loss = d_loss_real + d_loss_fake;

            // Backward pass and optimize
            d_optimizer.zero_grad();
            d_loss.backward();
            d_optimizer.step();

            // ================================================================== #
            //                        Train the generator                         #
            // ================================================================== #

            // Compute loss with fake images
            z = torch::randn({batch_size, latent_size}).to(device);
            fake_images = G->forward(z);
            outputs = D->forward(fake_images);

            // We train G to maximize log(D(G(z)) instead of minimizing log(1 - D(G(z)))
            // For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
            auto g_loss = torch::nn::functional::binary_cross_entropy(outputs, real_labels);

            // Backward pass and optimize
            g_optimizer.zero_grad();
            g_loss.backward();
            g_optimizer.step();

            if ((batch_index + 1) % 200 == 0)
            {
                std::cout << "Epoch [" << epoch << "/" << num_epochs << "], Step [" << batch_index + 1 << "/"
                          << num_samples / batch_size << "], d_loss: " << d_loss.item<double>() << ", g_loss: "
                          << g_loss.item<double>() << ", D(x): " << real_score
                          << ", D(G(z)): " << fake_score << "\n";
            }

            ++batch_index;
        }

        // Save real images once
        if (epoch == 0)
        {
            images = denorm(images.reshape({images.size(0), 1, 28, 28}));
            save_image(images, sample_output_dir_path + "real_images.png");
        }

        // Save generated fake images
        if ((epoch + 1) % 10 == 0)
        {
            fake_images = denorm(fake_images.reshape({fake_images.size(0), 1, 28, 28}));
            save_image(fake_images, sample_output_dir_path + "fake_images-" + std::to_string(epoch + 1) + ".png");
        }
    }

    std::cout << "Training finished!\n";
}

void variational_autoencoder()
{
    std::cout << "Variational Autoencoder\n\n";

    // Hyper parameters
    torch::Device device = choose_device();
    const int64_t h_dim = 400;
    const int64_t z_dim = 20;
    const int64_t image_size = 28 * 28;
    const int64_t batch_size = 100;
    const size_t num_epochs = 100;
    const double learning_rate = 1e-3;

    const std::string MNIST_data_path = "/home/chenziwei2021/cpp/cpt/data/mnist";
    const std::string sample_output_dir_path = "/home/chenziwei2021/cpp/cpt/output/";

    // MNIST dataset
    auto dataset = torch::data::datasets::MNIST(MNIST_data_path)
                       .map(torch::data::transforms::Stack<>());

    // Number of samples in the dataset
    auto num_samples = dataset.size().value();

    // Data loader
    auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(dataset), batch_size);

    // Model
    VAE model(image_size, h_dim, z_dim);
    model->to(device);

    // Optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch)
    {
        torch::Tensor images;
        size_t batch_index = 0;

        model->train();

        for (auto &batch : *dataloader)
        {
            // Transfer images to device
            images = batch.data.reshape({-1, image_size}).to(device);

            // Forward pass
            auto output = model->forward(images);

            // Compute reconstruction loss and kl divergence
            // For KL divergence, see Appendix B in VAE paper https://arxiv.org/pdf/1312.6114.pdf
            auto reconstruction_loss = torch::nn::functional::binary_cross_entropy(output.reconstruction, images,
                                                                                   torch::nn::functional::BinaryCrossEntropyFuncOptions().reduction(torch::kSum));
            auto kl_divergence = -0.5 * torch::sum(1 + output.log_var - output.mu.pow(2) - output.log_var.exp());

            // Backward pass and optimize
            auto loss = reconstruction_loss + kl_divergence;
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            if ((batch_index + 1) % 100 == 0)
            {
                std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], Step [" << batch_index + 1 << "/"
                          << num_samples / batch_size << "], Reconstruction loss: "
                          << reconstruction_loss.item<double>() / batch.data.size(0)
                          << ", KL-divergence: " << kl_divergence.item<double>() / batch.data.size(0)
                          << "\n";
            }
            ++batch_index;
        }

        model->eval();

        // Sample a batch of codings from the unit Gaussian Distribution, then decode them using the Decoder
        // and save the resulting images.
        if ((epoch + 1) % 10 == 0)
        {
            auto z = torch::randn({batch_size, z_dim}).to(device);
            auto images_decoded = model->decode(z).view({-1, 1, 28, 28});
            save_image(images_decoded, sample_output_dir_path + "sampled-" + std::to_string(epoch + 1) + ".png");
        }
    }
}

void neural_style_transfer()
{
    std::cout << "Neural Style Transfer\n\n";

    // Hyper parameters
    torch::Device device = choose_device();
    const int64_t max_image_size = 300;
    const double learning_rate = 3e-3;
    const double style_loss_weight = 100;
    const size_t num_total_steps = 2000;
    const size_t log_step = 10;
    const size_t sample_step = 500;

    // Paths to content and style images
    const std::string content_image_path = "/home/chenziwei2021/cpp/cpt/data/ContentStyle/C_image12.jpg";
    const std::string style_image_path = "/home/chenziwei2021/cpp/cpt/data/ContentStyle/S_image9.jpg";

    const std::string vgg19_layers_scriptmodule_path =
        "/home/chenziwei2021/cpp/cpt/vgg19_layers.pt";

    if (!std::ifstream(vgg19_layers_scriptmodule_path))
    {
        std::cout << "Could not open the required VGG19 layers scriptmodule file from path: "
                  << vgg19_layers_scriptmodule_path << ".\nThis file must be created using the provided python script at "
                                                       "pytorch-cpp/tutorials/advanced/neural_style_transfer/model/create_vgg19_layers_scriptmodule.py."
                  << std::endl;
        return;
    }

    // Create necessary normalization and denormalization transforms
    torch::data::transforms::Normalize<> normalize_transform({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});
    torch::data::transforms::Normalize<> denormalize_transform({-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225},
                                                               {1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225});

    // Load content and style images and resize style image to the same size as content image
    auto content = load_image(content_image_path, max_image_size, normalize_transform).unsqueeze_(0);
    auto style = load_image(style_image_path, {content.size(2), content.size(3)}, normalize_transform).unsqueeze_(0);

    auto target = content.clone();

    // Move tensors to device
    content = content.to(device);
    style = style.to(device);
    target = target.to(device);

    // Model
    VGGNet19 model(vgg19_layers_scriptmodule_path);
    model->to(device);
    model->eval();

    // Optimizer
    torch::optim::Adam optimizer(std::vector<torch::Tensor>{target.requires_grad_(true)},
                                 torch::optim::AdamOptions(learning_rate).betas(std::make_tuple(0.5, 0.999)));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    for (size_t step = 0; step != num_total_steps; ++step)
    {

        // Forward pass and extract feature tensors from some Conv2d layers
        auto target_features = model->forward(target);
        auto content_features = model->forward(content);
        auto style_features = model->forward(style);

        auto style_loss = torch::zeros({1}, torch::TensorOptions(device));
        auto content_loss = torch::zeros({1}, torch::TensorOptions(device));

        for (size_t f_id = 0; f_id != target_features.size(); ++f_id)
        {
            // Compute content loss between target and content feature images
            content_loss += torch::nn::functional::mse_loss(target_features[f_id], content_features[f_id]);

            auto c = target_features[f_id].size(1);
            auto h = target_features[f_id].size(2);
            auto w = target_features[f_id].size(3);

            // Reshape convolutional feature maps
            auto target_feature = target_features[f_id].view({c, h * w});
            auto style_feature = style_features[f_id].view({c, h * w});

            // Compute gram matrices
            target_feature = torch::mm(target_feature, target_feature.t());
            style_feature = torch::mm(style_feature, style_feature.t());

            // Compute style loss
            style_loss += torch::nn::functional::mse_loss(target_feature, style_feature) / (c * h * w);
        }

        // Compute total loss
        auto total_loss = content_loss + style_loss_weight * style_loss;

        // Backward pass and optimize
        optimizer.zero_grad();
        total_loss.backward();
        optimizer.step();

        if ((step + 1) % log_step == 0)
        {
            // Print losses
            std::cout << "Step [" << step + 1 << "/" << num_total_steps
                      << "], Content Loss: " << content_loss.item<double>()
                      << ", Style Loss: " << style_loss.item<double>() << "\n";
        }

        if ((step + 1) % sample_step == 0)
        {
            // Save the generated image
            auto image = denormalize_transform(target.to(torch::kCPU).clone().squeeze(0)).clamp_(0, 1);
            save_image(image, "/home/chenziwei2021/cpp/cpt/output/output-" + std::to_string(step + 1) + ".png", 1, 0);
        }
    }

    std::cout << "Training finished!\n";
}
