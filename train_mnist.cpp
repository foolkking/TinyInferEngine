
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <random>

#include "tensor.h"
#include "layer.h"
#include "model.h"
#include "loss.h"
#include "optimizer.h"
#include "scheduler.h"

uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xff000000) | ((val << 8) & 0x00ff0000)
        | ((val >> 8) & 0x0000ff00) | ((val >> 24) & 0x000000ff);
}

std::vector<float> read_mnist_images(const std::string& path, int& num_images) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open MNIST image file: " << path << std::endl;
        exit(1);
    }
    uint32_t magic_number = 0, num_items = 0, num_rows = 0, num_cols = 0;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&num_items), sizeof(num_items));
    file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
    num_images = swap_endian(num_items);
    int rows = swap_endian(num_rows);
    int cols = swap_endian(num_cols);
    std::vector<unsigned char> raw_data(static_cast<size_t>(num_images) * rows * cols);
    file.read(reinterpret_cast<char*>(raw_data.data()),
              static_cast<std::streamsize>(raw_data.size()));
    std::vector<float> float_data(raw_data.size());
    for (size_t i = 0; i < raw_data.size(); ++i) {
        float_data[i] = raw_data[i] / 255.0f;
    }
    return float_data;
}

std::vector<int> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open MNIST label file: " << path << std::endl;
        exit(1);
    }
    uint32_t magic_number = 0, num_items = 0;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&num_items), sizeof(num_items));
    int num_images = swap_endian(num_items);
    std::vector<unsigned char> raw_labels(static_cast<size_t>(num_images));
    file.read(reinterpret_cast<char*>(raw_labels.data()),
              static_cast<std::streamsize>(raw_labels.size()));
    std::vector<int> labels(num_images);
    for (int i = 0; i < num_images; ++i) {
        labels[i] = raw_labels[static_cast<size_t>(i)];
    }
    return labels;
}

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "  TinyInferEngine: Autograd & AdamW !    " << std::endl;
    std::cout << "=========================================" << std::endl;

    std::string data_dir = "./data/MNIST/raw/";
    int total_images = 0;
    std::vector<float> all_images =
        read_mnist_images(data_dir + "train-images-idx3-ubyte", total_images);
    std::vector<int> all_labels = read_mnist_labels(data_dir + "train-labels-idx1-ubyte");

    Sequential model;
    auto conv1 = std::make_shared<Conv2D>(1, 8, 3, 1, 0, true);
    auto pool1 = std::make_shared<MaxPool2D>(2, 2, 0);
    auto flatten = std::make_shared<Flatten>();
    auto fc1 = std::make_shared<Linear>(8 * 13 * 13, 128, true);
    auto relu = std::make_shared<ReLU>();
    auto fc2 = std::make_shared<Linear>(128, 10, true);

    conv1->weight()->randomize(-0.1f, 0.1f);
    conv1->bias()->fill(0.0f);
    fc1->weight()->randomize(-0.05f, 0.05f);
    fc1->bias()->fill(0.0f);
    fc2->weight()->randomize(-0.1f, 0.1f);
    fc2->bias()->fill(0.0f);

    model.add(conv1);
    model.add(pool1);
    model.add(flatten);
    model.add(fc1);
    model.add(relu);
    model.add(fc2);

    std::vector<ParamGroup> groups = {
        ParamGroup(model.named_parameters(), 1e-3f, 1e-4f),
    };
    AdamW optimizer(groups);
    CrossEntropyLoss criterion;

    int epochs = 5;
    CosineAnnealingLR scheduler(&optimizer, epochs, 1e-5f);

    int batch_size = 64;
    int num_batches = total_images / batch_size;

    std::vector<int> indices(static_cast<size_t>(total_images));
    for (int i = 0; i < total_images; ++i) {
        indices[static_cast<size_t>(i)] = i;
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int correct_predictions = 0;

        std::shuffle(indices.begin(), indices.end(),
                     std::default_random_engine(std::random_device{}()));

        for (int batch = 0; batch < num_batches; ++batch) {
            std::vector<int> b_shape = {batch_size, 1, 28, 28};
            TensorPtr input_batch = std::make_shared<Tensor>(b_shape, false);

            std::vector<int> t_shape = {batch_size};
            TensorPtr target_batch = std::make_shared<Tensor>(t_shape, false);

            for (int i = 0; i < batch_size; ++i) {
                int real_idx = indices[static_cast<size_t>(batch * batch_size + i)];
                target_batch->data()[i] = static_cast<float>(all_labels[static_cast<size_t>(real_idx)]);
                for (int p = 0; p < 28 * 28; ++p) {
                    input_batch->data()[i * 28 * 28 + p] =
                        all_images[static_cast<size_t>(real_idx * 28 * 28 + p)];
                }
            }

            optimizer.zero_grad();
            TensorPtr output = model.forward(input_batch);
            TensorPtr loss = criterion.forward(output, target_batch);
            loss->backward();
            optimizer.step();

            epoch_loss += loss->data()[0];

            for (int i = 0; i < batch_size; ++i) {
                int best_class = 0;
                float best_score = -1e9f;
                for (int c = 0; c < 10; ++c) {
                    float score = output->data()[i * 10 + c];
                    if (score > best_score) {
                        best_score = score;
                        best_class = c;
                    }
                }
                if (best_class == static_cast<int>(target_batch->data()[i])) {
                    ++correct_predictions;
                }
            }

            if (batch % 200 == 0) {
                std::cout << "  Epoch [" << epoch + 1 << "/" << epochs << "] "
                          << "Batch [" << batch << "/" << num_batches << "] "
                          << "Loss: " << std::fixed << std::setprecision(4) << loss->data()[0]
                          << std::endl;
            }
        }

        std::cout << ">>> Epoch " << epoch + 1 << " Summary: Avg Loss: "
                  << epoch_loss / static_cast<float>(num_batches) << " | Accuracy: "
                  << static_cast<float>(correct_predictions) / static_cast<float>(total_images)
                         * 100.0f
                  << "% <<<" << std::endl;

        scheduler.step();
    }

    std::cout << "\n[INFO] Training finished! Exporting weights..." << std::endl;
    conv1->weight()->save_to_bin("weights/cpp_conv1_weight.bin");
    conv1->bias()->save_to_bin("weights/cpp_conv1_bias.bin");
    fc1->weight()->save_to_bin("weights/cpp_fc1_weight.bin");
    fc1->bias()->save_to_bin("weights/cpp_fc1_bias.bin");
    fc2->weight()->save_to_bin("weights/cpp_fc2_weight.bin");
    fc2->bias()->save_to_bin("weights/cpp_fc2_bias.bin");

    return 0;
}
