#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#include "tensor.h"
#include "layer.h"
#include "model.h"
#include "conv2d.h"
#include "maxpool2d.h"
#include "flatten.h"
#include "linear.h"
#include "relu.h"
#include "loss.h"
#include "sgd.h"
#include "silu.h"

// 大端序转小端序辅助函数 (MNIST 格式规定使用大端序)
uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xff000000) |
           ((val <<  8) & 0x00ff0000) |
           ((val >>  8) & 0x0000ff00) |
           ((val >> 24) & 0x000000ff);
}

// 读取 MNIST 图像数据集 (直接读 PyTorch 下载好的原文件)
std::vector<float> read_mnist_images(const std::string& path, int& num_images) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open MNIST image file: " << path << std::endl;
        exit(1);
    }
    uint32_t magic_number=0, num_items=0, num_rows=0, num_cols=0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_items, sizeof(num_items));
    file.read((char*)&num_rows, sizeof(num_rows));
    file.read((char*)&num_cols, sizeof(num_cols));
    
    num_images = swap_endian(num_items);
    int rows = swap_endian(num_rows);
    int cols = swap_endian(num_cols);

    std::vector<unsigned char> raw_data(num_images * rows * cols);
    file.read((char*)raw_data.data(), raw_data.size());

    // 归一化到 0~1 之间
    std::vector<float> float_data(raw_data.size());
    for (size_t i = 0; i < raw_data.size(); ++i) {
        float_data[i] = raw_data[i] / 255.0f;
    }
    return float_data;
}

// 读取 MNIST 标签数据集
std::vector<int> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open MNIST label file: " << path << std::endl;
        exit(1);
    }
    uint32_t magic_number=0, num_items=0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_items, sizeof(num_items));
    
    int num_images = swap_endian(num_items);
    std::vector<unsigned char> raw_labels(num_images);
    file.read((char*)raw_labels.data(), raw_labels.size());

    std::vector<int> labels(num_images);
    for (int i = 0; i < num_images; ++i) {
        labels[i] = raw_labels[i];
    }
    return labels;
}

int main() {
    std::srand(std::time(nullptr)); // 初始化随机种子

    std::cout << "=========================================" << std::endl;
    std::cout << "  TinyInferEngine: Pure C++ Training!  " << std::endl;
    std::cout << "=========================================" << std::endl;

    // 1. 加载数据
    std::cout << "[INFO] Loading MNIST Dataset..." << std::endl;
    int total_images = 0;
    // 确保这个路径指向你运行 python 脚本下载的数据目录
    std::string data_dir = "./data/MNIST/raw/"; 
    std::vector<float> all_images = read_mnist_images(data_dir + "train-images-idx3-ubyte", total_images);
    std::vector<int> all_labels = read_mnist_labels(data_dir + "train-labels-idx1-ubyte");
    std::cout << "[INFO] Loaded " << total_images << " images." << std::endl;

    // 2. 搭建网络 (和 Python 完全对应)
    Sequential model;
    Conv2D* conv1 = new Conv2D(1, 8, 3, 1, 0, true); // 需要求导
    MaxPool2D* pool1 = new MaxPool2D(2, 2, 0);       // 池化不需要权重求导
    Flatten* flatten = new Flatten();
    Linear* fc1 = new Linear(8 * 13 * 13, 128, true); // 需要求导
    ReLU* relu = new ReLU();
    SiLU* silu = new SiLU();
    Linear* fc2 = new Linear(128, 10, true);          // 需要求导

    // 极简权重初始化 (打破对称性)
    conv1->weight()->randomize(-0.1f, 0.1f); conv1->bias()->fill(0.0f);
    fc1->weight()->randomize(-0.05f, 0.05f); fc1->bias()->fill(0.0f);
    fc2->weight()->randomize(-0.1f, 0.1f);   fc2->bias()->fill(0.0f);

    model.add(conv1);
    model.add(pool1);
    model.add(flatten);
    model.add(fc1);
    model.add(relu);
    // model.add(silu); // 可选：如果 PyTorch 侧使用了 SiLU 激活，这里也要加上
    model.add(fc2);

    // 3. 配置训练组件
    std::vector<Tensor*> params = {
        conv1->weight(), conv1->bias(),
        fc1->weight(), fc1->bias(),
        fc2->weight(), fc2->bias()
    };
    SGD optimizer(params, 0.05f); // 优化2，学习率加大，加快收敛
    CrossEntropyLoss criterion;

    // 4. 训练超参数
    int batch_size = 64;
    int epochs = 5;   //优化1，多训练几个epoch
    int num_batches = total_images / batch_size;

    std::vector<int> indices(total_images);   //优化三
    for (int i = 0; i < total_images; ++i) {
        indices[i] = i;
    }
    std::cout << "[INFO] Starting Training for " << epochs << " Epoch(s)..." << std::endl;

    // 5. 大循环开始
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int correct_predictions = 0;
        std::random_shuffle(indices.begin(), indices.end()); // 优化三：每个 epoch 打乱数据顺序
        for (int batch = 0; batch < num_batches; ++batch) {
            // -- a. 准备当前 Batch 的数据 --
            int b_shape[] = {batch_size, 1, 28, 28};
            Tensor* input_batch = new Tensor(b_shape, 4, false);
            std::vector<int> target_batch(batch_size);
            
            // 将数据切片塞入 Tensor
            int data_offset = batch * batch_size * 28 * 28;
            for(int i = 0; i < batch_size * 28 * 28; ++i) {
                input_batch->data()[i] = all_images[data_offset + i];
            }
            for(int i = 0; i < batch_size; ++i) {
                target_batch[i] = all_labels[batch * batch_size + i];
            }

            // -- b. 前向、反向、更新三部曲 --
            optimizer.zero_grad();
            
            Tensor* output = model.forward(input_batch);
            
            float loss = criterion.forward(*output, target_batch.data());
            epoch_loss += loss;

            // 计算准确率 (用于展示)
            for(int i=0; i<batch_size; ++i) {
                int best_class = 0;
                float best_score = -1e9f;
                for(int c=0; c<10; ++c) {
                    float score = output->data()[i * 10 + c];
                    if(score > best_score) { best_score = score; best_class = c; }
                }
                if(best_class == target_batch[i]) correct_predictions++;
            }

            Tensor* loss_grad = criterion.backward();
            
            model.backward(loss_grad);
            optimizer.step();

            // -- c. 清理当前 batch 内存 --
            delete input_batch;

            if (batch % 200 == 0) {
                std::cout << "  Epoch [" << epoch+1 << "/" << epochs << "] "
                          << "Batch [" << batch << "/" << num_batches << "] "
                          << "Loss: " << std::fixed << std::setprecision(4) << loss << std::endl;
            }
        }
        
        std::cout << ">>> Epoch " << epoch+1 << " Summary: "
                  << "Avg Loss: " << epoch_loss / num_batches 
                  << " | Accuracy: " << (float)correct_predictions / total_images * 100.0f << "% <<<" << std::endl;
    }

    // 6. 保存 C++ 自己训练出的权重！
    std::cout << "\n[INFO] Training finished! Exporting C++ weights..." << std::endl;
    // 使用 cpp_ 前缀，防止覆盖 Python 导出的权重
    conv1->weight()->save_to_bin("weights/cpp_conv1_weight.bin");
    conv1->bias()->save_to_bin("weights/cpp_conv1_bias.bin");
    fc1->weight()->save_to_bin("weights/cpp_fc1_weight.bin");
    fc1->bias()->save_to_bin("weights/cpp_fc1_bias.bin");
    fc2->weight()->save_to_bin("weights/cpp_fc2_weight.bin");
    fc2->bias()->save_to_bin("weights/cpp_fc2_bias.bin");

    std::cout << "All C++ weights saved successfully!" << std::endl;
    return 0;
}