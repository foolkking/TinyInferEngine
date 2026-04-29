
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <chrono> // 新增：用于监控训练耗时

#include "tensor.h"
#include "layer.h"
#include "model.h"
#include "loss.h"
#include "optimizer.h"  // 包含 ParamGroup, SGD, AdamW
#include "scheduler.h"  // 包含 LRScheduler, CosineAnnealingLR

// ... swap_endian, read_mnist_images, read_mnist_labels 保持完全不变 ...
uint32_t swap_endian(uint32_t val) { /* 你的原有代码 */ return ((val << 24) & 0xff000000) | ((val << 8) & 0x00ff0000) | ((val >> 8) & 0x0000ff00) | ((val >> 24) & 0x000000ff); }
std::vector<float> read_mnist_images(const std::string& path, int& num_images) { /* 你的原有代码 */ std::ifstream file(path, std::ios::binary); if (!file) { std::cerr << "Cannot open MNIST image file: " << path << std::endl; exit(1); } uint32_t magic_number=0, num_items=0, num_rows=0, num_cols=0; file.read((char*)&magic_number, sizeof(magic_number)); file.read((char*)&num_items, sizeof(num_items)); file.read((char*)&num_rows, sizeof(num_rows)); file.read((char*)&num_cols, sizeof(num_cols)); num_images = swap_endian(num_items); int rows = swap_endian(num_rows); int cols = swap_endian(num_cols); std::vector<unsigned char> raw_data(num_images * rows * cols); file.read((char*)raw_data.data(), raw_data.size()); std::vector<float> float_data(raw_data.size()); for (size_t i = 0; i < raw_data.size(); ++i) { float_data[i] = raw_data[i] / 255.0f; } return float_data; }
std::vector<int> read_mnist_labels(const std::string& path) { /* 你的原有代码 */ std::ifstream file(path, std::ios::binary); if (!file) { std::cerr << "Cannot open MNIST label file: " << path << std::endl; exit(1); } uint32_t magic_number=0, num_items=0; file.read((char*)&magic_number, sizeof(magic_number)); file.read((char*)&num_items, sizeof(num_items)); int num_images = swap_endian(num_items); std::vector<unsigned char> raw_labels(num_images); file.read((char*)raw_labels.data(), raw_labels.size()); std::vector<int> labels(num_images); for (int i = 0; i < num_images; ++i) { labels[i] = raw_labels[i]; } return labels; }

int main() {
    std::srand(std::time(nullptr)); 

    std::cout << "=========================================" << std::endl;
    std::cout << "  TinyInferEngine: Autograd & AdamW !    " << std::endl;
    std::cout << "=========================================" << std::endl;

    std::string data_dir = "./data/MNIST/raw/"; 
    int total_images = 0;
    std::vector<float> all_images = read_mnist_images(data_dir + "train-images-idx3-ubyte", total_images);
    std::vector<int> all_labels = read_mnist_labels(data_dir + "train-labels-idx1-ubyte");

    // 1. 搭建网络 (全面升级为 std::shared_ptr)
    Sequential model;
    auto conv1 = std::make_shared<Conv2D>(1, 8, 3, 1, 0, true);
    auto pool1 = std::make_shared<MaxPool2D>(2, 2, 0); 
    auto flatten = std::make_shared<Flatten>();
    auto fc1 = std::make_shared<Linear>(8 * 13 * 13, 128, true);
    auto relu = std::make_shared<ReLU>();
    auto fc2 = std::make_shared<Linear>(128, 10, true);

    // 随机初始化
    conv1->weight()->randomize(-0.1f, 0.1f); conv1->bias()->fill(0.0f);
    fc1->weight()->randomize(-0.05f, 0.05f); fc1->bias()->fill(0.0f);
    fc2->weight()->randomize(-0.1f, 0.1f);   fc2->bias()->fill(0.0f);

    model.add(conv1);
    model.add(pool1);
    model.add(flatten);
    model.add(fc1);
    model.add(relu);
    model.add(fc2);

    // 2. 配置大模型标配组件：参数组 + AdamW + 余弦退火
    std::vector<ParamGroup> groups = {
        // 学习率 1e-3, 权重衰减 1e-4
        ParamGroup(model.named_parameters(), 1e-3f, 1e-4f) 
    };
    AdamW optimizer(groups); // 替换了原来的 SGD
    CrossEntropyLoss criterion;
    
    int epochs = 5;
    CosineAnnealingLR scheduler(&optimizer, epochs, 1e-5f); // 学习率退火下限 1e-5

    int batch_size = 64;
    int num_batches = total_images / batch_size;

    std::vector<int> indices(total_images);   
    for (int i = 0; i < total_images; ++i) indices[i] = i;
    
    // 3. 动态图大循环
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int correct_predictions = 0;
        
        std::random_shuffle(indices.begin(), indices.end()); 

        for (int batch = 0; batch < num_batches; ++batch) {
            // -- a. 准备当前 Batch 的数据 (升级为 TensorPtr) --
            std::vector<int>b_shape = {batch_size, 1, 28, 28};
            TensorPtr input_batch = std::make_shared<Tensor>(b_shape, false);
            
            std::vector<int>t_shape= {batch_size};
            TensorPtr target_batch = std::make_shared<Tensor>(t_shape, false);
            
            // 修复打乱 Bug：根据洗牌后的 indices 抽取像素和标签
            for(int i = 0; i < batch_size; ++i) {
                int real_idx = indices[batch * batch_size + i];
                target_batch->data()[i] = all_labels[real_idx];
                for(int p = 0; p < 28 * 28; ++p) {
                    input_batch->data()[i * 28 * 28 + p] = all_images[real_idx * 28 * 28 + p];
                }
            }

            // -- b. 动态图 5 步曲！--
            optimizer.zero_grad();                                  // 1. 清空旧梯度
            TensorPtr output = model.forward(input_batch);          // 2. 前向建图
            TensorPtr loss = criterion.forward(output, target_batch); // 3. 算出误差源
            loss->backward();                                       // 4. 一键核爆！沿图反向回传
            optimizer.step();                                       // 5. 权重更新

            epoch_loss += loss->data()[0];

            // 计算准确率
            for(int i = 0; i < batch_size; ++i) {
                int best_class = 0;
                float best_score = -1e9f;
                for(int c = 0; c < 10; ++c) {
                    float score = output->data()[i * 10 + c];
                    if(score > best_score) { best_score = score; best_class = c; }
                }
                if(best_class == target_batch->data()[i]) correct_predictions++;
            }

            // -- c. 内存自动释放 -- 
            // 根本不需要写 delete！智能指针出作用域后，全网临时的梯度和缓存自动清空！

            if (batch % 200 == 0) {
                std::cout << "  Epoch [" << epoch+1 << "/" << epochs << "] "
                          << "Batch [" << batch << "/" << num_batches << "] "
                          << "Loss: " << std::fixed << std::setprecision(4) << loss->data()[0] << std::endl;
            }
        }
        
        std::cout << ">>> Epoch " << epoch+1 << " Summary: Avg Loss: " << epoch_loss / num_batches 
                  << " | Accuracy: " << (float)correct_predictions / total_images * 100.0f << "% <<<" << std::endl;
        
        // Epoch 结束，调度器自动调低 AdamW 的学习率
        scheduler.step();
    }

    // 4. 导出权重
    std::cout << "\n[INFO] Training finished! Exporting weights..." << std::endl;
    conv1->weight()->save_to_bin("weights/cpp_conv1_weight.bin");
    conv1->bias()->save_to_bin("weights/cpp_conv1_bias.bin");
    fc1->weight()->save_to_bin("weights/cpp_fc1_weight.bin");
    fc1->bias()->save_to_bin("weights/cpp_fc1_bias.bin");
    fc2->weight()->save_to_bin("weights/cpp_fc2_weight.bin");
    fc2->bias()->save_to_bin("weights/cpp_fc2_bias.bin");

    return 0;
}