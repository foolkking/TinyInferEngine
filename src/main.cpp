/*
 * @Author: fool
 * @Date: 2026-04-20 21:27:12
 * @LastEditors: fool
 * @LastEditTime: 2026-04-28 20:29:56
 * @FilePath: \TinyInferEngine\src\main.cpp
 * @Description:  
 * @Note:  
 */
#include <iostream>
#include <iomanip>
#include <cmath>
#include <memory>

#include "tensor.h"
#include "layer.h"
#include "model.h"
#include "conv2d.h"
#include "maxpool2d.h"
#include "flatten.h"
#include "linear.h"
#include "relu.h"

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "   TinyInferEngine: MNIST Inference      " << std::endl;
    std::cout << "=========================================" << std::endl;

    // 1. 组装计算图 (推理时全部 requires_grad = false)
    Sequential model;
    
    auto conv1 = std::make_shared<Conv2D>(1, 8, 3, 1, 0, false);
    auto pool1 = std::make_shared<MaxPool2D>(2, 2, 0);
    auto flatten = std::make_shared<Flatten>();
    auto fc1 = std::make_shared<Linear>(8 * 13 * 13, 128, false);
    auto relu = std::make_shared<ReLU>();
    auto fc2 = std::make_shared<Linear>(128, 10, false);

    model.add(conv1);
    model.add(pool1);
    model.add(flatten);
    model.add(fc1);
    model.add(relu);
    model.add(fc2);

    // 2. 加载训练好的真实权重
    std::cout << "[INFO] Loading weights trained by Autograd..." << std::endl;
    if (!conv1->weight()->load_from_file("weights/cpp_conv1_weight.bin")) return -1;
    if (!conv1->bias()->load_from_file("weights/cpp_conv1_bias.bin")) return -1;
    if (!fc1->weight()->load_from_file("weights/cpp_fc1_weight.bin")) return -1;
    if (!fc1->bias()->load_from_file("weights/cpp_fc1_bias.bin")) return -1;
    if (!fc2->weight()->load_from_file("weights/cpp_fc2_weight.bin")) return -1;
    if (!fc2->bias()->load_from_file("weights/cpp_fc2_bias.bin")) return -1;

    // 3. 准备输入图像
    std::cout << "[INFO] Loading test image..." << std::endl;
    std::vector<int> input_shape= {1, 1, 28, 28};
    TensorPtr input = std::make_shared<Tensor>(input_shape, false);
    
    if (!input->load_from_file("test_image_pixels.bin")) return -1;
    
    // 4. 引擎启动：前向传播！
    std::cout << "[INFO] Engine Running Forward Pass..." << std::endl;
    TensorPtr output = model.forward(input);

    // 5. 解析结果
    std::cout << "\n--- Confidence Scores ---" << std::endl;
    float max_score = -1e9f;
    int predicted_class = -1;
    
    for (int i = 0; i < 10; ++i) {
        float score = output->data()[i];
        if (score > max_score) {
            max_score = score;
            predicted_class = i;
        }
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < 10; ++i) sum_exp += std::exp(output->data()[i] - max_score); 
    
    for (int i = 0; i < 10; ++i) {
        float prob = std::exp(output->data()[i] - max_score) / sum_exp;
        std::cout << "Digit " << i << " : " << std::fixed << std::setprecision(5) << (prob * 100.0f) << "%" << std::endl;
    }

    std::cout << "\n=========================================" << std::endl;
    std::cout << ">>>  AI PREDICTION: It's the number [" << predicted_class << "] ! <<<" << std::endl;
    std::cout << "=========================================\n" << std::endl;

    // 没有任何 delete！程序结束时智能指针完美卸载内存。
    return 0;
}