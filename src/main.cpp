/*
 * @Author: fool
 * @Date: 2026-04-20 21:27:12
 * @LastEditors: fool
 * @LastEditTime: 2026-04-20 21:58:59
 * @FilePath: \TinyInferEngine\src\main.cpp
 * @Description:  
 * @Note:  
 */
#include <iostream>
#include <iomanip> // 用于控制打印格式

// 引入我们的全部心血
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
    std::cout << "   TinyInferEngine: MNIST Classification   " << std::endl;
    std::cout << "=========================================" << std::endl;

    // 1. 组装计算图 (必须与 PyTorch 侧的结构一模一样)
    Sequential model;
    
    // 输入 [1, 1, 28, 28] -> 经过 Conv2D(out=8, k=3) -> 输出 [1, 8, 26, 26]
    Conv2D* conv1 = new Conv2D(1, 8, 3, 1, 0);
    // 经过 MaxPool2D(k=2, stride=2) -> 输出 [1, 8, 13, 13]
    MaxPool2D* pool1 = new MaxPool2D(2, 2, 0);
    // 展平 -> 输出 [1, 8 * 13 * 13] 即 [1, 1352]
    Flatten* flatten = new Flatten();
    // 全连接 1352 -> 128
    Linear* fc1 = new Linear(8 * 13 * 13, 128);
    // 激活函数
    ReLU* relu = new ReLU();
    // 最终分类 128 -> 10
    Linear* fc2 = new Linear(128, 10);

    model.add(conv1);
    model.add(pool1);
    model.add(flatten);
    model.add(fc1);
    model.add(relu);
    model.add(fc2);

    // 2. 注入灵魂：加载训练好的真实权重
    std::cout << "[INFO] Loading weights from PyTorch..." << std::endl;
    if (!conv1->weight()->load_from_file("weights/conv1_weight.bin")) return -1;
    if (!conv1->bias()->load_from_file("weights/conv1_bias.bin")) return -1;
    
    if (!fc1->weight()->load_from_file("weights/fc1_weight.bin")) return -1;
    if (!fc1->bias()->load_from_file("weights/fc1_bias.bin")) return -1;
    
    if (!fc2->weight()->load_from_file("weights/fc2_weight.bin")) return -1;
    if (!fc2->bias()->load_from_file("weights/fc2_bias.bin")) return -1;

    // 3. 准备输入图像
    std::cout << "[INFO] Loading test image..." << std::endl;
    int input_shape[] = {1, 1, 28, 28};
    Tensor* input = new Tensor(input_shape, 4);
    // 直接把图像的浮点像素作为二进制读入
    if (!input->load_from_file("test_image_pixels.bin")) return -1;
    
    // 4. 引擎启动：前向传播！
    std::cout << "[INFO] Engine Running Forward Pass..." << std::endl;
    Tensor* output = model.forward(input);

    // 5. 解析并打印结果
    std::cout << "\n--- Confidence Scores ---" << std::endl;
    float max_score = -1e9f;
    int predicted_class = -1;
    
    for (int i = 0; i < 10; ++i) {
        float score = output->data()[i];
        std::cout << "Digit " << i << " : " << std::fixed << std::setprecision(4) << score << std::endl;
        
        // 寻找得分最高的类别 (ArgMax)
        if (score > max_score) {
            max_score = score;
            predicted_class = i;
        }
    }

    std::cout << "\n=========================================" << std::endl;
    std::cout << ">>>  AI PREDICTION: It's the number [" << predicted_class << "] ! <<<" << std::endl;
    std::cout << "=========================================\n" << std::endl;

    // 清理战场
    delete input;
    delete output; // Sequential 中间产物已经自动清理，这里只需要清理最终输出

    return 0;
}