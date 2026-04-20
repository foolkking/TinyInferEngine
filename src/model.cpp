/*
 * @Author: fool
 * @Date: 2026-04-17 19:56:03
 * @LastEditors: fool
 * @LastEditTime: 2026-04-20 21:30:04
 * @FilePath: \TinyInferEngine\src\model.cpp
 * @Description:  
 * @Note:  
 */
#include "model.h"

Sequential::~Sequential() {
    // 这里我们不负责删除 Layer*，因为它们可能在外部被共享或管理
    layers_.clear();
}
void Sequential::add(Layer* layer) {
    layers_.push_back(layer);
}
Tensor* Sequential::forward(Tensor* input) {
    Tensor* current_input = input;
    Tensor* current_output = nullptr;

    for (Layer* layer : layers_) {
        // 先根据当前输入的形状计算输出形状
        std::vector<int> input_shape;
        for (int i = 0; i < current_input->ndims(); ++i) {
            input_shape.push_back(current_input->shape(i));
        }
        std::vector<int> output_shape = layer->compute_output_shape(input_shape);

        // 创建一个新的 Tensor 来存储当前层的输出
        current_output = new Tensor(output_shape.data(), output_shape.size());

        // 执行前向传播
        layer->forward(*current_input, *current_output);
        if(current_input != input) {
            delete current_input; // 删除上一个中间输出，释放内存
        }
        // 当前输出成为下一层的输入
        current_input = current_output;
    }

    return current_output; // 最后一个输出就是整个模型的输出
}

std::vector<int> Sequential::compute_output_shape(const std::vector<int>& input_shape) const {
    std::vector<int> current_shape = input_shape;
    for (Layer* layer : layers_) {
        current_shape = layer->compute_output_shape(current_shape);
    }
    return current_shape;
}

