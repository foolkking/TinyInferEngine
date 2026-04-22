/*
 * @Author: fool
 * @Date: 2026-04-17 19:56:03
 * @LastEditors: fool
 * @LastEditTime: 2026-04-22 16:23:11
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
    clear_intermediate_outputs (); // 每次前向传播前清理上一次的中间输出，释放内存
    Tensor* current_input = input;
    Tensor* current_output = nullptr;
    intermediate_outputs_.push_back(input); // 缓存输入，方便反向传播时使用
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
        intermediate_outputs_.push_back(current_output); // 缓存当前输出，方便反向传播时使用

        // 当前输出成为下一层的输入
        current_input = current_output;
    }
    intermediate_outputs_.pop_back(); // 最后一个输出是模型的最终输出，不需要缓存了，弹出并返回
    return current_output; // 最后一个输出就是整个模型的输出
}

std::vector<int> Sequential::compute_output_shape(const std::vector<int>& input_shape) const {
    std::vector<int> current_shape = input_shape;
    for (Layer* layer : layers_) {
        current_shape = layer->compute_output_shape(current_shape);
    }
    return current_shape;
}

void Sequential::backward(Tensor* grad_output) {
    
    Tensor* current_grad_output = grad_output;
    // 反向遍历层，从最后一层开始
    for (int i = layers_.size() - 1; i >= 0; --i) {
        //std::cout << "    [DEBUG] Preparing backward for layer index: " << i << std::endl;
        Layer* layer = layers_[i];
        Tensor* current_grad_input = nullptr;
        // 上一层的输出就是当前层的输入
        std::vector<int> input_shape;
        for (int j = 0; j < intermediate_outputs_[i]->ndims(); ++j) {
            input_shape.push_back(intermediate_outputs_[i]->shape(j));
        }
        current_grad_input = new Tensor(input_shape.data(), input_shape.size()); // 创建一个新的 Tensor 来存储当前层的输入梯度
  
        //std::cout << "    [DEBUG] Executing layer->backward() for layer " << i << "..." << std::endl;
        // 核心微积分计算
        layer->backward(*current_grad_output, *current_grad_input);
        // std::cout << "    [DEBUG] Layer " << i << " backward SUCCESS!" << std::endl;

        // 执行反向传播
        if(current_grad_output != grad_output) {
            delete current_grad_output; // 删除上一个中间输出梯度，释放内存
        }
        // 当前输入梯度成为下一层的输出梯度
        current_grad_output = current_grad_input;
    }
    delete grad_output; // 删除最终输出的梯度，释放内存
}

void Sequential::clear_intermediate_outputs() { 
    // 第 0 个元素是借来的 input，我们无权释放！从索引 1 开始删！
    for (size_t i = 1; i < intermediate_outputs_.size(); ++i) {
        delete intermediate_outputs_[i];
    }
    intermediate_outputs_.clear(); 
}