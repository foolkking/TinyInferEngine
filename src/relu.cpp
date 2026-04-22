/*
 * @Author: fool
 * @Date: 2026-04-20 22:14:50
 * @LastEditors: fool
 * @LastEditTime: 2026-04-22 13:09:51
 * @FilePath: \TinyInferEngine\src\relu.cpp
 * @Description:  
 * @Note:  
 */
#include "relu.h"
#include "tensor.h"
#include<vector>
void ReLU::forward(const Tensor& input, Tensor& output) {
    if(cache_input_ != nullptr) {
        delete cache_input_;
    } //目的是删除不同batch之间的缓存，避免相互干扰
    std::vector<int> new_shape;
    for(int i = 0; i < input.ndims(); ++i) {
        new_shape.push_back(input.shape(i));
    }
    cache_input_ = new Tensor(new_shape.data(), input.ndims(), false); // 缓存输入用于反向传播
    // 先把输入数据复制到 cache_input_ 中
    for (int i = 0; i < input.size(); ++i) {    
        cache_input_->data()[i] = input.data()[i];
    }

    // 直接进行 ReLU 操作
    // 假设输入输出形状已经匹配，直接进行 ReLU 操作
    int size = input.size();
    const float* input_data = input.data();
    float* output_data = output.data();
    
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        output_data[i] = input_data[i] > 0 ? input_data[i] : 0;
    }
}

void ReLU::backward(const Tensor& grad_output, Tensor& grad_input) {
    const float* grad_output_data = grad_output.data();
    float* cache_input_data = cache_input_->data();
    float* grad_input_data = grad_input.data();

    // ReLU 的反向传播：如果输入 > 0，梯度不变；否则梯度为 0
    for (int i = 0; i < grad_input.size(); ++i) {
        grad_input_data[i] = (cache_input_data[i] > 0) ? grad_output_data[i] : 0.0f;
    }
}

