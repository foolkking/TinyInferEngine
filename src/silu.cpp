/*
 * @Author: fool
 * @Date: 2026-04-23 12:41:14
 * @LastEditors: fool
 * @LastEditTime: 2026-04-23 13:16:12
 * @FilePath: \TinyInferEngine\src\silu.cpp
 * @Description:  
 * @Note:  
 */
#include "silu.h"
#include <cmath>

void SiLU::forward(const Tensor& input, Tensor& output) {// SiLU(x) = x * sigmoid(x)
    const float* input_data = input.data();
    float* output_data = output.data();
    int num_elements = input.size();
    for (int i = 0; i < num_elements; ++i) {
        float x = input_data[i];
        float sigmoid_x = 1.0f / (1.0f + std::exp (-x)); // sigmoid(x)
        output_data[i] = x * sigmoid_x; // SiLU(x)
    }
    
}
void SiLU::backward(const Tensor& grad_output, Tensor& grad_input) {
    const float* input_data = cache_input_->data();
    const float* grad_output_data = grad_output.data();
    float* grad_input_data = grad_input.data();
    int num_elements = cache_input_->size();
    for (int i = 0; i < num_elements; ++i) {
        float x = input_data[i];
        float sigmoid_x = 1.0f / (1.0f + std::exp(-x)); // sigmoid(x)
        float grad_sigmoid = sigmoid_x * (1 - sigmoid_x); // sigmoid'(x)
        grad_input_data[i] = grad_output_data[i] * (sigmoid_x + x * grad_sigmoid); // 链式法则
    }
}