/*
 * @Author: fool
 * @Date: 2026-04-23 12:41:14
 * @LastEditors: fool
 * @LastEditTime: 2026-04-28 20:15:26
 * @FilePath: \TinyInferEngine\src\silu.cpp
 * @Description:  
 * @Note:  
 */
#include "silu.h"
#include <cmath>

TensorPtr SiLU(TensorPtr input){// SiLU(x) = x * sigmoid(x)
    TensorPtr output = std::make_shared<Tensor> (input->shape(),input->requires_grad());
    const float* input_data = input->data();
    float* output_data = output->data();
    int num_elements = input->size();
    for (int i = 0; i < num_elements; ++i) {
        float x = input_data[i];
        float sigmoid_x = 1.0f / (1.0f + std::exp (-x)); // sigmoid(x)
        output_data[i] = x * sigmoid_x; // SiLU(x)
    }
    if(input->requires_grad()){
        Tensor* output_ptr = output.get();
        std::function<void()> backward_fn = [output_ptr,input](){
            const float* input_data = input->data();
            const float* grad_output_data = output_ptr->grad();
            float* grad_input_data = input->grad();
            int num_elements = input->size();
            for (int i = 0; i < num_elements; ++i) {
                float x = input_data[i];
                float sigmoid_x = 1.0f / (1.0f + std::exp(-x)); // sigmoid(x)
                float grad_sigmoid = sigmoid_x * (1 - sigmoid_x); // sigmoid'(x)
                grad_input_data[i] = grad_output_data[i] * (sigmoid_x + x * grad_sigmoid); // 链式法则
            }
        };
        output->set_auto_grad(backward_fn,{input});
    }
    return output;
}