/*
 * @Author: fool
 * @Date: 2026-04-20 22:14:50
 * @LastEditors: fool
 * @LastEditTime: 2026-04-29 11:43:07
 * @FilePath: \TinyInferEngine\src\relu.cpp
 * @Description:  
 * @Note:  
 */
#include "relu.h"
#include "tensor.h"
#include<vector>
TensorPtr ReLU::forward(TensorPtr input){
    TensorPtr output = std::make_shared<Tensor> (input->shape(),input->requires_grad());
    // 直接进行 ReLU 操作
    // 假设输入输出形状已经匹配，直接进行 ReLU 操作
    int size = input->size();
    const float* input_data = input->data();
    float* output_data = output->data();
    
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        output_data[i] = input_data[i] > 0 ? input_data[i] : 0;
    }
    if(input->requires_grad()){
        Tensor* output_ptr = output.get();
        std::function<void()> backward_fn =[output_ptr,input](){
            const float* grad_output_data = output_ptr->grad();
            float* input_data = input->data();
            float* grad_input_data = input->grad();

            // ReLU 的反向传播：如果输入 > 0，梯度不变；否则梯度为 0
            for (int i = 0; i < input->size(); ++i) {
                grad_input_data[i] = (input_data[i] > 0) ? grad_output_data[i] : 0.0f;
            }
        };
        output->set_auto_grad(backward_fn,{input});
    }
    return output;
}

