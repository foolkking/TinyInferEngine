/*
 * @Author: fool
 * @Date: 2026-04-20 22:14:51
 * @LastEditors: fool
 * @LastEditTime: 2026-04-29 11:42:25
 * @FilePath: \TinyInferEngine\src\maxpool2d.cpp
 * @Description:  
 * @Note:  
 */

#include "maxpool2d.h"
#include <limits>
MaxPool2D::MaxPool2D(int k_size, int stride, int padding) {
    kernel_size_ = k_size;
    stride_ = stride;
    padding_ = padding;
}
MaxPool2D::~MaxPool2D() {
    if (max_indices_ != nullptr) {
        delete[] max_indices_;
        max_indices_ = nullptr;
    }
}
/**
 * @brief 最大池化操作
 * @param input 输入张量，形状为 [batch_size, in_channels, in_height, in_width]
 * @param output 输出张量，形状为 [batch_size, in_channels, out_height, out_width]
 */
TensorPtr MaxPool2D::forward(TensorPtr input) {
    
    if (input->ndims() != 4) {
        std::cerr << "Error: Input shape must be [batch_size, in_channels, in_height, in_width]!" << std::endl;
        exit(EXIT_FAILURE);
    }
    int batch_size = input->shape(0);
    int in_channels = input->shape(1);
    int in_height = input->shape(2);
    int in_width = input->shape(3);
    int out_height = (in_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    int out_width = (in_width + 2 * padding_ - kernel_size_) / stride_ + 1;
    std::vector<int> output_shape = {batch_size, in_channels, out_height, out_width};
    TensorPtr output = std::make_shared<Tensor>(output_shape, input->requires_grad());
    const float* input_data = input->data(); 
    float* output_data = output->data();
    
    
    if(max_indices_ != nullptr) {
        delete[] max_indices_;
    }
    max_indices_ = new int[output->size()]; // 用于记录每个输出位置对应的输入索引
    #pragma omp parallel for
    for (int n = 0; n < batch_size; ++n) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity(); // 初始化为负无穷
                    int max_index = -1; // 初始化为无效索引
                    for (int kh = 0; kh < kernel_size_; ++kh) {
                        for (int kw = 0; kw < kernel_size_; ++kw) {
                            int ih = oh * stride_ - padding_ + kh;
                            int iw = ow * stride_ - padding_ + kw;
                            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                int current_index = n * in_channels * in_height * in_width + 
                                                ic * in_height * in_width + 
                                                ih * in_width + 
                                                iw;
                                float val = input_data[current_index];
                                if (val > max_val) {
                                    max_val = val;
                                    max_index = current_index; // 记录最大值对应的输入索引
                                }
                            }
                        }
                    }
                    int output_index = n * in_channels * out_height * out_width + 
                                        ic * out_height * out_width + 
                                        oh * out_width + 
                                        ow;
                    max_indices_[output_index] = max_index; // 存储最大值索引
                    output_data[output_index] = max_val;
                }
            }
        }
    }
    if(input->requires_grad()) {
       Tensor* output_ptr = output.get(); // 获取输出张量的裸指针
       std::function<void()> backward_fn = [this, input, output_ptr,kernel_size_=kernel_size_,padding_=padding_,stride_=stride_,max_indices_=max_indices_]() {
            const float* grad_output_data = output_ptr->grad();     
            float* grad_input_data = input->grad();
            
            // 遍历每一个传回来的误差梯度，把它直接塞回当年那个赢家的口袋里
            // 这里不要加 #pragma omp parallel for，因为如果有窗口重叠，多个输出可能会把梯度累加给同一个输入，导致线程冲突
            for (int i = 0; i < output_ptr->size(); ++i) {
                int winner_input_index = max_indices_[i];  //这个索引是前向传播时记录的最大值位置，位置是相对Tensor的一维索引
                if (winner_input_index >= 0) { // 安全检查
                    grad_input_data[winner_input_index] += grad_output_data[i]; 
                }
            }

            float* input_grad_data = input->grad(); // 获取输入张量的梯度指针
            for (int i = 0; i < input->size(); ++i) {
                input_grad_data[i] += grad_input_data[i]; // 累加到输入张量的梯度中
            }
        };
        output->set_auto_grad(backward_fn, {input}); // 将反向传播函数和前驱节点绑定到输出张量上
    }
    return output;
}

