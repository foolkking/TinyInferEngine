/*
 * @Author: fool
 * @Date: 2026-04-20 22:14:51
 * @LastEditors: fool
 * @LastEditTime: 2026-04-22 15:46:43
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
void MaxPool2D::forward(const Tensor& input, Tensor& output) {
    const float* input_data = input.data(); 
    float* output_data = output.data();
    int batch_size = input.shape(0);
    int in_channels = input.shape(1);
    int in_height = input.shape(2);
    int in_width = input.shape(3);
    int out_height = output.shape(2);
    int out_width = output.shape(3);
    if(cache_input_ != nullptr) {
        delete cache_input_;
    }
    int new_cache_shape[] = {batch_size, in_channels, in_height, in_width};
    cache_input_ = new Tensor(new_cache_shape, 4);
    for(int i = 0; i < input.size(); ++i) {
        cache_input_->data()[i] = input_data[i];
    }
    
    if(max_indices_ != nullptr) {
        delete[] max_indices_;
    }
    max_indices_ = new int[output.size()]; // 用于记录每个输出位置对应的输入索引
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
}


std::vector<int> MaxPool2D::compute_output_shape(const std::vector<int>& input_shape) const {
    if (input_shape.size() != 4) {
        std::cerr << "Error: Input shape must be [batch_size, in_channels, in_height, in_width]!" << std::endl;
        return {};
    }
    int batch_size = input_shape[0];
    int in_channels = input_shape[1];
    int in_height = input_shape[2];
    int in_width = input_shape[3];
    
    int out_height = (in_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    int out_width = (in_width + 2 * padding_ - kernel_size_) / stride_ + 1;
    
    return {batch_size, in_channels, out_height, out_width};
}

void MaxPool2D::backward(const Tensor& grad_output, Tensor& grad_input) {
    const float* grad_output_data = grad_output.data();     
    float* grad_input_data = grad_input.data();
    // 将输入梯度初始化为0
    for (int i = 0; i < grad_input.size(); ++i) {
        grad_input_data[i] = 0.0f;
    }
    
    // 遍历每一个传回来的误差梯度，把它直接塞回当年那个赢家的口袋里
    // 这里不要加 #pragma omp parallel for，因为如果有窗口重叠，多个输出可能会把梯度累加给同一个输入，导致线程冲突
    for (int i = 0; i < grad_output.size(); ++i) {
        int winner_input_index = max_indices_[i];  //这个索引是前向传播时记录的最大值位置，位置是相对Tensor的一维索引
        if (winner_input_index >= 0) { // 安全检查
            grad_input_data[winner_input_index] += grad_output_data[i]; 
        }
    }
}