/*
 * @Author: fool
 * @Date: 2026-04-18 00:31:01
 * @LastEditors: fool
 * @LastEditTime: 2026-04-18 00:32:01
 * @FilePath: \TinyInferEngine\src\conv2d.cpp
 * @Description:  
 * @Note:  
 */
#include "conv2d.h"
Conv2D::Conv2D(int in_ch, int out_ch, int k_size, int stride, int padding) {
    in_channels_ = in_ch;
    out_channels_ = out_ch;
    kernel_size_ = k_size;
    stride_ = stride;
    padding_ = padding;

    // 初始化权重和偏置张量
    int weight_shape[] = {out_channels_, in_channels_, kernel_size_, kernel_size_}; // 权重张量的形状
    weight_ = new Tensor(weight_shape, 4); // 四维张量
    
    int bias_shape[] = {out_channels_}; // 偏置向量的形状
    bias_ = new Tensor(bias_shape, 1); // 一维张量
}

Conv2D::~Conv2D() {
    delete weight_;
    delete bias_;
}

void Conv2D::forward(const Tensor& input, Tensor& output) {
    for(int i = 0; i < output.size(); ++i) {
        output.data()[i] = 0.0f; // 先清零
    }
    // 这里我们需要实现卷积操作：output = conv2d(input, weight) + bias
    // 注意 input 的形状是 [batch_size, in_channels, in_height, in_width]
    // weight 的形状是 [out_channels, in_channels, kernel_size, kernel_size]
    // output 的形状是 [batch_size, out_channels, out_height, out_width]
    const float* input_data = input.data();
    const float* weight_data = weight_->data();
    const float* bias_data = bias_->data();
    float* output_data = output.data();
    int batch_size = input.shape(0);
    int in_height = input.shape(2); 
    int in_width = input.shape(3);
    int out_height = output.shape(2);
    int out_width = output.shape(3);
    #pragma omp parallel for
    for (int n = 0; n < batch_size; ++n) {
        for (int oc = 0; oc < out_channels_; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float sum = bias_data[oc]; // 从偏置开始累加
                    for (int ic = 0; ic < in_channels_; ++ic) {
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                int ih = oh * stride_ - padding_ + kh;
                                int iw = ow * stride_ - padding_ + kw;
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    sum += input_data[n * in_channels_ * in_height * in_width + 
                                                        ic * in_height * in_width + 
                                                        ih * in_width + 
                                                        iw] * 
                                           weight_data[oc * in_channels_ * kernel_size_ * kernel_size_ + 
                                                       ic * kernel_size_ * kernel_size_ + 
                                                       kh * kernel_size_ + 
                                                       kw];
                                }
                            }
                        }
                    }
                    output_data[n * out_channels_ * out_height * out_width + 
                                oc * out_height * out_width + 
                                oh * out_width + 
                                ow] = sum;
                }
            }
        }
    }
}
std::vector<int> Conv2D::compute_output_shape(const std::vector<int>& input_shape) const {
    // 输入形状应该是 [batch_size, in_channels, in_height, in_width]
    if (input_shape.size() != 4 || input_shape[1] != in_channels_) {
        std::cerr << "Error: Input shape must be [batch_size, " << in_channels_ << ", height, width]!" << std::endl;
        return {};
    }
    int batch_size = input_shape[0];
    int in_height = input_shape[2];
    int in_width = input_shape[3];

    // 计算输出的高度和宽度
    int out_height = (in_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    int out_width = (in_width + 2 * padding_ - kernel_size_) / stride_ + 1;

    return {batch_size, out_channels_, out_height, out_width};
}