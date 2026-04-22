/*
 * @Author: fool
 * @Date: 2026-04-18 00:31:01
 * @LastEditors: fool
 * @LastEditTime: 2026-04-22 15:16:47
 * @FilePath: \TinyInferEngine\src\conv2d.cpp
 * @Description:  
 * @Note:  
 */
#include "conv2d.h"
#include<vector>
Conv2D::Conv2D(int in_ch, int out_ch, int k_size, int stride, int padding, bool requires_grad) {
    in_channels_ = in_ch;
    out_channels_ = out_ch;
    kernel_size_ = k_size;
    stride_ = stride;
    padding_ = padding;
    
    int weight_shape[] = {out_channels_, in_channels_, kernel_size_, kernel_size_}; // 权重张量的形状
    weight_ = new Tensor(weight_shape, 4, requires_grad); // 四维张量
    int bias_shape[] = {out_channels_}; // 偏置向量的形状
    bias_ = new Tensor(bias_shape, 1, requires_grad); // 一维张量
}
Conv2D::~Conv2D() {
    clearup(); // 调用清理函数释放资源
}

void Conv2D::clearup() {
    Layer::clearup(); // 调用基类清理输入缓存
    if (weight_ != nullptr) {
        delete weight_;
        weight_ = nullptr;
    }
    if (bias_ != nullptr) {
        delete bias_;
        bias_ = nullptr;
    }
}

void Conv2D::forward(const Tensor& input, Tensor& output) {
    if(cache_input_ != nullptr) { // 只有在第一次前向传播时才缓存输入
        delete cache_input_;
    }
    std::vector<int> new_shape;
    for(int i = 0; i < input.ndims(); ++i) {
        new_shape.push_back(input.shape(i));
    }
    cache_input_ = new Tensor(new_shape.data(), input.ndims()); // 深拷贝输入张量，后续反向传播使用
    for (int i = 0; i < input.size(); ++i) {
        cache_input_->data()[i] = input.data()[i];
    }
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

// 反向传播：计算输入梯度、权重梯度和偏置梯度
void Conv2D::backward(const Tensor& grad_output, Tensor& grad_input) {
    // 获取输入、输出梯度和缓存的输入数据指针
    float* grad_input_data = grad_input.data();
    for(int i = 0; i < grad_input.size(); ++i) {
        grad_input_data[i] = 0.0f; // 先清零输入梯度
    }
    const float* grad_output_data = grad_output.data();
    const float* input_data = cache_input_->data();

    // 获取权重、权重梯度和偏置梯度指针
    const float* weight_data = weight_->data();
    float* grad_weight_data = weight_->grad();
    float* grad_bias_data = bias_->grad();

    // 获取各维度大小
    int batch_size = cache_input_->shape(0);
    int in_height = cache_input_->shape(2); 
    int in_width = cache_input_->shape(3);
    int out_height = grad_output.shape(2);
    int out_width = grad_output.shape(3);

    // 计算梯度
    //不能用#pragma omp parallel for，保证单线程计算的绝对正确性，因为权重梯度和偏置梯度的更新存在数据竞争，后续可以考虑加锁或者使用线程局部变量累加后再合并
    for (int n = 0; n < batch_size; ++n) {
        for (int oc = 0; oc < out_channels_; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float grad_out_val = grad_output_data[n * out_channels_ * out_height * out_width + 
                                                          oc * out_height * out_width + 
                                                          oh * out_width + 
                                                          ow];
                    grad_bias_data[oc] += grad_out_val; // 累加偏置梯度
                    for (int ic = 0; ic < in_channels_; ++ic) {   
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                int ih = oh * stride_ - padding_ + kh;
                                int iw = ow * stride_ - padding_ + kw;
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    // 累加权重梯度
                                    grad_weight_data[oc * in_channels_ * kernel_size_ * kernel_size_ + 
                                                     ic * kernel_size_ * kernel_size_ + 
                                                     kh * kernel_size_ + 
                                                     kw] += input_data[n * in_channels_ * in_height * in_width + 
                                                                       ic * in_height * in_width + 
                                                                       ih * in_width + 
                                                                       iw] * grad_out_val;
                                    // 累加输入梯度
                                    grad_input_data[n * in_channels_ * in_height * in_width + 
                                                    ic * in_height * in_width + 
                                                    ih * in_width + 
                                                    iw] += weight_data[oc * in_channels_ * kernel_size_ * kernel_size_ + 
                                                                       ic * kernel_size_ * kernel_size_ + 
                                                                       kh * kernel_size_ + 
                                                                       kw] * grad_out_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}  