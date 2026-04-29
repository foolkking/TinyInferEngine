/*
 * @Author: fool
 * @Date: 2026-04-18 00:31:01
 * @LastEditors: fool
 * @LastEditTime: 2026-04-29 11:41:14
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
    std::vector<int> weight_shape = {out_channels_, in_channels_, kernel_size_, kernel_size_}; // 权重张量的形状
    weight_ = std::make_shared<Tensor>(weight_shape, requires_grad); // 四维张量
    std::vector<int> bias_shape = {out_channels_}; // 偏置向量的形状
    bias_ = std::make_shared<Tensor>(bias_shape, requires_grad); // 一维张量
}
Conv2D::~Conv2D() {
    clearup(); // 调用清理函数释放资源
}

void Conv2D::clearup() {
    Layer::clearup(); // 调用基类清理输入缓存
    if (weight_ != nullptr) {
        weight_.reset();
        weight_ = nullptr;
    }
    if (bias_ != nullptr) {
        bias_.reset();
        bias_ = nullptr;
    }
}

TensorPtr Conv2D::forward(TensorPtr input) {
    // 这里我们需要实现卷积操作：output = conv2d(input, weight) + bias
    // 注意 input 的形状是 [batch_size, in_channels, in_height, in_width]
    // weight 的形状是 [out_channels, in_channels, kernel_size, kernel_size]
    // output 的形状是 [batch_size, out_channels, out_height, out_width]    
    if (input->ndims() != 4) {
        return {};
    }
    int batch_size = input->shape(0);
    int in_height = input->shape(2);
    int in_width = input->shape(3);

    // 计算输出的高度和宽度
    int out_height = (in_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    int out_width = (in_width + 2 * padding_ - kernel_size_) / stride_ + 1;
    std::vector<int> output_shape = {batch_size, out_channels_, out_height, out_width};

    bool out_req_grad = input->requires_grad() || weight_->requires_grad() || bias_->requires_grad();
    auto output = std::make_shared<Tensor>(output_shape, out_req_grad); 

    float* output_data = output->data();
    for(int i = 0; i < output->size(); ++i) {
        output_data[i] = 0.0f; // 先清零
    }

    const float* input_data = input->data();
    const float* weight_data = weight_->data();
    const float* bias_data = bias_->data();
    //#pragma omp parallel for
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
    
    if(out_req_grad){
        Tensor* output_ptr = output.get(); // 捕获输出张量的智能指针
        std::vector<NamedParameter> params = this->parameters(); // 获取权重和偏置作为命名参数
        std::function<void()> backward_fn = [input, params, output_ptr,padding=padding_,stride=stride_,kernel_size=kernel_size_]() {
            float* grad_input_data = input->grad();
            //升级自动图后，清零工作必须由优化器（Optimizer）在每一轮迭代开始前统一调用 zero_grad() 完成。因为可能有多个连接这个张量
            // for(int i = 0; i < input->size(); ++i) {
            //     grad_input_data[i] = 0.0f; // 先清零输入梯度
            // }
            const float* grad_output_data = output_ptr->grad();
            const float* input_data = input->data();

            // 获取权重、权重梯度和偏置梯度指针
            const float* weight_data = params[0].tensor->data();
            float* grad_weight_data = params[0].tensor->grad();
            float* grad_bias_data = params[1].tensor->grad();

            // 获取各维度大小
            int batch_size = input->shape(0);
            int in_channels = input->shape(1);
            int out_channels = output_ptr->shape(1);
            int in_height = input->shape(2); 
            int in_width = input->shape(3);
            int out_height = output_ptr->shape(2);
            int out_width = output_ptr->shape(3);

            // 计算梯度
            //不能用#pragma omp parallel for，保证单线程计算的绝对正确性，因为权重梯度和偏置梯度的更新存在数据竞争，后续可以考虑加锁或者使用线程局部变量累加后再合并
            for (int n = 0; n < batch_size; ++n) {
                for (int oc = 0; oc < out_channels; ++oc) {
                    for (int oh = 0; oh < out_height; ++oh) {
                        for (int ow = 0; ow < out_width; ++ow) {
                            float grad_out_val = grad_output_data[n * out_channels * out_height * out_width + 
                                                                oc * out_height * out_width + 
                                                                oh * out_width + 
                                                                ow];
                            grad_bias_data[oc] += grad_out_val; // 累加偏置梯度
                            for (int ic = 0; ic < in_channels; ++ic) {   
                                for (int kh = 0; kh < kernel_size; ++kh) {
                                    for (int kw = 0; kw < kernel_size; ++kw) {
                                        int ih = oh * stride - padding + kh;
                                        int iw = ow * stride - padding + kw;
                                        if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                            // 累加权重梯度
                                            grad_weight_data[oc * in_channels * kernel_size * kernel_size + 
                                                            ic * kernel_size * kernel_size + 
                                                            kh * kernel_size + 
                                                            kw] += input_data[n * in_channels * in_height * in_width + 
                                                                            ic * in_height * in_width + 
                                                                            ih * in_width + 
                                                                            iw] * grad_out_val;
                                            // 累加输入梯度
                                            if(input->requires_grad()) {
                                                grad_input_data[n * in_channels * in_height * in_width + 
                                                                ic * in_height * in_width + 
                                                                ih * in_width + 
                                                                iw] += weight_data[oc * in_channels * kernel_size * kernel_size + 
                                                                                ic * kernel_size * kernel_size + 
                                                                                kh * kernel_size + 
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
        };
        
        
        output->set_auto_grad(backward_fn, {input, params[0].tensor, params[1].tensor}); // 绑定计算图
        
    }
    return  output;
}

std::vector<NamedParameter> Conv2D::parameters() {
    return {{"weight", weight_}, {"bias", bias_}};
}