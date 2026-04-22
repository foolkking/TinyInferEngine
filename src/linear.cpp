/*
 * @Author: fool
 * @Date: 2026-04-20 22:14:50
 * @LastEditors: fool
 * @LastEditTime: 2026-04-22 15:06:32
 * @FilePath: \TinyInferEngine\src\linear.cpp
 * @Description:  
 * @Note:  
 */

#include "linear.h"
#include<vector>
Linear::Linear(int in_features, int out_features, bool requires_grad) {
    in_features_ = in_features;
    out_features_ = out_features;
    
    // 初始化权重和偏置张量
    int weight_shape[] = {out_features_, in_features_}; // 权重矩阵的形状
    weight_ = new Tensor(weight_shape, 2, requires_grad); // 二维张量
    
    int bias_shape[] = {out_features_}; // 偏置向量的形状
    bias_ = new Tensor(bias_shape, 1, requires_grad); // 一维张量
}

Linear::~Linear() {
    delete weight_;
    delete bias_;
}

void Linear::forward(const Tensor& input, Tensor& output) {
    //用于反向传播
    if(cache_input_ != nullptr) {
        delete cache_input_;
    }
    std::vector<int> new_shape;
    for(int i = 0; i < input.ndims(); ++i) {    
        new_shape.push_back(input.shape(i));    
    } 
    cache_input_ = new Tensor(new_shape.data(), input.ndims(), false); // 缓存输入用于反向传播
    for(int i = 0; i < input.size(); ++i) {
        cache_input_->data()[i] = input.data()[i];
    }

    //前向传播
    // 假设输入形状是 [batch_size, in_features_]
    // 输出形状应该是 [batch_size, out_features_]
    int batch_size = input.shape(0);
    // 这里我们需要实现矩阵乘法：output = input * weight^T + bias
    // 注意 weight 的形状是 [out_features_, in_features_], 需要转置成 [in_features_, out_features_]
    const float* input_data = input.data();
    const float* weight_data = weight_->data();
    const float* bias_data = bias_->data();
    float* output_data = output.data();
    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < out_features_; ++j) {
            float sum = bias_data[j]; // 从偏置开始累加
            for (int k = 0; k < in_features_; ++k) {
                sum += input_data[i * in_features_ + k] * weight_data[j * in_features_ + k];
            }
            output_data[i * out_features_ + j] = sum;
        }
    }
}

void Linear::backward(const Tensor& grad_output, Tensor& grad_input) {
    // 这里我们需要实现矩阵乘法的反向传播，计算 grad_input 和更新 grad_weight、grad_bias
    int batch_size = cache_input_->shape(0);
    const float* input_data = cache_input_->data();
    const float* weight_data = weight_->data();
    const float* grad_output_data = grad_output.data();
    float* grad_input_data = grad_input.data();
    float* grad_weight_data = weight_->grad();
    float* grad_bias_data = bias_->grad();

    // 计算 grad_input = grad_output * weight ,用于上一层的反向传播
    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        for (int k = 0; k < in_features_; ++k) {
            float sum = 0.0f;
            for (int j = 0; j < out_features_; ++j) {
                sum += grad_output_data[i * out_features_ + j] * weight_data[j * in_features_ + k];
            }
            grad_input_data[i * in_features_ + k] = sum;
        }
    }

    // 计算 grad_weight = grad_output^T * input ,用于更新权重，乘input是因为越大的输入对输出的影响越大，权重应该更新得越多
    #pragma omp parallel for
    for (int j = 0; j < out_features_; ++j) {
        for (int k = 0; k < in_features_; ++k) {
            float sum = 0.0f;
            for (int i = 0; i < batch_size; ++i) {
                sum += grad_output_data[i * out_features_ + j] * input_data[i * in_features_ + k];
            }
            grad_weight_data[j * in_features_ + k] += sum; // 累加梯度
        }
    }

    // 计算 grad_bias = sum(grad_output, axis=0) ,用于更新偏置
    #pragma omp parallel for
    for (int j = 0; j < out_features_; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            sum += grad_output_data[i * out_features_ + j];
        }
        grad_bias_data[j] += sum; // 累加梯度
    }
}