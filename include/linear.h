/*
 * @Author: fool
 * @Date: 2026-04-17 16:34:52
 * @LastEditors: fool
 * @LastEditTime: 2026-04-22 15:05:49
 * @FilePath: \TinyInferEngine\include\linear.h
 * @Description:  
 * @Note:  
 */
#ifndef LINEAR_H
#define LINEAR_H

#include "layer.h"

class Linear : public Layer {
private:
    int in_features_;
    int out_features_;
    
    Tensor* weight_; // 内部的权重张量，形状 [out_features, in_features]
    Tensor* bias_;   // 内部的偏置张量，形状 [out_features]

public:
    // 构造函数：告诉这个层输入维度和输出维度是多少
    Linear(int in_features, int out_features,bool requires_grad = false);
    
    // 析构函数：释放 weight_ 和 bias_
    ~Linear();
    // 暴露内部张量的指针，方便外部初始化权重 (比如用随机数或加载模型文件)
    Tensor* weight() { return weight_; }
    Tensor* bias() { return bias_; }

    // 矩阵乘法前向传播
    void forward(const Tensor& input, Tensor& output) override;
    // 形状推导：输入 [batch_size, in_features] 输出 [batch_size, out_features]
    std::vector<int> compute_output_shape(const std::vector<int>& input_shape) const override {
        if (input_shape.size() != 2 || input_shape[1] != in_features_) {
            std::cerr << "Error: Input shape must be [batch_size, " << in_features_ << "]!" << std::endl;
            return {};
        }
        return {input_shape[0], out_features_};
    }
    void Linear::backward(const Tensor& grad_output, Tensor& grad_input) override ;
};

#endif // LINEAR_H