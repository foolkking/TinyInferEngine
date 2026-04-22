/*
 * @Author: fool
 * @Date: 2026-04-20 22:14:51
 * @LastEditors: fool
 * @LastEditTime: 2026-04-21 21:12:38
 * @FilePath: \TinyInferEngine\include\conv2d.h
 * @Description:  
 * @Note:  
 */
#ifndef CONV2D_H
#define CONV2D_H

#include "layer.h"

class Conv2D : public Layer {
private:
    int in_channels_;  //输入通道数
    int out_channels_; //有多少个卷积核 → 输出就有多少个通道，oc个3维卷积核
    int kernel_size_; // 假设是正方形核
    int stride_;
    int padding_;

    Tensor* weight_; // 形状: [out_channels, in_channels, kernel_size, kernel_size]  
    Tensor* bias_;   // 形状: [out_channels]
protected:
    void clearup() override; // 重写清理函数，释放权重和偏置
public:
    Conv2D(int in_ch, int out_ch, int k_size, int stride = 1, int padding = 0, bool requires_grad = false);
    ~Conv2D(); // 析构函数，释放权重和偏置
    Tensor* weight() { return weight_; }
    Tensor* bias() { return bias_; }
    
    void forward(const Tensor& input, Tensor& output) override;
    std::vector<int> compute_output_shape(const std::vector<int>& input_shape) const override;
    void backward(const Tensor& grad_output, Tensor& grad_input) override;
};

#endif