/*
 * @Author: fool
 * @Date: 2026-04-20 22:14:50
 * @LastEditors: fool
 * @LastEditTime: 2026-04-21 13:07:29
 * @FilePath: \TinyInferEngine\include\relu.h
 * @Description:  
 * @Note:  
 */
#ifndef RELU_H
#define RELU_H

#include "layer.h"

class ReLU : public Layer {
public:
    // 覆盖基类的 forward 方法
    void forward(const Tensor& input, Tensor& output) override;
    // 覆盖基类的 compute_output_shape 方法
    std::vector<int> compute_output_shape(const std::vector<int>& input_shape) const override {
        // ReLU 的输出形状和输入形状完全一样
        return input_shape;
    }
    void backward(const Tensor& grad_output, Tensor& grad_input) override ;
};

#endif // RELU_H