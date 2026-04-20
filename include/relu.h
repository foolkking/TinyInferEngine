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
};

#endif // RELU_H