/*
 * @Author: fool
 * @Date: 2026-04-20 20:35:36
 * @LastEditors: fool
 * @LastEditTime: 2026-04-22 13:11:33
 * @FilePath: \TinyInferEngine\include\flatten.h
 * @Description:  
 * @Note:  
 */
#ifndef FLATTEN_H
#define FLATTEN_H

#include "layer.h"

class Flatten : public Layer {
public:
    Flatten() = default;
    ~Flatten() = default;

    void forward(const Tensor& input, Tensor& output) override;
    std::vector<int> compute_output_shape(const std::vector<int>& input_shape) const override;
    void backward(const Tensor& grad_output, Tensor& grad_input) override {
        // 展平层没有参数，所以反向传播时直接将 grad_output 传递回去即可
        const float* grad_out_ptr = grad_output.data();
        float* grad_in_ptr = grad_input.data();
        for (int i = 0; i < grad_output.size(); ++i) {
            grad_in_ptr[i] = grad_out_ptr[i];
        }
    }
};
#endif