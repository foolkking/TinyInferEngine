/*
 * @Author: fool
 * @Date: 2026-04-20 20:35:36
 * @LastEditors: fool
 * @LastEditTime: 2026-04-20 20:35:47
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
};

#endif