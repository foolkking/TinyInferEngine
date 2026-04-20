/*
 * @Author: fool
 * @Date: 2026-04-17 15:46:44
 * @LastEditors: fool
 * @LastEditTime: 2026-04-17 20:22:34
 * @FilePath: \TinyInferEngine\include\layer.h
 * @Description:  
 * @Note:  
 */
#ifndef LAYER_H
#define LAYER_H
#include <vector>
#include "tensor.h"

class Layer {
public:
    virtual ~Layer() = default;
    // 前向传播接口：常量输入，可变输出
    virtual void forward(const Tensor& input, Tensor& output) = 0; 
    // 形状推导接口。输入上一层的形状，返回本层的输出形状
    virtual std::vector<int> compute_output_shape(const std::vector<int>& input_shape) const = 0;

};

#endif // LAYER_H