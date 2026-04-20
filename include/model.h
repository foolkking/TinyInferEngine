/*
 * @Author: fool
 * @Date: 2026-04-17 19:55:27
 * @LastEditors: fool
 * @LastEditTime: 2026-04-18 15:06:44
 * @FilePath: \TinyInferEngine\include\moudle.h
 * @Description:  
 * @Note:  
 */
#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include "layer.h"

class Sequential {
private:
    std::vector<Layer*> layers_; // 存储“算子序列”

public:
    Sequential() = default;
    ~Sequential();

    // 往模型里按顺序添加算子
    void add(Layer* layer);
    std::vector<int> compute_output_shape(const std::vector<int>& input_shape) const;
    // 统筹前向传播：接收一个初始 input，自动流转所有 layer，返回最终的 output Tensor
    Tensor* forward(Tensor* input);
};

#endif // MODEL_H