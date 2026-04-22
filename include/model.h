/*
 * @Author: fool
 * @Date: 2026-04-17 19:55:27
 * @LastEditors: fool
 * @LastEditTime: 2026-04-22 13:58:01
 * @FilePath: \TinyInferEngine\include\model.h
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
    std::vector<Tensor*> intermediate_outputs_; // 存储前向传播过程中产生的中间输出，方便反向传播时使用

public:
    Sequential() = default;
    ~Sequential();

    // 往模型里按顺序添加算子
    void add(Layer* layer);
    std::vector<int> compute_output_shape(const std::vector<int>& input_shape) const;
    // 统筹前向传播：接收一个初始 input，自动流转所有 layer，返回最终的 output Tensor
    Tensor* forward(Tensor* input);
    void backward(Tensor* grad_output); // 反向传播接口：接收最终输出的梯度，自动流转所有 layer，计算每层的输入梯度
    void clear_intermediate_outputs(); // 清理前向传播过程中产生的中间输出，释放内存
};

#endif // MODEL_H