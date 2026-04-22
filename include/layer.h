/*
 * @Author: fool
 * @Date: 2026-04-17 15:46:44
 * @LastEditors: fool
 * @LastEditTime: 2026-04-21 20:26:45
 * @FilePath: \TinyInferEngine\include\layer.h
 * @Description:  
 * @Note:  
 */
#ifndef LAYER_H
#define LAYER_H
#include <vector>
#include <utility>
#include "tensor.h"

class Layer {
protected:
    Tensor* cache_input_ = nullptr; // 前向传播时缓存输入，反向传播时使用
    virtual void clearup() { // 清理缓存
        if (cache_input_ != nullptr) {
            delete cache_input_;
            cache_input_ = nullptr;
        }
    }
public:
    virtual ~Layer() {
        Layer::clearup();//调用基类清理，子类可以重写
    }
    // 前向传播接口：常量输入，可变输出
    virtual void forward(const Tensor& input, Tensor& output) = 0; 
    // 形状推导接口。输入上一层的形状，返回本层的输出形状
    virtual std::vector<int> compute_output_shape(const std::vector<int>& input_shape) const = 0;
    virtual void backward(const Tensor& grad_output, Tensor& grad_input) = 0; // 反向传播接口：常量输出梯度，可变输入梯度
};

#endif // LAYER_H