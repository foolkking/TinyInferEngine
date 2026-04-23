/*
 * @Author: fool
 * @Date: 2026-04-23 12:32:38
 * @LastEditors: fool
 * @LastEditTime: 2026-04-23 13:19:10
 * @FilePath: \TinyInferEngine\include\silu.h
 * @Description:  
 * @Note:  
 */
#ifndef SILU_H
#define SILU_H
#include "layer.h"    
class SiLU : public Layer {
public:
    void forward(const Tensor& input, Tensor& output) override ;
    std::vector<int> compute_output_shape(const std::vector<int>& input_shape)const override {
        return input_shape; // SiLU 的输出形状与输入相同   
    }
    void backward(const Tensor& grad_output, Tensor& grad_input) override ; 

};
#endif // SILU_H