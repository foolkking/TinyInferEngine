/*
 * @Author: fool
 * @Date: 2026-04-20 22:14:51
 * @LastEditors: fool
 * @LastEditTime: 2026-04-27 12:38:19
 * @FilePath: \TinyInferEngine\include\maxpool2d.h
 * @Description:  
 * @Note:  
 */
#ifndef MAXPOOL2D_H
#define MAXPOOL2D_H

#include "layer.h"

class MaxPool2D : public Layer {
private:
    int kernel_size_;// 假设是正方形核,池化不改变通道数,所以通道数可以直接读取张量的第二维
    int stride_;
    int padding_;
    int* max_indices_ = nullptr; // 用于反向传播时记录最大值的位置
public:
    MaxPool2D( int k_size, int stride = 1, int padding = 0);
    ~MaxPool2D()override;

    TensorPtr forward(TensorPtr input) override;
    std::vector<NamedParameter> parameters()  override {
        // MaxPool2D 没有可训练的权重，所以返回空向量
        return {};
    }
};

#endif