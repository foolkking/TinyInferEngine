#ifndef MAXPOOL2D_H
#define MAXPOOL2D_H

#include "layer.h"

class MaxPool2D : public Layer {
private:
    int kernel_size_;// 假设是正方形核,池化不改变通道数,所以通道数可以直接读取张量的第二维
    int stride_;
    int padding_;

public:
    MaxPool2D( int k_size, int stride = 1, int padding = 0);
    ~MaxPool2D()=default;

    void forward(const Tensor& input, Tensor& output) override;
    std::vector<int> compute_output_shape(const std::vector<int>& input_shape) const override;
};

#endif