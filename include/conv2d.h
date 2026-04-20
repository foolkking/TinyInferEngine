#ifndef CONV2D_H
#define CONV2D_H

#include "layer.h"

class Conv2D : public Layer {
private:
    int in_channels_;
    int out_channels_;
    int kernel_size_; // 假设是正方形核
    int stride_;
    int padding_;

    Tensor* weight_; // 形状: [out_channels, in_channels, kernel_size, kernel_size]
    Tensor* bias_;   // 形状: [out_channels]

public:
    Conv2D(int in_ch, int out_ch, int k_size, int stride = 1, int padding = 0);
    ~Conv2D();

    Tensor* weight() { return weight_; }
    Tensor* bias() { return bias_; }

    void forward(const Tensor& input, Tensor& output) override;
    std::vector<int> compute_output_shape(const std::vector<int>& input_shape) const override;
};

#endif