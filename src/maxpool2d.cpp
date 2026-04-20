/*
 * @Author: fool
 * @Date: 2026-04-20 16:56:34
 * @LastEditors: fool
 * @LastEditTime: 2026-04-20 20:28:12
 * @FilePath: \TinyInferEngine\src\maxpool2d.cpp
 * @Description:  
 * @Note:  
 */
/*
 * @Author: fool
 * @Date: 2026-04-20 16:56:34
 * @LastEditors: fool
 * @LastEditTime: 2026-04-20 17:03:29
 * @FilePath: \TinyInferEngine\src\maxpool2d.cpp
 * @Description:  
 * @Note:  
 */
#include "maxpool2d.h"
#include <limits>
MaxPool2D::MaxPool2D(int k_size, int stride, int padding) {
    kernel_size_ = k_size;
    stride_ = stride;
    padding_ = padding;
}
/**
 * @brief 最大池化操作
 * @param input 输入张量，形状为 [batch_size, in_channels, in_height, in_width]
 * @param output 输出张量，形状为 [batch_size, in_channels, out_height, out_width]
 */
void MaxPool2D::forward(const Tensor& input, Tensor& output) {
    const float* input_data = input.data(); 
    float* output_data = output.data();
    int batch_size = input.shape(0);
    int in_channels = input.shape(1);
    int in_height = input.shape(2);
    int in_width = input.shape(3);
    int out_height = output.shape(2);
    int out_width = output.shape(3);
    #pragma omp parallel for
    for (int n = 0; n < batch_size; ++n) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity(); // 初始化为负无穷
                    for (int kh = 0; kh < kernel_size_; ++kh) {
                        for (int kw = 0; kw < kernel_size_; ++kw) {
                            int ih = oh * stride_ - padding_ + kh;
                            int iw = ow * stride_ - padding_ + kw;
                            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                float val = input_data[n * in_channels * in_height * in_width + 
                                                        ic * in_height * in_width + 
                                                        ih * in_width + 
                                                        iw];
                                if (val > max_val) {
                                    max_val = val;
                                }
                            }
                        }
                    }
                    output_data[n * in_channels * out_height * out_width + 
                                ic * out_height * out_width + 
                                oh * out_width + 
                                ow] = max_val;
                }
            }
        }
    }
}

std::vector<int> MaxPool2D::compute_output_shape(const std::vector<int>& input_shape) const {
    if (input_shape.size() != 4) {
        std::cerr << "Error: Input shape must be [batch_size, in_channels, in_height, in_width]!" << std::endl;
        return {};
    }
    int batch_size = input_shape[0];
    int in_channels = input_shape[1];
    int in_height = input_shape[2];
    int in_width = input_shape[3];
    
    int out_height = (in_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    int out_width = (in_width + 2 * padding_ - kernel_size_) / stride_ + 1;
    
    return {batch_size, in_channels, out_height, out_width};
}