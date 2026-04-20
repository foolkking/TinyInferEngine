/*
 * @Author: fool
 * @Date: 2026-04-20 20:35:53
 * @LastEditors: fool
 * @LastEditTime: 2026-04-20 21:01:56
 * @FilePath: \TinyInferEngine\src\flatten.cpp
 * @Description:  
 * @Note:  
 */
 #include "flatten.h"

std::vector<int> Flatten::compute_output_shape(const std::vector<int>& input_shape) const {
    // 将 [Batch, C, H, W] 展平为 [Batch, C * H * W]
    int batch_size = input_shape[0];
    int features = 1;
    for (size_t i = 1; i < input_shape.size(); ++i) {
        features *= input_shape[i];
    }
    return {batch_size, features};
}

void Flatten::forward(const Tensor& input, Tensor& output) {
    // 物理内存上的数据完全一样，直接拷贝即可
    const float* in_ptr = input.data();
    float* out_ptr = output.data();
    for (int i = 0; i < input.size(); ++i) {
        out_ptr[i] = in_ptr[i];
    }
}