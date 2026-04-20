#include "relu.h"
#include "tensor.h"
void ReLU::forward(const Tensor& input, Tensor& output) {
    // 假设输入输出形状已经匹配，直接进行 ReLU 操作
    int size = input.size();
    const float* input_data = input.data();
    float* output_data = output.data();
    
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        output_data[i] = input_data[i] > 0 ? input_data[i] : 0;
    }
}


