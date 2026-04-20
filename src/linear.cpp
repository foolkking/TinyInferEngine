
#include "linear.h"

Linear::Linear(int in_features, int out_features) {
    in_features_ = in_features;
    out_features_ = out_features;
    
    // 初始化权重和偏置张量
    int weight_shape[] = {out_features_, in_features_}; // 权重矩阵的形状
    weight_ = new Tensor(weight_shape, 2); // 二维张量
    
    int bias_shape[] = {out_features_}; // 偏置向量的形状
    bias_ = new Tensor(bias_shape, 1); // 一维张量
}

Linear::~Linear() {
    delete weight_;
    delete bias_;
}

void Linear::forward(const Tensor& input, Tensor& output) {
    // 假设输入形状是 [batch_size, in_features_]
    // 输出形状应该是 [batch_size, out_features_]
    
    int batch_size = input.shape(0);
    
    // 这里我们需要实现矩阵乘法：output = input * weight^T + bias
    // 注意 weight 的形状是 [out_features_, in_features_], 需要转置成 [in_features_, out_features_]
    
    const float* input_data = input.data();
    const float* weight_data = weight_->data();
    const float* bias_data = bias_->data();
    float* output_data = output.data();
    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < out_features_; ++j) {
            float sum = bias_data[j]; // 从偏置开始累加
            for (int k = 0; k < in_features_; ++k) {
                sum += input_data[i * in_features_ + k] * weight_data[j * in_features_ + k];
            }
            output_data[i * out_features_ + j] = sum;
        }
    }
}