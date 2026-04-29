/*
 * @Author: fool
 * @Date: 2026-04-20 22:14:50
 * @LastEditors: fool
 * @LastEditTime: 2026-04-29 11:01:25
 * @FilePath: \TinyInferEngine\src\linear.cpp
 * @Description:  
 * @Note:  
 */

#include "linear.h"
#include<vector>
Linear::Linear(int in_features, int out_features, bool requires_grad) {
    in_features_ = in_features;
    out_features_ = out_features;
    
    // 初始化权重和偏置张量
    std::vector<int> weight_shape = {out_features_, in_features_}; // 权重矩阵的形状
    weight_ = std::make_shared<Tensor>(weight_shape, requires_grad); // 二维张量
    
    std::vector<int> bias_shape = {out_features_}; // 偏置向量的形状
    bias_ = std::make_shared<Tensor>(bias_shape, requires_grad); // 一维张量
}


std::vector<NamedParameter> Linear::parameters() {
    return {NamedParameter{"weight", weight_}, NamedParameter{"bias", bias_}};
}

TensorPtr Linear::forward(TensorPtr input) {
    //前向传播
    // 假设输入形状是 [batch_size, in_features_]
    // 输出形状应该是 [batch_size, out_features_]    
    std::vector<int> output_shape = {input->shape(0), out_features_};
    //Autograd 图引擎会从最后一层一直遍历到不用求导的叶子节点，所以只要输入或权重需要梯度，输出就需要梯度。
    bool out_req_grad = input->requires_grad() || weight_->requires_grad(); 
    TensorPtr output = std::make_shared<Tensor>(output_shape, out_req_grad);

    int batch_size = input->shape(0);
    // 这里我们需要实现矩阵乘法：output = input * weight^T + bias
    // 注意 weight 的形状是 [out_features_, in_features_], 需要转置成 [in_features_, out_features_]
    const float* input_data = input->data();
    const float* weight_data = weight_->data();
    const float* bias_data = bias_->data();
    float* output_data = output->data();
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
    if(out_req_grad) {
        auto parameters = this->parameters(); // 获取权重和偏置
        Tensor* output_ptr = output.get(); // 不能直接传入output，output含有backward_fn_，而backward_fn_又需要访问output，这样的循环引用会导致引用计数永远不可能为0，内存泄露
        std::function<void()> backward_fn = [input, parameters, output_ptr,in_features_=in_features_,out_features_=out_features_]() {
            // 反向传播函数的实现
            // 这里我们需要计算输入的梯度和权重、偏置的梯度
            int batch_size = input->shape(0);
            const float* grad_output = output_ptr->grad(); // 输出的梯度
            
            const float* weight_data = parameters[0].tensor->data(); // 权重数据
            float* grad_weight = parameters[0].tensor->grad(); // 权重的梯度
            float* grad_bias = parameters[1].tensor->grad(); // 偏置的梯度
            // 计算输入的梯度：grad_input = grad_output * weight
            if(input->requires_grad()) {  //特别重要！只有输入需要梯度时才计算输入的梯度，否则就白算了
                float* grad_input = input->grad(); // 输入的梯度
                #pragma omp parallel for
                for (int i = 0; i < batch_size; ++i) {
                    for (int k = 0; k < in_features_; ++k) {
                        float sum = 0.0f;
                        for (int j = 0; j < out_features_; ++j) {
                            sum += grad_output[i * out_features_ + j] * weight_data[j * in_features_ + k];
                        }
                        grad_input[i * in_features_ + k] += sum; // 累加输入的梯度
                    }
                }
            }
            // 计算权重的梯度：grad_weight = grad_output^T * input
            #pragma omp parallel for
            for (int j = 0; j < out_features_; ++j) {
                for (int k = 0; k < in_features_; ++k) {
                    float sum = 0.0f;
                    for (int i = 0; i < batch_size; ++i) {
                        sum += grad_output[i * out_features_ + j] * input->data()[i * in_features_ + k];
                    }
                    grad_weight[j * in_features_ + k] += sum; // 累加权重的梯度
                }
            }
            // 计算偏置的梯度：grad_bias = grad_output^T * 1
            #pragma omp parallel for
            for(int j = 0; j < out_features_; ++j) {
                float sum = 0.0f;
                for (int i = 0; i < batch_size; ++i) {
                    sum += grad_output[i * out_features_ + j];
                }
                grad_bias[j] += sum; // 累加偏置的梯度
            }
        };
        // 【架构细节】：在拓扑图中，权重等需要更新的参数其实也是 output 的“父母”(前驱节点)
        // 只有把它们加进 prev 列表，拓扑排序引擎才能遍历到它们
        output->set_auto_grad(backward_fn, {input,parameters[0].tensor,parameters[1].tensor});
    }
    
    return output;
}

