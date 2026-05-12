/*
 * @Author: fool
 * @Date: 2026-04-20 22:14:50
 * @LastEditors: fool
 * @LastEditTime: 2026-05-04 13:27:31
 * @FilePath: \TinyInferEngine\src\layer.cpp
 * @Description:  
 * @Note:  
 */

#include "layer.h"
#include<vector>
#include<limits>
#define _USE_MATH_DEFINES
#include<cmath>
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

Conv2D::Conv2D(int in_ch, int out_ch, int k_size, int stride, int padding, bool requires_grad) {
    in_channels_ = in_ch;
    out_channels_ = out_ch;
    kernel_size_ = k_size;
    stride_ = stride;
    padding_ = padding;
    std::vector<int> weight_shape = {out_channels_, in_channels_, kernel_size_, kernel_size_}; // 权重张量的形状
    weight_ = std::make_shared<Tensor>(weight_shape, requires_grad); // 四维张量
    std::vector<int> bias_shape = {out_channels_}; // 偏置向量的形状
    bias_ = std::make_shared<Tensor>(bias_shape, requires_grad); // 一维张量
}
Conv2D::~Conv2D() {
    clearup(); // 调用清理函数释放资源
}

void Conv2D::clearup() {
    Layer::clearup(); // 调用基类清理输入缓存
    if (weight_ != nullptr) {
        weight_.reset();
        weight_ = nullptr;
    }
    if (bias_ != nullptr) {
        bias_.reset();
        bias_ = nullptr;
    }
}

TensorPtr Conv2D::forward(TensorPtr input) {
    // 这里我们需要实现卷积操作：output = conv2d(input, weight) + bias
    // 注意 input 的形状是 [batch_size, in_channels, in_height, in_width]
    // weight 的形状是 [out_channels, in_channels, kernel_size, kernel_size]
    // output 的形状是 [batch_size, out_channels, out_height, out_width]    
    if (input->ndims() != 4) {
        return {};
    }
    int batch_size = input->shape(0);
    int in_height = input->shape(2);
    int in_width = input->shape(3);

    // 计算输出的高度和宽度
    int out_height = (in_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    int out_width = (in_width + 2 * padding_ - kernel_size_) / stride_ + 1;
    std::vector<int> output_shape = {batch_size, out_channels_, out_height, out_width};

    bool out_req_grad = input->requires_grad() || weight_->requires_grad() || bias_->requires_grad();
    auto output = std::make_shared<Tensor>(output_shape, out_req_grad); 

    float* output_data = output->data();
    for(int i = 0; i < output->size(); ++i) {
        output_data[i] = 0.0f; // 先清零
    }

    const float* input_data = input->data();
    const float* weight_data = weight_->data();
    const float* bias_data = bias_->data();
    //#pragma omp parallel for
    for (int n = 0; n < batch_size; ++n) {
        for (int oc = 0; oc < out_channels_; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float sum = bias_data[oc]; // 从偏置开始累加
                    for (int ic = 0; ic < in_channels_; ++ic) {   
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                int ih = oh * stride_ - padding_ + kh;
                                int iw = ow * stride_ - padding_ + kw;
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    sum += input_data[n * in_channels_ * in_height * in_width + 
                                                        ic * in_height * in_width + 
                                                        ih * in_width + 
                                                        iw] * 
                                           weight_data[oc * in_channels_ * kernel_size_ * kernel_size_ + 
                                                       ic * kernel_size_ * kernel_size_ + 
                                                       kh * kernel_size_ + 
                                                       kw];
                                }
                            }
                        }
                    }
                    output_data[n * out_channels_ * out_height * out_width + 
                                oc * out_height * out_width + 
                                oh * out_width + 
                                ow] = sum;
                }
            }
        }
    }
    
    if(out_req_grad){
        Tensor* output_ptr = output.get(); // 捕获输出张量的智能指针
        std::vector<NamedParameter> params = this->parameters(); // 获取权重和偏置作为命名参数
        std::function<void()> backward_fn = [input, params, output_ptr,padding=padding_,stride=stride_,kernel_size=kernel_size_]() {
            float* grad_input_data = input->grad();
            //升级自动图后，清零工作必须由优化器（Optimizer）在每一轮迭代开始前统一调用 zero_grad() 完成。因为可能有多个连接这个张量
            // for(int i = 0; i < input->size(); ++i) {
            //     grad_input_data[i] = 0.0f; // 先清零输入梯度
            // }
            const float* grad_output_data = output_ptr->grad();
            const float* input_data = input->data();

            // 获取权重、权重梯度和偏置梯度指针
            const float* weight_data = params[0].tensor->data();
            float* grad_weight_data = params[0].tensor->grad();
            float* grad_bias_data = params[1].tensor->grad();

            // 获取各维度大小
            int batch_size = input->shape(0);
            int in_channels = input->shape(1);
            int out_channels = output_ptr->shape(1);
            int in_height = input->shape(2); 
            int in_width = input->shape(3);
            int out_height = output_ptr->shape(2);
            int out_width = output_ptr->shape(3);

            // 计算梯度
            //不能用#pragma omp parallel for，保证单线程计算的绝对正确性，因为权重梯度和偏置梯度的更新存在数据竞争，后续可以考虑加锁或者使用线程局部变量累加后再合并
            for (int n = 0; n < batch_size; ++n) {
                for (int oc = 0; oc < out_channels; ++oc) {
                    for (int oh = 0; oh < out_height; ++oh) {
                        for (int ow = 0; ow < out_width; ++ow) {
                            float grad_out_val = grad_output_data[n * out_channels * out_height * out_width + 
                                                                oc * out_height * out_width + 
                                                                oh * out_width + 
                                                                ow];
                            grad_bias_data[oc] += grad_out_val; // 累加偏置梯度
                            for (int ic = 0; ic < in_channels; ++ic) {   
                                for (int kh = 0; kh < kernel_size; ++kh) {
                                    for (int kw = 0; kw < kernel_size; ++kw) {
                                        int ih = oh * stride - padding + kh;
                                        int iw = ow * stride - padding + kw;
                                        if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                            // 累加权重梯度
                                            grad_weight_data[oc * in_channels * kernel_size * kernel_size + 
                                                            ic * kernel_size * kernel_size + 
                                                            kh * kernel_size + 
                                                            kw] += input_data[n * in_channels * in_height * in_width + 
                                                                            ic * in_height * in_width + 
                                                                            ih * in_width + 
                                                                            iw] * grad_out_val;
                                            // 累加输入梯度
                                            if(input->requires_grad()) {
                                                grad_input_data[n * in_channels * in_height * in_width + 
                                                                ic * in_height * in_width + 
                                                                ih * in_width + 
                                                                iw] += weight_data[oc * in_channels * kernel_size * kernel_size + 
                                                                                ic * kernel_size * kernel_size + 
                                                                                kh * kernel_size + 
                                                                            kw] * grad_out_val;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
        
        
        output->set_auto_grad(backward_fn, {input, params[0].tensor, params[1].tensor}); // 绑定计算图
        
    }
    return  output;
}

std::vector<NamedParameter> Conv2D::parameters() {
    return {{"weight", weight_}, {"bias", bias_}};
}


MaxPool2D::MaxPool2D(int k_size, int stride, int padding) {
    kernel_size_ = k_size;
    stride_ = stride;
    padding_ = padding;
}
MaxPool2D::~MaxPool2D() {
    if (max_indices_ != nullptr) {
        delete[] max_indices_;
        max_indices_ = nullptr;
    }
}
/**
 * @brief 最大池化操作
 * @param input 输入张量，形状为 [batch_size, in_channels, in_height, in_width]
 * @param output 输出张量，形状为 [batch_size, in_channels, out_height, out_width]
 */
TensorPtr MaxPool2D::forward(TensorPtr input) {
    
    if (input->ndims() != 4) {
        std::cerr << "Error: Input shape must be [batch_size, in_channels, in_height, in_width]!" << std::endl;
        exit(EXIT_FAILURE);
    }
    int batch_size = input->shape(0);
    int in_channels = input->shape(1);
    int in_height = input->shape(2);
    int in_width = input->shape(3);
    int out_height = (in_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    int out_width = (in_width + 2 * padding_ - kernel_size_) / stride_ + 1;
    std::vector<int> output_shape = {batch_size, in_channels, out_height, out_width};
    TensorPtr output = std::make_shared<Tensor>(output_shape, input->requires_grad());
    const float* input_data = input->data(); 
    float* output_data = output->data();
    
    
    if(max_indices_ != nullptr) {
        delete[] max_indices_;
    }
    max_indices_ = new int[output->size()]; // 用于记录每个输出位置对应的输入索引
    #pragma omp parallel for
    for (int n = 0; n < batch_size; ++n) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity(); // 初始化为负无穷
                    int max_index = -1; // 初始化为无效索引
                    for (int kh = 0; kh < kernel_size_; ++kh) {
                        for (int kw = 0; kw < kernel_size_; ++kw) {
                            int ih = oh * stride_ - padding_ + kh;
                            int iw = ow * stride_ - padding_ + kw;
                            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                int current_index = n * in_channels * in_height * in_width + 
                                                ic * in_height * in_width + 
                                                ih * in_width + 
                                                iw;
                                float val = input_data[current_index];
                                if (val > max_val) {
                                    max_val = val;
                                    max_index = current_index; // 记录最大值对应的输入索引
                                }
                            }
                        }
                    }
                    int output_index = n * in_channels * out_height * out_width + 
                                        ic * out_height * out_width + 
                                        oh * out_width + 
                                        ow;
                    max_indices_[output_index] = max_index; // 存储最大值索引
                    output_data[output_index] = max_val;
                }
            }
        }
    }
    if(input->requires_grad()) {
       Tensor* output_ptr = output.get(); // 获取输出张量的裸指针
       std::function<void()> backward_fn = [this, input, output_ptr,kernel_size_=kernel_size_,padding_=padding_,stride_=stride_,max_indices_=max_indices_]() {
            const float* grad_output_data = output_ptr->grad();     
            float* grad_input_data = input->grad();
            
            // 遍历每一个传回来的误差梯度，把它直接塞回当年那个赢家的口袋里
            // 这里不要加 #pragma omp parallel for，因为如果有窗口重叠，多个输出可能会把梯度累加给同一个输入，导致线程冲突
            for (int i = 0; i < output_ptr->size(); ++i) {
                int winner_input_index = max_indices_[i];  //这个索引是前向传播时记录的最大值位置，位置是相对Tensor的一维索引
                if (winner_input_index >= 0) { // 安全检查
                    grad_input_data[winner_input_index] += grad_output_data[i]; 
                }
            }

            float* input_grad_data = input->grad(); // 获取输入张量的梯度指针
            for (int i = 0; i < input->size(); ++i) {
                input_grad_data[i] += grad_input_data[i]; // 累加到输入张量的梯度中
            }
        };
        output->set_auto_grad(backward_fn, {input}); // 将反向传播函数和前驱节点绑定到输出张量上
    }
    return output;
}

TensorPtr Flatten::forward(TensorPtr input) {
    int features = 1;
    for (size_t i = 1; i < input->ndims(); ++i) {
        features *= input->shape(i);
    }
    std::vector<int> output_shape = {input->shape(0), features};
    TensorPtr output = std::make_shared<Tensor>(output_shape, input->requires_grad());
    // 物理内存上的数据完全一样，直接拷贝即可
    const float* in_data = input->data();
    float* out_data = output->data();
    for (int i = 0; i < input->size(); ++i) {
        out_data[i] = in_data[i];
    }
    Tensor* out_ptr = output.get(); // 捕获输入张量的智能指针，确保它在闭包中有效
    std::function<void()> backward_fn = [input, out_ptr]() {
        // 展平层没有参数，所以反向传播时直接将输出的梯度传递回输入即可
        const float* grad_out_ptr = out_ptr->grad();
        float* grad_in_ptr = input->grad();
        for (int i = 0; i < out_ptr->size(); ++i) {
            grad_in_ptr[i] += grad_out_ptr[i]; // 累加梯度
        }
    };

    output->set_auto_grad(backward_fn, {input});
    return output;
}

TensorPtr ReLU::forward(TensorPtr input){
    TensorPtr output = std::make_shared<Tensor> (input->shape(),input->requires_grad());
    // 直接进行 ReLU 操作
    // 假设输入输出形状已经匹配，直接进行 ReLU 操作
    int size = input->size();
    const float* input_data = input->data();
    float* output_data = output->data();
    
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        output_data[i] = input_data[i] > 0 ? input_data[i] : 0;
    }
    if(input->requires_grad()){
        Tensor* output_ptr = output.get();
        std::function<void()> backward_fn =[output_ptr,input](){
            const float* grad_output_data = output_ptr->grad();
            float* input_data = input->data();
            float* grad_input_data = input->grad();

            // ReLU 的反向传播：如果输入 > 0，梯度不变；否则梯度为 0
            for (int i = 0; i < input->size(); ++i) {
                grad_input_data[i] = (input_data[i] > 0) ? grad_output_data[i] : 0.0f;
            }
        };
        output->set_auto_grad(backward_fn,{input});
    }
    return output;
}

TensorPtr SiLU::forward(TensorPtr input){// SiLU(x) = x * sigmoid(x)
    TensorPtr output = std::make_shared<Tensor> (input->shape(),input->requires_grad());
    const float* input_data = input->data();
    float* output_data = output->data();
    int num_elements = input->size();
    for (int i = 0; i < num_elements; ++i) {
        float x = input_data[i];
        float sigmoid_x = 1.0f / (1.0f + std::exp (-x)); // sigmoid(x)
        output_data[i] = x * sigmoid_x; // SiLU(x)
    }
    if(input->requires_grad()){
        Tensor* output_ptr = output.get();
        std::function<void()> backward_fn = [output_ptr,input](){
            const float* input_data = input->data();
            const float* grad_output_data = output_ptr->grad();
            float* grad_input_data = input->grad();
            int num_elements = input->size();
            for (int i = 0; i < num_elements; ++i) {
                float x = input_data[i];
                float sigmoid_x = 1.0f / (1.0f + std::exp(-x)); // sigmoid(x)
                float grad_sigmoid = sigmoid_x * (1 - sigmoid_x); // sigmoid'(x)
                grad_input_data[i] = grad_output_data[i] * (sigmoid_x + x * grad_sigmoid); // 链式法则
            }
        };
        output->set_auto_grad(backward_fn,{input});
    }
    return output;
}


/// @param num_features 输出特征数，整个网络每一层输出的“通道数”在一开始就必须是完全确定的
/// @param eps 防止除零，默认1e-5
/// @param momentum  动量，越大越偏向之前batch结果，默认0.1
BatchNorm2D::BatchNorm2D(int num_features, float eps, float momentum)
    : num_features_(num_features), eps_(eps), momentum_(momentum) {
    std::vector<int> w_shape = {num_features};
    weight_ = std::make_shared<Tensor>(w_shape, true);
    bias_ = std::make_shared<Tensor>(w_shape, true);
    weight_->ones();
    bias_->zeros();
    running_mean_.assign(num_features, 0.0f);
    running_var_.assign(num_features, 1.0f);
}

std::vector<NamedParameter> BatchNorm2D::parameters(){
    return { {"weight",weight_ }, {"bias",bias_}};
}

TensorPtr BatchNorm2D::forward(TensorPtr input) {
    TensorPtr output = std::make_shared<Tensor>(input->shape(),input->requires_grad());
    int batch_size = input->shape(0);
    int num_features = input->shape(1);
    int height = input->shape(2);
    int width = input->shape(3);
    int spacial_size = height * width;
    int N = batch_size * spacial_size; // 每个通道的元素总数

    float* input_data = input->data();
    float* output_data = output->data();

    std::vector<float> batch_mean(num_features, 0.0f);
    std::vector<float> batch_var(num_features, 0.0f);
    std::vector<float> normalized_input(input->size(), 0.0f); // 用于存储标准化后的输入值，反向传播时需要用到,空间换时间
    std::vector<float> inv_var_cache(num_features, 0.0f); // 用于存储每个通道的方差倒数乘γ，反向传播时需要用到，空间换时间
    if(training_){
        for(int c=0; c<num_features;++c){
            float sum = 0.0f;
            for(int b=0;b<batch_size;++b){
                for(int hw=0;hw<spacial_size;++hw){
                        int index = b*num_features*spacial_size+c* spacial_size+hw;
                        sum += input_data[index];
                }
            }
            batch_mean[c] = sum/(batch_size*spacial_size);
            running_mean_[c] = running_mean_[c]*(momentum_)+sum/( batch_size*spacial_size)*(1 -momentum_);
            float var_sum = 0.0;
            for(int b=0;b<batch_size;++b ){
                for(int hw=0;hw<spacial_size;++hw){
                    int index = b*num_features *spacial_size+c* spacial_size+hw;
                    float diff = input_data[index]-running_mean_[c];
                    var_sum += diff*diff;
                }
            }
            batch_var[c] = var_sum/(batch_size*spacial_size);
            running_var_[c] = running_var_[c]*(momentum_)+var_sum/(batch_size*spacial_size)*(1 -momentum_);
            inv_var_cache[c] = 1.0f/sqrt(batch_var[c]+eps_)*weight_->data(c); // 预先计算方差倒数乘γ，反向传播时直接用
        }
    }
    for(int c=0; c<num_features;++c){
        float mean =training_? batch_mean[c]: running_mean_[c];
        float var = training_? batch_var[c]: running_var_[c];
        var = sqrt(var+eps_);

        float weight = weight_->data(c);
        float bias = bias_->data(c);
        for(int b=0;b<batch_size;++b){
            for(int hw=0;hw<spacial_size;++hw){
                int index = b*num_features*spacial_size+c*spacial_size+hw;
                normalized_input[index] = (input_data[index]-mean)/var;
                output_data[index] = normalized_input[index] * weight + bias;
            }
        }
    }
    if(input->requires_grad()){
        Tensor* output_ptr = output.get();
        std::function<void()> backward_fn = [input,output_ptr,batch_size,num_features,spacial_size,normalized_input,weight=weight_.get(),bias=bias_.get(),inv_var_cache,N]() {
            const float* grad_output_data = output_ptr->grad();
            float* grad_input_data = input->grad();
            float* grad_weight = weight->requires_grad() ? weight->grad() : nullptr;
            float* grad_bias   = bias->requires_grad() ? bias->grad() : nullptr;
            for(int c=0; c<num_features;++c){
                float col_grad_bias = 0.0f;
                float col_grad_weight = 0.0f;
                for(int b=0;b<batch_size;++b){
                    for(int hw=0;hw<spacial_size;++hw){
                        int index = b*num_features*spacial_size+c*spacial_size+hw;
                        col_grad_bias += grad_output_data[index];
                        col_grad_weight += grad_output_data[index]*normalized_input[index];
                    }
                }
                if(bias->requires_grad()){ grad_bias[c] += col_grad_bias;} // 累加偏置的梯度
                if(weight->requires_grad()){grad_weight[c] += col_grad_weight;}// 累加权重的梯度
               
                // 计算输入的梯度,与 momentum_无关，momentum_是验证才用的
                // xi​→μ→σ2→x^i​
                // ​∂L/​∂x_j = (​∂L/(∂x^_j)) * (​∂x^_j/​∂x_j)  ​= (​∂L/​∂y_j) * ​γ * (​∂x^_j/​∂x_j) 
                // 最重要求：(​∂x^_j/​∂x_j)  ，由归一化表达式.链式求导法则。
                //
                // 归一化表达式：x^i​=(xi​−μ)/σ
                // ​∂x^_j/​∂x_j = 1/σ * (1 - 1/N) - 1/N * (xi​−μ)/σ^3 * (xi​−μ) = 1/σ * (1 - 1/N - (xi​−μ)^2/(N*σ^2))
                // 其中 N 是每个通道的元素总数，即 batch_size * spacial_size
                if(input->requires_grad()) {
                    float gamma_over_std = inv_var_cache[c];
                    float inv_N = 1.0f / N;
                    for(int b=0;b<batch_size;++b){
                        for(int hw=0;hw<spacial_size;++hw){
                            int index = b*num_features*spacial_size+c*spacial_size+hw;
                            float dout = grad_output_data[index];
                            float x_hat = normalized_input[index];
                            // dx = (gamma / var) * (dout - mean(dout) - x_hat * mean(dout * x_hat))
                            // 极致优雅：复用算好的 col_grad_bias 和 col_grad_weight
                            grad_input_data[index] += gamma_over_std*(dout - col_grad_bias*inv_N - x_hat*col_grad_weight/N);
                        }
                    }
                }
            }
        };
        output->set_auto_grad(backward_fn,{input,weight_,bias_});
    }
    return output;     
}


