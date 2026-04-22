/*
 * @Author: fool
 * @Date: 2026-04-22 12:10:01
 * @LastEditors: fool
 * @LastEditTime: 2026-04-22 15:21:52
 * @FilePath: \TinyInferEngine\src\loss.cpp
 * @Description:  
 * @Note:  
 */

#include "loss.h"
#include<limits>
#include<cmath>
#include<vector>
CrossEntropyLoss::~CrossEntropyLoss() {
    if (cached_probs_ != nullptr) {
        delete cached_probs_;
        cached_probs_ = nullptr;
    }
    if (cached_labels_ != nullptr) {
        delete[] cached_labels_;
        cached_labels_ = nullptr;
    }
}
float CrossEntropyLoss::forward(const Tensor& logits, const int* target_labels) {
    batch_size_ = logits.shape(0);
    num_classes_ = logits.shape(1);
    if (cached_probs_ != nullptr) {
        delete cached_probs_;
    }
    std::vector<int> new_shape;
    for(int i = 0; i < logits.ndims(); ++i) {
        new_shape.push_back(logits.shape(i));
    }
    cached_probs_ = new Tensor(new_shape.data(), logits.ndims()); // 缓存概率张量，形状与 logits 相同
    if (cached_labels_ != nullptr) {
        delete[] cached_labels_;
    }
    cached_labels_ = new int[batch_size_]; // 缓存标签数组
    for (int i = 0; i < batch_size_; ++i) {
        cached_labels_[i] = target_labels[i];
    }
    // 计算 Softmax 概率并缓存
    float* probs_data = cached_probs_->data();
    const float* logits_data = logits.data();
    for (int n = 0; n < batch_size_; ++n) {
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < num_classes_; ++c) { // 先找到当前样本的最大 logit，作为数值稳定性的基准
            float logit_val = logits_data[n * num_classes_ + c];
            if (logit_val > max_logit) {
                max_logit = logit_val;
            }
        }
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes_; ++c) {
            float exp_val = std::exp(logits_data[n * num_classes_ + c] - max_logit);
            sum_exp += exp_val;
            probs_data[n * num_classes_ + c] = exp_val; // 先存储 exp(logit)，后续除以 sum_exp
        }
        for (int c = 0; c < num_classes_; ++c) {
            probs_data[n * num_classes_ + c] /= sum_exp; // 最终得到概率
        }
    }
    // 计算交叉熵损失
    float loss = 0.0f;
    for (int n = 0; n < batch_size_; ++n) {
        int label = target_labels[n];  //真实标签是one-hot编码中的索引，对应类别概率直接是1
        float prob = probs_data[n * num_classes_ + label];
        loss -= std::log(prob + 1e-8f); // 加一个小常数避免 log(0)
    }
    loss /= batch_size_; // 平均损失
    return loss;
}

Tensor* CrossEntropyLoss::backward() {
    if (cached_probs_ == nullptr || cached_labels_ == nullptr) {
        std::cerr << "Error: No cached data for backward pass!" << std::endl;
        return nullptr;
    }
    std::vector<int> new_shape;
    for(int i = 0; i < cached_probs_->ndims(); ++i) {
        new_shape.push_back(cached_probs_->shape(i));
    }
    Tensor* grad_input = new Tensor(new_shape.data(), cached_probs_->ndims(), true); // 创建梯度张量，形状与 logits 相同
    float* grad_data = grad_input->data();
    const float* probs_data = cached_probs_->data();
    for (int n = 0; n < batch_size_; ++n) {
        int label = cached_labels_[n];
        for (int c = 0; c < num_classes_; ++c) {
            grad_data[n * num_classes_ + c] = probs_data[n * num_classes_ + c]; // 先复制概率值
        }
        grad_data[n * num_classes_ + label] -= 1.0f; // 对正确类别的概率减去1，得到初始梯度
    }
    for (int i = 0; i < grad_input->size(); ++i) {
        grad_data[i] /= batch_size_; // 平均梯度
    }
    return grad_input;
}