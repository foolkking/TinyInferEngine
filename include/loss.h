/*
 * @Author: fool
 * @Date: 2026-04-22 00:43:10
 * @LastEditors: fool
 * @LastEditTime: 2026-04-22 00:43:17
 * @FilePath: \TinyInferEngine\include\loss.h
 * @Description:  
 * @Note:  
 */
#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

class CrossEntropyLoss {
private:
    Tensor* cached_probs_ = nullptr; // 缓存 Softmax 算出的概率
    int* cached_labels_ = nullptr;   // 缓存真实的标签
    int batch_size_;
    int num_classes_;

public:
    CrossEntropyLoss() = default;
    ~CrossEntropyLoss();

    // 前向传播：计算 Loss 标量值
    // logits: 网络的最后一层输出
    // target_labels: 真实的类别索引数组 (例如 [2, 0, 1])
    float forward(const Tensor& logits, const int* target_labels);

    // 反向传播：返回初始的误差梯度 Tensor
    Tensor* backward();
};

#endif // LOSS_H