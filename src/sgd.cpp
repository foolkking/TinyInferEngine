/*
 * @Author: fool
 * @Date: 2026-04-21 16:33:20
 * @LastEditors: fool
 * @LastEditTime: 2026-04-21 16:34:32
 * @FilePath: \TinyInferEngine\src\sgd.cpp
 * @Description:  
 * @Note:  
 */
#include "sgd.h"
void SGD::step() {
    for (Tensor* param : parameters_) {
        if(!param->requires_grad()|| param->grad() == nullptr) {
            continue; // 如果这个参数不需要梯度，就跳过
        }
        float* data = param->data();
        float* grad = param->grad();
        #pragma omp parallel for
        for (int i = 0; i < param->size(); ++i) {
            data[i] -= lr_ * grad[i]; // 更新参数：param = param - lr * grad
        }
    }
}

