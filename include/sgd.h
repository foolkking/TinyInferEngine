/*
 * @Author: fool
 * @Date: 2026-04-21 16:22:33
 * @LastEditors: fool
 * @LastEditTime: 2026-04-21 16:33:12
 * @FilePath: \TinyInferEngine\include\sgd.h
 * @Description:  
 * @Note:  
 */
#ifndef SGD_H
#define SGD_H
#include "tensor.h"
#include <vector>
class SGD {
private:
    float lr_;
    std::vector<Tensor*> parameters_; // 需要更新的参数列表，通常是权重和偏置
public:
    SGD(const std::vector<Tensor*>& params, float lr) : parameters_(params), lr_(lr) {};
    ~SGD()=default;
    void zero_grad() {
        for (Tensor* param : parameters_) {
            if(param->requires_grad()) {
                param->zero_grad();
            }
        }   
    }
    void step();
};
#endif // SGD_H