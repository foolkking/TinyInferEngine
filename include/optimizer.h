/*
 * @Author: fool
 * @Date: 2026-04-28 18:28:15
 * @LastEditors: fool
 * @LastEditTime: 2026-04-28 20:18:06
 * @FilePath: \TinyInferEngine\include\optimizer.h
 * @Description:  
 * @Note:  
 */
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>

#include "tensor.h"
#include "layer.h"
// 之前定义的 ParamGroup 原封不动
struct ParamGroup {
    std::vector<NamedParameter> params;
    float lr;
    float weight_decay;
    ParamGroup(std::vector<NamedParameter> p, float l, float wd = 0.0f)
        : params(std::move(p)), lr(l), weight_decay(wd) {}
};

class Optimizer {
public:
    // 注意：为了让 Scheduler 能够修改它，通常开放 public 或者提供 getters/setters
    std::vector<ParamGroup> param_groups;

    Optimizer(const std::vector<ParamGroup>& groups) : param_groups(groups) {}
    virtual ~Optimizer() = default;

    // 所有优化器必须实现的通用接口
    virtual void zero_grad() {
        for (auto& group : param_groups) {
            for (auto& param : group.params) {
                if (param.tensor->requires_grad()) param.tensor->zero_grad();
            }
        }
    }

    // 核心虚函数：每个子类自己决定怎么更新！
    virtual void step() = 0; 
};

class SGD: public Optimizer {
public:
    // 构造函数1：接收多个参数组
    // 使用Scheduler可以修改学习率
    SGD(const std::vector<ParamGroup>& groups) : Optimizer(groups) {}

    // 构造函数2：兼容以前的"一刀切"用法
    // 设置统一的学习率和weight_decay策略进行更新
    SGD(const std::vector<NamedParameter>& params, float lr, float weight_decay = 0.0f)
        : Optimizer({ParamGroup(params, lr, weight_decay)}) {}
    ~SGD() = default;

    void step() override;
};

class AdamW : public Optimizer {
private:
    float beta1_;
    float beta2_;
    float eps_;
    int step_t_; // 记录全局更新次数，用于偏差校正
    // 【核心设计】：状态字典。使用 Tensor 的内存地址作为 Key
    std::unordered_map<Tensor*, std::vector<float>> exp_avg_;    // 一阶矩 m
    std::unordered_map<Tensor*, std::vector<float>> exp_avg_sq_; // 二阶矩 v
public:
    AdamW(const std::vector<ParamGroup>& groups, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
        : Optimizer(groups), beta1_(beta1), beta2_(beta2), eps_(eps), step_t_(0) {
        // 初始化所有需要求导的参数的状态
        for (auto& group : param_groups) {
            for (auto& param : group.params) {
                Tensor* t = param.tensor.get();
                if (t->requires_grad()) {
                    exp_avg_[t] = std::vector<float>(t->size(), 0.0f);
                    exp_avg_sq_[t] = std::vector<float>(t->size(), 0.0f);
                }
            }
        }
    }
    void step();
};


#endif // OPTIMIZER_H