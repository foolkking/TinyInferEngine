/*
 * @Author: fool
 * @Date: 2026-04-28 18:16:20
 * @LastEditors: fool
 * @LastEditTime: 2026-04-28 19:45:50
 * @FilePath: \TinyInferEngine\include\scheduler.h
 * @Description:  
 * @Note:  
 */
#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "optimizer.h"
#include <vector>

class LRScheduler {//实现StepLR与CosineAnnealingLR
protected:
    Optimizer* optimizer_;
    int last_epoch_ = 0;
    std::vector<float> base_lrs_; // 记录每一组最初始的学习率

public:
    LRScheduler(Optimizer* optimizer) : optimizer_(optimizer) {
        for (const auto& group : optimizer_->param_groups) {
            base_lrs_.push_back(group.lr);
        }
    }
    virtual ~LRScheduler() = default;
    virtual void step() = 0;// 留给子类去实现具体的衰减数学公式
};

class StepLR :public LRScheduler {
private:
    int step_size_;
    float gamma_;
public:
    StepLR(Optimizer* optimizer, int step_size, float gamma = 0.1f)
        : LRScheduler(optimizer), step_size_(step_size), gamma_(gamma) {}
    void step()override;    // 每个 Epoch 结束后调用
};

class CosineAnnealingLR : public LRScheduler {
private:
    int T_max_;      // 周期长度 (比如总的 Epoch 数)
    float eta_min_;  // 学习率的下限
public:
    CosineAnnealingLR(Optimizer* optimizer, int T_max, float eta_min = 0.0f)
        : LRScheduler(optimizer), T_max_(T_max), eta_min_(eta_min) {}
    void step() override ;
};

#endif // SCHEDULER_H