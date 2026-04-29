#include "scheduler.h"
#define _USE_MATH_DEFINES
#include <cmath>
void StepLR::step() {
        last_epoch_++;
        if (last_epoch_ % step_size_ == 0) {
            // 触发衰减：遍历优化器里的所有参数组，修改它们的学习率
            for (int i = 0; i < optimizer_->param_groups.size(); ++i) {
                optimizer_->param_groups[i].lr *= gamma_;
            }
        }
}

void CosineAnnealingLR::step(){
    last_epoch_++;
    
    // 核心数学公式：eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(T_cur / T_max * PI))
    float progress = static_cast<float>(last_epoch_) / T_max_;
    float cosine_factor = 0.5f * (1.0f + std::cos(progress * acos(-1.0f)));

    for (size_t i = 0; i < optimizer_->param_groups.size(); ++i) {
        float initial_lr = base_lrs_[i];
        
        // 重新计算并覆盖优化器内部的学习率
        optimizer_->param_groups[i].lr = eta_min_ + (initial_lr - eta_min_) * cosine_factor;
    }
}