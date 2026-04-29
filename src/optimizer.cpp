/*
 * @Author: fool
 * @Date: 2026-04-21 16:33:20
 * @LastEditors: fool
 * @LastEditTime: 2026-04-28 20:18:20
 * @FilePath: \TinyInferEngine\src\optimizer.cpp
 * @Description:  
 * @Note:  
 */
#include "optimizer.h"

void SGD::step() {
    // 遍历每一个参数组
    for (auto& group : param_groups) {
        float current_lr = group.lr; // 提取该组专属 LR
        float group_wd = group.weight_decay;

        // 遍历该组内的所有参数
        for (auto& param : group.params) {
            if (!param.tensor->requires_grad()) continue;
            float* data = param.tensor->data();
            const float* grad = param.tensor->grad();
            // 依然保留你的智能判定：即使组里设置了 WD，Bias 也免死
            float final_wd = (param.name.find("bias") != std::string::npos) ? 0.0f : group_wd;
            #pragma omp parallel for
            for (int i = 0; i < param.tensor->size(); ++i) {
                // W_new = W_old - lr * (Grad + final_wd * W_old)
                data[i] -= current_lr * (grad[i] + final_wd * data[i]);
            }
        }
    }
}

    
void AdamW::step() {
    step_t_++;
    // 偏差校正系数
    float bias_correction1 = 1.0f - std::pow(beta1_, step_t_);
    float bias_correction2 = 1.0f - std::pow(beta2_, step_t_);

    for (auto& group : param_groups) {
        float lr = group.lr;
        float weight_decay = group.weight_decay;

        for (auto& param : group.params) {
            Tensor* t = param.tensor.get();
            if (!t->requires_grad()) continue;

            float* data = t->data();
            const float* grad = t->grad();
            std::vector<float>& m = exp_avg_[t];
            std::vector<float>& v = exp_avg_sq_[t];

            float final_wd = (param.name.find("bias") != std::string::npos) ? 0.0f : weight_decay;

            // 具体的 AdamW 物理公式实现
            #pragma omp parallel for
            for (int i = 0; i < t->size(); ++i) {
                // 1. AdamW 专属：解耦的权重衰减 (直接修改权重，不混入梯度)
                data[i] -= lr * final_wd * data[i];

                // 2. 更新一阶矩 (动量) 和二阶矩 (能量)
                m[i] = beta1_ * m[i] + (1.0f - beta1_) * grad[i];
                v[i] = beta2_ * v[i] + (1.0f - beta2_) * grad[i] * grad[i];

                // 3. 偏差校正
                float m_hat = m[i] / bias_correction1;
                float v_hat = v[i] / bias_correction2;

                // 4. 更新权重
                data[i] -= lr * m_hat / (std::sqrt(v_hat) + eps_);
            }
        }
    }
}


