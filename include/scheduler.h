/*
 * @Author: fool
 * @Date: 2026-04-28 18:16:20
 * @LastEditors: fool
 * @LastEditTime: 2026-04-29 12:58:35
 * @FilePath: \TinyInferEngine\include\scheduler.h
 * @Description:  
 * @Note:  
 */
#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "optimizer.h"
#include <vector>

/// 学习率调度器基类
/// 
/// LRScheduler是所有学习率调度器的抽象基类。
/// 学习率调度器用于在训练过程中动态调整优化器的学习率。
/// 
/// 主要职责：
/// - 在训练的不同阶段调整学习率
/// - 保存初始学习率供参考
/// - 定义学习率衰减策略
/// 
/// 常见的调度策略：
/// - StepLR: 每隔N个epoch，学习率乘以一个衰减因子
/// - CosineAnnealingLR: 学习率按余弦曲线从初始值衰减到最小值
/// - ReduceLROnPlateau: 当指标停止改进时降低学习率
/// 
/// 使用流程：
/// 1. 创建优化器
/// 2. 创建调度器，传入优化器
/// 3. 每个epoch结束后调用scheduler.step()更新学习率
class LRScheduler {
protected:
    Optimizer* optimizer_;              ///< 指向优化器的指针
    int last_epoch_ = 0;                ///< 记录当前epoch数
    std::vector<float> base_lrs_;       ///< 存储每个参数组的初始学习率
    
public:
    /// 构造学习率调度器
    /// @param optimizer 指向优化器的指针，将在其上应用学习率调度
    /// @note 自动从优化器的参数组中提取初始学习率
    LRScheduler(Optimizer* optimizer) : optimizer_(optimizer) {
        for (const auto& group : optimizer_->param_groups) {
            base_lrs_.push_back(group.lr);
        }
    }
    
    /// 虚析构函数
    virtual ~LRScheduler() = default;
    
    /// 执行一步学习率调度
    /// 根据调度策略更新优化器中各参数组的学习率
    /// @note 此为纯虚函数，所有子类必须实现具体的衰减公式
    virtual void step() = 0;
};

/// 阶跃学习率调度器(Step Learning Rate Scheduler)
/// 
/// StepLR按固定间隔步骤对学习率进行阶跃性衰减。
/// 每隔指定的epoch数，学习率乘以一个衰减因子(通常为0.1)。
/// 
/// 学习率更新公式：
/// - 如果 (last_epoch + 1) % step_size == 0:
///   - new_lr = old_lr * gamma
/// - 其中gamma通常为0.1(表示学习率降至原来的1/10)
/// 
/// 效果：
/// - 学习率呈阶跃下降
/// - 便于控制衰减时机
/// - 适合有明确训练阶段的任务
/// 
/// 使用示例：
/// ```
/// optimizer = SGD(params, lr=0.1)
/// scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
/// for epoch in range(100):
///     train()
///     scheduler.step()  // 每10个epoch学习率乘以0.1
/// ```
/// 
/// @note 常用于需要明确学习率调整时机的场景
class StepLR : public LRScheduler {
private:
    int step_size_;     ///< 阶跃间隔，每隔此数个epoch更新一次学习率
    float gamma_;       ///< 衰减因子，学习率乘以此值(默认0.1)
    
public:
    /// 构造StepLR调度器
    /// @param optimizer 指向优化器的指针
    /// @param step_size 阶跃间隔(epoch数)，每隔此数个epoch衰减一次学习率
    /// @param gamma 衰减因子，默认为0.1，表示学习率降至1/10
    StepLR(Optimizer* optimizer, int step_size, float gamma = 0.1f)
        : LRScheduler(optimizer), step_size_(step_size), gamma_(gamma) {}
    
    /// 执行一步学习率调度
    /// 检查是否应该进行衰减，若是则更新所有参数组的学习率
    void step() override;
};

/// 余弦退火学习率调度器(Cosine Annealing Learning Rate Scheduler)
/// 
/// CosineAnnealingLR按余弦曲线动态调整学习率。
/// 学习率从初始值平滑地衰减至最小值，遵循余弦函数的形状。
/// 
/// 学习率更新公式：
/// - new_lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(π * last_epoch / T_max))
/// - 其中:
///   - eta_min为学习率最小值
///   - base_lr为初始学习率
///   - T_max为周期长度(通常为总epoch数)
///   - last_epoch为当前epoch数
/// 
/// 效果：
/// - 学习率平滑衰减，避免阶跃跳跃
/// - 训练初期较大的学习率用于快速收敛
/// - 训练后期较小的学习率用于精细优化
/// - 可减少学习率对超参数的敏感性
/// 
/// 使用示例：
/// ```
/// optimizer = AdamW(params, lr=1e-3)
/// scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
/// for epoch in range(100):
///     train()
///     scheduler.step()  // 学习率按余弦曲线从1e-3衰减到1e-5
/// ```
/// 
/// 优势：
/// - 广泛应用于现代深度学习模型(如Vision Transformer)
/// - 相比线性衰减，精细优化效果更好
/// - 避免过早的学习率衰减
/// 
/// @note 推荐用于需要平滑学习率衰减的任务，特别是Transformer类模型
class CosineAnnealingLR : public LRScheduler {
private:
    int T_max_;         ///< 周期长度(通常为总epoch数)
    float eta_min_;     ///< 学习率的最小值下限
    
public:
    /// 构造CosineAnnealingLR调度器
    /// @param optimizer 指向优化器的指针
    /// @param T_max 周期长度(epoch数)，通常设为总训练epoch数
    /// @param eta_min 学习率最小值，默认为0.0
    /// 
    /// @note 推荐 eta_min 设置为一个较小的值(如1e-5)，避免学习率过小影响训练
    CosineAnnealingLR(Optimizer* optimizer, int T_max, float eta_min = 0.0f)
        : LRScheduler(optimizer), T_max_(T_max), eta_min_(eta_min) {}
    
    /// 执行一步学习率调度
    /// 按余弦曲线更新所有参数组的学习率
    void step() override;
};

#endif // SCHEDULER_H