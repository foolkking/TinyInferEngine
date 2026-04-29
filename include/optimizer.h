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

/// 参数分组结构体
/// 
/// 用于管理具有不同学习率和权重衰减策略的参数组。
/// 支持为不同的层或参数设置不同的优化策略。
/// 
/// 应用场景：
/// - 为不同层设置不同的学习率（如预训练层用低学习率，新层用高学习率）
/// - 为权重和偏置设置不同的权重衰减
/// - 微调(Fine-tuning)中的差异化学习率策略
struct ParamGroup {
    std::vector<NamedParameter> params; ///< 参数组中的所有参数
    float lr;                           ///< 学习率
    float weight_decay;                 ///< 权重衰减系数（L2正则化）
    
    /// 构造参数分组
    /// @param p 参数列表
    /// @param l 学习率
    /// @param wd 权重衰减系数，默认为0.0
    ParamGroup(std::vector<NamedParameter> p, float l, float wd = 0.0f)
        : params(std::move(p)), lr(l), weight_decay(wd) {}
};

/// 优化器基类
/// 
/// Optimizer是所有优化器的抽象基类，定义了优化器必须实现的接口。
/// 优化器用于根据梯度信息更新模型参数。
/// 
/// 主要职责：
/// - 管理参数分组和学习率
/// - 清零梯度（zero_grad）
/// - 执行参数更新（step）
/// - 与学习率调度器配合工作
/// 
/// 支持的优化器：
/// - SGD: 随机梯度下降
/// - AdamW: Adam优化器的权重衰减变体
class Optimizer {
public:
    /// 参数分组列表（公开以便学习率调度器修改）
    std::vector<ParamGroup> param_groups;

    /// 构造优化器
    /// @param groups 参数分组列表，包含不同学习率的参数组
    Optimizer(const std::vector<ParamGroup>& groups) : param_groups(groups) {}
    
    /// 虚析构函数
    virtual ~Optimizer() = default;

    /// 清零所有参数的梯度
    /// 应在每个训练步骤开始时调用，防止梯度累积
    /// @note 此为虚函数，可被子类覆盖以提供优化实现
    virtual void zero_grad() {
        for (auto& group : param_groups) {
            for (auto& param : group.params) {
                if (param.tensor->requires_grad()) param.tensor->zero_grad();
            }
        }
    }

    /// 执行一步参数更新
    /// 根据梯度信息和优化器的特定算法更新所有参数
    /// @note 此为纯虚函数，所有子类必须实现
    virtual void step() = 0; 
};

/// 随机梯度下降(SGD)优化器
/// 
/// 实现标准的随机梯度下降算法。
/// 支持动量、权重衰减等功能。
/// 
/// 参数更新公式：
/// - W = W - lr * (∇L + weight_decay * W)
/// - 其中∇L为梯度，weight_decay为权重衰减系数
/// 
/// 特点：
/// - 简单快速
/// - 内存占用小
/// - 适合大数据集
/// 
/// @note 支持参数分组和学习率调度
class SGD : public Optimizer {
public:
    /// 构造函数1：多参数组构造
    /// 
    /// 用于需要为不同参数设置不同学习率的场景
    /// 
    /// @param groups 参数分组列表，每个分组可有不同的学习率和权重衰减
    /// @note 学习率可通过学习率调度器(Scheduler)动态修改
    SGD(const std::vector<ParamGroup>& groups) : Optimizer(groups) {}

    /// 构造函数2：单参数组构造（向后兼容）
    /// 
    /// 简化的构造方式，所有参数共用同一个学习率和权重衰减策略
    /// 
    /// @param params 所有参数
    /// @param lr 学习率
    /// @param weight_decay 权重衰减系数，默认为0.0
    SGD(const std::vector<NamedParameter>& params, float lr, float weight_decay = 0.0f)
        : Optimizer({ParamGroup(params, lr, weight_decay)}) {}
    
    /// 析构函数
    ~SGD() = default;

    /// 执行SGD更新步骤
    /// 更新所有参数组中的参数
    void step() override;
};

/// AdamW优化器（权重衰减解耦的Adam）
/// 
/// 实现AdamW算法，是Adam的改进版本，改进了权重衰减(L2正则化)的处理。
/// 相比标准Adam，AdamW将权重衰减与梯度更新解耦，效果更好。
/// 
/// 核心特性：
/// - 自适应学习率：根据梯度的一阶和二阶矩自动调整学习率
/// - 动量机制：使用指数加权移动平均计算梯度的动量
/// - 偏差校正：对初期迭代进行偏差校正
/// - 解耦权重衰减：直接修改权重而不影响梯度计算
/// 
/// 参数更新公式：
/// - m = β1*m + (1-β1)*∇L        (一阶矩估计)
/// - v = β2*v + (1-β2)*∇L²       (二阶矩估计)
/// - m_hat = m / (1-β1^t)        (偏差校正)
/// - v_hat = v / (1-β2^t)        (偏差校正)
/// - W = W - lr*weight_decay*W - lr*m_hat/(√v_hat + ε)
/// 
/// 适用场景：
/// - 深度神经网络训练
/// - Transformer模型
/// - 需要快速收敛的任务
/// 
/// @note 对学习率和其他超参数相对鲁棒，是推荐的通用优化器
class AdamW : public Optimizer {
private:
    float beta1_;                                          ///< 一阶矩指数衰减因子，默认0.9
    float beta2_;                                          ///< 二阶矩指数衰减因子，默认0.999
    float eps_;                                            ///< 数值稳定性小常数，默认1e-8
    int step_t_;                                           ///< 全局更新步数计数器，用于偏差校正
    
    /// 梯度一阶矩(动量)，键为参数张量指针，值为该参数的一阶矩向量
    std::unordered_map<Tensor*, std::vector<float>> exp_avg_;
    
    /// 梯度二阶矩(能量)，键为参数张量指针，值为该参数的二阶矩向量
    std::unordered_map<Tensor*, std::vector<float>> exp_avg_sq_;
    
public:
    /// 构造AdamW优化器
    /// @param groups 参数分组列表
    /// @param beta1 一阶矩指数衰减因子，默认0.9
    /// @param beta2 二阶矩指数衰减因子，默认0.999
    /// @param eps 数值稳定性小常数，默认1e-8
    /// 
    /// @note 会自动为所有参数初始化一阶矩和二阶矩状态
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
    
    /// 执行AdamW更新步骤
    /// 更新所有参数组中的参数
    void step() override;
};


#endif // OPTIMIZER_H