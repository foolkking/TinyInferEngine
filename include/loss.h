/*
 * @Author: fool
 * @Date: 2026-04-22 00:43:10
 * @LastEditors: fool
 * @LastEditTime: 2026-04-29 12:43:37
 * @FilePath: \TinyInferEngine\include\loss.h
 * @Description:  
 * @Note:  
 */
#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

/// 损失函数基类
/// 
/// Loss是所有损失函数的抽象基类，定义了损失函数必须实现的接口。
/// 损失函数用于衡量模型预测值与目标值之间的差异。
/// 
/// 已实现loss: CrossEntropyLoss  、 MSELoss
/// 
/// 主要职责：
/// - 计算预测值与真实值之间的误差
/// - 返回可进行自动求导的TensorPtr，用于反向传播
/// 
/// 关键特性：
/// - 返回的是一个标量张量（用于自动求导）
/// - 支持动态计算图构建
/// - 可与优化器无缝配合
class Loss {
public:
    /// 前向传播：计算损失
    /// @param preds 模型的预测输出张量
    /// @param targets 真实标签张量
    /// @return 损失值张量（标量），支持自动求导
    /// @note 此为虚函数，具体实现由子类提供
    virtual TensorPtr forward(TensorPtr preds, TensorPtr targets) { return nullptr; };
    
    /// 虚析构函数
    virtual ~Loss() = default;
};

/// 交叉熵损失函数（多分类任务）
/// 
/// CrossEntropyLoss计算分类任务的交叉熵损失。
/// 适用于多分类问题，模型输出应为logits（未经softmax的原始分数）。
/// 
/// 损失计算：
/// - 对预测值进行softmax转换为概率分布
/// - 计算真实分布与预测分布之间的KL散度
/// 
/// 输入要求：
/// - preds: [batch_size, num_classes] - 模型的logits输出
/// - targets: [batch_size] - 类别标签（0到num_classes-1的整数）
/// 
/// 输出：
/// - 标量损失值，表示平均交叉熵
/// 
/// @note 常用于分类任务的多类别目标
class CrossEntropyLoss : public Loss {
public:
    /// 计算交叉熵损失
    /// @param preds 预测logits张量，形状 [batch_size, num_classes]
    /// @param targets 真实标签张量，形状 [batch_size]，包含0到num_classes-1的整数
    /// @return 标量损失张量，支持自动求导用于反向传播
    virtual TensorPtr forward(TensorPtr preds, TensorPtr targets) override;
};

/// 均方误差损失函数（回归任务）
/// 
/// MSELoss计算回归任务的均方误差。
/// 适用于连续值预测，衡量预测值与目标值的平方差。
/// 
/// 损失计算：
/// - MSE = mean((preds - targets)^2)
/// - 即计算所有元素差的平方的平均值
/// 
/// 输入要求：
/// - preds: [batch_size, ...] - 模型的预测输出，任意维度
/// - targets: [batch_size, ...] - 真实目标值，与preds形状相同
/// 
/// 输出：
/// - 标量损失值，表示平均均方误差
/// 
/// 特点：
/// - 对异常值（离群点）敏感，因为平方项
/// - 梯度为2*(preds - targets)，反向传播清晰
/// 
/// @note 常用于回归任务或连续值预测
class MSELoss : public Loss {
public:
    /// 计算均方误差损失
    /// @param preds 预测值张量，可为任意形状
    /// @param targets 真实值张量，形状与preds相同
    /// @return 标量损失张量，支持自动求导用于反向传播
    virtual TensorPtr forward(TensorPtr preds, TensorPtr targets) override;
};

#endif // LOSS_H