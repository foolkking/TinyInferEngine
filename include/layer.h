/*
 * @Author: fool
 * @Date: 2026-04-17 15:46:44
 * @LastEditors: fool
 * @LastEditTime: 2026-04-29 12:38:32
 * @FilePath: \TinyInferEngine\include\layer.h
 * @Description:  
 * @Note:  
 */
#ifndef LAYER_H
#define LAYER_H
#include <vector>
#include <utility> // for std::pair
#include "tensor.h"

/// 命名参数结构体，为Tensor赋予身份标识和名称
/// 用于优化器追踪和管理模型参数，支持分组学习率策略
struct NamedParameter {
    std::string name;    ///< 参数名称（如"layer_0.weight"、"conv1.bias"）
    TensorPtr tensor;    ///< 指向张量的智能指针
};

class Layer;
using LayerPtr = std::shared_ptr<Layer>;

/// 所有神经网络层的基类
/// 
/// Layer是一个抽象基类，定义了神经网络层必须实现的接口。
/// 每个具体层都继承自此类。
/// 
/// 包括的层有：
/// 常规层： Linear 、 Conv2D  、MaxPool2D  、 Flatten
/// 激活函数层：ReLU  、 SiLU 
///
/// 主要职责：
/// - 前向传播计算
/// - 参数管理和暴露
/// - 内存清理
/// 
/// 关键特性：
/// - 基于自动求导图构建动态计算图
/// - 使用智能指针管理内存，无需手动delete
/// - 支持参数提取用于优化器
class Layer {
protected:
    /// 清理层内部资源（缓存、临时数据等）
    /// 默认实现什么都不做，子类可重写来释放自己的缓存
    virtual void clearup() { 
        // 默认实现：什么都不做。子类可以重写这个方法来清理它们自己的缓存。
    }
public:
    virtual ~Layer() {
        Layer::clearup();
    }
    
    /// 前向传播计算
    /// @param input 输入张量，维度取决于具体层的设计
    /// @return 输出张量，维度由层的计算规则确定
    /// @note 该方法是纯虚函数，所有子类必须实现
    virtual TensorPtr forward(TensorPtr input) = 0; 
    
    /// 获取该层的所有可训练参数
    /// @return 包含所有权重和偏置的NamedParameter向量
    /// @note 无参数的层（ReLU、Flatten等）返回空向量
    /// @note 有参数的层（Linear、Conv2D等）返回{weight, bias, ...}
    virtual std::vector<NamedParameter> parameters(){ 
        // 默认返回空数组。像 ReLU, Flatten 这种没有权重的层直接继承即可。
        // 而 Linear, Conv2D 这种有权重的层，需要重写它，返回 {{"weight", weight_}, {"bias", bias_}}
        return {}; 
    }
};

/// 全连接层（线性层）
/// 
/// 实现标准的线性变换: y = Wx + b
/// 其中 W 是权重矩阵，b 是偏置向量
/// 
/// 参数：
/// - weight_: 形状 [out_features, in_features]，权重矩阵
/// - bias_: 形状 [out_features]，偏置向量
/// 
/// 前向传播：
/// - 输入形状: [batch_size, ..., in_features]
/// - 输出形状: [batch_size, ..., out_features]
class Linear : public Layer {

private:
    int in_features_;    ///< 输入特征维度
    int out_features_;   ///< 输出特征维度
    
    TensorPtr weight_;   ///< 权重张量，形状 [out_features, in_features]，需要求梯度
    TensorPtr bias_;     ///< 偏置张量，形状 [out_features]，需要求梯度
    
protected:
    /// 清理内部资源
    void clearup() override {
        // 这里不需要手动 delete 了，智能指针会自动管理内存
        weight_.reset();
        bias_.reset();
    }
    
public:
    /// 构造全连接层
    /// @param in_features 输入特征数
    /// @param out_features 输出特征数
    /// @param requires_grad 是否需要计算梯度（默认false，通常在构造时设为true以启用自动求导）
    Linear(int in_features, int out_features, bool requires_grad = false);
    
    /// 获取权重张量指针
    /// @return 权重张量的智能指针，可用于初始化或访问权重
    TensorPtr weight(){ return weight_; }
    
    /// 获取偏置张量指针
    /// @return 偏置张量的智能指针，可用于初始化或访问偏置
    TensorPtr bias(){ return bias_; }

    /// 默认析构函数
    /// 智能指针会自动清理weight_和bias_资源
    ~Linear() = default;

    /// 前向传播：矩阵乘法
    /// @param input 输入张量，最后一维应为in_features
    /// @return 输出张量，最后一维为out_features
    TensorPtr forward(TensorPtr input) override;
    
    /// 获取可训练参数
    /// @return 包含权重和偏置的NamedParameter向量
    std::vector<NamedParameter> parameters() override;
};

/// 二维卷积层
/// 
/// 实现标准的2D卷积操作：y = Conv2D(x) + b
/// 
/// 参数：
/// - weight_: 形状 [out_channels, in_channels, kernel_size, kernel_size]
/// - bias_: 形状 [out_channels]
/// 
/// 前向传播：
/// - 输入形状: [batch_size, in_channels, height, width]
/// - 输出形状: [batch_size, out_channels, new_height, new_width]
/// - new_height = (height + 2*padding - kernel_size) / stride + 1
/// - new_width = (width + 2*padding - kernel_size) / stride + 1
class Conv2D : public Layer {
private:
    int in_channels_;    ///< 输入通道数
    int out_channels_;   ///< 输出通道数（卷积核个数）
    int kernel_size_;    ///< 卷积核大小（假设为正方形）
    int stride_;         ///< 步长
    int padding_;        ///< 填充大小

    TensorPtr weight_;   ///< 权重张量，形状: [out_channels, in_channels, kernel_size, kernel_size]
    TensorPtr bias_;     ///< 偏置张量，形状: [out_channels]
    
protected:
    /// 清理内部资源
    void clearup() override;
    
public:
    /// 构造二维卷积层
    /// @param in_ch 输入通道数
    /// @param out_ch 输出通道数
    /// @param k_size 卷积核大小（正方形）
    /// @param stride 步长，默认为1
    /// @param padding 填充大小，默认为0
    /// @param requires_grad 是否需要计算梯度，默认为false
    Conv2D(int in_ch, int out_ch, int k_size, int stride = 1, int padding = 0, bool requires_grad = false);
    
    /// 析构函数，释放权重和偏置
    ~Conv2D();
    
    /// 获取权重张量指针
    /// @return 权重张量的智能指针
    TensorPtr weight() { return weight_; }
    
    /// 获取偏置张量指针
    /// @return 偏置张量的智能指针
    TensorPtr bias() { return bias_; }

    /// 获取可训练参数
    /// @return 包含权重和偏置的NamedParameter向量
    std::vector<NamedParameter> parameters() override;
    
    /// 前向传播：二维卷积
    /// @param input 输入张量，形状 [batch_size, in_channels, height, width]
    /// @return 输出张量，形状 [batch_size, out_channels, new_height, new_width]
    TensorPtr forward(TensorPtr input) override;
};

/// 二维最大池化层
/// 
/// 对输入进行最大池化操作，不改变通道数。
/// 在反向传播时，梯度只回传到最大值位置。
/// 
/// 前向传播：
/// - 输入形状: [batch_size, channels, height, width]
/// - 输出形状: [batch_size, channels, new_height, new_width]
/// - new_height = (height + 2*padding - kernel_size) / stride + 1
/// - new_width = (width + 2*padding - kernel_size) / stride + 1
class MaxPool2D : public Layer {
private:
    int kernel_size_;    ///< 池化核大小（正方形）
    int stride_;         ///< 步长
    int padding_;        ///< 填充大小
    int* max_indices_ = nullptr; ///< 用于反向传播时记录最大值的位置
    
public:
    /// 构造最大池化层
    /// @param k_size 池化核大小（正方形）
    /// @param stride 步长，默认为1
    /// @param padding 填充大小，默认为0
    MaxPool2D(int k_size, int stride = 1, int padding = 0);
    
    /// 析构函数
    ~MaxPool2D() override;

    /// 前向传播：最大池化
    /// @param input 输入张量，形状 [batch_size, channels, height, width]
    /// @return 输出张量，形状 [batch_size, channels, new_height, new_width]
    TensorPtr forward(TensorPtr input) override;
    
    /// 获取可训练参数（MaxPool2D无参数）
    /// @return 空向量，因为最大池化层没有可训练参数
    std::vector<NamedParameter> parameters() override {
        // MaxPool2D 没有可训练的权重，所以返回空向量
        return {};
    }
};

/// 展平层
/// 
/// 将多维张量展平为二维张量（保留batch维度）。
/// 例如：[batch_size, C, H, W] → [batch_size, C*H*W]
/// 
/// 无可训练参数。
class Flatten : public Layer {
public:
    Flatten() = default;
    ~Flatten() = default;
    
    /// 前向传播：展平
    /// @param input 输入张量，任意维度
    /// @return 展平后的张量，形状 [batch_size, flattened_size]
    TensorPtr forward(TensorPtr input) override;
    
    /// 获取可训练参数（Flatten无参数）
    /// @return 空向量，因为Flatten层没有可训练参数
    std::vector<NamedParameter> parameters() override {
        // Flatten 层没有可训练参数，直接返回空向量
        return {};
    }

};

/// ReLU激活函数层
/// 
/// 实现整流线性单元(Rectified Linear Unit)激活函数。
/// 公式: y = max(0, x)
/// 
/// 无可训练参数。
class ReLU : public Layer {
public:
    /// 前向传播：ReLU激活
    /// @param input 输入张量，任意维度
    /// @return 激活后的张量，形状与输入相同，负值被置为0
    TensorPtr forward(TensorPtr input) override;
    
    /// 获取可训练参数（ReLU无参数）
    /// @return 空向量
    std::vector<NamedParameter> parameters() override {
        return {};
    }

};

/// SiLU激活函数层
/// 
/// 实现Sigmoid Linear Unit激活函数。
/// 公式: y = x * sigmoid(x)
/// 相比ReLU，SiLU是光滑的，梯度更稳定。
/// 
/// 无可训练参数。
class SiLU : public Layer {
public:
    /// 前向传播：SiLU激活
    /// @param input 输入张量，任意维度
    /// @return 激活后的张量，形状与输入相同
    TensorPtr forward(TensorPtr input) override;
    
    /// 获取可训练参数（SiLU无参数）
    /// @return 空向量
    std::vector<NamedParameter> parameters() override {
        return {};
    }

};

#endif // LAYER_H