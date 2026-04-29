/*
 * @Author: fool
 * @Date: 2026-04-17 19:55:27
 * @LastEditors: fool
 * @LastEditTime: 2026-04-29 12:59:15
 * @FilePath: \TinyInferEngine\include\model.h
 * @Description:  
 * @Note:  
 */
#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include "layer.h"

/// 顺序堆叠模型容器
/// 
/// Sequential是一个简单而强大的神经网络容器，
/// 它将多个层按照添加顺序堆叠在一起，形成完整的神经网络模型。
/// 
/// **工作机制：**
/// 1. 使用 add() 方法添加层，按照添加顺序形成计算链
/// 2. 调用 forward() 时，输入会依次通过每一层
/// 3. 第n层的输出作为第n+1层的输入
/// 4. 最后一层的输出就是整个模型的输出
/// 
/// **特点：**
/// - 支持任意数量的层堆叠
/// - 自动构建从输入到输出的计算图
/// - 通过 named_parameters() 聚合所有层的参数
/// - 配合优化器进行训练和反向传播
/// 
/// **典型使用模式：**
/// ```cpp
/// Sequential model;
/// model.add(std::make_shared<Linear>(784, 128));
/// model.add(std::make_shared<ReLU>());
/// model.add(std::make_shared<Linear>(128, 10));
/// 
/// auto output = model.forward(input);
/// output->backward();  // 反向传播自动工作
/// optimizer.step();    // 更新参数
/// ```
class Sequential {
private:
    std::vector<LayerPtr> layers_; ///< 存储有序的网络层，形成计算链

public:
    /// 默认构造函数，创建空模型
    Sequential() = default;
    
    /// 析构函数，自动释放所有层资源
    ~Sequential();

    /// 向模型添加一个新层
    /// 层会按照添加顺序形成计算链
    /// 
    /// @param layer 要添加的层对象(LayerPtr 智能指针)
    /// @note 添加顺序影响计算流程，通常设计为：输入 → 特征提取 → 分类头
    /// @note 层可以是 Linear、Conv2D、ReLU、MaxPool2D、Flatten 等任意 Layer 子类
    /// 
    /// **示例：**
    /// ```cpp
    /// model.add(std::make_shared<Linear>(784, 128));
    /// model.add(std::make_shared<ReLU>());
    /// model.add(std::make_shared<Linear>(128, 10));
    /// ```
    void add(LayerPtr layer);
    
    /// 执行前向传播
    /// 输入张量会依次通过所有层，生成最终输出
    /// 
    /// @param input 输入张量，形状由第一层决定
    /// @return 最后一层的输出张量
    /// @note 张量的计算图会在此过程中动态构建
    /// @note 输入是常量引用，允许传入临时张量
    /// 
    /// **工作流程：**
    /// 1. 第0层接收input，计算output[0]
    /// 2. 第1层接收output[0]，计算output[1]
    /// 3. ... 循环进行 ...
    /// 4. 返回最后一层的输出
    /// 
    /// **示例：**
    /// ```cpp
    /// auto output = model.forward(input);
    /// // output 已包含完整的计算图，可以调用 backward()
    /// ```
    TensorPtr forward(TensorPtr input);
    
    /// 聚合所有层的参数
    /// 遍历所有层，收集它们的可训练参数
    /// 
    /// @return 包含所有参数的 NamedParameter 向量
    /// @note 返回的参数是指针，修改不会改变原层中的参数
    /// @note 参数名格式为 "layer_index.param_name"，如 "0.weight", "1.bias"
    /// 
    /// **参数聚合过程：**
    /// 1. 遍历所有层
    /// 2. 对每一层调用 parameters()
    /// 3. 合并所有参数到一个向量
    /// 4. 返回供优化器使用
    /// 
    /// **示例：**
    /// ```cpp
    /// auto params = model.named_parameters();
    /// for (const auto& param : params) {
    ///     std::cout << param.name << ": " << param.tensor->size() << " elements\n";
    /// }
    /// ```
    std::vector<NamedParameter> named_parameters() const;
};

#endif // MODEL_H
