/*
 * @Author: fool
 * @Date: 2026-04-17 15:46:44
 * @LastEditors: fool
 * @LastEditTime: 2026-04-29 10:58:48
 * @FilePath: \TinyInferEngine\include\layer.h
 * @Description:  
 * @Note:  
 */
#ifndef LAYER_H
#define LAYER_H
#include <vector>
#include <utility> // for std::pair
#include "tensor.h"

// 【新增】：命名参数结构体，给冷冰冰的 Tensor 赋予身份！
struct NamedParameter {
    std::string name;
    TensorPtr tensor;
};

class Layer {
protected:
    //不需要再记住cache_input_来计算当前节点的梯度了，反向传播可以直接通过前驱节点访问输入Tensor的梯度。
    virtual void clearup() { 
        // 默认实现：什么都不做。子类可以重写这个方法来清理它们自己的缓存。
    }
public:
    virtual ~Layer() {
        Layer::clearup();
    }
    // 前向传播接口：常量输入，可变输出
    virtual TensorPtr forward(TensorPtr input) = 0; 
    // 获取该层的所有可训练权重（极大地简化优化器逻辑）
    virtual std::vector<NamedParameter> parameters(){ 
        // 默认返回空数组。像 ReLU, Flatten 这种没有权重的层直接继承即可。
        // 而 Linear, Conv2D 这种有权重的层，需要重写它，返回 {{"weight", weight_}, {"bias", bias_}}
        return {}; 
    }
    // 只有forward使用到，也没必要实现。可以在具体类的内部作为私有辅助函数实现，不再强制要求作为公开的虚函数。
    //virtual std::vector<int> compute_output_shape(const std::vector<int>& input_shape) const = 0;
};
using LayerPtr = std::shared_ptr<Layer>;
#endif // LAYER_H