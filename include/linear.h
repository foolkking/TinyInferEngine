/*
 * @Author: fool
 * @Date: 2026-04-17 16:34:52
 * @LastEditors: fool
 * @LastEditTime: 2026-04-28 19:57:28
 * @FilePath: \TinyInferEngine\include\linear.h
 * @Description:  
 * @Note:  
 */
#ifndef LINEAR_H
#define LINEAR_H

#include "layer.h"

class Linear : public Layer {

private:
    int in_features_;
    int out_features_;
    
    TensorPtr weight_; // 内部的权重张量，形状 [out_features, in_features]
    TensorPtr bias_;   // 内部的偏置张量，形状 [out_features]
protected:
    void clearup() override {
        // 这里不需要手动 delete 了，智能指针会自动管理内存
        weight_.reset();
        bias_.reset();
    }
public:
    // 构造函数：告诉这个层输入维度和输出维度是多少
    Linear(int in_features, int out_features,bool requires_grad = false);
    TensorPtr weight(){ return weight_; }
    TensorPtr bias(){ return bias_; }

    // 析构函数：释放 weight_ 和 bias_
    ~Linear()=default; // 使用默认析构函数，智能指针会自动清理资源
    // 暴露内部张量的指针，方便外部初始化权重 (比如用随机数或加载模型文件)

    // 矩阵乘法前向传播
    TensorPtr forward(TensorPtr input) override;
    // 获取可训练权重
    std::vector<NamedParameter> parameters()override;

};

#endif // LINEAR_H