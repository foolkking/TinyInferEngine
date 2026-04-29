/*
 * @Author: fool
 * @Date: 2026-04-17 19:55:27
 * @LastEditors: fool
 * @LastEditTime: 2026-04-28 21:54:29
 * @FilePath: \TinyInferEngine\include\model.h
 * @Description:  
 * @Note:  
 */
#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include "layer.h"

class Sequential {
private:
    std::vector<LayerPtr> layers_; // 存储“算子序列”

public:
    Sequential() = default;
    ~Sequential();

    // 往模型里按顺序添加算子
    void add(LayerPtr layer);
    TensorPtr forward(TensorPtr input);
    std::vector<NamedParameter>named_parameters()const;
};

#endif // MODEL_H