/*
 * @Author: fool
 * @Date: 2026-04-20 20:35:36
 * @LastEditors: fool
 * @LastEditTime: 2026-04-28 21:57:54
 * @FilePath: \TinyInferEngine\include\flatten.h
 * @Description:  
 * @Note:  
 */
#ifndef FLATTEN_H
#define FLATTEN_H

#include "layer.h"

class Flatten : public Layer {
public:
    Flatten() = default;
    ~Flatten() = default;
    
    TensorPtr forward( TensorPtr input) override;
    std::vector<NamedParameter> parameters()  override {
        // Flatten 层没有可训练参数，直接返回空向量
        return {};
    }

};
#endif