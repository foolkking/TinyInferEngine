/*
 * @Author: fool
 * @Date: 2026-04-20 22:14:50
 * @LastEditors: fool
 * @LastEditTime: 2026-04-21 13:07:29
 * @FilePath: \TinyInferEngine\include\relu.h
 * @Description:  
 * @Note:  
 */
#ifndef RELU_H
#define RELU_H

#include "layer.h"

class ReLU : public Layer {
public:
    // 覆盖基类的 forward 方法
    TensorPtr ReLU::forward(TensorPtr input) override;
    std::vector<NamedParameter> parameters(){
        return {};
    }

};

#endif // RELU_H