/*
 * @Author: fool
 * @Date: 2026-04-23 12:32:38
 * @LastEditors: fool
 * @LastEditTime: 2026-04-23 13:19:10
 * @FilePath: \TinyInferEngine\include\silu.h
 * @Description:  
 * @Note:  
 */
#ifndef SILU_H
#define SILU_H
#include "layer.h"    
class SiLU : public Layer {
public:
    TensorPtr SiLU::forward(TensorPtr input)override ;
    std::vector<NamedParameter>parameters(){
        return {};
    }

};
#endif // SILU_H