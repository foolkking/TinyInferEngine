/*
 * @Author: fool
 * @Date: 2026-04-17 19:56:03
 * @LastEditors: fool
 * @LastEditTime: 2026-04-29 11:32:55
 * @FilePath: \TinyInferEngine\src\model.cpp
 * @Description:  
 * @Note:  
 */
#include "model.h"
#include <string>
Sequential::~Sequential() {
    // 这里我们不负责删除 Layer*，因为它们可能在外部被共享或管理
    layers_.clear();
}
void Sequential::add(LayerPtr layer){
    layers_.push_back(layer);
}

TensorPtr Sequential::forward( TensorPtr input) {
    TensorPtr current_output = input; // 当前输出初始为输入
    // int layerindex = 0;
    for (LayerPtr& layer : layers_) {
        
        current_output = layer->forward(current_output); // 前向传播
        // std::cout<<"layer "<<layerindex++<<" is ok\n";
        // current_output->print_info();
    }
    return current_output; // 返回最终输出
}

std::vector<NamedParameter> Sequential::named_parameters() const {
    std::vector<NamedParameter> named_params;
    for (int i = 0; i < layers_.size();i++) {
        auto layer_para = layers_[i]->parameters();
        for(auto& parameter : layer_para) {
            parameter.name = "Layer_" + std::to_string(i) + " " + parameter.name;
            named_params.push_back(parameter);
        }
    }
    return  named_params; // 返回所有参数的名称和值
}