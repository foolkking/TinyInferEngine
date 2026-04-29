/*
 * @Author: fool
 * @Date: 2026-04-22 12:10:01
 * @LastEditors: fool
 * @LastEditTime: 2026-04-29 11:01:45
 * @FilePath: \TinyInferEngine\src\loss.cpp
 * @Description:  
 * @Note:  
 */

#include "loss.h"
#include<limits>
#include<cmath>
#include<vector>
TensorPtr  CrossEntropyLoss::forward(TensorPtr preds, TensorPtr targets) {
    int batch_size = preds->shape(0);
    int num_classes = preds->shape(1);

    std::vector<float> softmax_probs = std::vector<float>(batch_size * num_classes,0.0f);
    std::vector<int> loss_shape = {1}; //Loss 的职责是返回一个标量（只有一个数字），代表全网的平均误差。
    TensorPtr loss = std::make_shared<Tensor>(loss_shape, preds->requires_grad());

    const float* preds_data = preds->data();
    const float* targets_data = targets->data();
    float total_loss = 0.0f;
    for (int n = 0; n < batch_size; ++n) { //计算 Softmax 概率并缓存
        float max_logit = -std::numeric_limits<float>::infinity();
        for(int c = 0; c < num_classes;++c){
            float logit_val = preds_data[n*num_classes+c]; // 获取第n个样本的第 c 类别预测
            if(logit_val>max_logit){
                max_logit = logit_val;
            }
        }
        float sun_exp = 0.0f;
        for(int c = 0; c < num_classes;++c){
            float exp_val = std::exp(preds_data[n*num_classes + c] - max_logit); 
            sun_exp += exp_val;
            softmax_probs[n*num_classes+c] = exp_val;
        }
        for(int c = 0; c < num_classes;++c){
            softmax_probs[n*num_classes+c] = softmax_probs[n*num_classes+c]/sun_exp; // 计算softmax概率

            if(c==targets_data[n]){
                total_loss += -std::log(softmax_probs[n*num_classes+c]+1e-7); //加1e7防止log(0)
            }
        }
    }
    float avge_loss = total_loss/batch_size;
    loss->data()[0]=avge_loss;
    
    if(preds->requires_grad()){
        Tensor* loss_ptr = loss.get();
        std::function<void()>backward_fn = [preds, targets, softmax_probs, loss_ptr,batch_size,num_classes](){
            float* preds_grad = preds->grad();
            const float* targets_data = targets->data();
            const float* loss_grad = loss_ptr->grad();

            for(int n = 0; n < batch_size;++n){
                for(int c = 0; c < num_classes;++c){
                    float grad_val = softmax_probs[n*num_classes+c];
                    if(c==targets_data[n]){
                        grad_val -= 1.0f;
                    }
                    grad_val /= batch_size;
                    preds_grad[n*num_classes+c] += grad_val*loss_grad[0];
                }
            }
        };
        loss->set_auto_grad(backward_fn,{preds});
    }
        
    return loss;
}

TensorPtr MSELoss::forward(TensorPtr preds,TensorPtr targets){
    // 假设 preds 和 targets 维度完全一致
    auto preds_data = preds->data();
    auto targets_data = targets->data();
    auto total_elem = preds->size();
    std::vector<int> loss_shape = {1}; 
    TensorPtr loss = std::make_shared<Tensor>(loss_shape,preds->requires_grad());
    float total_loss = 0.0f;
    for(int i=0;i<total_elem;i++){
        float diff=preds_data[i]-targets_data[i];
        total_loss = diff*diff+total_loss;
    }
    float avge_loss = total_loss/total_elem;
    loss->data()[0] = avge_loss;
    if(preds->requires_grad()){
        Tensor* loss_ptr = loss.get();
        std::function<void()>backward_fn = [total_elem,loss_ptr,preds,targets](){
            float* preds_grad = preds->grad();
            auto preds_data = preds->data();
            auto targets_data = targets->data();
            for(int i=0;i<total_elem;++i){
                float diff = preds_data[i]-targets_data[i];
                preds_grad[i] += (2.0f*diff)/total_elem;
            }
        };
        loss->set_auto_grad(backward_fn,{preds});
    }
    return loss;
}

