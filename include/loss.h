/*
 * @Author: fool
 * @Date: 2026-04-22 00:43:10
 * @LastEditors: fool
 * @LastEditTime: 2026-04-29 11:05:49
 * @FilePath: \TinyInferEngine\include\loss.h
 * @Description:  
 * @Note:  
 */
#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"
class Loss{
public :
    virtual TensorPtr forward(TensorPtr preds, TensorPtr targets){return nullptr;};  //返回的是一个可以触发自动求导的 TensorPtr！
    virtual ~Loss()=default;
};
class CrossEntropyLoss : public Loss{
/*
交叉熵损失 (多分类任务损失计算)
*/
public:
    virtual TensorPtr forward(TensorPtr preds, TensorPtr targets) override;
};

class MSELoss : public Loss{
/*
均方误差损失 (回归任务损失计算)
*/
public:
    virtual TensorPtr forward(TensorPtr preds, TensorPtr targets)  override;
};

#endif // LOSS_H