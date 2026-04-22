/*
 * @Author: fool
 * @Date: 2026-04-15 18:30:24
 * @LastEditors: fool
 * @LastEditTime: 2026-04-22 14:47:45
 * @FilePath: \TinyInferEngine\include\tensor.h
 * @Description:  
 * @Note:  
 */

#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>

class Tensor {
private:
    float* data_;       
    int* shape_;        
    int* strides_;
    int ndims_;         
    int size_; 
    float *grad_;
    bool requires_grad_;         

public:
    Tensor(const int* shape, int ndims, bool requires_grad = false);  //必须在一开始就决定好形状和是否需要梯度
    ~Tensor();

    int size() const; 
    int shape(int index) const;
    void print_info() const;
    float* data(); // 1. 返回内部的 float* 裸指针
    const float* data() const; // 常量版本
    int ndims() const { return ndims_; } //  获取维度数量
    int stride(int index) const; // 获取指定维度的 stride
    float& at(const int* indices);  // 根据多维索引访问元素
    const float& at(const int* indices) const; // 常量版本

    float * grad(){ return grad_; } // 获取梯度指针
    bool requires_grad() const { return requires_grad_; }
    void zero_grad(){ // 将梯度清零
        if (grad_) {
            for (int i = 0; i < size_; ++i) {
                grad_[i] = 0.0f;
            }
        }
    }

    void fill(float value); // 2. 将张量里的所有元素都填充为指定的 value
    bool load_from_file(const std::string& filename); // 5. 从文件加载数据
    void randomize(float min_val, float max_val); // 随机初始化权重
    bool save_to_bin(const std::string& file_path) const; // 导出二进制权重
    
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
};
#endif // TENSOR_H