/*
 * @Author: fool
 * @Date: 2026-04-15 18:30:24
 * @LastEditors: fool
 * @LastEditTime: 2026-04-17 20:35:33
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

public:
    Tensor(const int* shape, int ndims);
    ~Tensor();

    int size() const; 
    int shape(int index) const;
    void print_info() const;
    float* data(); // 1. 返回内部的 float* 裸指针
    const float* data() const; // 常量版本
    void fill(float value); // 2. 将张量里的所有元素都填充为指定的 value
    int stride(int index) const; 
    float& at(const int* indices);  // 3. 根据多维索引访问元素
    const float& at(const int* indices) const; // 常量版本
    int ndims() const { return ndims_; } // 4. 获取维度数量
    bool load_from_file(const std::string& filename); // 5. 从文件加载数据
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
};
#endif // TENSOR_H