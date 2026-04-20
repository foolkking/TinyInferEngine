/*
 * @Author: fool
 * @Date: 2026-04-15 18:30:24
 * @LastEditors: fool
 * @LastEditTime: 2026-04-17 21:00:50
 * @FilePath: \TinyInferEngine\src\tensor.cpp
 * @Description:  
 * @Note:  
 */

#include "tensor.h"
#include <fstream>
// 注意这里要加上 Tensor:: 作用域解析符
Tensor::Tensor(const int* shape, int ndims) {
    ndims_ = ndims;
    shape_ = new int[ndims_];
    strides_ = new int[ndims_]; // 分配步长数组的内存
    size_ = 1;
    // 拷贝形状并计算总大小
    for (int i = 0; i < ndims_; ++i) {
        shape_[i] = shape[i];
        size_ *= shape[i];
    }
    
    data_ = new float[size_];
    
    // 【核心优化】：从后往前，一次性计算并缓存所有维度的步长
    int current_stride = 1;
    for (int i = ndims_ - 1; i >= 0; --i) {
        strides_[i] = current_stride;
        current_stride *= shape_[i];
    }
}

Tensor::~Tensor() {
    delete[] data_;  
    delete[] shape_; 
    delete[] strides_; // 记得释放 strides_ 内存
}

int Tensor::size() const { return size_; }

int Tensor::shape(int index) const {
    // 可以在这里加一个简单的越界检查，保证引擎的鲁棒性
    if (index < 0 || index >= ndims_) {
        std::cerr << "Error: Dimension index out of bounds!" << std::endl;
        return -1;
    }
    return shape_[index];
}

void Tensor::print_info() const {
    std::cout << "Tensor Info: " << std::endl;
    std::cout << "Dimensions: " << ndims_ << std::endl;
    std::cout << "Shape: [";
    for (int i = 0; i < ndims_; ++i) {
        std::cout << shape_[i];
        if (i < ndims_ - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Total Size: " << size_ << std::endl;
}

float* Tensor::data() {
    return data_;
}

const float* Tensor::data() const {
    return data_;
}

void Tensor::fill(float value) {
    for (int i = 0; i < size_; ++i) {
        data_[i] = value;
    }
}

int Tensor::stride(int index) const {
    if (index < 0 || index >= ndims_) {
        std::cerr << "Error: Dimension index out of bounds!" << std::endl;
        return -1;
    }
    return strides_[index];
}

float& Tensor::at(const int* indices) {
    int offset = 0;
    for (int i = 0; i < ndims_; ++i) {
        offset += indices[i] * stride(i);
    }
    return data_[offset];
}

const float& Tensor::at(const int* indices) const {
    int offset = 0;
    for (int i = 0; i < ndims_; ++i) {
        offset += indices[i] * stride(i);
    }
    return data_[offset];
}



bool Tensor::load_from_file(const std::string& filename) {
    std::ifstream infile(filename,std::ios::binary); // 以二进制模式打开文件,确保换行等不被错误转义
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    } 
    infile.read(reinterpret_cast<char*>(data_), size_ * sizeof(float));
    int total_elements = size_*sizeof(float);;
    if (infile.gcount() != total_elements) {
        std::cerr << "Error: Expected to read " << total_elements << " bytes, but got " << infile.gcount() << " bytes." << std::endl;
        return false;
    }

    infile.close();
    return true;
}
