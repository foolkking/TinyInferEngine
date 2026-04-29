/*
 * @Author: fool
 * @Date: 2026-04-15 18:30:24
 * @LastEditors: fool
 * @LastEditTime: 2026-04-26 14:25:24
 * @FilePath: \TinyInferEngine\include\tensor.h
 * @Description:  
 * @Note:  
 */

#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <memory>
#include <vector>
#include <functional>

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
//升级智能指针，在动态计算图中，一个 Tensor 可能会被多个操作共享（比如SwiGLU门机制里的X兵分两路）。
class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    float* data_ = nullptr;       
    std::vector<int> shape_;        
    int* strides_ = nullptr;      
    int size_ = 0; 

    float *grad_ = nullptr;
    bool requires_grad_ = false;    
    
    std::vector<TensorPtr> prev_; // 记录前驱节点，反向传播时需要访问它们的梯度
    std::function<void()> backward_fn_; //当前这个节点的“反向传播函数”,它是一个闭包，捕获了前驱节点和求导法则才初始化。
    //引入它之后Layer 变成了纯粹的“造图工厂”：Linear::forward 不再只是算一个乘法和加法。它在算完数学结果后，
    //会动态地生成一个闭包（Lambda 函数，也就是 backward_fn_），把求导法则打包塞进结果 Tensor 里，
    //并把输入的 X 和 W 连上线（存入 prev_）。 然后当调用结果 Tensor 的 backward() 时，这个闭包就会被调用，
    //自动地把输出的梯度变成输入的梯度，递归地传播下去。自动完成反向传播

    bool is_view_ = false; // 是否是视图（比如切片或转置），视图不拥有数据，不保留梯度
public:
    Tensor(const std::vector<int>& shape, bool requires_grad = false);  //必须在一开始就决定好形状和是否需要梯度
    ~Tensor();

    int size() const; 
    int shape(int index) const;
    std::vector<int> shape() const { return shape_; } // 获取完整形状信息
    void print_info() const;
    float* data(); // 1. 返回内部的 float* 裸指针
    const float* data() const; // 常量版本
    int ndims() const { return shape_.size(); } //  获取维度数量,ndims()也不显式存储
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

    void set_auto_grad(const std::function<void()>& backward_fn, const std::vector<TensorPtr>& prev) ; //绑定计算图:在发生前向计算(如加法、乘法)时，由算子调用此方法
    bool is_leaf() const { return prev_.empty(); } //不需要显式存储is_leaf_，通过检查前驱节点是否为空来判断是否为叶子节点
    bool is_view() const { return is_view_; }   
    std::vector<TensorPtr> prev() const { return prev_; } // 获取前驱节点列表
    void set_view(bool is_view) { is_view_ = is_view; } // 设置是否为视图
    void backward(); // 反向传播接口，调用当前节点的 backward_fn_，并递归调用前驱节点的 backward()
    
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
};
#endif // TENSOR_H


//以后的算子（如加法、乘法、线性层等）不需要再含有backward()函数了。它们只负责前向计算。
//真正的反向传播逻辑完全封装在 Tensor 类的backward_fn中里由backward()自动求导调用