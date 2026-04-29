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

/// 张量智能指针类型别名
/// 使用 shared_ptr 管理张量生命周期，支持多个操作共享同一张量
/// （如在SwiGLU门机制中，同一个输入张量可被多个分支使用）
using TensorPtr = std::shared_ptr<Tensor>;

/// 多维数据张量类 - 深度学习框架的核心数据结构
/// 
/// Tensor是整个深度学习框架的核心数据结构。
/// 它不仅存储多维数据，还支持自动求导(Autograd)来构建动态计算图。
/// 
/// **核心功能：**
/// 1. **数据存储**：以连续的浮点数数组存储，支持任意维度(shape)
/// 2. **自动求导**：通过记录计算过程和梯度，支持反向传播
/// 3. **动态计算图**：通过 set_auto_grad 在前向计算时动态构建计算图
/// 4. **内存管理**：使用智能指针自动管理生命周期，无需手动释放
/// 
/// **工作机制：**
/// - **前向传播**：算子(Layer)调用 set_auto_grad 将求导函数打包进张量
/// - **反向传播**：调用 backward() 会自动递归执行所有 backward_fn，逐层回传梯度
/// - **计算图**：动态的，由前驱节点(prev_)和反向函数(backward_fn_)组成
/// 
/// **内存布局：**
/// - 数据以行优先(row-major)格式存储
/// - stride 数组记录每个维度的跳跃步长，用于多维索引转换
/// 
/// @note 禁用拷贝构造和赋值，防止数据不一致。张量共享通过 TensorPtr 进行。
/// @note 支持视图(view)概念，视图不拥有数据，仅提供不同的访问方式。
class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    float* data_ = nullptr;              ///< 数据指针，指向浮点数数组
    std::vector<int> shape_;             ///< 张量形状，如[2, 3, 4]表示2x3x4的张量
    int* strides_ = nullptr;             ///< stride数组，用于多维索引到线性地址的转换
    int size_ = 0;                       ///< 数据总元素数

    float* grad_ = nullptr;              ///< 梯度数据指针，与data_形状相同
    bool requires_grad_ = false;         ///< 是否需要计算梯度

    std::vector<TensorPtr> prev_;        ///< 前驱节点列表，记录该张量由哪些张量计算而来
    
    /// 反向传播闭包函数，捕获前驱节点和求导法则
    /// 在反向传播时调用此函数，自动将输出梯度变换为输入梯度
    std::function<void()> backward_fn_;

    bool is_view_ = false;               ///< 是否为视图(如切片或转置)，视图不拥有数据也不保留梯度
    
public:
    /// 构造张量
    /// @param shape 张量形状向量，如{2, 3, 4}创建2x3x4的张量
    /// @param requires_grad 是否需要计算梯度(用于自动求导)，默认false
    /// @note shape必须在构造时确定，之后不可改变
    Tensor(const std::vector<int>& shape, bool requires_grad = false);
    
    /// 析构函数，释放数据、梯度和stride数组
    ~Tensor();

    /// 获取张量元素总数
    /// @return 张量的元素总数
    int size() const;
    
    /// 获取指定维度的大小
    /// @param index 维度索引(0-based)
    /// @return 该维度的大小
    int shape(int index) const;
    
    /// 获取完整的形状向量
    /// @return 包含所有维度大小的向量
    std::vector<int> shape() const { return shape_; }
    
    /// 打印张量信息(形状、大小等)到标准输出
    void print_info() const;
    
    /// 获取数据指针(可写)
    /// @return 指向内部数据的可写指针，用于直接访问和修改数据
    float* data();
    
    /// 获取数据指针(只读)
    /// @return 指向内部数据的只读常量指针
    const float* data() const;
    
    /// 获取张量维度数
    /// @return 张量的维度数(如3D张量返回3)
    int ndims() const { return shape_.size(); }
    
    /// 获取指定维度的步长(stride)
    /// @param index 维度索引
    /// @return 该维度的步长，用于多维索引转换
    int stride(int index) const;
    
    /// 根据多维索引获取元素值(可写)
    /// @param indices 多维索引数组，长度应等于ndims()
    /// @return 对应位置元素的引用，可读写
    float& at(const int* indices);
    
    /// 根据多维索引获取元素值(只读)
    /// @param indices 多维索引数组，长度应等于ndims()
    /// @return 对应位置元素的常量引用
    const float& at(const int* indices) const;

    /// 获取梯度指针
    /// @return 指向梯度数据的指针，如果未计算梯度则为nullptr
    float* grad(){ return grad_; }
    
    /// 检查是否需要计算梯度
    /// @return true表示该张量需要计算梯度，参与自动求导
    bool requires_grad() const { return requires_grad_; }
    
    /// 清零梯度
    /// 在反向传播前调用，防止梯度累积
    /// @note 通常由优化器的zero_grad()调用
    void zero_grad(){
        if (grad_) {
            for (int i = 0; i < size_; ++i) {
                grad_[i] = 0.0f;
            }
        }
    }

    /// 用指定值填充张量的所有元素
    /// @param value 填充值
    void fill(float value);
    
    /// 从文件加载数据
    /// @param filename 文件路径
    /// @return 加载成功返回true，否则返回false
    bool load_from_file(const std::string& filename);
    
    /// 用随机值初始化张量
    /// @param min_val 随机值的最小值
    /// @param max_val 随机值的最大值
    /// @note 常用于初始化网络权重
    void randomize(float min_val, float max_val);
    
    /// 将张量数据导出为二进制文件
    /// @param file_path 输出文件路径
    /// @return 保存成功返回true，否则返回false
    bool save_to_bin(const std::string& file_path) const;

    /// 为张量绑定自动求导信息
    /// 在前向计算时由算子(Layer)调用，动态构建计算图
    /// @param backward_fn 反向传播闭包函数，捕获求导法则
    /// @param prev 前驱节点列表，该张量的输入张量
    /// @note 此方法是自动求导的关键，算子在计算后立即调用此方法
    void set_auto_grad(const std::function<void()>& backward_fn, const std::vector<TensorPtr>& prev);
    
    /// 检查是否为叶子节点
    /// 叶子节点是前驱节点为空的张量(通常是参数或输入)
    /// @return true表示是叶子节点(无前驱节点)
    bool is_leaf() const { return prev_.empty(); }
    
    /// 检查是否为视图(view)
    /// 视图是不拥有数据的张量，如切片或转置
    /// @return true表示是视图
    bool is_view() const { return is_view_; }
    
    /// 获取前驱节点列表
    /// @return 该张量的输入张量(前驱节点)列表
    std::vector<TensorPtr> prev() const { return prev_; }
    
    /// 设置是否为视图
    /// @param is_view true表示这是一个视图，不拥有数据
    void set_view(bool is_view) { is_view_ = is_view; }
    
    /// 执行反向传播
    /// 调用此方法会自动递归执行所有 backward_fn_，计算梯度
    /// @note 仅在输出张量(损失值)上调用
    void backward();
    
    // 禁用拷贝构造和赋值，防止数据共享导致的问题
    Tensor(const Tensor&) = delete;              ///< 删除拷贝构造函数
    Tensor& operator=(const Tensor&) = delete;   ///< 删除赋值操作符
};

#endif // TENSOR_H

/// **算子(Layer)的职责：**
/// 算子只负责前向计算，不包含反向传播函数。
/// 反向传播逻辑完全通过 Tensor::set_auto_grad 动态打包进计算图。
/// 
/// **示例(Linear层的forward)：**
/// 算子在计算完结果后，定义一个闭包来捕获求导法则，
/// 然后调用 set_auto_grad 将其绑定到输出张量。
/// 当调用 backward() 时，这个闭包会被自动执行，
/// 自动地把输出的梯度变成输入的梯度，递归地传播下去。
