/*
 * @Author: fool
 * @Date: 2026-04-15 18:30:32
 * @LastEditors: fool
 * @LastEditTime: 2026-04-17 21:28:33
 * @FilePath: \TinyInferEngine\tests\test_tensor.cpp
 * @Description:  
 * @Note:  
 */

#include "tensor.h"
#include "relu.h"
#include "linear.h"
#include "moudle.h"
#include <cassert>
#include <iostream>

int main() {
    std::cout << "--- Running Tensor Tests ---" << std::endl;

    // 模拟一个 Batch=2, Channel=3, Height=224, Width=224 的图像张量
    int shape[] = {2, 3, 224, 224};
    Tensor t(shape, 4);

    t.print_info();

    // 验证内存大小是否计算正确 (2 * 3 * 224 * 224 = 301056)
    assert(t.size() == 301056);
    // 验证通道数维度是否正确
    assert(t.shape(1) == 3);

    std::cout << "--- All Tensor Init Tests Passed! ---" << std::endl;
    // --- Step 2 测试：数据写入与访问 ---
    std::cout << "--- Running Step 2 Tests ---" << std::endl;
    t.fill(1.5f); // 填充 1.5
    
    // 验证起始指针是否有值
    assert(t.data() != nullptr);
    // 验证第一个元素和最后一个元素是否都被正确填充
    assert(t.data()[0] == 1.5f);
    assert(t.data()[t.size() - 1] == 1.5f);
    
    std::cout << "--- Step 2 Tests Passed! ---" << std::endl;
    // --- Step 3 测试：多维坐标寻址 ---
    std::cout << "--- Running Step 3 Tests ---" << std::endl;
    
    // 假设我们有一个 2x3 的二维矩阵 (2行3列)
    int shape2d[] = {2, 3};
    Tensor t2d(shape2d, 2);
    t2d.fill(0.0f); // 先全部清零

    // 修改坐标为 (1, 2) 的元素，即第2行第3列 (索引从0开始)
    int indices[] = {1, 2};
    t2d.at(indices) = 9.9f;

    // 验证底层一维数组是否正确被修改
    // 偏移量推导: index = 1 * stride[0] + 2 * stride[1] = 1 * 3 + 2 * 1 = 5
    assert(t2d.data()[5] == 9.9f);
    
    std::cout << "--- Step 3 Tests Passed! Tensor is fully operational! ---" << std::endl;
    // --- Step 4/5 测试：解耦后的 ReLU 算子 ---
    std::cout << "--- Running Step 4/5 Tests (ReLU Layer) ---" << std::endl;
    
    int shape_relu[] = {4};
    Tensor t_relu(shape_relu, 1);
    
    // 手动塞入一些正数和负数
    t_relu.data()[0] = 3.14f;
    t_relu.data()[1] = -2.5f;
    t_relu.data()[2] = 0.0f;
    t_relu.data()[3] = -9.9f;

    // 实例化 ReLU 算子并执行前向传播
    ReLU relu_layer;
    relu_layer.forward(t_relu, t_relu); // 直接在原地修改 t_relu 的数据

    // 验证：负数必须变 0，正数和 0 保持不变
    assert(t_relu.data()[0] == 3.14f);
    assert(t_relu.data()[1] == 0.0f);
    assert(t_relu.data()[2] == 0.0f);
    assert(t_relu.data()[3] == 0.0f);

    std::cout << "--- Step 4/5 Tests Passed! Architecture is clean! ---" << std::endl;
    std::cout << "--- Running Step 6 Tests (Linear Layer) ---" << std::endl;
    
    // 假设输入特征是 2 维，输出特征是 3 维
    // 输入 X: [1, 2] (Batch=1)
    int in_shape[] = {1, 2};
    Tensor x(in_shape, 2);
    x.data()[0] = 1.0f; x.data()[1] = 2.0f;

    // 创建 Linear 层
    Linear linear(2, 3);
    
    // 手动设置权重 W (形状 [3, 2]):
    // [[1, 1],
    //  [2, 2],
    //  [3, 3]]
    float* w_ptr = linear.weight()->data();
    w_ptr[0] = 1.0f; w_ptr[1] = 1.0f;
    w_ptr[2] = 2.0f; w_ptr[3] = 2.0f;
    w_ptr[4] = 3.0f; w_ptr[5] = 3.0f;

    // 手动设置偏置 B (形状 [3]): [0.1, 0.2, 0.3]
    float* b_ptr = linear.bias()->data();
    b_ptr[0] = 0.1f; b_ptr[1] = 0.2f; b_ptr[2] = 0.3f;

    // 准备输出 Tensor (形状 [1, 3])
    int out_shape[] = {1, 3};
    Tensor y(out_shape, 2);

    // 执行前向推理
    linear.forward(x, y);

    // 验证数学结果
    // y_0 = (1*1 + 2*1) + 0.1 = 3.1
    // y_1 = (1*2 + 2*2) + 0.2 = 6.2
    // y_2 = (1*3 + 2*3) + 0.3 = 9.3
    // 注意：浮点数比较通常需要一个极小的 epsilon，这里为了简单直接比较
    assert(std::abs(y.data()[0] - 3.1f) < 1e-5);
    assert(std::abs(y.data()[1] - 6.2f) < 1e-5);
    assert(std::abs(y.data()[2] - 9.3f) < 1e-5);

    std::cout << "--- Step 6 Tests Passed! Math is solid! ---" << std::endl;


    Sequential model = Sequential();
    Linear* linear1 = new Linear(64, 128);
    linear1->weight()->fill(0.001f); // 初始化权重
    linear1->bias()->fill(0.001f); // 初始化偏置    
    ReLU* relu1 = new ReLU();     
    Linear* linear2 = new Linear(128, 10);
    linear2->weight()->fill(0.001f); // 初始化权重
    linear2->bias()->fill(0.001f); // 初始化偏置
    model.add(linear1);
    model.add(relu1);
    model.add(linear2); // 最后输出 10 类的概率分布
    int input_shape[] = {1, 64};
    Tensor* input = new Tensor(input_shape, 2); // Batch=1, Features=64
    input->fill(1.0f); // 输入全 1
    Tensor* output = model.forward(input);
    output->print_info(); // 输出的形状应该是 [1, 10]
    for(int i = 0; i < output->size(); ++i) {
        std::cout << "Output[" << i << "] = " << output->data()[i] << std::endl;
    }

    return 0; // 退出时会自动调用析构函数，如果没有段错误(Segmentation Fault)，说明内存释放成功
}