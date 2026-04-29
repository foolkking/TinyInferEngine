# TinyInferEngine

一个轻量级的 C++ 深度学习推理框架，支持动态计算图和自动求导。

**语言**: 中文 | [English](README_EN.md)

## 📋 目录

- [项目简介](#项目简介)
- [主要特性](#主要特性)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [构建和运行](#构建和运行)
- [API 文档](#api-文档)
- [示例](#示例)
- [架构设计](#架构设计)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 项目简介

TinyInferEngine 是一个从零开始构建的 C++ 深度学习框架，专注于：

- **动态计算图**：前向计算时动态构建，支持灵活的网络结构
- **自动求导(Autograd)**：通过动态闭包实现梯度反向传播
- **纯粹的架构设计**：Layer 是"造图工厂"，专注前向计算；反向逻辑完全封装在 Tensor 的闭包中
- **零依赖实现**：仅使用标准库，无需外部深度学习框架

适合用于：
- 学习深度学习框架的内部原理
- 轻量级推理应用
- 教学和研究

## 主要特性

### 核心功能

| 功能 | 说明 |
|------|------|
| **Tensor** | 多维数组，支持自动求导的计算图节点 |
| **自动求导** | 通过动态闭包实现反向传播，无需显式定义反向函数 |
| **Layer 基类** | 标准神经网络层接口，包含前向计算和参数管理 |
| **优化器** | SGD、AdamW 等多种优化算法 |
| **学习率调度** | StepLR、CosineAnnealingLR 等调度策略 |
| **损失函数** | CrossEntropyLoss、MSELoss 等 |

### 支持的层类型

- **Linear** - 全连接层
- **Conv2D** - 2D 卷积层
- **MaxPool2D** - 最大池化层
- **Flatten** - 展平层
- **ReLU** - ReLU 激活函数
- **SiLU** - SiLU 激活函数

### 优化器

- **SGD** - 随机梯度下降，支持权重衰减
- **AdamW** - Adam 优化器，带 decoupled weight decay

### 学习率调度

- **StepLR** - 每隔固定步数衰减学习率
- **CosineAnnealingLR** - 余弦退火学习率调度

## 项目结构

```
TinyInferEngine/
├── CMakeLists.txt              # CMake 构建配置
├── README.md                   # 本文件
├── include/                    # 头文件目录
│   ├── tensor.h               # 核心张量类（含Autograd）
│   ├── layer.h                # 基础层和所有层实现
│   ├── model.h                # Sequential 顺序模型容器
│   ├── optimizer.h            # 优化器基类和实现
│   ├── scheduler.h            # 学习率调度器
│   ├── loss.h                 # 损失函数
│   ├── conv2d.h               # 卷积层
│   ├── maxpool2d.h            # 池化层
│   ├── flatten.h              # 展平层
│   ├── relu.h                 # ReLU 激活
│   └── silu.h                 # SiLU 激活
├── src/                        # 源文件目录
│   ├── main.cpp               # 推理引擎主程序
│   ├── tensor.cpp             # 张量实现
│   ├── layer.cpp              # 层实现
│   ├── model.cpp              # Sequential 实现
│   ├── optimizer.cpp          # 优化器实现
│   ├── scheduler.cpp          # 调度器实现
│   ├── loss.cpp               # 损失函数实现
│   ├── conv2d.cpp             # 卷积层实现
│   ├── maxpool2d.cpp          # 池化层实现
│   ├── flatten.cpp            # 展平层实现
│   ├── relu.cpp               # ReLU 实现
│   ├── silu.cpp               # SiLU 实现
│   ├── train_minist.cpp       # MNIST 训练脚本
│   └── export_model.py        # 模型导出工具
├── tests/                      # 测试目录
│   └── test_tensor.cpp        # 张量功能测试
├── data/                       # 数据目录
│   └── MNIST/                 # MNIST 数据集
├── weights/                    # 预训练权重目录
└── build/                      # CMake 构建输出目录

```

## 快速开始

### 环境要求

- **C++ 标准**: C++17 或更高
- **编译器**: GCC 7.0+、Clang 5.0+、MSVC 2017+
- **构建工具**: CMake 3.10+
- **操作系统**: Windows、Linux、macOS

### 安装依赖

本项目无第三方依赖，仅使用 C++ 标准库。

### 克隆项目

```bash
git clone <repository-url>
cd TinyInferEngine
```

## 构建和运行

### 使用 CMake 构建

```bash
# 创建构建目录
mkdir build
cd build

# 配置和编译
cmake ..
cmake --build . --config Release

# 运行推理引擎
.\build\Debug\infer_engine.exe

# 运行 MNIST 训练
.\build\Debug\train_minist.exe
```

### 构建目标

| 目标 | 说明 |
|------|------|
| `infer_engine` | 推理引擎可执行文件 |
| `test_tensor` | 张量功能单元测试 |
| `train_minist` | MNIST 模型训练脚本 |
| `core_lib` | 核心库（静态库） |

## API 文档

所有头文件都包含详细的 /// 格式文档注释，支持 IDE 的智能提示。

### 核心类

#### Tensor（张量）

```cpp
#include "tensor.h"

// 创建张量
std::vector<int> shape = {2, 3, 4};
auto tensor = std::make_shared<Tensor>(shape, true);  // requires_grad=true

// 数据操作
tensor->fill(1.0f);                    // 填充值
tensor->randomize(-1.0f, 1.0f);        // 随机初始化
tensor->zero_grad();                   // 清零梯度

// 自动求导
tensor->backward();                    // 反向传播

// 数据访问
float* data = tensor->data();
int size = tensor->size();
int ndims = tensor->ndims();
```

#### Layer（层）

```cpp
#include "layer.h"

// 创建线性层
auto linear = std::make_shared<Linear>(784, 128);
auto output = linear->forward(input);

// 获取参数
auto params = linear->parameters();
```

#### Sequential（模型）

```cpp
#include "model.h"

Sequential model;
model.add(std::make_shared<Linear>(784, 128));
model.add(std::make_shared<ReLU>());
model.add(std::make_shared<Linear>(128, 10));

auto output = model.forward(input);
```

#### Optimizer（优化器）

```cpp
#include "optimizer.h"

// 创建 SGD 优化器
auto optimizer = std::make_shared<SGD>(model.named_parameters(), 0.01f);

// 训练循环
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    optimizer->zero_grad();           // 清零梯度
    auto pred = model.forward(input); // 前向传播
    auto loss = loss_fn(pred, target);// 计算损失
    loss->backward();                 // 反向传播
    optimizer->step();                // 参数更新
}
```

#### Scheduler（学习率调度）

```cpp
#include "scheduler.h"

auto scheduler = std::make_shared<StepLR>(optimizer, 10, 0.1f);

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    // ... 训练代码 ...
    scheduler->step();  // 更新学习率
}
```

## 示例

### 简单的 MNIST 分类器

```cpp
#include "model.h"
#include "optimizer.h"
#include "loss.h"
#include "scheduler.h"
#include <iostream>

int main() {
    // 构建模型
    Sequential model;
    model.add(std::make_shared<Flatten>());
    model.add(std::make_shared<Linear>(784, 128));
    model.add(std::make_shared<ReLU>());
    model.add(std::make_shared<Linear>(128, 10));
    
    // 创建优化器和损失函数
    auto optimizer = std::make_shared<SGD>(model.named_parameters(), 0.01f);
    auto loss_fn = CrossEntropyLoss();
    
    // 创建学习率调度器
    auto scheduler = std::make_shared<StepLR>(optimizer, 10, 0.1f);
    
    // 加载数据（伪代码）
    auto train_images = LoadMNIST("train-images");
    auto train_labels = LoadMNIST("train-labels");
    
    // 训练循环
    int num_epochs = 20;
    int batch_size = 32;
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float total_loss = 0.0f;
        int correct = 0;
        
        for (int batch = 0; batch < num_batches; ++batch) {
            // 获取批次数据
            auto [batch_images, batch_labels] = GetBatch(batch);
            
            // 前向传播
            optimizer->zero_grad();
            auto pred = model.forward(batch_images);
            auto loss = loss_fn.forward(pred, batch_labels);
            
            // 反向传播
            loss->backward();
            optimizer->step();
            
            total_loss += loss->data()[0];
            // 计算准确率...
        }
        
        scheduler->step();  // 更新学习率
        
        std::cout << "Epoch " << epoch << ": Loss=" << total_loss 
                  << ", Accuracy=" << (correct * 100.0 / num_samples) << "%\n";
    }
    
    return 0;
}
```

### 张量操作示例

```cpp
#include "tensor.h"
#include <iostream>

int main() {
    // 创建张量
    auto x = std::make_shared<Tensor>(std::vector<int>{2, 3}, true);
    x->fill(2.0f);
    
    // 访问元素
    float* data = x->data();
    std::cout << "张量大小: " << x->size() << "\n";
    std::cout << "张量形状: [" << x->shape(0) << ", " << x->shape(1) << "]\n";
    
    // 梯度操作
    auto grad = x->grad();
    if (grad) {
        for (int i = 0; i < x->size(); ++i) {
            grad[i] += 1.0f;
        }
    }
    
    return 0;
}
```

## 架构设计

### 核心设计理念

#### 1. "造图工厂"模式（Graph Factory Pattern）

```
Layer 层 → 纯粹的前向计算函数
    ↓
    生成闭包（Lambda），打包求导法则
    ↓
Result Tensor ← 绑定闭包和前驱节点
    ↓
backward() 调用 ← 自动递归执行所有闭包
```

Layer 只负责：
- 创建输出张量
- 定义求导闭包
- 调用 `set_auto_grad()` 绑定计算图

**优势**：
- 代码清晰，关注点分离
- 易于扩展新的操作
- 自动求导逻辑集中在 Tensor 类

#### 2. 动态计算图（Dynamic Computation Graph）

与静态图框架不同，TinyInferEngine 的计算图在**前向传播时动态构建**：

```cpp
// 每次调用 forward() 都构建新的计算图
auto output = model.forward(input);  // 图构建
output->backward();                  // 沿图反向传播
```

**优势**：
- 支持条件分支和循环
- 调试更容易（实际执行路径明确）
- 灵活支持变长序列

#### 3. 智能指针和自动内存管理

使用 `std::shared_ptr<Tensor>` 实现自动生命周期管理：

```cpp
// 不需要手动 delete
auto tensor = std::make_shared<Tensor>(...);
// 当没有引用时自动释放
```

#### 4. 梯度流和反向传播

```cpp
// 前驱节点 X → Layer → Result
// Result.prev_ = {X}
// Result.backward_fn_ = [X的梯度更新法则]

Result.backward()
    → 执行 Result.backward_fn_
        → 更新 X.grad_
        → X.backward()（如果X不是叶子）
            → 递归继续...
```

## 训练流程

```
┌─────────────────────────────────┐
│ 初始化模型、优化器、损失函数      │
└────────────────┬────────────────┘
                 │
         ┌───────▼────────┐
         │ for each epoch │
         └───────┬────────┘
                 │
    ┌────────────▼─────────────┐
    │ for each batch:          │
    │  1. zero_grad()          │
    │  2. forward()            │ ← 构建计算图
    │  3. loss = loss_fn()     │
    │  4. backward()           │ ← 沿图反向传播
    │  5. optimizer.step()     │ ← 参数更新
    └────────────┬─────────────┘
                 │
    ┌────────────▼─────────────┐
    │ scheduler.step()         │
    │ （更新学习率）            │
    └────────────┬─────────────┘
                 │
         ┌───────▼────────┐
         │ 保存模型权重    │
         └────────────────┘
```

## 编码约定

### 命名规范

- **类名**: PascalCase（如 `Linear`、`ReLU`）
- **函数/方法**: snake_case（如 `forward()`、`zero_grad()`）
- **私有成员**: snake_case 加 `_` 后缀（如 `data_`、`shape_`）
- **常量**: UPPER_SNAKE_CASE（如 `MAX_SIZE`）

### 文档注释

所有公共接口使用 `///` 格式的 Doxygen 风格注释：

```cpp
/// 简短的单行描述
/// 
/// 详细的多行说明，可以跨越多个段落
/// 
/// @param name 参数说明
/// @return 返回值说明
/// @note 重要备注
/// @see 相关参考
void function(int name);
```

## 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发步骤

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

### 代码风格

- 遵循项目现有的命名约定
- 为新功能添加 `///` 格式文档注释
- 确保代码编译无警告
- 为新功能编写单元测试

## 已知限制

- 当前仅支持浮点32位(float)数据类型
- 不支持 GPU 计算
- 张量维度数有限制（通常 ≤ 10D）
- 部分操作仅支持连续张量

## 常见问题 (FAQ)

### Q: 为什么没有使用现有的深度学习框架（如 PyTorch、TensorFlow）？

A: 这个项目是教学和学习用途，目标是深入理解框架内部原理。使用现有框架会隐藏许多实现细节。

### Q: 性能如何？

A: TinyInferEngine 优先考虑代码清晰度和学习价值，而非性能。对于生产环境，建议使用优化的框架。

### Q: 可以在生产环境中使用吗？

A: 可以，但仅适合轻量级应用和推理任务。建议用于边缘设备或嵌入式系统。

### Q: 如何添加自己的 Layer？

A: 继承 `Layer` 基类，实现 `forward()` 和 `parameters()` 方法，在 `forward()` 中定义反向函数并调用 `set_auto_grad()`。



## 联系方式

- 问题提交: GitHub Issues
- 讨论: GitHub Discussions

---

**最后更新**: 2026-04-29
