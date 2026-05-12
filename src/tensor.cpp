/*
 * @Author: fool
 * @Date: 2026-04-15 18:30:24
 * @LastEditors: fool
 * @LastEditTime: 2026-04-29 20:20:08
 * @FilePath: \TinyInferEngine\src\tensor.cpp
 * @Description:  
 * @Note:  
 */

#include "tensor.h"
#include <fstream>
#include <cstdlib> // for rand() and srand()
#include <ctime>   // for time()
#include <algorithm>
// 注意这里要加上 Tensor:: 作用域解析符
Tensor::Tensor(const std::vector<int>& shape, bool requires_grad) {
    shape_ = shape;
    strides_ = new int[ndims()]; // 分配步长数组的内存
    requires_grad_ = requires_grad;
    size_ = 1;
    // 拷贝形状并计算总大小
    for (int i = 0; i < ndims(); ++i) {
        shape_[i] = shape[i];
        size_ *= shape[i];
    }
    
    data_ = new float[size_];
    if (requires_grad_) {
        grad_ = new float[size_];
        zero_grad(); // 刚分配完立刻清零，防止里面是内存垃圾
    } 
    else {
        grad_ = nullptr;
    }
    // 【核心优化】：从后往前，一次性计算并缓存所有维度的步长
    int current_stride = 1;
    for (int i = ndims() - 1; i >= 0; --i) {
        strides_[i] = current_stride;
        current_stride *= shape_[i];
    }
}
    
Tensor::~Tensor() {
    delete[] data_;  
    shape_.clear();
    delete[] strides_; // 记得释放 strides_ 内存
    if (requires_grad_ && grad_) {
        delete[] grad_;
    }
}

int Tensor::size() const { return size_; }

int Tensor::shape(int index) const {
    // 可以在这里加一个简单的越界检查，保证引擎的鲁棒性
    if (index < 0 || index >= ndims()) {
        std::cerr << "Error: Dimension index out of bounds!" << std::endl;
        return -1;
    }
    return shape_[index];
}

void Tensor::print_info() const {
    std::cout << "Tensor Info: " << std::endl;
    std::cout << "Dimensions: " << ndims() << std::endl;
    std::cout << "Shape: [";
    for (int i = 0; i < ndims(); ++i) {
        std::cout << shape_[i];
        if (i < ndims() - 1) std::cout << ", ";
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
float Tensor::data(int index) {
    return data_[index];

}

const float Tensor::data(int index) const {
    return data_[index];
}


void Tensor::fill(float value) {
    for (int i = 0; i < size_; ++i) {
        data_[i] = value;
    }
}

int Tensor::stride(int index) const {
    if (index < 0 || index >= ndims()) {
        std::cerr << "Error: Dimension index out of bounds!" << std::endl;
        return -1;
    }
    return strides_[index];
}

float& Tensor::at(const int* indices) {
    int offset = 0;
    for (int i = 0; i < ndims(); ++i) {
        offset += indices[i] * stride(i);
    }
    return data_[offset];
}

const float& Tensor::at(const int* indices) const {
    int offset = 0;
    for (int i = 0; i < ndims(); ++i) {
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


void Tensor::randomize(float min_val, float max_val) {
    for (int i = 0; i < size_; ++i) {
        data_[i] = min_val + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_val - min_val)));
    }
}

bool Tensor::save_to_bin(const std::string& file_path) const {
    std::ofstream outfile(file_path, std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << file_path << std::endl;
        return false;
    }
    outfile.write(reinterpret_cast<const char*>(data_), size_ * sizeof(float));
    outfile.close();
    return true;
}

void Tensor::set_auto_grad(const std::function<void()>& backward_fn, const std::vector<TensorPtr>& prev) {
    backward_fn_ = backward_fn;
    prev_ = prev;
}
static void build_topo(const TensorPtr& node, std::vector<TensorPtr>& topo) {
    if (!node->is_view()) {
        node->set_view(true); 
        for (const auto& prev_node : node->prev()) {
            build_topo(prev_node, topo);
        }
        topo.push_back(node);
    }

}

void Tensor::backward() {
    std::vector<TensorPtr> topo;
    build_topo(shared_from_this(), topo);
    std::reverse(topo.begin(), topo.end()); // 反转拓扑排序结果，确保从叶子节点开始反向传播
    for (int i= 0;i<this->size();++i){  // 反向传播的起点是输出节点，输出节点的梯度初始化为1，表示 d(output)/d(output) = 1
        if (this->grad_){
            this->grad_[i] = 1.0f;
        }
    }
    for (const auto& node : topo) {
        if (node->backward_fn_) {
            node->backward_fn_(); // 调用每个节点的反向传播函数
        }
    }
    for(const auto& node : topo){
        node->set_view(false); // 反向传播结束后重置视图标志，允许下次反向传播重新构建计算图
    }
}


TensorPtr Tensor::add(const TensorPtr& a, const TensorPtr& b){
    if(a->size()!=b->size()){
        throw std::runtime_error("[FATAL] add_tensors: Shape mismatch!");
    }
    int total_elem = a->size();
    bool out_req_grad = a->requires_grad()||b->requires_grad();
    TensorPtr output= std::make_shared<Tensor>(a->shape(),out_req_grad);
    float* output_data = output->data();
    float* a_data = a->data();
    float* b_data = b->data();
    for(int i =0;i<total_elem;++i){
        output_data[i] = a_data[i]+b_data[i];
    }
    if(out_req_grad){
        Tensor* output_ptr = output.get();
        std::function<void()> backward_fn = [output_ptr,a,b,total_elem](){
            float*a_grad,*b_grad;
            if(a->requires_grad())float* a_grad = a->grad();
            if(b->requires_grad())float* b_grad = b->grad(); 
            float* output_grad = output_ptr->grad();
            for(int i=0;i<total_elem;i++){
                if(a->requires_grad())a_grad[i]+= output_grad[i];
                if(b->requires_grad())b_grad[i]+= output_grad[i];
            }
        };
        output->set_auto_grad(backward_fn,{a,b});
    }
    return output;
}

