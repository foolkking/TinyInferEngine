/*
 * @Author: fool
 * @Date: 2026-04-20 20:35:53
 * @LastEditors: fool
 * @LastEditTime: 2026-04-26 19:56:52
 * @FilePath: \TinyInferEngine\src\flatten.cpp
 * @Description:  
 * @Note:  
 */
 #include "flatten.h"

TensorPtr Flatten::forward(TensorPtr input) {
    int features = 1;
    for (size_t i = 1; i < input->ndims(); ++i) {
        features *= input->shape(i);
    }
    std::vector<int> output_shape = {input->shape(0), features};
    TensorPtr output = std::make_shared<Tensor>(output_shape, input->requires_grad());
    // 物理内存上的数据完全一样，直接拷贝即可
    const float* in_data = input->data();
    float* out_data = output->data();
    for (int i = 0; i < input->size(); ++i) {
        out_data[i] = in_data[i];
    }
    Tensor* out_ptr = output.get(); // 捕获输入张量的智能指针，确保它在闭包中有效
    std::function<void()> backward_fn = [input, out_ptr]() {
        // 展平层没有参数，所以反向传播时直接将输出的梯度传递回输入即可
        const float* grad_out_ptr = out_ptr->grad();
        float* grad_in_ptr = input->grad();
        for (int i = 0; i < out_ptr->size(); ++i) {
            grad_in_ptr[i] += grad_out_ptr[i]; // 累加梯度
        }
    };

    output->set_auto_grad(backward_fn, {input});
    return output;
}
