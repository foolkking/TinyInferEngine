#include "tensor.h"
#include "layer.h"
#include "model.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

int main() {
    {
        std::vector<int> shape = {2, 3, 224, 224};
        auto t = std::make_shared<Tensor>(shape, false);
        assert(t->size() == 2 * 3 * 224 * 224);
        assert(t->shape(1) == 3);

        t->fill(1.5f);
        assert(t->data()[0] == 1.5f);
        assert(t->data()[t->size() - 1] == 1.5f);
        std::cout << "tensor init/fill OK\n";
    }

    {
        std::vector<int> shape2d = {2, 3};
        auto t2d = std::make_shared<Tensor>(shape2d, false);
        t2d->zeros();
        int idx[2] = {1, 2};
        t2d->at(idx) = 9.9f;
        assert(std::abs(t2d->data()[5] - 9.9f) < 1e-5f);
        std::cout << "tensor at() OK\n";
    }

    {
        std::vector<int> sh = {4};
        auto x = std::make_shared<Tensor>(sh, false);
        x->data()[0] = 3.14f;
        x->data()[1] = -2.5f;
        x->data()[2] = 0.0f;
        x->data()[3] = -9.9f;
        ReLU relu_layer;
        auto y = relu_layer.forward(x);
        assert(std::abs(y->data()[0] - 3.14f) < 1e-5f);
        assert(std::abs(y->data()[1]) < 1e-5f);
        assert(std::abs(y->data()[2]) < 1e-5f);
        assert(std::abs(y->data()[3]) < 1e-5f);
        std::cout << "ReLU OK\n";
    }

    {
        auto linear = std::make_shared<Linear>(2, 3, false);
        float* w_ptr = linear->weight()->data();
        w_ptr[0] = 1.0f;
        w_ptr[1] = 1.0f;
        w_ptr[2] = 2.0f;
        w_ptr[3] = 2.0f;
        w_ptr[4] = 3.0f;
        w_ptr[5] = 3.0f;
        float* b_ptr = linear->bias()->data();
        b_ptr[0] = 0.1f;
        b_ptr[1] = 0.2f;
        b_ptr[2] = 0.3f;

        std::vector<int> in_shape = {1, 2};
        auto x = std::make_shared<Tensor>(in_shape, false);
        x->data()[0] = 1.0f;
        x->data()[1] = 2.0f;

        auto y = linear->forward(x);
        assert(std::abs(y->data()[0] - 3.1f) < 1e-4f);
        assert(std::abs(y->data()[1] - 6.2f) < 1e-4f);
        assert(std::abs(y->data()[2] - 9.3f) < 1e-4f);
        std::cout << "Linear OK\n";
    }

    {
        Sequential model;
        model.add(std::make_shared<Linear>(64, 128, false));
        model.add(std::make_shared<ReLU>());
        model.add(std::make_shared<Linear>(128, 10, false));
        for (const auto& p : model.named_parameters()) {
            p.tensor->fill(0.001f);
        }
        std::vector<int> in_shape = {1, 64};
        auto input = std::make_shared<Tensor>(in_shape, false);
        input->fill(1.0f);
        auto output = model.forward(input);
        assert(output->shape(0) == 1);
        assert(output->shape(1) == 10);
        std::cout << "Sequential OK\n";
    }

    std::cout << "All tests passed.\n";
    return 0;
}
