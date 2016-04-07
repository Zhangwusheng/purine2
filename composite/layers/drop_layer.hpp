// Copyright Lin Min 2015
#ifndef PURINE_DROP_LAYER
#define PURINE_DROP_LAYER

#include "composite/layer.hpp"
#include "operations/include/drop.hpp"
#include "operations/include/eltwise.hpp"
#include "operations/include/random.hpp"
#include <stack>

namespace purine {

    class DropLayer : public Layer {
        protected:
            DTYPE ratio;
            bool test;
            bool inplace;
        public:
            typedef tuple<DTYPE, bool, bool> param_tuple;
            DropLayer(int rank, int device, const param_tuple& args)
                : Layer(rank, device) {
                    std::tie(ratio, test, inplace) = args;
                }
            virtual ~DropLayer() {}
        protected:
            virtual void setup() override {
                CHECK(bottom_setup_);
                CHECK_EQ(bottom_.size(), 2);
                Size bottom_size = bottom_[0]->tensor()->size();

                // check top
                if (top_.size() != 0) {
                    CHECK_EQ(top_.size(), 2);
                    for (auto top : top_) {
                        CHECK_EQ(top->tensor()->size(), bottom_size);
                    }
                } else {
                    if (!inplace) {
                        top_ = {
                            create("top", bottom_size),
                            create("top_diff", bottom_size)
                        };
                    } else {
                        top_ = {
                            create("top", bottom_[1]->shared_tensor()),
                            create("top_diff", bottom_[1]->shared_tensor())
                        };
                    }
                }
                float* dropVector = new float[bottom_size.num() * bottom_size.channels()];
                Op<Drop>* activation_up = create<Drop>("Drop", "main",
                        Drop::param_tuple(ratio, dropVector, test));
                Op<DropDown>* activation_down = create<DropDown>(
                        "Drop_down", "main", DropDown::param_tuple(ratio, dropVector));

                // forward
                B{ bottom_[0] } >> *activation_up >> B{ top_[0] };
                // backward
                B{ top_[1], top_[0], bottom_[0] } >> *activation_down >> B{ bottom_[1] };
            }
    };

}

#endif
