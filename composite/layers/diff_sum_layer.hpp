// Copyright Lin Min 2015
#ifndef PURINE_DIFF_SUM_LAYER
#define PURINE_DIFF_SUM_LAYER

#include "composite/layer.hpp"
#include "operations/include/eltwise.hpp"
#include "operations/include/random.hpp"
#include "operations/operation.hpp"

namespace purine {

    class DiffSumLayer: public Layer {
        protected:
        public:
            DiffSumLayer(int rank, int device)
                : Layer(rank, device) {
                }
            virtual ~DiffSumLayer() {}

        protected:
            //typedef tuple<> param_tuple;
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
                    top_ = {
                        create("top", bottom_[0]->shared_tensor()),
                        create("top_sum", bottom_[1]->shared_tensor())
                    };
                }

                Op<WeightedSum>* op_sum = 
                    create<WeightedSum>("weight_sum", "main", WeightedSum::param_tuple({1., 1.}) );
                B{bottom_[0], bottom_[1]} >> *op_sum >> B{top_[1]};
            }
    };

}

#endif
