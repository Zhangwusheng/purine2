// Copyright Lin Min 2015
#ifndef PURINE_Drop_INCEPTION_LAYER
#define PURINE_Drop_INCEPTION_LAYER

#include "composite/layer.hpp"
#include "composite/layers/pool_layer.hpp"
#include "composite/layers/conv_layer.hpp"
#include "composite/layers/concat_layer.hpp"

namespace purine {

    // Inception generates top
    class DropInceptionLayer;
    const vector<Blob*>& operator >> (DropInceptionLayer& inception,
            const vector<Blob*>& top) = delete;

    class DropInceptionLayer : public Layer {
        protected:
            int one;
            int three;
            int five;
            int three_reduce;
            int five_reduce;
            int pool_proj;
            float drop_rate;
            bool test;
        public:
            typedef vector<Blob*> B;
            typedef tuple<int, int, int, int, int, int, float, bool> param_tuple;
            DropInceptionLayer(int rank, int device, const param_tuple& args)
                : Layer(rank, device) {
                    std::tie(one, three, five, three_reduce, five_reduce, pool_proj, drop_rate, test) = args;
                }
            virtual ~DropInceptionLayer() override {}

        protected:
            virtual void setup() override {
                CHECK(bottom_setup_);
                CHECK_EQ(bottom_.size(), 2);
                Size bottom_size = bottom_[0]->tensor()->size();
                string activation = "relu";

                ConvLayer* one_reduce_ = createGraph<ConvLayer>("one",
                        ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, one, activation)); 
                ConvLayer* three_reduce_ = createGraph<ConvLayer>("three_reduce",
                        ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, three_reduce, activation));
                ConvLayer* five_reduce_ = createGraph<ConvLayer>("five_reduce",
                        ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, five_reduce, activation)); 
                PoolLayer* max_pool_ = createGraph<PoolLayer>("max_pool",
                        PoolLayer::param_tuple("max", 3, 3, 1, 1, 1, 1));

                PoolLayer* zero_ = createGraph<PoolLayer>("max_pool",
                        PoolLayer::param_tuple("max", 1, 1, 1, 1, 0, 0));
                ConvLayer* one_ = createGraph<ConvLayer>("one",
                        ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, one, ""));
                ConvLayer* three_ = createGraph<ConvLayer>("three",
                        ConvLayer::param_tuple(1, 1, 1, 1, 3, 3, three, ""));
                ConvLayer* five_ = createGraph<ConvLayer>("five",
                        ConvLayer::param_tuple(2, 2, 1, 1, 5, 5, five, ""));
                ConvLayer* pool_proj_ = createGraph<ConvLayer>("pool_proj",
                        ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, pool_proj, ""));

                ConcatLayer* concat = createGraph<ConcatLayer>("concat",
                        ConcatLayer::param_tuple(Split::CHANNELS));
                ActivationLayer* act = createGraph<ActivationLayer>("act",
                        ActivationLayer::param_tuple(activation, true));

                DropLayer* drop_ = createGraph<DropLayer>("dropLayer", DropLayer::param_tuple(drop_rate, test, false));
                
                bottom_ >> *zero_; 
                bottom_ >> *one_;
                bottom_ >> *three_reduce_ >> *three_;
                bottom_ >> *five_reduce_>> *five_;
                bottom_ >> *max_pool_ >> *pool_proj_;

                vector<Blob*>{ zero_->top()[0], one_->top()[0], three_->top()[0], five_->top()[0], pool_proj_->top()[0],
                    zero_->top()[1], one_->top()[1], three_->top()[1],
                    five_->top()[1], pool_proj_->top()[1] } >> *concat >> *drop_ >> *act;
                top_ = act->top();

                vector<Layer*> layers = { one_reduce_, three_reduce_, five_reduce_,
                    one_, three_, five_, pool_proj_ };
                for (auto layer : layers) {
                    const vector<Blob*>& w = layer->weight_data();
                    weight_.insert(weight_.end(), w.begin(), w.end());
                }
                for (auto layer : layers) {
                    const vector<Blob*>& w = layer->weight_diff();
                    weight_.insert(weight_.end(), w.begin(), w.end());
                }
            }
    };

}

#endif
