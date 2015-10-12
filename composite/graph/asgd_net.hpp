// Copyright Lin Min 2015
#ifndef PURINE_ASGD_NET_
#define PURINE_ASGD_NET_

#include <vector>
#include <utility>
#include <thread>
#include "dispatch/runnable.hpp"
#include "dispatch/op.hpp"
#include "composite/composite.hpp"

using namespace std;

namespace purine {
    
    template <typename Net>
    class asgd_net: public Runnable {
        protected:
            std::vector<Blob*> weight_diff_sum_;
            std::vector<Blob*> data_;
            std::vector<Blob*> labels_;
            int weight_diff_count_;
            int rank_;
            int device_;
            int batch_;
            Net* net_;
        public:
            asgd_net(int rank, int device, int batch);
            virtual ~asgd_net() override {}
        public:
            void feed(const std::vector<Blob*> data, const std::vector<Blob*> labels);
            void add_weight_diff_sum();
            std::vector<Blob*>& get_weight_diff();
            std::vector<Blob*> loss(){ return net_->loss();}
            inline Net* net() { return net_; }
            inline int rank(){return rank_;}
            inline int device(){return device_;}
    };

    template <typename Net>
    void asgd_net<Net>::feed(const std::vector<Blob*> data, const std::vector<Blob*> labels){
        CHECK_EQ(data.size(), data_.size());
        CHECK_EQ(labels.size(), labels_.size());
        for (int i = 0; i < data.size(); ++i) {
            if (current_rank() == data_[i]->rank()) {
                data_[i]->tensor()->swap_memory(data[i]->tensor());
            }
        }

        for (int i = 0; i < labels.size(); ++i) {
            if (current_rank() == labels_[i]->rank()) {
                labels_[i]->tensor()->swap_memory(labels[i]->tensor());
            }
        }
    }

    template <typename Net>
        asgd_net<Net>::asgd_net(int rank, int device, int batch): rank_(rank), device_(device), batch_(batch){
            net_ = createGraph<Net>("replica" + to_string(rank_) + " " + to_string(device_),
                    rank_, device_, batch_);

            const vector<Blob*>& dt = net_->data();
            const vector<Blob*>& lb = net_->label();
            data_.insert(data_.end(), dt.begin(), dt.end());
            labels_.insert(labels_.end(), lb.begin(), lb.end());

            std::vector<Blob*> weight_diff = net_->weight_diff();
            Runnable filler(rank_, device_);
            for(int i = 0; i < net_->weight_diff().size(); i++){
                Blob* diff = create("weight_diff_sum_", rank_, device_, 
                        weight_diff[i]->shared_tensor()->size());

                /*申请新的weight_diff_sum_*/
                weight_diff_sum_ .push_back(diff);
                Blob* to_fill = filler.create("weight_diff_sum", diff->shared_tensor());
                *filler.create<Constant>("fill_weight_diff_sum", "main", Constant::param_tuple(0.))
                    >> vector<Blob*>{ to_fill };
            }
            filler.run();
        }

    template <typename Net>
        std::vector<Blob*>& asgd_net<Net>::get_weight_diff()
        {
            return weight_diff_sum_;
        }

    template<typename Net>
        void asgd_net<Net>::add_weight_diff_sum(){
            Runnable apply_sum_run(rank_, device_);
            std::vector<Blob*> weight_diff = net_->weight_diff();
            for(int i = 0; i < weight_diff_sum_.size(); i++){
                Blob* s1 = apply_sum_run.create("diff_s1", weight_diff_sum_[i]->shared_tensor());
                Blob* s2 = apply_sum_run.create("diff_s2", weight_diff[i]->shared_tensor());
                Op<WeightedSum>* op_sum = 
                    apply_sum_run.create<WeightedSum>("weight_sum", "main", WeightedSum::param_tuple({1., 1.}) );
                std::vector<Blob*>{s1, s2} >> *op_sum
                    >>std::vector<Blob*>{s1};
            }
            apply_sum_run.run();
        }
}
#endif
