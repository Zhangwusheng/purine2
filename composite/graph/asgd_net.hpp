// Copyright Lin Min 2015
#ifndef PURINE_ASGD_NET_
#define PURINE_ASGD_NET_

#include <vector>
#include <utility>
#include <thread>
#include "dispatch/runnable.hpp"
#include "dispatch/op.hpp"
#include "composite/composite.hpp"
#include "common/common.hpp"

using namespace std;

namespace purine {
    
    template <typename Net>
    class asgd_net: public Runnable {
        protected:
            std::vector<Blob*> weight_diff_sum_;
            std::vector<Blob*> data_;
            std::vector<Blob*> labels_;
            Blob* weight_diff_count_;
            Runnable* father;   
            int rank_;
            int device_;
            int batch_;
            Net* net_;
            double period;
        public:
            asgd_net(int rank, int device, int batch);
            virtual ~asgd_net() override {
            }
        public:
            void feed();
            std::vector<Blob*> loss(){ return net_->loss();}
            inline Net* net() { return net_; }
            inline int rank(){return rank_;}
            inline int device(){return device_;}
            inline Blob* get_weight_diff_count(){return weight_diff_count_;}
            void clear_weight_diff();
            inline std::vector<Blob*>& get_weight_diff(){return weight_diff_sum_;}
            virtual void sync() override;
            virtual void run_async() override;
            inline void set_period(double p){ 
                period = p;
            }
            virtual void run() override;
    };

    template <typename Net>
    void asgd_net<Net>::feed(){
        const std::vector<Blob*>data = net_->fetch()->images();
        const std::vector<Blob*>labels = net_->fetch()->labels();
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
            period = 0.1;
            net_ = createGraph<Net>("replica" + to_string(rank_) + " " + to_string(device_),
                    rank_, device_, batch_);
            const vector<Blob*>& data_diff = net_->data_diff();
            vector<Node*> to_prune(data_diff.size());
            transform(data_diff.begin(), data_diff.end(), to_prune.begin(),
                    [](Blob* b)->Node* {
                    return dynamic_cast<Node*>(b);
                    });
            net_->prune(to_prune);
            // get the data and labels
            const vector<Blob*>& dt = net_->data();
            const vector<Blob*>& lb = net_->label();
            data_.insert(data_.end(), dt.begin(), dt.end());
            labels_.insert(labels_.end(), lb.begin(), lb.end());

            std::vector<Blob*> weight_diff = net_->weight_diff();
            Runnable filler(rank_, device_);
            weight_diff_sum_.clear();
            for(int i = 0; i < net_->weight_diff_sum().size(); i++){
                weight_diff_sum_.push_back(create("weight_diff_sum_", net_->weight_diff_sum()[i]->shared_tensor()));
                Blob* diff = filler.create("weight_diff_sum", weight_diff_sum_[i]->shared_tensor());
                Blob* to_fill = filler.create("weight_diff_sum", diff->shared_tensor());
                *filler.create<Constant>("fill_weight_diff_sum", "main", Constant::param_tuple(0.))
                    >> vector<Blob*>{ to_fill };
            }
            weight_diff_count_ = create("count", net_->diff_sum_count()->shared_tensor());
            Blob* weight_diff_count_output_ = filler.create("count_output", weight_diff_count_->shared_tensor());
            *filler.create<Constant>("fill_weight_diff_count", rank_, -1,  "main", Constant::param_tuple(0.))
                >> vector<Blob*>{weight_diff_count_output_};

            filler.run();
        }


    template<typename Net>
        void asgd_net<Net>::clear_weight_diff(){
            if(current_rank() == rank_){
                Runnable filler(rank_, device_);
                for(int i = 0; i < weight_diff_sum_.size(); i++){
                    Blob* to_fill = filler.create("weight_diff_sum", weight_diff_sum_[i]->shared_tensor());
                    *filler.create<Constant>("fill_weight_diff_sum", "main", Constant::param_tuple(0.))
                        >> vector<Blob*>{ to_fill };
                }

                Blob* weight_diff_count_output_ = filler.create("count_output", weight_diff_count_->shared_tensor());
                *filler.create<Constant>("fill_weight_diff_count", rank_, -1, "main", Constant::param_tuple(0.))
                    >> vector<Blob*>{weight_diff_count_output_};

                filler.run();
            }
        }

    template<typename Net>
        void asgd_net<Net>::sync(){
            Runnable::sync();
            net_->fetch()->sync();
        }

    template<typename Net>
        void asgd_net<Net>::run_async(){
            feed();
            Runnable::run_async();
            net_->fetch()->run_async();
        }

    template<typename Net>
        void asgd_net<Net>::run(){
            struct timeval start, end;
            gettimeofday(&start,0);
            double avr = 0;
            while(true){
                gettimeofday(&end, 0);
                avr = time_subtract(&start, &end);
                if(avr < period){
                    run_async();
                    sync();
                }
                else{
                    break;
                }
            }
        }
}
#endif
