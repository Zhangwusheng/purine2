// Copyright Lin Min 2015
#ifndef PURINE_ASYN_NIN_CIFAR10
#define PURINE_ASYN_NIN_CIFAR10

#include <glog/logging.h>
#include "common/common.hpp"
#include "dispatch/runnable.hpp"
#include "composite/composite.hpp"


extern string source;
extern string mean_file;

template <bool test>
class Asyn_NIN_Cifar10 : public Graph {
    protected:
        Blob* data_;
        Blob* label_;
        Blob* data_diff_;
        Blob* diff_sum_count_;
        Blob* const_one_;
        vector<Blob*> weight_data_;
        vector<Blob*> weight_diff_;
        vector<Blob*> weight_diff_sum_;
        vector<Blob*> copy_diff_;
        vector<Blob*> loss_;
        vector<Blob*> probs_;
        int batch_size_;
        shared_ptr<LocalFetchImage> fetch_;
    public:
        explicit Asyn_NIN_Cifar10(int rank, int device, int bs);
        virtual ~Asyn_NIN_Cifar10() override {fetch_.reset();}
        inline const vector<Blob*>& weight_data() { return weight_data_; }
        inline const vector<Blob*>& weight_diff() { return weight_diff_; }
        inline vector<Blob*>& weight_diff_sum() { return weight_diff_sum_; }
        inline Blob* diff_sum_count(){return diff_sum_count_;}
        inline Blob* const_one(){return const_one_;}
        inline vector<Blob*> data() { return { data_ }; }
        inline vector<Blob*> label() { return { label_ }; }
        inline vector<Blob*> data_diff() { return { data_diff_ }; }
        inline vector<Blob*> loss() { return loss_; }
        inline vector<Blob*> get_probs() { return probs_;}
        inline shared_ptr<LocalFetchImage> fetch(){ return fetch_;}
};

    template <bool test>
Asyn_NIN_Cifar10<test>::Asyn_NIN_Cifar10(int rank, int device, int bs)
    : Graph(rank, device) {
        batch_size_ = bs;

        data_ = create("data", { batch_size_, 3, 32, 32 });
        data_diff_ = create("data_diff", { batch_size_, 3, 32, 32 });
        label_ = create("label", { batch_size_, 1, 1, 1 });
        diff_sum_count_ = create("diff_sum_count",rank_, -1, {1, 1, 1, 1});

        std::string activation_function = "lrelu";

        // creating layers
        NINLayer* nin1 = createGraph<NINLayer>("nin1",
                NINLayer::param_tuple(2, 2, 1, 1, 5, 5, activation_function, {192, 160, 96}));
        PoolLayer* pool1 = createGraph<PoolLayer>("pool1",
                PoolLayer::param_tuple("max", 3, 3, 2, 2, 0, 0));
        DropoutLayer* dropout1 = createGraph<DropoutLayer>("dropout1",
                DropoutLayer::param_tuple(0.5, test, false));

        NINLayer* nin2 = createGraph<NINLayer>("nin2",
                NINLayer::param_tuple(2, 2, 1, 1, 5, 5, activation_function, {192, 192, 192}));
        PoolLayer* pool2 = createGraph<PoolLayer>("pool2",
                PoolLayer::param_tuple("max", 3, 3, 2, 2, 0, 0));
        DropoutLayer* dropout2 = createGraph<DropoutLayer>("dropout2",
                DropoutLayer::param_tuple(0.5, test, false));

        NINLayer* nin3 = createGraph<NINLayer>("nin3",
                NINLayer::param_tuple(1, 1, 1, 1, 3, 3, activation_function, {192, 192, 10}));

        GlobalAverageLayer* global_ave = createGraph<GlobalAverageLayer>("global_avg",
                GlobalAverageLayer::param_tuple());
        SoftmaxLossLayer* softmaxloss = createGraph<SoftmaxLossLayer>("softmaxloss",
                SoftmaxLossLayer::param_tuple(1.));

        Acc* acc = createGraph<Acc>("acc", rank_, -1, Acc::param_tuple(1));

        // connecting layers
        B{ data_,  data_diff_ } >> *nin1 >> *pool1 >> *dropout1
            >> *nin2 >> *pool2 >> *dropout2 >> *nin3 >> *global_ave;

        // loss layer
        softmaxloss->set_label(label_);
        *global_ave >> *softmaxloss;
        acc->set_label(label_);
        vector<Blob*>{ global_ave->top()[0] } >> *acc;

        // loss
        loss_  = { softmaxloss->loss()[0], acc->loss()[0] };
        probs_ = { softmaxloss->get_probs()};

        // weight
        vector<Layer*> layers = { nin1, nin2, nin3 };
        for (auto layer : layers) {
            const vector<Blob*>& w = layer->weight_data();
            weight_data_.insert(weight_data_.end(), w.begin(), w.end());
        }
        for (auto layer : layers) {
            const vector<Blob*>& w = layer->weight_diff();
            weight_diff_.insert(weight_diff_.end(), w.begin(), w.end());
        }

        for(auto weight_diff : weight_diff_){
            Blob* sum = create("diff_sum", weight_diff->tensor()->size());
            Blob* sum_output = create("diff_sum", sum->shared_tensor()); 
            Op<WeightedSum>* op_sum = 
                create<WeightedSum>("weight_sum", "main", WeightedSum::param_tuple({1., 1.}) );
            weight_diff_sum_.push_back(sum_output);
            B{sum, weight_diff}>> *op_sum >> B{sum_output};
        }

        {
            Runnable filler(rank_, -1);
            const_one_ = create("const_one", rank_, -1, {1,1,1,1});
            Blob* const_one_output = filler.create("count_output", const_one_->shared_tensor());
            *filler.create<Constant>("fill_const_one_count", "main", Constant::param_tuple(1.0))
                >> vector<Blob*>{const_one_output};
            filler.run();
        }

        Op<WeightedSum>* op_sum = 
            create<WeightedSum>("weight_sum", rank_, -1,  "main", WeightedSum::param_tuple({1., 1.}) );
        Blob* diff_sum_count_output = create("count_output", diff_sum_count_->shared_tensor());
        B{diff_sum_count_, const_one_}>> *op_sum >> B{diff_sum_count_output};

        fetch_ = make_shared<LocalFetchImage>(source, mean_file,
                    true, true, true, 1.1, 10,
                    32, std::vector<int>{rank_, device_, batch_size_});
        fetch_->run();
    }
#endif
