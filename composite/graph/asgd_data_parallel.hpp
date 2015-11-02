// Copyright Lin Min 2015
#ifndef PURINE_ASGD_DATA_PARALLEL
#define PURINE_ASGD_DATA_PARALLEL

#include <iomanip>
#include <fstream>
#include <set>
#include "composite/composite.hpp"
#include "operations/include/eltwise.hpp"
#include "composite/graph/copy.hpp"


using namespace std;

namespace purine {

    template <typename Net, typename PS>
        class AsgdDataParallel : public Runnable {
            protected:
                Vectorize<PS>* param_server_ = NULL;
                vector<Blob*> loss_;
                Blob* fetch_count_;
                vector<vector<Blob*> > new_weights_;
                vector<vector<Blob*> > weights_;
                vector<vector<Blob*> > weights_diff_;
                vector<shared_ptr<asgd_net<Net > > >nets_;
                double period;
            public:
                AsgdDataParallel(const vector<vector<int> >& locations);
                virtual ~AsgdDataParallel() override {};
                virtual void run_async() override;
                virtual void run() override;
                virtual void sync() override;
                inline int fetch_count(){return fetch_count_->shared_tensor()->cpu_data()[0];}
                inline void set_period(double p){period = p;}
                vector<DTYPE> loss();
                // init weight using random number.
                template <typename Random>
                    void init(vector<int> index, const typename Random::param_tuple& args);

                /**
                 * @brief load weights from snapshot file
                 */
                void load(const string& filename);

                /**
                 * @brief save weights to snapshot file
                 */
                void save(const string& filename);

                void print_weight_info();
                void feed(const vector<Blob*>& data, const vector<Blob*>& labels);
                PS* param_server(int index) {
                    return param_server_->element(index);
                }
                template <typename... Args>
                    void setup_param_server(const Args&... args) {
                        //PS->AllReduce
                        //...Args == vector<int>(18, 0), vector<int>(18, -1), param = {0.9, learning_rate, global_decay}
                        param_server_ = createAny<Vectorize<PS> >("param_server", args...);
                        weights_diff_ >> *param_server_;
                        new_weights_ = param_server_->top();
                    }
        };

    template <typename Net, typename PS>
        vector<DTYPE> AsgdDataParallel<Net, PS>::loss() {
            CHECK_EQ(current_rank(), 0);
            vector<DTYPE> ret(loss_.size());
            transform(loss_.begin(), loss_.end(), ret.begin(), [](Blob* b)->DTYPE {
                    return b->tensor()->cpu_data()[0];
                    });
            return ret;
        }

    template <typename Net, typename PS>
        void AsgdDataParallel<Net, PS>::print_weight_info() {
            if (current_rank() == this->rank_) {
                const vector<Blob*>& weight = nets_[0]->weight_data();
                int max_len = 0;
                for (int i = 0; i < weight.size(); ++i) {
                    int len = weight[i]->cached_name().length();
                    max_len = max_len > len ? max_len : len;
                }
                for (int i = 0; i < param_server_->size(); ++i) {
                    shared_ptr<Tensor> h = param_server_->element(i)->history();
                    shared_ptr<Tensor> w = param_server_->element(i)->weight();
                    // shared_ptr<Tensor> h = param_server_->element(i)->weight_diff();
                    DTYPE h_abs_sum = 0;
                    const DTYPE* data = h->cpu_data();
                    for (int j = 0; j < h->size().count(); ++j) {
                        h_abs_sum += abs(data[j]);
                    }
                    h_abs_sum /= h->size().count();
                    DTYPE w_abs_sum = 0;
                    data = w->cpu_data();
                    for (int j = 0; j < w->size().count(); ++j) {
                        w_abs_sum += abs(data[j]);
                    }
                    w_abs_sum /= w->size().count();

                    const string& name = weight[i]->cached_name();
                    size_t pos = name.find("::");
                    LOG(INFO) << std::left << std::setw(max_len - pos + 1) <<
                        std::setfill(' ') << name.substr(pos + 2) << std::scientific <<
                        "(" << w_abs_sum << ") " << " [" << h_abs_sum << "]";
                }
            }
        }

    template <typename Net, typename PS>
        template <typename Random>
        void AsgdDataParallel<Net, PS>::init(vector<int> index,
                const typename Random::param_tuple& args) {
            Runnable initializer(0, -1);
            Op<Random>* rnd = initializer.create<Random>("init", "main", args);
            vector<Blob*> tmp(index.size());
            vector<vector<Blob*> > weights(nets_.size() + 1);
            for (int i = 0; i < index.size(); ++i) {
                tmp[i] = initializer.create("tmp",
                        param_server_->element(index[i])->weight()->size());
            }
            for (int i = 0; i < nets_.size(); ++i) {
                weights[i] = vector<Blob*>(index.size());
                for (int j = 0; j < index.size(); ++j) {
                    weights[i][j] = initializer.create("weight",
                            nets_[i]->net()->weight_data()[index[j]]->shared_tensor());
                }
            }
            weights[nets_.size()] = vector<Blob*>(index.size());
            for (int j = 0; j < index.size(); ++j) {
                weights[nets_.size()][j] = initializer.create("weight_ps",
                        param_server_->element(index[j])->weight());
            }
            //在本地初始化好数据
            *rnd >> tmp;
            vector<vector<Blob*> >{ tmp }
            //分发给其他所有节点.
            >> *initializer.createAny<Vectorize<Distribute> >("init_distribute",
                    vector<Distribute::param_tuple>(index.size(), Distribute::param_tuple()))
                >> weights;
            initializer.run();
        }

    template <typename Net, typename PS>
        void AsgdDataParallel<Net, PS>::load(const string& filename) {
            Runnable loader(0, -1);
            int num_param = param_server_->size();
            vector<Blob*> tmp(num_param);
            vector<vector<Blob*> > weights(nets_.size() + 1);
            for (int i = 0; i < param_server_->size(); ++i) {
                tmp[i] = loader.create("tmp",
                        param_server_->element(i)->weight()->size());
            }
            for (int i = 0; i < nets_.size(); ++i) {
                weights[i] = vector<Blob*>(param_server_->size());
                for (int j = 0; j < param_server_->size(); ++j) {
                    weights[i][j] = loader.create("weight",
                            nets_[i]->weight_data()[j]->shared_tensor());
                }
            }
            weights[nets_.size()] = vector<Blob*>(param_server_->size());
            for (int j = 0; j < param_server_->size(); ++j) {
                weights[nets_.size()][j] = loader.create("weight_ps",
                        param_server_->element(j)->weight());
            }
            vector<vector<Blob*> >{ tmp }
            >> *loader.createAny<Vectorize<Distribute> >("init_distribute",
                    vector<Distribute::param_tuple>(param_server_->size(),
                        Distribute::param_tuple()))
                >> weights;
            // fill with the binary data
            if (current_rank() == 0) {
                LOG(INFO) << "Loading snapshot " << filename;
                // read file into binary string raw
                ifstream in(filename, ios::binary);
                stringstream ss;
                ss << in.rdbuf();
                const string& raw = ss.str();

                int total_len = 0;
                for (Blob* b : tmp) {
                    total_len += b->tensor()->size().count() * sizeof(DTYPE);
                }
                CHECK_EQ(raw.length(), total_len) <<
                    "Snapshot size incompatible with network weight";
                int offset = 0;
                for (Blob* b : tmp) {
                    int len = b->tensor()->size().count() * sizeof(DTYPE);
                    memcpy(b->tensor()->mutable_cpu_data(), raw.c_str() + offset, len);
                    offset += len;
                }
            }
            // run
            loader.run();
            MPI_LOG( << "Snapshot loaded" );
        }

    template <typename Net, typename PS>
        void AsgdDataParallel<Net, PS>::save(const string& filename) {
            Runnable saver;
            int param_num = param_server_->size();
            vector<Blob*> param(param_num);
            for (int i = 0; i < param_num; ++i) {
                param[i] = saver.create("param", param_server_->element(i)->weight());
            }
            auto copier = saver.createAny<Vectorize<Copy> >("copy_here",
                    vector<Copy::param_tuple>(param_num, Copy::param_tuple(0, -1)));
            vector<vector<Blob*> >{ param } >> *copier;
            vector<Blob*> copied = copier->top()[0];
            saver.run();

            if (current_rank() == 0) {
                ofstream out(filename);
                for (int i = 0; i < param_num; ++i) {
                    const char* data = reinterpret_cast<const char*>(
                            copied[i]->tensor()->cpu_data());
                    int len = copied[i]->tensor()->size().count() * sizeof(DTYPE);
                    out.write(data, len);
                }
                LOG(INFO) << "Saving snapshot " << filename;
            }
        }

    template <typename Net, typename PS>
        AsgdDataParallel<Net, PS>::AsgdDataParallel(const vector<vector<int> >& locations)
        : Runnable() {
            period = 0.1;
            for(int i = 0; i < locations.size(); i++){
                nets_.push_back(
                        make_shared<asgd_net<Net > >
                        (locations[i][0], locations[i][1], locations[i][2])
                        );
            }
            vector<vector<Blob*> > losses(locations.size());
            weights_ = vector<vector<Blob*> >(locations.size());
            for (int i = 0; i < locations.size(); ++i) {
                // get the data and labels
                for(int j = 0; j < nets_[i]->loss().size(); j++){
                    losses[i].push_back(
                            create("loss", nets_[i]->net()->loss()[j]->shared_tensor()));
                }
                std::vector<Blob*>weights = nets_[i]->net()->weight_data();
                for(int j = 0; j < weights.size(); j++){
                    weights_[i].push_back(create("weights", weights[j]->shared_tensor()));
                }
            }
            // agg loss to rank 0 device -1.
            Vectorize<Aggregate>* agg = createAny<Vectorize<Aggregate> >("agg_loss",
                    vector<Aggregate::param_tuple>(losses[0].size(),
                        Aggregate::param_tuple(Aggregate::AVERAGE, 0, -1)));
            losses >> *agg;
            loss_ = agg->top()[0];

            //agg fetch_counts to rank 0 device -1
            std::vector<Blob*>fetches;
            std::vector<Blob*>fetches_dis;
            for(int j = 0; j < nets_.size(); j++){
                fetches.push_back(create("fetches", nets_[j]->get_weight_diff_count()->shared_tensor()));
                fetches_dis.push_back(create("fetches", nets_[j]->rank(), nets_[j]->device(), nets_[j]->get_weight_diff_count()->shared_tensor()->size()));
            }
            fetch_count_ = create("fetches_output", 0, -1, Size(1,1,1,1));
            fetches >> *createAny<Aggregate>("aggregate_weight_diff_count", 
                    Aggregate::param_tuple(Aggregate::SUM, 0, -1))
                >> std::vector<Blob*>{fetch_count_};
            vector<Blob*>{ fetch_count_ } >> *createAny<Distribute>("dist_new_weight",
                    Distribute::param_tuple()) >> fetches_dis;

            weights_diff_.resize(nets_.size());

            for (int i = 0; i < nets_.size(); ++i) {
                vector<Blob*>weight_diff_sum = nets_[i]->net()->weight_diff_sum();
                for(int j = 0; j < weight_diff_sum.size(); j++){
                    Blob* diff = create("weight_diff_sum", weight_diff_sum[j]->shared_tensor());
                    Blob* diff_output = create("weight_diff_sum_output", diff->shared_tensor());
                    Op<ScaleA>* scalea = create<ScaleA>("wait_for_fetch_count", nets_[i]->rank(), nets_[i]->device(), "main", ScaleA::param_tuple());
                    std::vector<Blob*>{diff, fetches_dis[i]}>> *scalea >> std::vector<Blob*>{diff_output};
                    weights_diff_[i].push_back(diff_output);
                }
            }
        }

    template <typename Net, typename PS>
        void AsgdDataParallel<Net, PS>::sync() {
            Runnable::sync();
            // update the weights
            for (int i = 0; i < nets_.size(); ++i) {
                if (nets_[i]->rank() == current_rank()) {
                    for (int j = 0; j < weights_[0].size(); ++j) {
                        CHECK_EQ(new_weights_[i][j]->tensor()->size(),
                                weights_[i][j]->tensor()->size());
                        CHECK_EQ(new_weights_[i][j]->rank(), weights_[i][j]->rank());
                        CHECK_EQ(new_weights_[i][j]->device(), weights_[i][j]->device());
                        new_weights_[i][j]->tensor()->swap_memory(weights_[i][j]->tensor());
                    }
                }
            }
            for(int i = 0; i < nets_.size(); i++){
                nets_[i]->clear_weight_diff();
            }
        }

    template <typename Net, typename PS>
        void AsgdDataParallel<Net, PS>::run_async(){
            std::vector<std::thread>threads;
            for(auto& net : nets_){
                if(net->is_empty() == false){
                    threads.push_back(            
                            std::thread([&](){
                                net->set_period(period);
                                net->run();}));
                }
            }
            for(auto& t : threads)t.join();
            Runnable::run_async();
        }

    template<typename Net, typename PS>
        void AsgdDataParallel<Net, PS>::run(){
            Runnable::run();
        }

}

#endif
