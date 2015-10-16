// Copyright Lin Min 2015

#include <mpi.h>
#include <vector>
#include <glog/logging.h>
#include "examples/nin_cifar10.hpp"
#include "composite/graph/all_reduce.hpp"
#include "composite/graph/asgd_net.hpp"
#include "composite/composite.hpp"
#include "operations/tensor.hpp"

string data_path = "/home/zhxfl/purine2/data/cifar-10/";
string source =    data_path + "cifar-10-train-lmdb";
string mean_file = data_path + "mean.binaryproto";

using namespace purine;

std::vector<int> param_rank;
std::vector<int> param_device;
std::vector<std::vector<DTYPE> >param(18);
std::vector<Blob*>weights_server;
std::vector<Blob*>weights_diff_server;
std::vector<Blob*>history_server;
std::vector<std::vector<Blob*> > weights;
std::vector<std::vector<Blob*> > weights_diff;

void save(const string& filename){
    if(current_rank() == 0){
        ofstream out(filename);
        for(int i = 0; i < weights_server.size(); i++){
            const char* data = reinterpret_cast<const char*>(
                    weights_server[i]->tensor()->cpu_data());
            int len = weights_server[i]->tensor()->size().count() * sizeof(DTYPE);
            out.write(data, len);
        }
        LOG(INFO) << "save snapshot" << filename;
    }    
}

void load(const string& filename){
    if(current_rank() == 0){
        LOG(INFO) << "loading snapshot" << filename;
        ifstream in(filename, ios::binary);
        stringstream ss;
        ss << in.rdbuf();
        const string &raw = ss.str();

        int total_len = 0;
        for(Blob* b: weights_server){
            total_len += b->tensor()->size().count() * sizeof(DTYPE);
        }
        CHECK_EQ(raw.length(), total_len) << 
            "Snapshot size incompatible with network weight";
        int offset = 0;
        for(Blob* b: weights_server){
            int len = b->tensor()->size().count() * sizeof(DTYPE);
            memcpy(b->tensor()->mutable_cpu_data(), raw.c_str() + offset, len);
            offset += len;
        }
        MPI_LOG(<<"Snapshot loaded");
    } 
}

void setup_param_server(
        DTYPE global_learning_rate,
        DTYPE global_decay){
    param.clear();
    param_rank.clear();
    param_device.clear();
    for (int i = 0; i < 18; ++i) {
        DTYPE learning_rate = global_learning_rate * (i % 2 ? 2.f : 1.f);
        if (i == 16 || i == 17) {
            learning_rate /= 10.f;
        }
        param[i] = vector<DTYPE>{0.9f, learning_rate,
            learning_rate * global_decay * (i % 2 ? 0.f : 1.f)};
    }
    param_rank = vector<int>(18, 0);
    param_device = vector<int>(18, -1);
}

void read_parallel_config(vector<vector<int>>& parallels){
    FILE* file = fopen("parallel_config", "r+");
    int rank, device, batch_size;
    while(fscanf(file, "%d %d %d", &rank, &device, &batch_size) != EOF){
        parallels.push_back({rank, device, batch_size});
        printf("rank %d device %d batch_size %d\n", rank, device, batch_size);
    }
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    // initilize MPI
    int ret;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &ret));
    // parallels
    vector<vector<int> > parallels;
    read_parallel_config(parallels);  
    // parameter server
    // fetch image
    vector<shared_ptr<FetchImage>> fetch;
    vector<shared_ptr<asgd_net<NIN_Cifar10<false> > > >parallel_nin_cifar;
    DTYPE global_learning_rate = 0.05;
    DTYPE global_decay = 0.0001;
    for(int i = 0; i < parallels.size(); i++){
        fetch.push_back(make_shared<FetchImage>(source, mean_file,
                    true, true, true, 1.1, 32, vector<vector<int>>{parallels[i]}));
        fetch[i]->run();
        parallel_nin_cifar.push_back(
                make_shared<asgd_net<NIN_Cifar10<false> > >
                (parallels[i][0], parallels[i][1], parallels[i][2])
                );
        //shape_ptr.get()获取共享指针里面的内容
        setup_param_server(global_learning_rate, global_decay);
    }
    // do the initialization
#define RANDOM
#ifdef RANDOM
    // 初始化为0,-1表示只有机器
    Runnable init(0, -1);
    for(int i = 0; i < parallel_nin_cifar[0]->net()->weight_data().size(); i++){
        Size size_ = parallel_nin_cifar[0]->net()->weight_data()[i]->shared_tensor()->size();
        Blob* to_fill = init.create("weight_init", size_);
        Blob* diff = init.create("weight_diff", size_);
        Blob* history = init.create("weight_history", size_);
        if(i % 2 == 0){
            *init.create<Gaussian>("fill_weight", "init", Gaussian::param_tuple(0.f, 0.05f)) >>
                vector<Blob*>{to_fill};
        }
        else{
            *init.create<Constant>("fill_weight", "init", Constant::param_tuple(0.f)) >>
                vector<Blob*>{to_fill};
        }
        *init.create<Constant>("fill_history", "init", Constant::param_tuple(0.f)) >>
            vector<Blob*>{history};
        weights_server.push_back(to_fill);
        weights_diff_server.push_back(diff);
        history_server.push_back(history);
    }
    weights = std::vector<std::vector<Blob*>>(parallel_nin_cifar.size());
    weights_diff = std::vector<std::vector<Blob*>>(parallel_nin_cifar.size());

    for(int i = 0; i < parallel_nin_cifar.size(); i++){
        weights[i] = std::vector<Blob*>(parallel_nin_cifar[i]->net()->weight_data().size()); 
        weights_diff[i] = std::vector<Blob*>(parallel_nin_cifar[i]->net()->weight_diff().size()); 
        for(int j = 0; j < parallel_nin_cifar[i]->net()->weight_data().size(); j++){
            weights[i][j] = init.create("weights",
                    parallel_nin_cifar[i]->net()->weight_data()[j]->shared_tensor());
            weights_diff[i][j] = init.create("weights",
                    parallel_nin_cifar[i]->net()->weight_diff()[j]->shared_tensor());
        }
    }
    // weights的结果分发到其他所有机器
    // 分发给其他所有节点.
    std::vector<std::vector<Blob*> >{weights_server} >> *init.createAny<Vectorize<Distribute> >("init_distribute",
            vector<Distribute::param_tuple>(18, Distribute::param_tuple()))
        >> weights;

    /*
       std::vector<std::vector<Blob*> >{weights[0]} >> *init.createAny<Vectorize<Distribute> >("init_distribute",
       vector<Distribute::param_tuple>(18, Distribute::param_tuple()))>> std::vector<std::vector<Blob*>>{weights_server};
    */

    init.run();
    //weights_server[0]->shared_tensor()->print();
#else
    load("./nin_cifar_dump_iter_50000.snapshot");
#endif
    // iteration
    int avr_iter = 2;
    for (int iter = 1; iter <= 50000; ++iter) {
        if(iter % 10000 == 0){
            avr_iter ++;
        }
        for(int j = 0; j < parallels.size(); j++){
            if(iter == 40000 || iter == 45000){
                global_learning_rate /= 10.;
                setup_param_server(
                        global_learning_rate,
                        global_decay);
            }
            // feed prefetched data to nin_cifar
            parallel_nin_cifar[j]->feed(fetch[j]->images(), fetch[j]->labels());
            // start nin_cifar and next fetch
            parallel_nin_cifar[j]->run_async();
            fetch[j]->run_async();
        }
        for(int j = 0; j < parallel_nin_cifar.size(); j++){
            parallel_nin_cifar[j]->sync();
            parallel_nin_cifar[j]->add_weight_diff_sum();
            fetch[j]->sync();
        }
        if(iter % 5000 == 0){
            save("./nin_cifar_dump_iter_" + to_string(iter) + ".snapshot");
        }
        if(iter % avr_iter == 0)
        {
            std::vector<std::vector<Blob*> > losses = std::vector<std::vector<Blob*>>(parallel_nin_cifar.size());
            //get loss
            Runnable get_loss_run(0, -1);
            for(int n = 0; n < parallel_nin_cifar.size(); n++){
                std::vector<Blob*>tmp = parallel_nin_cifar[n]->loss();
                for(int m = 0; m < tmp.size(); m++){
                    losses[n].push_back(get_loss_run.create("loss", tmp[m]->shared_tensor()));
                }
            }
            std::vector<Blob*>loss_output;
            loss_output.push_back(get_loss_run.create("loss_output", 0, -1, losses[0][0]->shared_tensor()->size()));
            loss_output.push_back(get_loss_run.create("loss_output", 0, -1, losses[0][1]->shared_tensor()->size()));

            Vectorize<Aggregate>* agg = get_loss_run.createAny<Vectorize<Aggregate> >("agg_loss",
                    vector<Aggregate::param_tuple>(losses[0].size(),
                        Aggregate::param_tuple(Aggregate::AVERAGE, 0, -1)));/*rank device*/
            losses >> *agg >> std::vector<std::vector<Blob*>>{loss_output};
            get_loss_run.run();
            if(current_rank() == 0){
                std::vector<Blob*>loss = agg->top()[0];
                vector<DTYPE> ret(loss.size());
                transform(loss.begin(), loss.end(), ret.begin(), [](Blob* b)->DTYPE {
                        return b->tensor()->cpu_data()[0];
                        });
                printf("global_learning_rate %.4f, global_decay %.8f\niter %5d, loss %.4f, accuracy %.4f\n",
                        global_learning_rate, global_decay, iter, ret[0], ret[1]);
            }
            // reduce weight_diff_
            Runnable reduce_weight_diff(0, -1);
            for(int i = 0; i < 18; i++){
                std::vector<Blob*> tmp_;
                for(int j = 0; j < parallel_nin_cifar.size(); j++){
                    tmp_.push_back(reduce_weight_diff.create("reduce_weight_diff_tmp", parallel_nin_cifar[j]->get_weight_diff()[i]->shared_tensor()));
                }
                Aggregate* agg = reduce_weight_diff.createAny<Aggregate>("aggregate_weight_diff_tmp",
                        Aggregate::param_tuple(Aggregate::SUM, 0, -1));
                Blob* output = reduce_weight_diff.create("output", weights_diff_server[i]->shared_tensor());
                tmp_>> *agg >> std::vector<Blob*>{output};
                //Blob* output = reduce_weight_diff.create("output", weights_diff_server[i]->shared_tensor());
                //agg->top() >> *reduce_weight_diff.create<Scale>("scale", agg->top()[0]->rank(), agg->top()[0]->device(),
                //        "main", Scale::param_tuple(static_cast<DTYPE>(1.0f / static_cast<DTYPE>(avr_iter)))) >> std::vector<Blob*>{output};
            }
            reduce_weight_diff.run();
            //update weight
            //new_history = history * momentum + weight_diff * learning_rate + weight * weight_decay;
            Runnable update_weight_history(0, -1);
            //param[i] = {momentum, learning_rate, learning_rate * global_decay};
            std::vector<Blob*>new_weight_;
            for(int i = 0; i < 18; i++){
                Op<WeightedSum>* update_history = update_weight_history.create<WeightedSum>("update_history", 
                        "main", WeightedSum::param_tuple({param[i][0], param[i][1] / avr_iter / parallel_nin_cifar.size(), param[i][2]}));
                Blob* weight = update_weight_history.create("wegiht", weights_server[i]->shared_tensor());
                Blob* weight_diff = update_weight_history.create("weight_diff", weights_diff_server[i]->shared_tensor());
                Blob* history = update_weight_history.create("history", history_server[i]->shared_tensor());
                Blob* new_history = update_weight_history.create("new_history", history_server[i]->shared_tensor());
                std::vector<Blob*>{history, weight_diff, weight} >> *update_history >> std::vector<Blob*>{new_history};
                //new_weight = weight - new_history
                Op<WeightedSum>* update_weight = update_weight_history.create<WeightedSum>("update_weight",  "main",
                        WeightedSum::param_tuple({1., -1.}));
                Blob* new_weight = update_weight_history.create("new_weight", weights_server[i]->shared_tensor());
                new_weight_.push_back(new_weight);
                std::vector<Blob*>{weight, new_history} >> *update_weight >> std::vector<Blob*>{new_weight};
            }
            //分发weight到节点上，和参数初始化类似
            std::vector<std::vector<Blob*>> weights_ = std::vector<std::vector<Blob*>>(parallel_nin_cifar.size());
            for(int i = 0; i < parallel_nin_cifar.size(); i++){
                weights_[i] = std::vector<Blob*>(parallel_nin_cifar[i]->net()->weight_data().size()); 
                for(int j = 0; j < parallel_nin_cifar[i]->net()->weight_data().size(); j++){
                    weights_[i][j] = update_weight_history.create("weights",
                            parallel_nin_cifar[i]->net()->weight_data()[j]->shared_tensor());
                }
            }
            // weights的结果分发到其他所有机器
            // 分发给其他所有节点.
            std::vector<std::vector<Blob*> >{new_weight_} >> *update_weight_history.createAny<Vectorize<Distribute> >("init_distribute",
                    vector<Distribute::param_tuple>(18, Distribute::param_tuple()))
                >> weights_;
            update_weight_history.run();
            for(int i = 0; i < parallel_nin_cifar.size(); i++){
                parallel_nin_cifar[i]->clear_weight_diff();
            }
        }
    }
    // delete
    for(int i = 0; i < fetch.size(); i++){
        fetch[i].reset();
    }
    for(int i = 0; i < parallel_nin_cifar.size(); i++){
        parallel_nin_cifar[i].reset();
    }
    // Finalize MPI
    MPI_CHECK(MPI_Finalize());
    return 0;
}
