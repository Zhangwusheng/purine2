// Copyright Lin Min 2015

#include <mpi.h>
#include <vector>
#include <time.h>
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
        MPI_LOG(<<"rank " << rank << " device " << device << " batch_size " << batch_size);
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
    vector<shared_ptr<LocalFetchImage>> fetch;
    vector<shared_ptr<asgd_net<NIN_Cifar10<false> > > >parallel_nin_cifar;
    DTYPE global_learning_rate = 0.05;
    DTYPE global_decay = 0.0001;
    setup_param_server(global_learning_rate, global_decay);
    for(int i = 0; i < parallels.size(); i++){
        fetch.push_back(make_shared<LocalFetchImage>(source, mean_file,
                    true, true, true, 1.1, 32, parallels[i]));
        fetch[i]->run();
        parallel_nin_cifar.push_back(
                make_shared<asgd_net<NIN_Cifar10<false> > >
                (parallels[i][0], parallels[i][1], parallels[i][2])
                );
    }
    // do the initialization
    Runnable init(0, -1);
    // 初始化为0,-1表示只有机器
    for(int i = 0; i < parallel_nin_cifar[0]->net()->weight_data().size(); i++){
        Size size_ = parallel_nin_cifar[0]->net()->weight_data()[i]->shared_tensor()->size();
        Blob* weight = init.create("weight_init", size_);
        Blob* diff = init.create("weight_diff", size_);
        Blob* history = init.create("weight_history", size_);
        if(i % 2 == 0){
            *init.create<Gaussian>("fill_weight", "init", Gaussian::param_tuple(0.f, 0.05f)) >>
                vector<Blob*>{weight};
        }
        else{
            *init.create<Constant>("fill_weight", "init", Constant::param_tuple(0.f)) >>
                vector<Blob*>{weight};
        }
        *init.create<Constant>("fill_diff", "init", Constant::param_tuple(0.f)) >>
            vector<Blob*>{history};
        *init.create<Constant>("fill_history", "init", Constant::param_tuple(0.f)) >>
            vector<Blob*>{diff};
        weights_server.push_back(weight);
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
    std::vector<std::vector<Blob*> >{weights_diff_server} >> *init.createAny<Vectorize<Distribute> >("init_distribute",
            vector<Distribute::param_tuple>(18, Distribute::param_tuple()))
        >> weights_diff;

    init.run();

//#define LOAD
#ifdef LOAD 
    load("./nin_cifar_dump_iter_50000.snapshot");
#endif

    int fetch_count = 0;
    int save_fetch = 5000;
    double period = 0.05;

    int iter = 0;

    while(iter < 30000){
        if(iter == 20000 || iter == 25000){
            global_learning_rate /= 10;
            setup_param_server(global_learning_rate, global_decay);
        }
        if(iter == 1000){
            period += 0.05;
        }
        iter++;
        int ttt = 1;
        double start_t = clock();
        while(true){
            for(int net_id = 0; net_id < parallel_nin_cifar.size(); net_id++){
                auto net = parallel_nin_cifar[net_id];
                auto fetch_image = fetch[net_id];
                net->feed(fetch_image->images(), fetch_image->labels());

                net->run_async();
                fetch_image->run_async();

                net->sync();
                fetch_image->sync();
                net->add_weight_diff_sum();
            }
            double end_t = clock();
            double p = (end_t - start_t) / (double)CLOCKS_PER_SEC;
            if(p > period)
                break;
        }

        if(save_fetch < fetch_count){
            save("./nin_cifar_dump_iter_" + to_string(save_fetch) + ".snapshot");
            save_fetch += 5000;
        }

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
        std::vector<Blob*>fetches;
        for(int j = 0; j < parallel_nin_cifar.size(); j++){
            fetches.push_back(get_loss_run.create("fetches", parallel_nin_cifar[j]->get_weight_diff_count()->shared_tensor()));
        }
        Blob* fetches_output = get_loss_run.create("fetches_output", 0, -1, Size(1,1,1,1));
        fetches >> *get_loss_run.createAny<Aggregate>("aggregate_weight_diff_count", 
                Aggregate::param_tuple(Aggregate::SUM, 0, -1))
            >> std::vector<Blob*>{fetches_output};

        get_loss_run.run();
        int cur_fetch_count;
        if(current_rank() == 0){
            vector<DTYPE> ret(loss_output.size());
            transform(loss_output.begin(), loss_output.end(), ret.begin(), [](Blob* b)->DTYPE {
                    return b->tensor()->cpu_data()[0];
                    });
            cur_fetch_count = fetches_output->tensor()->cpu_data()[0];
            fetch_count += cur_fetch_count;
            MPI_LOG(<< "iter " << iter <<
                    " loss " << ret[0] << 
                    " accuracy " << ret[1] << 
                    " period " << period << 
                    " fetch_count " << cur_fetch_count << "\\" << fetch_count);
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
            tmp_>> *agg;
            Blob* output1 = reduce_weight_diff.create("output", weights_diff_server[i]->shared_tensor());
            agg->top() >> *reduce_weight_diff.create<Scale>("scale", 0, -1, 
                    "main", Scale::param_tuple(static_cast<DTYPE>(1.0f))) >> std::vector<Blob*>{output1};
        }

        reduce_weight_diff.run();

        //update weight
        //new_history = history * momentum + weight_diff * learning_rate + weight * weight_decay;
        Runnable update_weight_history(0, -1);
        //param[i] = {momentum, learning_rate, learning_rate * global_decay};
        std::vector<Blob*>new_weight_;
        for(int i = 0; i < 18; i++){
            Op<WeightedSum>* update_history = update_weight_history.create<WeightedSum>("update_history", 
                    "main", WeightedSum::param_tuple({param[i][0], param[i][1] / cur_fetch_count, param[i][2]}));
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
