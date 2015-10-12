// Copyright Lin Min 2015

#include <mpi.h>
#include <vector>
#include <glog/logging.h>
#include "examples/nin_cifar10.hpp"
#include "composite/graph/all_reduce.hpp"
#include "composite/graph/asgd_net.hpp"
#include "composite/composite.hpp"

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

void setup_param_server(asgd_net<NIN_Cifar10<false> > *parallel_nin_cifar,
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
    shared_ptr<FetchImage> fetch;
    vector<shared_ptr<asgd_net<NIN_Cifar10<false> > > >parallel_nin_cifar;
    DTYPE global_learning_rate = 0.05;
    DTYPE global_decay = 0.0001;
    for(int i = 0; i < parallels.size(); i++){
        fetch = make_shared<FetchImage>(source, mean_file,
                    true, true, true, 1.1, 32, vector<vector<int>>{parallels[i]});
        fetch->run();
        parallel_nin_cifar.push_back(
                make_shared<asgd_net<NIN_Cifar10<false> > >
                (parallels[i][0], parallels[i][1], parallels[i][2])
                );
        //shape_ptr.get()获取共享指针里面的内容
        setup_param_server(parallel_nin_cifar[i].get(), global_learning_rate, global_decay);
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
    init.run();
#else
    // parallel_nin_cifar[i]->load("./nin_cifar_dump_iter_50000.snapshot");
#endif
    // iteration
    for (int iter = 1; iter <= 50000; ++iter) {
        for(int j = 0; j < parallels.size(); j++){
            if(iter == 40000 || iter == 45000){
                global_learning_rate /= 10.;
                setup_param_server(parallel_nin_cifar[j].get(),
                        global_learning_rate,
                        global_decay);
            }
            // feed prefetched data to nin_cifar
            parallel_nin_cifar[j]->feed(fetch->images(), fetch->labels());
            // start nin_cifar and next fetch
            parallel_nin_cifar[j]->run_async();
            fetch->run_async();
        }
        for(int j = 0; j < parallel_nin_cifar.size(); j++){
            parallel_nin_cifar[j]->sync();
            parallel_nin_cifar[j]->add_weight_diff_sum();
            fetch->sync();
        }
        // verbose
        /*MPI_LOG( << "iteration: " << iter << ", loss: "
          << parallel_nin_cifar[j]->loss()[0]);
          */
        /*if(iter % 1 == 0 && current_rank() == 0)
          {
          printf("global_learning_rate %.4f, global_decay %.8f\niter %5d, loss %.4f, accuracy %.4f\n",
          global_learning_rate,
          global_decay, 
          iter, 
          parallel_nin_cifar[j]->loss()[0],
          parallel_nin_cifar[j]->loss()[1]);
          }
          if (iter % 100 == 0 && current_rank() == 0) {
          parallel_nin_cifar[j]->print_weight_info();
          }
          if (iter % 5000 == 0 && current_rank() == 0) {
          parallel_nin_cifar[j]->save("./nin_cifar_dump_iter_"
          + to_string(iter) + ".snapshot");
          }*/

        if(iter % 10 == 0 && current_rank() == 0){
            std::vector<std::vector<Blob*> > losses;
            for(int n = 0; n < parallel_nin_cifar.size(); n++){
                losses.push_back(parallel_nin_cifar[n]->loss());
            }
            //get loss
            Runnable get_loss_run(0, -1);
            Vectorize<Aggregate>* agg = get_loss_run.createAny<Vectorize<Aggregate> >("agg_loss",
                    vector<Aggregate::param_tuple>(losses[0].size(),
                        Aggregate::param_tuple(Aggregate::AVERAGE, 0, -1)));
            losses >> *agg;
            std::vector<Blob*>loss = agg->top()[0];
            vector<DTYPE> ret(loss.size());
            transform(loss.begin(), loss.end(), ret.begin(), [](Blob* b)->DTYPE {
                    return b->tensor()->cpu_data()[0];
                    });
            printf("global_learning_rate %.4f, global_decay %.8f\niter %5d, loss %.4f, accuracy %.4f\n",
                    global_learning_rate,
                    global_decay, 
                    iter, 
                    ret[0],
                    ret[1]);

            // reduce weight_diff_
            Runnable reduce_weight_diff(0, -1);
            for(int i = 0; i < 18; i++){
                std::vector<Blob*> tmp_;
                for(int j = 0; j < parallel_nin_cifar.size(); j++){
                    tmp_.push_back(reduce_weight_diff.create("reduce_weight_diff_tmp", weights[j][i]->shared_tensor()));
                }

                Aggregate* agg = reduce_weight_diff.createAny<Aggregate>("aggregate_weight_diff_tmp",
                        Aggregate::param_tuple(Aggregate::AVERAGE, 0, -1));
                tmp_>> *agg >> vector<Blob*>{weights_diff_server[i]};
            }
            reduce_weight_diff.run();
            //update weight
        }
    }
    // delete
    fetch.reset();
    for(int i = 0; i < parallel_nin_cifar.size(); i++){
        parallel_nin_cifar[i].reset();
    }
    // Finalize MPI
    MPI_CHECK(MPI_Finalize());
    return 0;
}
