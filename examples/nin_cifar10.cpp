// Copyright Lin Min 2015

#include <mpi.h>
#include <glog/logging.h>
#include "examples/nin_cifar10.hpp"
#include "composite/graph/all_reduce.hpp"

string data_path = "/home/zhxfl/purine2/data/cifar-10/";

string source =    data_path + "cifar-10-train-lmdb";
string mean_file = data_path + "mean.binaryproto";

using namespace purine;

void setup_param_server(DataParallel<NIN_Cifar10<false>, AllReduce> *parallel_nin_cifar,
        DTYPE global_learning_rate,
        DTYPE global_decay){

    vector<AllReduce::param_tuple> param(18);
    for (int i = 0; i < 18; ++i) {
        DTYPE learning_rate = global_learning_rate * (i % 2 ? 2. : 1.);
        if (i == 16 || i == 17) {
            learning_rate /= 10.;
        }
        param[i] = AllReduce::param_tuple(0.9, learning_rate,
                learning_rate * global_decay * (i % 2 ? 0. : 1.));
    }
    parallel_nin_cifar->setup_param_server(vector<int>(18, 0),
            vector<int>(18, -1), param);
}

void update_param_server(DataParallel<NIN_Cifar10<false>, AllReduce> *parallel_nin_cifar,
        DTYPE global_learning_rate,
        DTYPE global_decay){

    for (int i = 0; i < 18; ++i) {
        DTYPE learning_rate = global_learning_rate * (i % 2 ? 2. : 1.);
        if (i == 16 || i == 17) {
            learning_rate /= 10.;
        }
        DTYPE weight_decay = learning_rate * global_decay * (i % 2 ? 0. : 1.);

        parallel_nin_cifar->param_server(i)->set_param(
                make_tuple<vector<DTYPE> >({0.9, learning_rate, weight_decay})
                );
    }
}

void read_parallel_config(vector<vector<int>>& parallels){
    FILE* file = fopen("parallel_config", "r+");
    int rank, device, batch_size;
    while(fscanf(file, "%d %d %d", &rank, &device, &batch_size) != EOF){
        parallels.push_back({rank, device, batch_size});
        MPI_LOG(<<"rank " << rank << "device" << device << "batch_size" << batch_size);
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
    shared_ptr<FetchImage> fetch = make_shared<FetchImage>(source, mean_file,
            true, true, true, 1.1, 32, parallels);
    fetch->run();
    // create data parallelism of Nin_Cifar;
    shared_ptr<DataParallel<NIN_Cifar10<false>, AllReduce> > parallel_nin_cifar
        = make_shared<DataParallel<NIN_Cifar10<false>, AllReduce> >(parallels);
    // set learning rate etc
    DTYPE global_learning_rate = 0.05;
    DTYPE global_decay = 0.0001;
    // shape_ptr.get()获取共享指针里面的内容
    setup_param_server(parallel_nin_cifar.get(), global_learning_rate, global_decay);

    // do the initialization
#define RANDOM
#ifdef RANDOM
    vector<int> indice(9);
    iota(indice.begin(), indice.end(), 0);//递增
    vector<int> weight_indice(9);
    vector<int> bias_indice(9);
    transform(indice.begin(), indice.end(), weight_indice.begin(),
            [](int i)->int {
            return i * 2;
            });
    transform(indice.begin(), indice.end(), bias_indice.begin(),
            [](int i)->int {
            return i * 2 + 1;
            });
    parallel_nin_cifar->init<Constant>(bias_indice, Constant::param_tuple(0.));
    parallel_nin_cifar->init<Gaussian>(weight_indice,
            Gaussian::param_tuple(0., 0.05));
#else
    parallel_nin_cifar->load("./nin_cifar_dump_iter_50000.snapshot");
#endif
    // iteration
    for (int iter = 1; iter <= 50000; ++iter) {
        if(iter == 40000 || iter == 45000){
            global_learning_rate /= 10.;
            update_param_server(parallel_nin_cifar.get(),
                    global_learning_rate,
                    global_decay);
        }
        // feed prefetched data to nin_cifar
        parallel_nin_cifar->feed(fetch->images(), fetch->labels());
        // start nin_cifar and next fetch
        parallel_nin_cifar->run_async();
        fetch->run_async();
        parallel_nin_cifar->sync();
        fetch->sync();
        // verbose
        MPI_LOG( << "iteration: " << iter << ", loss: "
                << parallel_nin_cifar->loss()[0]);
        if(iter % 1 == 0 && current_rank() == 0)
        {
            MPI_LOG(<<"iter " << iter << "loss " << parallel_nin_cifar->loss()[0] << "accuracy " << parallel_nin_cifar->loss()[1]);
        }
        if (iter % 100 == 0 && current_rank() == 0) {
            parallel_nin_cifar->print_weight_info();
        }
        if (iter % 5000 == 0 && current_rank() == 0) {
            parallel_nin_cifar->save("./nin_cifar_dump_iter_"
                    + to_string(iter) + ".snapshot");
        }
    }
    // delete
    fetch.reset();
    parallel_nin_cifar.reset();
    // Finalize MPI
    MPI_CHECK(MPI_Finalize());
    return 0;
}
