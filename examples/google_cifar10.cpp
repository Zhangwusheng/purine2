// Copyright Lin Min 2015

#include <mpi.h>
#include <glog/logging.h>
#include "examples/google_cifar10.hpp"
#include "composite/graph/all_reduce.hpp"

int batch_size = 128;
string data_path = "data/cifar-10/";

string source =    data_path + "cifar-10-train-lmdb";
string mean_file = data_path + "mean.binaryproto";

using namespace purine;
const int nParams = 12;

void setup_param_server(DataParallel<google_cifar10<false>, AllReduce> *parallel_nin_cifar,
        DTYPE global_learning_rate,
        DTYPE global_decay){

    vector<AllReduce::param_tuple> param(nParams * 2);
    for (int i = 0; i < nParams * 2; ++i) {
        DTYPE learning_rate = global_learning_rate * (i % 2 ? 2. : 1.);
        if (i == nParams * 2 - 2 || i == nParams * 2 - 1) {
            learning_rate /= 10.;
        }
        param[i] = AllReduce::param_tuple(0.9, learning_rate,
                learning_rate * global_decay * (i % 2 ? 0. : 1.));
    }
    parallel_nin_cifar->setup_param_server(vector<int>(nParams * 2, 0),
            vector<int>(nParams * 2, -1), param);
}

void update_param_server(DataParallel<google_cifar10<false>, AllReduce> *parallel_nin_cifar,
        DTYPE global_learning_rate,
        DTYPE global_decay){

    for (int i = 0; i < nParams * 2; ++i) {
        DTYPE learning_rate = global_learning_rate * (i % 2 ? 2. : 1.);
        if (i == nParams * 2 - 2 || i == nParams * 2 - 1) {
            learning_rate /= 10.;
        }
        DTYPE weight_decay = learning_rate * global_decay * (i % 2 ? 0. : 1.);

        parallel_nin_cifar->param_server(i)->set_param(
                make_tuple<vector<DTYPE> >({0.9, learning_rate, weight_decay})
                );
    }
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    // initilize MPI
    int ret;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &ret));
    // parallels
    vector<vector<int> > parallels;
    for (int rank : {0}) {
        for (int device : {0}) {
            parallels.push_back({rank, device, 128});
        }
    }
    // parameter server
    pair<int, int> param_server = {0, -1};
    // fetch image
    shared_ptr<FetchImage > fetch = make_shared<FetchImage>(source, mean_file,
            false, false, true, 1.1,  32, parallels);
    fetch->run();
    // create data parallelism of Nin_Cifar;
    shared_ptr<DataParallel<google_cifar10<false>, AllReduce> > parallel_nin_cifar
        = make_shared<DataParallel<google_cifar10<false>, AllReduce> >(parallels);
    // set learning rate etc
    DTYPE global_learning_rate = 0.05;
    DTYPE global_decay = 0.0001;
    setup_param_server(parallel_nin_cifar.get(), global_learning_rate, global_decay);

    // do the initialization
#define RANDOM
#ifdef RANDOM
    vector<int> indice( nParams );
    iota(indice.begin(), indice.end(), 0);
    vector<int> weight_indice( nParams );
    vector<int> bias_indice( nParams );
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
    parallel_nin_cifar->load("./nin_cifar_dump_iter_30000.snapshot");
#endif
    // iteration
    for (int iter = 1; iter <= 50000; ++iter) {
        if(iter == 45000 || iter == 47500){
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
        fetch->sync();
        parallel_nin_cifar->sync();
        // verbose
        MPI_LOG( << "iteration: " << iter << ", loss: "
                << parallel_nin_cifar->loss()[0]);
        if(iter % 50 == 0)
            printf("global_learning_rate %.4f, global_decay %.8f\niter %5d, loss %.4f, accuracy %.4f\n",
                    global_learning_rate,
                    global_decay, 
                    iter, 
                    parallel_nin_cifar->loss()[0],
                    parallel_nin_cifar->loss()[1]);

        if (iter % 100 == 0) {
            parallel_nin_cifar->print_weight_info();
        }
        if (iter % 5000 == 0) {
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
