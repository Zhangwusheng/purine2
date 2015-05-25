// Copyright Lin Min 2015

#include <mpi.h>
#include <glog/logging.h>
#include "examples/nin_cifar10.hpp"
#include "composite/graph/compute_loss.hpp"

int batch_size = 100;
string source = "/home/zhxfl/purine2/data/cifar-10/cifar-10-test-lmdb";
string mean_file = "/home/zhxfl/purine2/data/cifar-10/mean.binaryproto";

using namespace purine;

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    // initilize MPI
    int ret;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &ret));
    // parallels
    // parameter server
    // fetch image
    shared_ptr<FetchImage> fetch = make_shared<FetchImage>(source, mean_file,
            false, false, true, batch_size, 32, vector<pair<int, int> >{{0, 0}});
    fetch->run();
    // create data parallelism of Nin_Cifar;
    shared_ptr<ComputeLoss<NIN_Cifar10<true> > > nin_cifar_test
        = make_shared<ComputeLoss<NIN_Cifar10<true> > >(0, 0);
    // do the initialization
    nin_cifar_test->load("./nin_cifar_dump_iter_50000.snapshot");
    // iteration
    DTYPE loss = 0.0;
    DTYPE acc  = 0.0;
    for (int iter = 1; iter <= 500; ++iter) {
        // feed prefetched data to nin_cifar
        nin_cifar_test->feed(fetch->images(), fetch->labels());
        // start nin_cifar and next fetch
        nin_cifar_test->run_async();
        fetch->run_async();
        fetch->sync();
        nin_cifar_test->sync();
        // verbose
        nin_cifar_test->print_loss();
        vector<DTYPE>test_loss = nin_cifar_test->get_loss();

        loss += test_loss[0];
        acc  += test_loss[1];
    }
    printf("loss %f, acc %f\n", loss, acc / 500);
    // delete
    fetch.reset();
    nin_cifar_test.reset();
    // Finalize MPI
    MPI_CHECK(MPI_Finalize());
    return 0;
}
