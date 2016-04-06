// Copyright Lin Min 2015

#include <mpi.h>
#include <glog/logging.h>
#include "examples/nin_cifar10.hpp"
#include "composite/graph/compute_loss.hpp"
#include "dispatch/blob.hpp"

int batch_size = 100;
string data_path = "/home/zhxfl/purine2/data/cifar-10/";
string source = data_path + "cifar-10-test-lmdb";
string mean_file = data_path + "mean.binaryproto";

using namespace purine;

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    // initilize MPI
    int ret;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &ret));
    // create data parallelism of Nin_Cifar;
    shared_ptr<ComputeLoss<NIN_Cifar10<true> > > nin_cifar_test
        = make_shared<ComputeLoss<NIN_Cifar10<true> > >(0, 0, batch_size);
    // do the initialization
    nin_cifar_test->load("./nin_cifar_dump_iter_50000.snapshot");

    // iteration
    DTYPE loss = 0.0;
    DTYPE acc  = 0.0;
    std::vector<vector<DTYPE> >probs(10000, vector<DTYPE>(10, 0.0f));
    std::vector<int>labels(10000);
    for(int multi_view_id = 0; multi_view_id < 10; multi_view_id++){
        shared_ptr<FetchImage> fetch = make_shared<FetchImage>(source, mean_file,
                true, multi_view_id, 1.1,1.0,
                32, vector<vector<int> >{{0, 0, batch_size}});
        
        fetch->run();
        loss = 0.0;
        acc = 0.0;
        for (int iter = 0; iter < 100; ++iter) {
            if(multi_view_id == 0){
                Blob* sub_lables = fetch->get_master_cpu_labels();
                const DTYPE* data = sub_lables->tensor()->data();
                for(int i = 0; i < batch_size; i++){
                    labels[iter * batch_size + i] = data[i]; 
                    //printf("%f ", data[i]);
                }//printf("\n");
            }
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

            const DTYPE* sub_probs = nin_cifar_test->get_probs()[0]->tensor()->cpu_data();
            for(int j = 0; j < batch_size; j++){
                for(int k = 0; k < 10; k++){
                    probs[iter * batch_size + j][k] += sub_probs[j * 10 + k];
                }
            }
        }
        printf("cur: loss %f, acc %f\n", loss, acc / 100);
        acc = 0.0;
        for(int i = 0; i < 10000; i++){
            float max_ = -1;
            float cur_id = 0;
            for(int j = 0; j < 10; j++){
                if(max_ < probs[i][j]){
                    max_ = probs[i][j];
                    cur_id = j;
                }        
            }
            if(cur_id == labels[i]){
                acc += 1.0;            
            }
        }
        printf("voting: acc %f\n", acc / 10000);
    }

    // delete
    nin_cifar_test.reset();
    // Finalize MPI
    MPI_CHECK(MPI_Finalize());
    return 0;
}
