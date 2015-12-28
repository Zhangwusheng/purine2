// Copyright Lin Min 2015

#include <mpi.h>
#include <vector>
#include <time.h>
#include <sys/time.h>
#include <string>
#include <glog/logging.h>
#include "common/common.hpp"
#include "examples/asyn_nin_cifar10.hpp"
#include "composite/graph/all_reduce.hpp"
#include "composite/graph/asgd_net.hpp"
#include "composite/composite.hpp"
#include "operations/tensor.hpp"
#include "composite/graph/asgd_data_parallel.hpp"

string data_path = "/home/zhxfl/purine2/data/cifar-10/";
string source =    data_path + "cifar-10-train-lmdb";
string mean_file = data_path + "mean.binaryproto";

using namespace purine;

std::vector<int> param_rank;
std::vector<int> param_device;
std::vector<AllReduce::param_tuple >param(18);
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
        param.push_back(AllReduce::param_tuple{0.9f, learning_rate,
            learning_rate * global_decay * (i % 2 ? 0.f : 1.f)}
            );
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
    DTYPE global_learning_rate = 0.05;
    DTYPE global_decay = 0.0001;
    setup_param_server(global_learning_rate, global_decay);
    AsgdDataParallel<Asyn_NIN_Cifar10<false>, AllReduce>* data_parallel = new AsgdDataParallel<Asyn_NIN_Cifar10<false>, AllReduce>(parallels);
    data_parallel->setup_param_server(param_rank, param_device, param);
    // do the initialization
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
    data_parallel->init<Constant>(bias_indice, Constant::param_tuple(0.));
    data_parallel->init<Gaussian>(weight_indice,
            Gaussian::param_tuple(0., 0.05));

//#define LOAD
#ifdef LOAD 
    load("./nin_cifar_dump_iter_50000.snapshot");
#endif
    int fetch_count = 0;
    int save_fetch = 5000;
    double period = 2.0; // second

    int cur_fetch_count = 1;
    int iter = 0;

    while(iter < 30000){
        if(iter == 20000 || iter == 25000){
            global_learning_rate /= 10;
        }

        if(iter == 100){
            period += 0.5;
        }
        if(iter == 200){
            period += 0.5;
        }

        iter++;
        data_parallel->set_period(period);
        data_parallel->run();
        if(current_rank() == 0)
        {
            cur_fetch_count = data_parallel->fetch_count();
            vector<DTYPE> ret = data_parallel->loss();
            fetch_count += cur_fetch_count;
            MPI_LOG(<< "iter " << iter <<
                    " loss " << ret[0] << 
                    " accuracy " << ret[1] << 
                    " period " << period << 
                    " fetch_count " << cur_fetch_count << "\\" << fetch_count);
        }

        if(save_fetch < fetch_count){
            data_parallel->save("./nin_cifar_dump_iter_" + to_string(save_fetch) + ".snapshot");
            save_fetch += 5000;
        }

    }
    // Finalize MPI
    MPI_CHECK(MPI_Finalize());
    return 0;
}
