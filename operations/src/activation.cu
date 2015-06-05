// Copyright Lin Min 2015
#include "operations/include/activation.hpp"
#include "caffeine/caffeine.hpp"

namespace purine {

    Activation::Activation(const vector<Tensor*>& inputs,
            const vector<Tensor*>& outputs, const param_tuple& args)
        : Operation(inputs, outputs) {
            std::tie(mode_) = args;
            CHECK_EQ(inputs_[0]->size(), outputs_[0]->size());
            Size bottom_size = inputs_[0]->size();
            Stride bottom_stride = inputs_[0]->stride();
            Size top_size = outputs_[0]->size();
            Stride top_stride = outputs_[0]->stride();
            cudnn::createTensor4dDesc<DTYPE>(&bottom_desc_, bottom_size, bottom_stride);
            cudnn::createTensor4dDesc<DTYPE>(&top_desc_, top_size, top_stride);
            if (mode_ == "relu") {
                activation_mode_ = CUDNN_ACTIVATION_RELU;
            } else if (mode_ == "sigmoid") {
                activation_mode_ = CUDNN_ACTIVATION_SIGMOID;
            } else if (mode_ == "tanh") {
                activation_mode_ = CUDNN_ACTIVATION_TANH;
            } else if (mode_ == "lrelu"){
            } else{
                LOG(FATAL) << "Unknown activation mode " << mode_;
            }
        }

    Activation::~Activation() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
    }

    /*
       lrelu forward
     */
    __global__ void LreluForward(const DTYPE* in, DTYPE* out, int data_size){
        int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
        int num_threads = blockDim.x * gridDim.x;
        for(int i = 0; i < data_size; i += num_threads){
            int index = i + thread_index;
            out[index] = in[index] > 0 ? in[index] : in[index] * 0.01;
        }
    }

    void Activation::compute_gpu(const vector<bool>& add) {
        
        if(mode_ == "lrelu"){
            Size s = inputs_[0]->size();
            int data_size = s.num() * s.channels() * s.height() * s.width();
            int num_blocks  = std::min(CAFFE_GET_BLOCKS(data_size), 1024);
            int num_threads = CAFFE_CUDA_NUM_THREADS;
            LreluForward<<<num_blocks, num_threads, 0, stream()>>> 
                (inputs_[0]->gpu_data(), 
                    outputs_[0]->mutable_gpu_data(),
                    data_size); 
            CUDA_POST_KERNEL_CHECK;
        }
        else if(mode_ == "relu" || mode_ == "sigmoid" || mode_ == "tanh"){
            DTYPE alpha = 1.;
            DTYPE beta = add[0] ? 1. : 0.;
            CUDNN_CHECK(cudnnActivationForward(cudnn_handle(), activation_mode_,
                        &alpha, bottom_desc_, inputs_[0]->gpu_data(), &beta, top_desc_,
                        outputs_[0]->mutable_gpu_data()));
        }
        else {
            LOG(FATAL) << "Unknown activation mode " << mode_;
        }
    }

    ActivationDown::ActivationDown(const vector<Tensor*>& inputs,
            const vector<Tensor*>& outputs, const param_tuple& args)
        : Operation(inputs, outputs) {
            std::tie(mode_) = args;
            CHECK_EQ(inputs_[0]->size(), outputs_[0]->size());
            Size bottom_size = outputs_[0]->size();
            Stride bottom_stride = outputs_[0]->stride();
            Size top_size = inputs_[0]->size();
            Stride top_stride = inputs_[0]->stride();
            cudnn::createTensor4dDesc<DTYPE>(&bottom_desc_, bottom_size, bottom_stride);
            cudnn::createTensor4dDesc<DTYPE>(&top_desc_, top_size, top_stride);
            if (mode_ == "relu") {
                activation_mode_ = CUDNN_ACTIVATION_RELU;
            } else if (mode_ == "sigmoid") {
                activation_mode_ = CUDNN_ACTIVATION_SIGMOID;
            } else if (mode_ == "tanh") {
                activation_mode_ = CUDNN_ACTIVATION_TANH;
            } else if (mode_ == "lrelu"){
            }
            else {
                LOG(FATAL) << "Unknown activation mode " << mode_;
            }
        }

    ActivationDown::~ActivationDown() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
    }

    /*
       lrelu backward
       B{ top_[1], top_[0], bottom_[0] } >> *activation_down >> B{ bottom_[1] }; 
     */
    __global__ void LreluBackward(const DTYPE* in_diff, const DTYPE* in_data, 
            DTYPE* out_diff, int data_size){
        
        int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
        int num_threads = blockDim.x * gridDim.x;

        for(int i = 0; i < data_size; i += num_threads){
            int index = i + thread_index;
            if(index < data_size){
                out_diff[index] = in_diff[index] * ((in_data[index] > 0)
                                + (in_data[index] <= 0) * 0.01);
            }
        }
    }

    /*
       lrelu backward
       B{ top_[1], top_[0], bottom_[0] } >> *activation_down >> B{ bottom_[1] }; 
     */
    void ActivationDown::compute_gpu(const vector<bool>& add) {
        if(mode_ == "lrelu"){
            Size s = inputs_[0]->size();
            int data_size = s.num() * s.channels() * s.height() * s.width();
            int num_blocks  = std::min(CAFFE_GET_BLOCKS(data_size), 1024);
            int num_threads = CAFFE_CUDA_NUM_THREADS;
            LreluBackward<<< num_blocks, num_threads, 0, stream()>>>(
                inputs_[0]->gpu_data(), 
                inputs_[1]->gpu_data(),
                outputs_[0]->mutable_gpu_data(),
                data_size);
            CUDA_POST_KERNEL_CHECK;
        }
        else if(mode_ == "relu" || mode_ == "sigmoid" || mode_ == "tanh"){
            DTYPE alpha = 1.;
            DTYPE beta = add[0] ? 1. : 0.;
            /*
             */
            CUDNN_CHECK(cudnnActivationBackward(cudnn_handle(), activation_mode_,
                        &alpha, top_desc_, inputs_[1]->gpu_data(), top_desc_,
                        inputs_[0]->gpu_data(), bottom_desc_, inputs_[2]->gpu_data(),
                        &beta, bottom_desc_, outputs_[0]->mutable_gpu_data()));
        } 
    }

}
