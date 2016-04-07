// Copyright Lin Min 2015
#include "operations/include/drop.hpp"
#include "caffeine/caffeine.hpp"
#include <cuda.h>

namespace purine {

    Drop::Drop(const vector<Tensor*>& inputs,
            const vector<Tensor*>& outputs, const param_tuple& args)
        : Operation(inputs, outputs) {
            std::tie(rate_, dropVector_, isTest_) = args;
            dDropVector_ = NULL;
            //CHECK_EQ(inputs_[0]->size(), outputs_[0]->size());
        }

    Drop::~Drop() {
    }

    void Drop::compute_cpu(const vector<bool>& add){
        printf("Drop::Compute_cpu\n");
    }

    struct cuSize{
        int n, c, h, w;
    };

    struct cuOffset{
        int n, c, h, w;
    };

    struct cuStride{
        int n,c,h,w;
    };

    cuSize getCuSize( Size size ){
        cuSize ret;
        ret.n = size.num();
        ret.c = size.channels();
        ret.w = size.width();
        ret.h = size.height();
        return ret;
    }

    cuOffset getCuOffset( Offset offset ){
        cuOffset ret;
        ret.n = offset.noffset();
        ret.c = offset.coffset();
        ret.h = offset.hoffset();
        ret.w = offset.woffset();
        return ret;
    }

    cuStride getCuStride( Stride stride ){
        cuStride ret;
        ret.n = stride.nstride();
        ret.c = stride.cstride();
        ret.h = stride.hstride();
        ret.w = stride.wstride();
        return ret;
    }

    __device__ inline int getNStride(cuSize size){
        return size.c * size.h * size.w;
    }

    __device__ inline int getCStride(cuSize size){
        return size.h * size.w;
    }

    __device__ inline int getHStride(cuSize size){
        return size.w;
    }

    __device__ inline int getN(int index, cuSize size){
        int cur = index / getNStride(size);
        return cur;
    }

    __device__ inline int getC(int index, cuSize size){
        int cur = index % getNStride(size);
        cur = cur / getCStride(size);
        return cur;
    }

    __device__ inline int getH(int index, cuSize size){
        int cur = index % getCStride(size);
        cur = cur / getHStride(size);
        return cur;
    }

    __device__ inline int getW(int index, cuSize size){
        int cur = index % size.w;
        return cur;
    }

    __device__ inline int getIndex(int n, int c, int h, int w, cuOffset offset, cuStride stride){
        int n1 = n + offset.n;
        int c1 = c + offset.c;
        int h1 = h + offset.h;
        int w1 = w + offset.w;
        return n1 * stride.n + c1 * stride.c + h1 * stride.h + w1 * stride.w;
    }
    /*
       drop forward
     */
    __global__ void DropForward(const DTYPE* in, cuSize input_size, cuOffset input_offset, cuStride input_stride,
            DTYPE* out, cuSize output_size, cuOffset output_offset, cuStride output_stride,
            float* dropVector, 
            int data_size){
        int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
        int num_threads = blockDim.x * gridDim.x;
        for(int i = 0; i < data_size; i += num_threads){
            int index = i + thread_index;

            if(index < data_size){
                int n = getN(index, input_size);
                int c = getC(index, input_size);
                int h = getH(index, input_size);
                int w = getW(index, input_size);
                int input_index = getIndex(n,c,h,w, input_offset, input_stride);
                int output_index = getIndex(n,c,h,w, output_offset, output_stride);
                /*
                if(input_index != output_index){
                    printf("haahah\n");
                }
                if(input_index != index){
                    printf("hihihi\n");
                }
                if( n * input_size.c + c > input_size.n * input_size.c) {
                    printf("dropvector error\n");
                }
                */
                
                //out[index] = in[index] * alpha;
                //out[output_index] = in[input_index] * alpha;
                /*
                if( dropVector[n * input_size.c + c] < 0.001){
                    printf("papapu\n");
                }*/
                out[output_index] = in[input_index] * dropVector[n * input_size.c + c];
                //out[index] = in[index];

                //out[output_index] = in[input_index];
                /*if(input_index != index){
                    printf("forword wocaocaocao\n");
                }*/
                /*if(index > data_size - 50){
                  printf("%d %d %d %d\n", n, c, h, w);
                  }*/
                //if(index < 50){
                //    printf("%f %f alpha%f\n", out[index], in[index], alpha);
                //}
            }
        }
    }

    void printStride(cuStride stride){
        printf("stride %d %d %d %d\n", stride.n, stride.c, stride.h, stride.w);
    }

    void printOffset(cuOffset offset){
        printf("offset%d %d %d %d\n", offset.n, offset.c, offset.h, offset.w);
    }

    void printSize( cuSize size){
        printf("size %d %d %d %d\n", size.n, size.c, size.h, size.w);
    }

    void Drop::compute_gpu(const vector<bool>& add) {
        std::lock_guard<std::mutex>lock_guard_(outputs_[0]->get_mutex());
        Size s = inputs_[0]->size();
        //CHECK_EQ(inputs_[0]->size().count(), outputs_[0]->size().count());
        //CHECK_EQ(inputs_[1]->size().count(), inputs_[0]->size().count());
        int data_size = s.count();
        int num_blocks  = CAFFE_GET_BLOCKS(data_size);
        int num_threads = CAFFE_CUDA_NUM_THREADS;

        int nSizeOfFeature = s.num() * s.channels();
        if( isTest_ == true ){
            for( int i = 0; i < nSizeOfFeature; i++){
                dropVector_[i] = 1.0 - rate_; 
            }
        }
        else{
            for(int i = 0; i < nSizeOfFeature; i++){
                float tmp = 1.0f * rand() / RAND_MAX;
                if( rate_ <= tmp ){
                    dropVector_[i] = 1.0;
                }
                else{
                    dropVector_[i] = 0.0;
                }
            }
        }

        if(dDropVector_ == NULL)
            cudaMalloc((void**)&dDropVector_, nSizeOfFeature * sizeof(float));
        cudaMemcpy(dDropVector_, dropVector_, nSizeOfFeature * sizeof(float), cudaMemcpyHostToDevice);  

        //printf("up alpha %d %f rate %f \n", rate_, alpha, rate_[0]);
        cuSize input_size = getCuSize(inputs_[0]->size());
        cuOffset input_offset = getCuOffset(inputs_[0]->offset());
        cuStride input_stride = getCuStride(inputs_[0]->stride());

        cuSize output_size = getCuSize(outputs_[0]->size());
        cuOffset output_offset = getCuOffset(outputs_[0]->offset());
        cuStride output_stride = getCuStride(outputs_[0]->stride());
        /*
           printSize(input_size);
           printSize(output_size);
           printOffset(input_offset);
           printOffset(output_offset);
           printStride(input_stride);
           printStride(output_stride);
           printf("\n\n");
         */
        //cudaMemcpy(outputs_[0]->mutable_gpu_data(), inputs_[0]->gpu_data(), data_size * sizeof(float), cudaMemcpyDeviceToDevice);
        DropForward<<<num_blocks, num_threads, 0, stream()>>> 
            (inputs_[0]->gpu_data(), input_size, input_offset, input_stride,
             outputs_[0]->mutable_gpu_data(), output_size, output_offset, output_stride,
             dDropVector_, 
             data_size); 
        CUDA_POST_KERNEL_CHECK;
        (cudaDeviceSynchronize());
    }

    DropDown::DropDown(const vector<Tensor*>& inputs,
            const vector<Tensor*>& outputs, const param_tuple& args)
        : Operation(inputs, outputs) {
            std::tie(rate_, dropVector_) = args;
            dDropVector_ = NULL;
            //CHECK_EQ(inputs_[0]->size(), outputs_[0]->size());
        }

    DropDown::~DropDown() {
    }

    void DropDown::compute_cpu(const vector<bool>& add){
        printf("wocao\n");
    }
    /*
       drop backward
       B{ top_[1], top_[0], bottom_[0] } >> *activation_down >> B{ bottom_[1] }; 
     */
    __global__ void DropBackward(
            const DTYPE* in_diff, cuSize in_diff_size, cuOffset in_diff_offset, cuStride in_diff_stride,
            const DTYPE* in_data, cuSize in_data_size, cuOffset in_data_offset, cuStride in_data_stride,
            DTYPE* out_diff, cuSize out_diff_size, cuOffset out_diff_offset, cuStride out_diff_stride, 
            float* dropVector,
            int data_size){

        int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
        int num_threads = blockDim.x * gridDim.x;

        for(int i = 0; i < data_size; i += num_threads){
            int index = i + thread_index;
            if(index < data_size){
                int n = getN(index, in_diff_size);
                int c = getC(index, in_diff_size);
                int h = getH(index, in_diff_size);
                int w = getW(index, in_diff_size);
                int in_diff_index = getIndex(n,c,h,w, in_diff_offset, in_diff_stride);
                int in_data_index = getIndex(n,c,h,w, in_data_offset, in_data_stride);
                int output_diff_index = getIndex(n,c,h,w, out_diff_offset, out_diff_stride);
                //out_diff[output_diff_index] = in_diff[in_diff_index] * alpha;
                //out_diff[index] = in_diff[index] * alpha;
                /*
                if( output_diff_index != in_diff_index ){
                    printf("backward  output != index");
                }
                */
                //out_diff[output_diff_index] = in_diff[in_diff_index] * alpha;
                out_diff[output_diff_index] = in_diff[in_diff_index] * dropVector[n * in_diff_size.c + c];
                /*
                   if( dropVector[n * in_diff_size.c + c] < 0.001){
                   printf("papapu\n");
                   }*/

                /*
                   if( output_diff_index != index){
                   printf("co\n");
                   }
                   if( output_diff_index != in_diff_index){
                   printf("ho\n");
                   }*/

                /*if(index < 50){
                //printf("%f %f %f\n", out_diff[output_diff_index], in_diff[in_diff_index], in_data[in_data_index]);
                }*/
            }
        }
    }

    /*
       lrelu backward
       B{ top_[1], top_[0], bottom_[0] } >> *activation_down >> B{ bottom_[1] }; 
     */
    void DropDown::compute_gpu(const vector<bool>& add){
        std::lock_guard<std::mutex>lock_guard_(outputs_[0]->get_mutex());
        CHECK_EQ(inputs_[0]->size().count(), outputs_[0]->size().count());
        Size s = inputs_[0]->size();
        int data_size = s.count();
        int num_blocks  = CAFFE_GET_BLOCKS(data_size);
        int num_threads = CAFFE_CUDA_NUM_THREADS;

        cuSize input0_size = getCuSize(inputs_[0]->size());
        cuOffset input0_offset = getCuOffset(inputs_[0]->offset());
        cuStride input0_stride = getCuStride(inputs_[0]->stride());

        cuSize input1_size = getCuSize(inputs_[1]->size());
        cuOffset input1_offset = getCuOffset(inputs_[1]->offset());
        cuStride input1_stride = getCuStride(inputs_[1]->stride());

        cuSize output_size = getCuSize(outputs_[0]->size());
        cuOffset output_offset = getCuOffset(outputs_[0]->offset());
        cuStride output_stride = getCuStride(outputs_[0]->stride());


        int nSizeOfFeature = s.num() * s.channels();
        if(dDropVector_ == NULL)
            cudaMalloc((void**)&dDropVector_, nSizeOfFeature * sizeof(float));
        cudaMemcpy(dDropVector_, dropVector_, nSizeOfFeature * sizeof(float), cudaMemcpyHostToDevice);  

        /*printSize(input0_size);
          printSize(input1_size);
          printSize(output_size);

          printOffset(input0_offset);
          printOffset(input1_offset);
          printOffset(output_offset);

          printStride(input0_stride);
          printStride(input1_stride);
          printStride(output_stride);
          printf("\n\n");
         */
        // cudaMemcpy(outputs_[0]->mutable_gpu_data(), inputs_[0]->gpu_data(), data_size * sizeof(float), cudaMemcpyDeviceToDevice);
        DropBackward<<< num_blocks, num_threads, 0, stream()>>>(
                inputs_[0]->gpu_data(), input0_size, input0_offset, input0_stride,
                inputs_[1]->gpu_data(), input1_size, input1_offset, input1_stride,
                outputs_[0]->mutable_gpu_data(), output_size, output_offset, output_stride,
                dDropVector_, 
                data_size);
        CUDA_POST_KERNEL_CHECK;
        (cudaDeviceSynchronize());
    }

}
