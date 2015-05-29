// Copyright Lin Min 2015
#include "operations/tensor.hpp"

namespace purine {

    Tensor::Tensor(int rank, int device, const Size& size, const Offset& offset,
            const Stride& stride) : size_(size), offset_(offset), stride_(stride),
    rank_(rank), device_(device) {
    }

    Tensor::Tensor(int rank, int device, const Size& size)
        : size_(size), rank_(rank), device_(device) {
            offset_ = Offset(0, 0, 0, 0);
            stride_ = Stride(size);
        }

    Tensor::~Tensor() {
        data_.reset();
    }

    int Tensor::offset(const Offset& off, const Stride& stride) {
        return off.noffset() * stride.nstride()
            + off.coffset() * stride.cstride()
            + off.hoffset() * stride.hstride()
            + off.woffset() * stride.wstride();
    }

    void Tensor::alloc_mem(DTYPE** data, const Size& size, int rank, int device) {
        CHECK_GT(size.count(), 0);
        CHECK_EQ(current_rank(), rank) << "Can't allocate memory on another machine";
        if (device < 0) {
            // #ifndef NDEBUG
            //     cudaHostAlloc(data, sizeof(DTYPE) * (1 + size.count()),
            //         cudaHostAllocPortable);
            // #else
            cudaHostAlloc(data, sizeof(DTYPE) * size.count(), cudaHostAllocPortable);
            // #endif
        } else {
            SWITCH_DEVICE(device);
            // #ifndef NDEBUG
            //     CUDA_CHECK(cudaMalloc(data, sizeof(DTYPE) * (1 + size.count())));
            // #else
            CUDA_CHECK(cudaMalloc(data, sizeof(DTYPE) * size.count()));
            // #endif
            SWITCH_BACK(device);
        }
    }

    void Tensor::free_mem(DTYPE* data, int rank, int device) {
        if (data == NULL) {
            return;
        }
        CHECK_EQ(current_rank(), rank) << "can't delete memory on another machine";
        if (device < 0) {
            cudaFreeHost(data);
        } else {
            SWITCH_DEVICE(device);
            CUDA_CHECK(cudaFree(data));
            SWITCH_BACK(device);
        }
    }

    void Tensor::swap_memory(Tensor* other) {
        // #ifndef NDEBUG
        //   DTYPE* tmp = other->past_the_end_;
        //   other->past_the_end_ = past_the_end_;
        //   past_the_end_ = tmp;
        // #endif
        CHECK_EQ(other->size_, size_);
        CHECK_EQ(other->stride_, stride_);
        CHECK_EQ(other->offset_, offset_);
        this->data_.swap(other->data_);
    }

    void Tensor::slice_from(Tensor* other, const Offset& off, const Size& size) {
        // #ifndef NDEBUG
        //   past_the_end_ = other->past_the_end_;
        // #endif
        rank_ = other->rank_;
        device_ = other->device_;
        stride_ = other->stride_;
        data_ = other->data_;
        size_ = size;
        offset_ += off;
    }

    void Tensor::share_from(Tensor* other) {
        // #ifndef NDEBUG
        //   past_the_end_ = other->past_the_end_;
        // #endif
        rank_ = other->rank_;
        device_ = other->device_;
        stride_ = other->stride_;
        data_ = other->data_;
        size_ = other->size_;
        offset_ = other->offset_;
    }

    void Tensor::delete_data() {
        data_.reset();
    }

    const DTYPE* Tensor::data() const {
        CHECK(data_);
        // #ifndef NDEBUG
        //   if (device_ < 0) {
        //     CHECK_EQ(*past_the_end_, 555.);
        //   } else {
        //     DTYPE flag = 0;
        //     SWITCH_DEVICE(device_);
        //     CUDA_CHECK(cudaMemcpy(&flag, past_the_end_, sizeof(DTYPE) * 1,
        //             cudaMemcpyDeviceToHost));
        //     SWITCH_BACK(device_);
        //     CHECK_EQ(flag, 555.);
        //   }
        // #endif
        return data_.get() + Tensor::offset(offset_, stride_);
    }

    DTYPE* Tensor::mutable_data() {
        CHECK_EQ(current_rank(), rank_) << "can't access data from a different rank";
        if (!data_) {
            CHECK(is_contiguous());
            DTYPE* ptr;
            Tensor::alloc_mem(&ptr, size_, rank_, device_);
            data_.reset(ptr, bind(Tensor::free_mem, std::placeholders::_1, rank_,
                        device_));
            // #ifndef NDEBUG
            //     past_the_end_ = data_.get() + size_.count();
            //     if (device_ < 0) {
            //       *past_the_end_ = 555.;
            //     } else {
            //       DTYPE flag = 555.;
            //       SWITCH_DEVICE(device_);
            //       CUDA_CHECK(cudaMemcpy(past_the_end_, &flag, sizeof(DTYPE) * 1,
            //               cudaMemcpyHostToDevice));
            //       SWITCH_BACK(device_);
            //     }
            // #endif
        }
        // #ifndef NDEBUG
        //   if (device_ < 0) {
        //     CHECK_EQ(*past_the_end_, 555.);
        //   } else {
        //     DTYPE flag = 0;
        //     SWITCH_DEVICE(device_);
        //     CUDA_CHECK(cudaMemcpy(&flag, past_the_end_, sizeof(DTYPE) * 1,
        //             cudaMemcpyDeviceToHost));
        //     SWITCH_BACK(device_);
        //     CHECK_EQ(flag, 555.);
        //   }
        // #endif
        return data_.get() + Tensor::offset(offset_, stride_);
    }

    bool Tensor::is_contiguous() const {
        return Stride(size_) == stride_;
    }

}
