// Copyright Lin Min 2015

#include "caffeine/io.hpp"
#include "caffeine/proto/caffe.pb.h"
#include "operations/include/image_label.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core_c.h>
#include "common/common.hpp"
using caffe::BlobProto;
using caffe::Datum;
using namespace std;

namespace purine {

    static void TensorFromBlob(const BlobProto& proto, Tensor* tensor) {
        Size s = tensor->size();
        DTYPE* data_vec = tensor->mutable_cpu_data();
        for (int i = 0; i < s.count(); ++i) {
            data_vec[i] = proto.data(i);
        }
    }

    ImageLabel::ImageLabel(const vector<Tensor*>& inputs,
            const vector<Tensor*>& outputs, const param_tuple& args)
        : Operation(inputs, outputs) {
            std::tie(source, mean, mirror, random, color, multi_view_id, scale, angle, 
                    offset, interval,
                    batch_size, crop_size)
                = args;

            CHECK_EQ(batch_size, outputs_[0]->size().num());
            CHECK_EQ(batch_size, outputs_[1]->size().num());
            CHECK_EQ(crop_size, outputs_[0]->size().height());
            CHECK_EQ(crop_size, outputs_[0]->size().width());
            mean_.reset(new Tensor(current_rank(), -1,
                        {1, color ? 3 : 1, crop_size, crop_size}));
            if (!(mean == "")) {
                BlobProto blob_proto;
                ReadProtoFromBinaryFileOrDie(mean, &blob_proto);
                TensorFromBlob(blob_proto, mean_.get());
            } else {
                caffe::caffe_memset(mean_->size().count() * sizeof(DTYPE), 0,
                        mean_->mutable_cpu_data());
            }
            CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS)
                << "mdb_env_create failed";
            CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);
            CHECK_EQ(mdb_env_open(mdb_env_, source.c_str(), MDB_RDONLY|MDB_NOTLS,
                        0664), MDB_SUCCESS) << "mdb_env_open failed";
            CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_),
                    MDB_SUCCESS) << "mdb_txn_begin failed";
            CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
                << "mdb_open failed";
            CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
                << "mdb_cursor_open failed";
            // LOG(INFO) << "Opening lmdb " << source;
            CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
                    MDB_SUCCESS) << "mdb_cursor_get failed";
            // go to the offset
            for (int i = 0; i < offset; ++i) {
                if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
                        != MDB_SUCCESS) {
                    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
                                MDB_FIRST), MDB_SUCCESS);
                }
            }
        }

    void rotateNScale(const cv::Mat &_from, cv::Mat &_to, double angle, double scale){
        cv::Point center = cv::Point(_from.cols / 2, _from.rows / 2);
        // Get the rotation matrix with the specifications above
        cv::Mat rot_mat = getRotationMatrix2D(center, angle, scale);
        // Rotate the warped image
        warpAffine(_from, _to, rot_mat, _to.size());
    }


    void resize_rotate_image(Datum& datum, std::vector<DTYPE>&out, const DTYPE* mean, float scale, float angle){
        int width = datum.width();
        int height = datum.height();
        int channels = datum.channels();
        int size = width * height * channels;
        if(fabs(scale - 1.0) < 0.000000001){
            out.resize(size);
            for(int index = 0; index < size; index++){
                out[index] = static_cast<DTYPE>(static_cast<uint8_t>(datum.data()[index])) - mean[index];
            }
            return;
        }
        else{
            std::vector<DTYPE>sub_mean_data(size);
            
            for(int index = 0; index < size; index++){
                sub_mean_data[index] = static_cast<DTYPE>(static_cast<uint8_t>(datum.data()[index])) - mean[index];
            }

            cv::Mat src = caffe::dtype2mat(sub_mean_data.data(), channels, width, height);
            //cv::Mat sik;
            cv::Mat sik(scale * src.rows,
                    scale * src.cols, 
                    channels == 3 ? CV_32FC3: CV_32FC1);
            rotateNScale(src, sik, angle, scale);
            //cv::resize(src, sik, cv::Size(scale* src.rows, scale * src.cols));
            caffe::mat2dtype(sik, out);
        }
    }

    void ImageLabel::compute_cpu(const vector<bool>& add) {
        Datum datum;
        const DTYPE* mean = mean_->data();
        DTYPE* top_data = outputs_[0]->mutable_cpu_data();
        DTYPE* top_label = outputs_[1]->mutable_cpu_data();


        for (int item_id = 0; item_id < batch_size; ++item_id) {
            CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
                        MDB_GET_CURRENT), MDB_SUCCESS);
            datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
            CHECK(mean_->size().channels() == datum.channels()) << " mean_.channels != datum.channels";
            CHECK(mean_->size().width() == datum.width()) << " mean_.width != datum.width";
            CHECK(mean_->size().height() == datum.height()) << " mean_.height != datum.height";

            std::vector<DTYPE>data;

            // scale
            float cur_scale = 1;
            if(multi_view_id == -1){
                cur_scale = 1.0 + (scale - 1.0) * rand() / RAND_MAX;
            }
            else{
                int tmp = multi_view_id / (10 * 3);
                multi_view_id = multi_view_id % (10 * 3);
                if( tmp == 0 ) cur_scale = scale;
                else if( tmp == 1 ) cur_scale = 1.0;
                else if( tmp == 2 ) cur_scale = (scale - 1.0) / 2 + scale;
                else{
                    printf("imagle_label cur_scale error\n");
                    exit(0);
                }
            }

            // rotate
            float cur_angle = 0.0;
            if( multi_view_id == -1){
                cur_angle = angle * ( 2.0 * rand() / RAND_MAX - 1.0);
            }
            else{
                int tmp = multi_view_id / 10;
                multi_view_id = multi_view_id % 10;
                if( tmp == 0 ) cur_angle = 0.0;
                else if( tmp == 1 ) cur_angle = angle;
                else if( tmp == 2 ) cur_angle = - angle;
                else{
                    printf("imagle_label cur_angle error\n");
                    exit(0);
                }
            }

            resize_rotate_image(datum, data, mean, cur_scale, cur_angle);
            

            //crop
            int height = cur_scale * datum.height();
            int width = cur_scale * datum.width();
            int channels = cur_scale * datum.channels();

            int h_off, w_off;
            if(multi_view_id == -1){// trainning 
                if (random) {
                    if(height == crop_size)
                        h_off = 0;
                    else 
                        h_off = caffe::caffe_rng_rand() % (height - crop_size);

                    if(width == crop_size)
                        w_off = 0;
                    else
                        w_off = caffe::caffe_rng_rand() % (width - crop_size);
                } else {
                    h_off = (height - crop_size) / 2;
                    w_off = (width - crop_size) / 2;
                }
            }
            else{// test
                int crop_w = width  - crop_size;
                int crop_h = height - crop_size;
                static const bool mirror_[10] = {1,      1,          1,      1,      1, 0,      0,          0,      0,      0};
                static const int  h_o[10]     = {0,      0, crop_h / 2, crop_h, crop_h, 0,      0, crop_h / 2, crop_h, crop_h};
                static const int  w_o[10]     = {0, crop_w, crop_w / 2,      0, crop_w, 0, crop_w, crop_w / 2,      0, crop_w};

                mirror = mirror_[multi_view_id];
                h_off = h_o[multi_view_id]; 
                w_off = w_o[multi_view_id];   
            }

            if ((multi_view_id == -1 && mirror && caffe::caffe_rng_rand() % 2) //training 
                    ||(multi_view_id >= 0 && mirror) // testing
               )
            { 
                // Copy mirrored version
                for (int c = 0; c < channels; ++c) {
                    for (int h = 0; h < crop_size; ++h) {
                        for (int w = 0; w < crop_size; ++w) {
                            int top_index = ((item_id * channels + c) * crop_size + h)
                                * crop_size + (crop_size - 1 - w);
                            int data_index = (c * height + h + h_off) * width + w + w_off;
                            top_data[top_index] = data[data_index];
                        }
                    }
                }
            } else {
                // Normal copy
                for (int c = 0; c < channels; ++c) {
                    for (int h = 0; h < crop_size; ++h) {
                        for (int w = 0; w < crop_size; ++w) {
                            int top_index = ((item_id * channels + c) * crop_size + h)
                                * crop_size + w;
                            int data_index = (c * height + h + h_off) * width + w + w_off;
                            top_data[top_index] = data[data_index];
                        }
                    }
                }
            }

            top_label[item_id] = datum.label();
            if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
                    != MDB_SUCCESS) {
                // We have reached the end. Restart from the first.
                // DLOG(INFO) << "Restarting data prefetching from start.";
                CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
                            MDB_FIRST), MDB_SUCCESS);
            }
        }

    }

    ImageLabel::~ImageLabel() {
        mdb_cursor_close(mdb_cursor_);
        mdb_close(mdb_env_, mdb_dbi_);
        mdb_txn_abort(mdb_txn_);
        mdb_env_close(mdb_env_);
    }

}
