#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "caffeine/io.hpp"
#include "caffeine/proto/caffe.pb.h"
#include "caffeine/math_functions.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core_c.h>
using caffe::Datum;
using caffe::BlobProto;

namespace caffe {

    using google::protobuf::io::FileInputStream;
    using google::protobuf::io::FileOutputStream;
    using google::protobuf::io::ZeroCopyInputStream;
    using google::protobuf::io::CodedInputStream;
    using google::protobuf::io::ZeroCopyOutputStream;
    using google::protobuf::io::CodedOutputStream;
    using google::protobuf::Message;

    bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
        int fd = open(filename, O_RDONLY);
        CHECK_NE(fd, -1) << "File not found: " << filename;
        ZeroCopyInputStream* raw_input = new FileInputStream(fd);
        CodedInputStream* coded_input = new CodedInputStream(raw_input);
        coded_input->SetTotalBytesLimit(1073741824, 536870912);

        bool success = proto->ParseFromCodedStream(coded_input);

        delete coded_input;
        delete raw_input;
        close(fd);
        return success;
    }
    void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
        fstream output(filename, ios::out | ios::trunc | ios::binary);
        CHECK(proto.SerializeToOstream(&output));
    }

    cv::Mat DatumToCVMat(const Datum& datum, bool is_color) {
        cv::Mat cv_img;
        const string& data = datum.data();
        int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
        cv_img = cv::imdecode(cv::Mat(datum.width(), datum.height(), is_color ? CV_8UC3: CV_8U, (void*)data.c_str() ), cv_read_flag);
        if (!cv_img.data) {
            LOG(ERROR) << "Could not decode datum ";
        }
        return cv_img;
    }

    void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
        CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
        datum->set_channels(cv_img.channels());
        datum->set_height(cv_img.rows);
        datum->set_width(cv_img.cols);
        datum->clear_data();
        datum->clear_float_data();
        
        int datum_channels = datum->channels();
        int datum_height = datum->height();
        int datum_width = datum->width();
        int datum_size = datum_channels * datum_height * datum_width;
        std::string buffer(datum_size, ' ');
        for (int h = 0; h < datum_height; ++h) {
            const uchar* ptr = cv_img.ptr<uchar>(h);
            int img_index = 0;
            for (int w = 0; w < datum_width; ++w) {
                for (int c = 0; c < datum_channels; ++c) {
                    int datum_index = (c * datum_height + h) * datum_width + w;
                    buffer[datum_index] = static_cast<char>(ptr[img_index++]);
                }
            } 
        }
        datum->set_data(buffer);
    }

    cv::Mat dtype2mat(DTYPE* in, int channels, int rows, int cols){
        if(channels == 3){
            cv::Mat out(rows, cols, CV_32FC3);
            for(int i = 0; i < rows; i++){
                for(int j = 0; j < cols; j++){
                    out.at<cv::Vec3f>(i, j) = cv::Vec3f(
                            in[0 * rows * cols + i * cols + j],
                            in[1 * rows * cols + i * cols + j],
                            in[2 * rows * cols + i * cols + j]);
                }
            }
            return out;
        }else{
            cv::Mat out(rows, cols, CV_32FC1);
            for(int i = 0; i < rows; i++){
                for(int j = 0; j < cols; j++){
                    out.at<DTYPE>(i, j) = in[rows * cols + i * cols + j];
                }
            }
            return out;
        }
    }

    void mat2dtype(cv::Mat& in, std::vector<DTYPE>& out){
        int channels = in.channels();
        int rows     = in.rows;
        int cols     = in.cols;
        
        out.resize(channels * rows * cols);
        
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                cv::Vec3f v3f = in.at<cv::Vec3f>(i, j);
                out[0 * rows * cols + i * cols + j] = v3f.val[0];
                out[1 * rows * cols + i * cols + j] = v3f.val[1];
                out[2 * rows * cols + i * cols + j] = v3f.val[2];
            }
        }
    }

}  // namespace caffe
