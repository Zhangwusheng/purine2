#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <unistd.h>
#include <string>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "google/protobuf/message.h"
#include "caffeine/proto/caffe.pb.h"
#include "caffeine/math_functions.hpp"

using caffe::Datum;
using caffe::BlobProto;

using namespace std;

namespace caffe {

    using ::google::protobuf::Message;

    bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

    inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
        return ReadProtoFromBinaryFile(filename.c_str(), proto);
    }

    inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
        CHECK(ReadProtoFromBinaryFile(filename, proto));
    }

    inline void ReadProtoFromBinaryFileOrDie(const string& filename,
            Message* proto) {
        ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
    }

    void WriteProtoToBinaryFile(const Message& proto, const char* filename);
    inline void WriteProtoToBinaryFile(
            const Message& proto, const string& filename) {
        WriteProtoToBinaryFile(proto, filename.c_str());
    }
    
    cv::Mat DatumToCVMat(const Datum& datum, bool is_color);
    void CVMatToDatum(const cv::Mat& cv_img, Datum* datum);

    void mat2dtype(cv::Mat& in, std::vector<DTYPE>& out);
    cv::Mat dtype2mat(DTYPE* in, int channels, int rows, int cols);

}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
