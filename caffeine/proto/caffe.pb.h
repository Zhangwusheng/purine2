// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: caffe.proto

#ifndef PROTOBUF_caffe_2eproto__INCLUDED
#define PROTOBUF_caffe_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 2004000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 2004001 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/generated_message_reflection.h>
// @@protoc_insertion_point(includes)

namespace caffe {

// Internal implementation detail -- do not call these.
void  protobuf_AddDesc_caffe_2eproto();
void protobuf_AssignDesc_caffe_2eproto();
void protobuf_ShutdownFile_caffe_2eproto();

class BlobProto;
class Datum;

// ===================================================================

class BlobProto : public ::google::protobuf::Message {
 public:
  BlobProto();
  virtual ~BlobProto();
  
  BlobProto(const BlobProto& from);
  
  inline BlobProto& operator=(const BlobProto& from) {
    CopyFrom(from);
    return *this;
  }
  
  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }
  
  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }
  
  static const ::google::protobuf::Descriptor* descriptor();
  static const BlobProto& default_instance();
  
  void Swap(BlobProto* other);
  
  // implements Message ----------------------------------------------
  
  BlobProto* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const BlobProto& from);
  void MergeFrom(const BlobProto& from);
  void Clear();
  bool IsInitialized() const;
  
  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:
  
  ::google::protobuf::Metadata GetMetadata() const;
  
  // nested types ----------------------------------------------------
  
  // accessors -------------------------------------------------------
  
  // optional int32 num = 1 [default = 0];
  inline bool has_num() const;
  inline void clear_num();
  static const int kNumFieldNumber = 1;
  inline ::google::protobuf::int32 num() const;
  inline void set_num(::google::protobuf::int32 value);
  
  // optional int32 channels = 2 [default = 0];
  inline bool has_channels() const;
  inline void clear_channels();
  static const int kChannelsFieldNumber = 2;
  inline ::google::protobuf::int32 channels() const;
  inline void set_channels(::google::protobuf::int32 value);
  
  // optional int32 height = 3 [default = 0];
  inline bool has_height() const;
  inline void clear_height();
  static const int kHeightFieldNumber = 3;
  inline ::google::protobuf::int32 height() const;
  inline void set_height(::google::protobuf::int32 value);
  
  // optional int32 width = 4 [default = 0];
  inline bool has_width() const;
  inline void clear_width();
  static const int kWidthFieldNumber = 4;
  inline ::google::protobuf::int32 width() const;
  inline void set_width(::google::protobuf::int32 value);
  
  // repeated float data = 5 [packed = true];
  inline int data_size() const;
  inline void clear_data();
  static const int kDataFieldNumber = 5;
  inline float data(int index) const;
  inline void set_data(int index, float value);
  inline void add_data(float value);
  inline const ::google::protobuf::RepeatedField< float >&
      data() const;
  inline ::google::protobuf::RepeatedField< float >*
      mutable_data();
  
  // repeated float diff = 6 [packed = true];
  inline int diff_size() const;
  inline void clear_diff();
  static const int kDiffFieldNumber = 6;
  inline float diff(int index) const;
  inline void set_diff(int index, float value);
  inline void add_diff(float value);
  inline const ::google::protobuf::RepeatedField< float >&
      diff() const;
  inline ::google::protobuf::RepeatedField< float >*
      mutable_diff();
  
  // @@protoc_insertion_point(class_scope:caffe.BlobProto)
 private:
  inline void set_has_num();
  inline void clear_has_num();
  inline void set_has_channels();
  inline void clear_has_channels();
  inline void set_has_height();
  inline void clear_has_height();
  inline void set_has_width();
  inline void clear_has_width();
  
  ::google::protobuf::UnknownFieldSet _unknown_fields_;
  
  ::google::protobuf::int32 num_;
  ::google::protobuf::int32 channels_;
  ::google::protobuf::int32 height_;
  ::google::protobuf::int32 width_;
  ::google::protobuf::RepeatedField< float > data_;
  mutable int _data_cached_byte_size_;
  ::google::protobuf::RepeatedField< float > diff_;
  mutable int _diff_cached_byte_size_;
  
  mutable int _cached_size_;
  ::google::protobuf::uint32 _has_bits_[(6 + 31) / 32];
  
  friend void  protobuf_AddDesc_caffe_2eproto();
  friend void protobuf_AssignDesc_caffe_2eproto();
  friend void protobuf_ShutdownFile_caffe_2eproto();
  
  void InitAsDefaultInstance();
  static BlobProto* default_instance_;
};
// -------------------------------------------------------------------

class Datum : public ::google::protobuf::Message {
 public:
  Datum();
  virtual ~Datum();
  
  Datum(const Datum& from);
  
  inline Datum& operator=(const Datum& from) {
    CopyFrom(from);
    return *this;
  }
  
  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }
  
  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }
  
  static const ::google::protobuf::Descriptor* descriptor();
  static const Datum& default_instance();
  
  void Swap(Datum* other);
  
  // implements Message ----------------------------------------------
  
  Datum* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const Datum& from);
  void MergeFrom(const Datum& from);
  void Clear();
  bool IsInitialized() const;
  
  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:
  
  ::google::protobuf::Metadata GetMetadata() const;
  
  // nested types ----------------------------------------------------
  
  // accessors -------------------------------------------------------
  
  // optional int32 channels = 1;
  inline bool has_channels() const;
  inline void clear_channels();
  static const int kChannelsFieldNumber = 1;
  inline ::google::protobuf::int32 channels() const;
  inline void set_channels(::google::protobuf::int32 value);
  
  // optional int32 height = 2;
  inline bool has_height() const;
  inline void clear_height();
  static const int kHeightFieldNumber = 2;
  inline ::google::protobuf::int32 height() const;
  inline void set_height(::google::protobuf::int32 value);
  
  // optional int32 width = 3;
  inline bool has_width() const;
  inline void clear_width();
  static const int kWidthFieldNumber = 3;
  inline ::google::protobuf::int32 width() const;
  inline void set_width(::google::protobuf::int32 value);
  
  // optional bytes data = 4;
  inline bool has_data() const;
  inline void clear_data();
  static const int kDataFieldNumber = 4;
  inline const ::std::string& data() const;
  inline void set_data(const ::std::string& value);
  inline void set_data(const char* value);
  inline void set_data(const void* value, size_t size);
  inline ::std::string* mutable_data();
  inline ::std::string* release_data();
  
  // optional int32 label = 5;
  inline bool has_label() const;
  inline void clear_label();
  static const int kLabelFieldNumber = 5;
  inline ::google::protobuf::int32 label() const;
  inline void set_label(::google::protobuf::int32 value);
  
  // repeated float float_data = 6;
  inline int float_data_size() const;
  inline void clear_float_data();
  static const int kFloatDataFieldNumber = 6;
  inline float float_data(int index) const;
  inline void set_float_data(int index, float value);
  inline void add_float_data(float value);
  inline const ::google::protobuf::RepeatedField< float >&
      float_data() const;
  inline ::google::protobuf::RepeatedField< float >*
      mutable_float_data();
  
  // @@protoc_insertion_point(class_scope:caffe.Datum)
 private:
  inline void set_has_channels();
  inline void clear_has_channels();
  inline void set_has_height();
  inline void clear_has_height();
  inline void set_has_width();
  inline void clear_has_width();
  inline void set_has_data();
  inline void clear_has_data();
  inline void set_has_label();
  inline void clear_has_label();
  
  ::google::protobuf::UnknownFieldSet _unknown_fields_;
  
  ::google::protobuf::int32 channels_;
  ::google::protobuf::int32 height_;
  ::std::string* data_;
  ::google::protobuf::int32 width_;
  ::google::protobuf::int32 label_;
  ::google::protobuf::RepeatedField< float > float_data_;
  
  mutable int _cached_size_;
  ::google::protobuf::uint32 _has_bits_[(6 + 31) / 32];
  
  friend void  protobuf_AddDesc_caffe_2eproto();
  friend void protobuf_AssignDesc_caffe_2eproto();
  friend void protobuf_ShutdownFile_caffe_2eproto();
  
  void InitAsDefaultInstance();
  static Datum* default_instance_;
};
// ===================================================================


// ===================================================================

// BlobProto

// optional int32 num = 1 [default = 0];
inline bool BlobProto::has_num() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void BlobProto::set_has_num() {
  _has_bits_[0] |= 0x00000001u;
}
inline void BlobProto::clear_has_num() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void BlobProto::clear_num() {
  num_ = 0;
  clear_has_num();
}
inline ::google::protobuf::int32 BlobProto::num() const {
  return num_;
}
inline void BlobProto::set_num(::google::protobuf::int32 value) {
  set_has_num();
  num_ = value;
}

// optional int32 channels = 2 [default = 0];
inline bool BlobProto::has_channels() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void BlobProto::set_has_channels() {
  _has_bits_[0] |= 0x00000002u;
}
inline void BlobProto::clear_has_channels() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void BlobProto::clear_channels() {
  channels_ = 0;
  clear_has_channels();
}
inline ::google::protobuf::int32 BlobProto::channels() const {
  return channels_;
}
inline void BlobProto::set_channels(::google::protobuf::int32 value) {
  set_has_channels();
  channels_ = value;
}

// optional int32 height = 3 [default = 0];
inline bool BlobProto::has_height() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void BlobProto::set_has_height() {
  _has_bits_[0] |= 0x00000004u;
}
inline void BlobProto::clear_has_height() {
  _has_bits_[0] &= ~0x00000004u;
}
inline void BlobProto::clear_height() {
  height_ = 0;
  clear_has_height();
}
inline ::google::protobuf::int32 BlobProto::height() const {
  return height_;
}
inline void BlobProto::set_height(::google::protobuf::int32 value) {
  set_has_height();
  height_ = value;
}

// optional int32 width = 4 [default = 0];
inline bool BlobProto::has_width() const {
  return (_has_bits_[0] & 0x00000008u) != 0;
}
inline void BlobProto::set_has_width() {
  _has_bits_[0] |= 0x00000008u;
}
inline void BlobProto::clear_has_width() {
  _has_bits_[0] &= ~0x00000008u;
}
inline void BlobProto::clear_width() {
  width_ = 0;
  clear_has_width();
}
inline ::google::protobuf::int32 BlobProto::width() const {
  return width_;
}
inline void BlobProto::set_width(::google::protobuf::int32 value) {
  set_has_width();
  width_ = value;
}

// repeated float data = 5 [packed = true];
inline int BlobProto::data_size() const {
  return data_.size();
}
inline void BlobProto::clear_data() {
  data_.Clear();
}
inline float BlobProto::data(int index) const {
  return data_.Get(index);
}
inline void BlobProto::set_data(int index, float value) {
  data_.Set(index, value);
}
inline void BlobProto::add_data(float value) {
  data_.Add(value);
}
inline const ::google::protobuf::RepeatedField< float >&
BlobProto::data() const {
  return data_;
}
inline ::google::protobuf::RepeatedField< float >*
BlobProto::mutable_data() {
  return &data_;
}

// repeated float diff = 6 [packed = true];
inline int BlobProto::diff_size() const {
  return diff_.size();
}
inline void BlobProto::clear_diff() {
  diff_.Clear();
}
inline float BlobProto::diff(int index) const {
  return diff_.Get(index);
}
inline void BlobProto::set_diff(int index, float value) {
  diff_.Set(index, value);
}
inline void BlobProto::add_diff(float value) {
  diff_.Add(value);
}
inline const ::google::protobuf::RepeatedField< float >&
BlobProto::diff() const {
  return diff_;
}
inline ::google::protobuf::RepeatedField< float >*
BlobProto::mutable_diff() {
  return &diff_;
}

// -------------------------------------------------------------------

// Datum

// optional int32 channels = 1;
inline bool Datum::has_channels() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void Datum::set_has_channels() {
  _has_bits_[0] |= 0x00000001u;
}
inline void Datum::clear_has_channels() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void Datum::clear_channels() {
  channels_ = 0;
  clear_has_channels();
}
inline ::google::protobuf::int32 Datum::channels() const {
  return channels_;
}
inline void Datum::set_channels(::google::protobuf::int32 value) {
  set_has_channels();
  channels_ = value;
}

// optional int32 height = 2;
inline bool Datum::has_height() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void Datum::set_has_height() {
  _has_bits_[0] |= 0x00000002u;
}
inline void Datum::clear_has_height() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void Datum::clear_height() {
  height_ = 0;
  clear_has_height();
}
inline ::google::protobuf::int32 Datum::height() const {
  return height_;
}
inline void Datum::set_height(::google::protobuf::int32 value) {
  set_has_height();
  height_ = value;
}

// optional int32 width = 3;
inline bool Datum::has_width() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void Datum::set_has_width() {
  _has_bits_[0] |= 0x00000004u;
}
inline void Datum::clear_has_width() {
  _has_bits_[0] &= ~0x00000004u;
}
inline void Datum::clear_width() {
  width_ = 0;
  clear_has_width();
}
inline ::google::protobuf::int32 Datum::width() const {
  return width_;
}
inline void Datum::set_width(::google::protobuf::int32 value) {
  set_has_width();
  width_ = value;
}

// optional bytes data = 4;
inline bool Datum::has_data() const {
  return (_has_bits_[0] & 0x00000008u) != 0;
}
inline void Datum::set_has_data() {
  _has_bits_[0] |= 0x00000008u;
}
inline void Datum::clear_has_data() {
  _has_bits_[0] &= ~0x00000008u;
}
inline void Datum::clear_data() {
  if (data_ != &::google::protobuf::internal::kEmptyString) {
    data_->clear();
  }
  clear_has_data();
}
inline const ::std::string& Datum::data() const {
  return *data_;
}
inline void Datum::set_data(const ::std::string& value) {
  set_has_data();
  if (data_ == &::google::protobuf::internal::kEmptyString) {
    data_ = new ::std::string;
  }
  data_->assign(value);
}
inline void Datum::set_data(const char* value) {
  set_has_data();
  if (data_ == &::google::protobuf::internal::kEmptyString) {
    data_ = new ::std::string;
  }
  data_->assign(value);
}
inline void Datum::set_data(const void* value, size_t size) {
  set_has_data();
  if (data_ == &::google::protobuf::internal::kEmptyString) {
    data_ = new ::std::string;
  }
  data_->assign(reinterpret_cast<const char*>(value), size);
}
inline ::std::string* Datum::mutable_data() {
  set_has_data();
  if (data_ == &::google::protobuf::internal::kEmptyString) {
    data_ = new ::std::string;
  }
  return data_;
}
inline ::std::string* Datum::release_data() {
  clear_has_data();
  if (data_ == &::google::protobuf::internal::kEmptyString) {
    return NULL;
  } else {
    ::std::string* temp = data_;
    data_ = const_cast< ::std::string*>(&::google::protobuf::internal::kEmptyString);
    return temp;
  }
}

// optional int32 label = 5;
inline bool Datum::has_label() const {
  return (_has_bits_[0] & 0x00000010u) != 0;
}
inline void Datum::set_has_label() {
  _has_bits_[0] |= 0x00000010u;
}
inline void Datum::clear_has_label() {
  _has_bits_[0] &= ~0x00000010u;
}
inline void Datum::clear_label() {
  label_ = 0;
  clear_has_label();
}
inline ::google::protobuf::int32 Datum::label() const {
  return label_;
}
inline void Datum::set_label(::google::protobuf::int32 value) {
  set_has_label();
  label_ = value;
}

// repeated float float_data = 6;
inline int Datum::float_data_size() const {
  return float_data_.size();
}
inline void Datum::clear_float_data() {
  float_data_.Clear();
}
inline float Datum::float_data(int index) const {
  return float_data_.Get(index);
}
inline void Datum::set_float_data(int index, float value) {
  float_data_.Set(index, value);
}
inline void Datum::add_float_data(float value) {
  float_data_.Add(value);
}
inline const ::google::protobuf::RepeatedField< float >&
Datum::float_data() const {
  return float_data_;
}
inline ::google::protobuf::RepeatedField< float >*
Datum::mutable_float_data() {
  return &float_data_;
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace caffe

#ifndef SWIG
namespace google {
namespace protobuf {


}  // namespace google
}  // namespace protobuf
#endif  // SWIG

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_caffe_2eproto__INCLUDED
