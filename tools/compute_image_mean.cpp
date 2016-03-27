#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <lmdb.h>

#include "boost/scoped_ptr.hpp"
#include "glog/logging.h"


#include "caffeine/proto/caffe.pb.h"
#include "caffeine/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

using std::max;
using std::pair;
using boost::scoped_ptr;
string data_path = "data/cifar-10/";
string db_path = data_path + "cifar-10-train-lmdb";
string save_path = data_path + "mean.binaryproto";

int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);

    // lmdb
    MDB_env *mdb_env;
    MDB_dbi mdb_dbi;
    MDB_val mdb_key, mdb_value;
    MDB_txn *mdb_txn;
    MDB_stat mdb_stat;
    MDB_cursor* mdb_cursor;

    LOG(INFO) << "Opening lmdb " << db_path;
    CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) 
        << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)
        << "mdb_env_set_mapsize failed";
    CHECK_EQ(mdb_env_open(mdb_env, db_path.c_str(), 0, 0664), MDB_SUCCESS)
        << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
        << "mdb_open failed. Does the lmdb already exist? ";
    CHECK_EQ(mdb_cursor_open(mdb_txn, mdb_dbi, &mdb_cursor), MDB_SUCCESS) 
        << "mdb_cursor_open failed";
    CHECK_EQ(mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_FIRST), MDB_SUCCESS) 
        << "mdb_cursor_get failed";
    CHECK_EQ(mdb_env_stat(mdb_env, &mdb_stat), MDB_SUCCESS)
        <<"mdb_evn_stat failed";
    
    int db_size = mdb_stat.ms_entries;

    BlobProto sum_blob;
    Datum datum;
    
    for(int item = 0; item < db_size; item++){
        CHECK_EQ(mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_GET_CURRENT), MDB_SUCCESS) 
            << "mdb_cursor_get failed";
        datum.ParseFromArray(mdb_value.mv_data, mdb_value.mv_size);
        const string& data = datum.data();

        const int data_size = datum.channels() * datum.height() * datum.width();
        printf("width, height, channels %d %d %d\n", datum.width(), datum.height(), datum.channels());

        if(item == 0){
            sum_blob.set_num(1);
            sum_blob.set_channels(datum.channels());
            sum_blob.set_height(datum.height());
            sum_blob.set_width(datum.width());
            for (int i = 0; i < data_size; ++i) {
                sum_blob.add_data(0.);
            }
        }

        for(int i = 0; i < data_size; i++){
            sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
        }
        
        if (item % 10000 == 0) {
            LOG(INFO) << "Processed " << item << " files.";
        }
        if (mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_NEXT) != MDB_SUCCESS) {
            CHECK_EQ(mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value,  MDB_FIRST), MDB_SUCCESS);
        }
        if(item + 1 == db_size){
            for (int i = 0; i < data_size; ++i) {
                sum_blob.set_data(i, sum_blob.data(i) / db_size);
                //printf("%f ", sum_blob.data(i));
            }//printf("\n");
        }
    }

    // Write to disk
    LOG(INFO) << "Write to " << save_path;
    WriteProtoToBinaryFile(sum_blob, save_path.c_str());
    return 0;
}
