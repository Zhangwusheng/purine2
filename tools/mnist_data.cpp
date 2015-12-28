#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <lmdb.h>
#include <stdint.h>
#include <sys/stat.h>
#include <fstream>
#include <string>
#include "caffeine/proto/caffe.pb.h"

using namespace caffe;
using namespace std;

string data_path = "/home/zhxfl/purine2/data/mnist/";
/*reverse the int*/
int reverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int) ch1 << 24) | ((int) ch2 << 16) | ((int) ch3 << 8) | ch4;
}

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);

    // lmdb
    MDB_env *mdb_env;
    MDB_dbi mdb_dbi;
    MDB_val mdb_key, mdb_data;
    MDB_txn *mdb_txn;

    string db_path = data_path + string("mnist-train-lmdb");

    LOG(INFO) << "Opening lmdb " << db_path;
    CHECK_EQ(mkdir(db_path.c_str(), 0744), 0)  << "mkdir " << db_path << "failed";
    CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS) << "mdb_env_set_mapsize failed";
    CHECK_EQ(mdb_env_open(mdb_env, db_path.c_str(), 0, 0664), MDB_SUCCESS)<< "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)<< "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)<< "mdb_open failed. Does the lmdb already exist? ";
    //CHECK_EQ(mdb_cursor_open(mdb_txn, mdb_dbi, &mdb_cursor), MDB_SUCCESS) << "mdb_cursor_open failed";
    //CHECK_EQ(mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_FIRST), MDB_SUCCESS) << "mdb_cursor_get failed";
    /*dbsize*/
    // Storing to db
    char label;
    const int kMaxKeyLength = 10;
    char key_cstr[kMaxKeyLength];
    string value;
    int channels = 1;
    int width    = 28;
    int height   = 28;
    int image_pixes_size = channels * width * height;

    unsigned char* pixels = (unsigned char*) malloc(image_pixes_size);

    Datum datum;
    datum.set_channels(1);
    datum.set_height(height);
    datum.set_width(width);

    {
        string train_data_name  = data_path + "/train-images-idx3-ubyte";
        string train_label_name = data_path + "/train-labels-idx1-ubyte";

        ifstream file_data(train_data_name.c_str(), ios::binary);
        ifstream file_label(train_label_name.c_str(), ios::binary);

        int number_of_images = 60000;
        if(file_data.is_open() && file_label.is_open()){
            file_data.read((char*)&number_of_images, sizeof(number_of_images));
            number_of_images = reverseInt(number_of_images);
            file_data.read((char*)&number_of_images, sizeof(number_of_images));
            number_of_images = reverseInt(number_of_images);
            printf("number_of_images %d\n", number_of_images);
            file_data.read((char*)&height, sizeof(height));
            height = reverseInt(height);
            printf("height %d\n", height);

            file_data.read((char*)&width, sizeof(width));
            width = reverseInt(width);
            printf("width %d\n", width);

            file_label.read((char*)&number_of_images, sizeof(number_of_images));
            file_label.read((char*)&number_of_images, sizeof(number_of_images));
            number_of_images = reverseInt(number_of_images);
            printf("number_of_images %d\n", number_of_images);

            for(int i = 0; i < number_of_images; i++){
                printf("width, height, channels %d %d %d\n", width, height, channels);
                file_data.read((char*)pixels, sizeof(char) * width * height * channels);
                file_label.read((char*)&label, sizeof(char) * 1);

                datum.set_label(label);
                datum.set_data(pixels, sizeof(char) * width * height * channels);
                int item_id = i;
                snprintf(key_cstr, kMaxKeyLength, "%08d", item_id);

                /*to serialize*/
                datum.SerializeToString(&value);
                string keystr(key_cstr);

                /*write db*/
                mdb_data.mv_size = value.size();
                mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
                mdb_key.mv_size = keystr.size();
                mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
                CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)<< "mdb_put failed";
            }
        }

        CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
        mdb_close(mdb_env, mdb_dbi);
        mdb_env_close(mdb_env);
    }
    {
        string db_path = data_path + string("mnist-test-lmdb");

        LOG(INFO) << "Opening lmdb " << db_path;
        CHECK_EQ(mkdir(db_path.c_str(), 0744), 0)  << "mkdir " << db_path << "failed";
        CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
        CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS) << "mdb_env_set_mapsize failed";
        CHECK_EQ(mdb_env_open(mdb_env, db_path.c_str(), 0, 0664), MDB_SUCCESS)<< "mdb_env_open failed";
        CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)<< "mdb_txn_begin failed";
        string test_data_name  = data_path + "/test-images-idx3-ubyte";
        string test_label_name = data_path + "/test-labels-idx1-ubyte";

        ifstream file_data(test_data_name.c_str(), ios::binary);
        ifstream file_label(test_label_name.c_str(), ios::binary);

        int number_of_images = 10000;
        if(file_data.is_open() && file_label.is_open()){
            file_data.read((char*)&number_of_images, sizeof(number_of_images));
            number_of_images = reverseInt(number_of_images);
            file_data.read((char*)&number_of_images, sizeof(number_of_images));
            number_of_images = reverseInt(number_of_images);
            printf("number_of_images %d\n", number_of_images);
            file_data.read((char*)&height, sizeof(height));
            height = reverseInt(height);
            printf("height %d\n", height);

            file_data.read((char*)&width, sizeof(width));
            width = reverseInt(width);
            printf("width %d\n", width);

            file_label.read((char*)&number_of_images, sizeof(number_of_images));
            file_label.read((char*)&number_of_images, sizeof(number_of_images));
            number_of_images = reverseInt(number_of_images);
            printf("number_of_images %d\n", number_of_images);

            for(int i = 0; i < number_of_images; i++){
                file_data.read((char*)pixels, sizeof(char) * width * height * channels);
                file_label.read((char*)&label, sizeof(char) * 1);
                datum.set_data(pixels, sizeof(char) * width * height * channels);
                datum.set_label(label);
                int item_id = i;
                snprintf(key_cstr, kMaxKeyLength, "%08d", item_id);
                printf("label %d\n", static_cast<int>(label));

                /*to serialize*/
                datum.SerializeToString(&value);
                string keystr(key_cstr);

                /*write db*/
                mdb_data.mv_size = value.size();
                mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
                mdb_key.mv_size = keystr.size();
                mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
                CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)<< "mdb_put failed";
            }
        }

        CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
        mdb_close(mdb_env, mdb_dbi);
        mdb_env_close(mdb_env);
    }
    return 0;
}


