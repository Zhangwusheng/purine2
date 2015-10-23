// Copyright Lin Min 2015
#include <lmdb.h>

#include "dispatch/runnable.hpp"
#include "dispatch/op.hpp"
#include "operations/include/image_label.hpp"
#include "composite/graph/local_fetch_image.hpp"
#include "composite/graph/split.hpp"
#include "composite/graph/copy.hpp"
#include "composite/vectorize.hpp"

using namespace std;

namespace purine {

    LocalFetchImage::LocalFetchImage(const string& source, const string& mean,
            bool mirror, bool random, bool color, float scale, int crop_size,
            const vector<int> & location)
    {
        images_.push_back(create("IMAGES", location[0], location[1],
                {location[2], color ? 3 : 1, crop_size, crop_size}));
        labels_.push_back(create("LABELS", location[0], location[1], {location[2], 1, 1, 1}));

        MDB_env* mdb_env_;
        MDB_stat mdb_stat_;
        CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS)
            << "mdb_env_create failed";
        CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);
        CHECK_EQ(mdb_env_open(mdb_env_, source.c_str(), MDB_RDONLY|MDB_NOTLS,
                    0664), MDB_SUCCESS) << "mdb_env_open failed";
        CHECK_EQ(mdb_env_stat(mdb_env_, &mdb_stat_), MDB_SUCCESS);
        int entries = mdb_stat_.ms_entries;
        mdb_env_close(mdb_env_);
        MPI_LOG( << "Lmdb contains " << entries << " entries." );
        int offset = rand() % entries;
        int batch_size = location[2];
        MPI_LOG( << " ============================= " );
        MPI_LOG( << " machine    " << location[0] );
        MPI_LOG( << " batch size " << batch_size);
        MPI_LOG( << " offset     " << offset);
        MPI_LOG( << " ============================= " );

        Blob* image = create("IMAGES", location[0], -1,
                {batch_size, color ? 3 : 1, crop_size, crop_size});
        Blob* label = create("LABELS", location[0], -1, {batch_size, 1, 1, 1});
        Op<ImageLabel>* image_label = create<ImageLabel>("FETCH", location[0], -1,
                "fetch", ImageLabel::param_tuple(source, mean, mirror, random, color, -1, scale, 
                    offset, batch_size, batch_size, crop_size));
        *image_label >> vector<Blob*>{ image, label };

        master_cpu_labels = label;

        // copy splitted images to the destination
        vector<vector<Blob*> >{ std::vector<Blob*>{image} }
        >> *createAny<Vectorize<Copy> >("copy_image_to_dest",
                vector<Copy::param_tuple>(1))
            >> vector<vector<Blob*> >{ images_ };

        vector<vector<Blob*> >{ std::vector<Blob*>{label} }
        >> *createAny<Vectorize<Copy> >("copy_label_to_dest",
                vector<Copy::param_tuple>(1))
            >> vector<vector<Blob*> >{ labels_ };

        offset += batch_size;
    }
}


