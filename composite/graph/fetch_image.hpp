// Copyright Lin Min 2015
#ifndef PURINE_FETCH_IMAGE
#define PURINE_FETCH_IMAGE

#include <vector>
#include <utility>
#include "dispatch/runnable.hpp"
#include "dispatch/op.hpp"
#include "operations/include/image_label.hpp"
#include "composite/graph/fetch_image.hpp"
#include "composite/graph/split.hpp"
#include "composite/graph/copy.hpp"
#include "composite/vectorize.hpp"

using namespace std;

namespace purine {

    class FetchImage : public Runnable {
        protected:
            vector<Blob*> images_;
            vector<Blob*> labels_;
            Blob* master_cpu_labels;
        public:
            /* @bref
             * for trainning
             */
            FetchImage(const string& source, const string& mean,
                    bool mirror, bool random, bool color, float scale, int batch_size, int crop_size,
                    const vector<pair<int, int> >& location);

            /* @bref
             * for testing
             * mulit_view_id = -1 is crop on middle
             * mulit_view_id >= 0 is multi-view test
             */
            FetchImage(const string& source, const string& mean,
                    bool color, int multi_view_id, float scale, int batch_size, int crop_size,
                    const vector<pair<int, int> >& location);

            virtual ~FetchImage() override {}
            const vector<Blob*>& images() { return images_; }
            const vector<Blob*>& labels() { return labels_; }
            Blob* get_master_cpu_labels() { 
                return master_cpu_labels;
            }
    };
}

#endif

