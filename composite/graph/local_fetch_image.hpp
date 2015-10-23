// Copyright Lin Min 2015
#ifndef PURINE_LOCAL_FETCH_IMAGE
#define PURINE_LOCAL_FETCH_IMAGE

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

    class LocalFetchImage : public Runnable {
        protected:
            vector<Blob*> images_;
            vector<Blob*> labels_;
            Blob* master_cpu_labels;
        public:
            /* @bref
             * for trainning
             */
            LocalFetchImage(const string& source, const string& mean,
                    bool mirror, bool random, bool color, float scale, int crop_size,
                    const vector<int> & location);

            virtual ~LocalFetchImage() override {}
            const vector<Blob*>& images() { return images_; }
            const vector<Blob*>& labels() { return labels_; }
            const Blob* get_master_cpu_labels() { 
                return master_cpu_labels;
            }
    };
}

#endif

