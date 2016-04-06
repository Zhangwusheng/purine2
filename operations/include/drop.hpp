// Copyright Lin Min 2015
#ifndef PURINE_DROP_
#define PURINE_DROP_

#include <string>
#include "operations/cudnn.hpp"
#include "operations/operation.hpp"
#include <map>
#include <vector>
#include <stack>

using std::string;

namespace purine {

    /**
     * { bottom } >> op >> { top }
     */
    class Drop: public Operation {
        protected:
            float rate_;
            bool isTest_;
            float*dropVector_;
            float*dDropVector_;
        public:
            typedef tuple<float,float*, bool> param_tuple;
            explicit Drop(const vector<Tensor*>& inputs,
                    const vector<Tensor*>& outputs, const param_tuple& args);
            virtual ~Drop();
            virtual void compute_gpu(const vector<bool>& add);
            virtual void compute_cpu(const vector<bool>& add);
    };

    /**
     * { top_diff, top, bottom } >> op >> { bottom_diff }
     */
    class DropDown: public Operation {
        protected:
            float rate_;
            float* dropVector_;
            float* dDropVector_;
        public:
            typedef tuple<float, float*> param_tuple;
            explicit DropDown(const vector<Tensor*>& inputs,
                    const vector<Tensor*>& outputs, const param_tuple& args);
            virtual ~DropDown();
            virtual void compute_gpu(const vector<bool>& add);
            virtual void compute_cpu(const vector<bool>& add);
    };

}

#endif
