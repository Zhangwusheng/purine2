// Copyright Lin Min 2015
#ifndef PURINE_CONNECTABLE
#define PURINE_CONNECTABLE

#include "dispatch/graph.hpp"

namespace purine {

    class Connectable : public Graph {
        friend Connectable& operator >> (Connectable& graph1,
                Connectable& graph2);
        friend Connectable& operator >> (const vector<Blob*>& bottom,
                Connectable& graph);
        friend const vector<Blob*>& operator >> (Connectable& graph,
                const vector<Blob*>& top);
        protected:
        vector<Blob*> bottom_;
        vector<Blob*> top_;
        bool bottom_setup_ = false;
        bool top_setup_ = false;
        public:
        Connectable(int rank = 0, int device = 0);
        virtual ~Connectable() override;

        const vector<Blob*>& bottom();
        const vector<Blob*>& top();
        virtual void set_bottom(const vector<Blob*>& bottom);
        virtual void set_top(const vector<Blob*>& top);
    };

}

#endif
