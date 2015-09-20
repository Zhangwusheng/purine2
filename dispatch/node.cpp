// Copyright Lin Min 2015
#include "dispatch/node.hpp"

namespace purine {

    Node::Node(int rank, int device)
        : Graph(rank, device), in_(0), out_(0) {
        }

    Node::~Node() {
        // disconnect from the graph
        for (Node* input : inputs_) {
            input->outputs_.erase(remove(input->outputs_.begin(),
                        input->outputs_.end(), this), input->outputs_.end());
        }
        for (Node* output : outputs_) {
            output->inputs_.erase(remove(output->inputs_.begin(),
                        output->inputs_.end(), this), output->inputs_.end());
        }
    }

    void Node::compute() {
        LOG(FATAL) << "Not Implemented";
    }

    int Node::in() const {
        return in_;
    }

    int Node::out() const {
        return out_;
    }

    void Node::setup() {
    }

    void Node::inc_in() {
        int in = in_.fetch_add(1);
        if (in + 1 == (int)inputs_.size()) {
            compute();
            for (Node* node : inputs_) {
                node->inc_out();
            }
            clear_in();
            // for (Node* node : outputs_) {
            //   node->inc_in();
            // }
        }
    }

    void Node::inc_out() {
        int out = out_.fetch_add(1);
        if (out + 1 >= (int)outputs_.size()) {
            clear_out();
        }
    }

    void Node::clear_in() {
        in_ = 0;
    }

    void Node::clear_out() {
        out_ = 0;
    }

}
