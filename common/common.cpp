// Copyright Lin Min 2014

#include <iostream>
#include <stdint.h>
#include <fcntl.h>
#include <string>
#include <algorithm>
#include <vector>
#include <fstream>  // NOLINT(readability/streams)
#include <glog/logging.h>
#include <cstdlib>
#include <sys/syscall.h>
#include <glog/logging.h>
#include <mpi.h>

#include "common/common.hpp"

using std::fstream;
using std::ios;
using std::string;
using namespace std;

namespace purine {

string mpi_strerror(int errorcode) {
  int len;
  char estring[MPI_MAX_ERROR_STRING];
  MPI_Error_string(errorcode, estring, &len);
  return string(estring, len);
}

string get_env(const string& env) {
  const char* env_value = getenv(env.c_str());
  CHECK(env_value) << "Environment Variable " << env << " is not defined";
  return env_value;
}

int64_t cluster_seedgen(void) {
  int64_t s, seed, pid, tid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }

  LOG(INFO) << "System entropy source not available, "
              "using fallback algorithm to generate seed instead.";
  if (f)
    fclose(f);

  pid = getpid();
  tid = syscall(SYS_gettid);
  s = time(NULL);
  // TODO: how to hash with tid
  seed = abs(((s * 181) * ((pid + tid - 83) * 359)) % 104729);
  return seed;
}

rng_t* caffe_rng() {
  static thread_local rng_t rng_(cluster_seedgen());
  return &rng_;
}

int current_rank() {
  int rank;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  return rank;
}

void print_graph(const vector<vector<string> >& print_out) {
  for (const vector<string>& ss : print_out) {
    for (vector<string>::const_iterator it = ss.begin(); it != ss.end(); it++) {
      if (it == ss.begin()) {
        cout << *it;
      } else {
        cout << "\033[1;37m --> \033[0m" << *it;
      }
      if (it == ss.end() - 1) {
        cout << endl;
      }
    }
  }
}
double time_subtract(struct timeval *x, struct timeval *y){
    if (x->tv_sec > y->tv_sec) return -1;
    if ((x->tv_sec==y->tv_sec) && (x->tv_usec>y->tv_usec)) return -1;
    double tv_sec = ( y->tv_sec-x->tv_sec );
    double tv_usec = ( y->tv_usec-x->tv_usec );
    if (tv_usec<0){
        tv_sec--;
        tv_usec+=1000000;
    }
    return tv_sec + tv_usec / 1000000;
}

}
