/*cppimport
<%
cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = []

setup_pybind11(cfg)
%>
*/
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pybind11/stl.h>
#include <algorithm>
#include <iostream>
#include <limits>

#include "sicdock/util/Timer.hpp"
#include "sicdock/util/types.hpp"

using namespace pybind11::literals;
using namespace Eigen;

namespace py = pybind11;

namespace sicdock {
namespace motif {

using namespace util;
using std::cout;
using std::endl;

typedef uint64_t Key;

VectorX<double> logsum_bins(VectorX<uint64_t> lbub, VectorX<double> vals) {
  py::gil_scoped_release release;
  VectorX<double> out(lbub.size());
  for (int i = 0; i < lbub.size(); ++i) {
    int ub = lbub[i] >> 32;
    int lb = (lbub[i] << 32) >> 32;
    double sum = 0;
    for (int j = lb; j < ub; ++j) {
      sum += std::exp(vals[j]);
    }
    out[i] = std::log(sum);
  }
  return out;
}

py::tuple jagged_bin(VectorX<Key> keys0) {
  py::gil_scoped_release release;

  std::vector<std::pair<Key, Key>> pairs;
  pairs.reserve(keys0.size());
  for (int i = 0; i < keys0.size(); ++i) {
    if (keys0[i] != 0) {
      pairs.push_back(std::make_pair(keys0[i], i));
    }
  }
  sort(pairs.begin(), pairs.end());

  std::vector<Key> uniqkeys;
  std::vector<uint64_t> breaks;
  VectorX<uint64_t> order(pairs.size());
  Key prev = std::numeric_limits<Key>::max();

  for (int i = 0; i < pairs.size(); ++i) {
    order[i] = pairs[i].second;
    Key cur = pairs[i].first;
    if (cur != prev) {
      breaks.push_back(i);
      uniqkeys.push_back(cur);
      prev = cur;
    }
  }
  breaks.push_back(pairs.size());
  assert(breaks.size() == uniqkeys.size() + 1);

  VectorX<uint64_t> ranges(uniqkeys.size());
  VectorX<Key> keys(uniqkeys.size());
  for (int i = 0; i < uniqkeys.size(); ++i) {
    keys[i] = uniqkeys[i];
    uint64_t lb = breaks[i], ub = breaks[i + 1];
    ranges[i] = ub << 32 | lb;
  }

  py::gil_scoped_acquire acquire;

  return py::make_tuple(order, keys, ranges);
}

PYBIND11_MODULE(_motif, m) {
  m.def("jagged_bin", &jagged_bin);
  m.def("logsum_bins", &logsum_bins, "lbub"_a, "vals"_a);
}

}  // namespace motif
}  // namespace sicdock
