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

using namespace pybind11::literals;
using namespace Eigen;

namespace py = pybind11;

namespace sicdock {
namespace motif {

using std::cout;
using std::endl;

typedef uint64_t Key;

py::array_t<double> logsum_bins(py::array_t<uint64_t> lbub,
                                py::array_t<double> vals) {
  uint64_t* plbub = (uint64_t*)lbub.request().ptr;
  double* pvals = (double*)vals.request().ptr;
  py::array_t<double> out(lbub.size());
  double* pout = (double*)out.request().ptr;
  for (int i = 0; i < lbub.size(); ++i) {
    int lb = plbub[i] >> 32;
    int ub = (plbub[i] << 32) >> 32;
    double sum = 0;
    for (int j = lb; j < ub; ++j) {
      sum += std::exp(pvals[j]);
    }
    pout[i] = std::log(sum);
  }
  return out;
}

py::tuple jagged_bin(py::array_t<Key> k) {
  Key* pk = (Key*)k.request().ptr;

  std::vector<std::pair<Key, Key>> pairs;
  pairs.reserve(k.size());
  for (int i = 0; i < k.size(); ++i) {
    if (pk[i] != 0) {
      pairs.push_back(std::make_pair(pk[i], i));
    }
  }
  sort(pairs.begin(), pairs.end());

  std::vector<Key> uniqkeys;
  std::vector<uint64_t> breaks;
  py::array_t<uint64_t> order(pairs.size());
  uint64_t* porder = (uint64_t*)order.request().ptr;
  Key prev = std::numeric_limits<Key>::max();

  for (int i = 0; i < pairs.size(); ++i) {
    porder[i] = pairs[i].second;
    Key cur = pairs[i].first;
    if (cur != prev) {
      breaks.push_back(i);
      uniqkeys.push_back(cur);
      prev = cur;
    }
  }
  breaks.push_back(pairs.size());
  assert(breaks.size() == uniqkeys.size() + 1);

  py::array_t<uint64_t> ranges(uniqkeys.size());
  uint64_t* pranges = (uint64_t*)ranges.request().ptr;
  py::array_t<Key> keys(uniqkeys.size());
  uint64_t* pkeys = (uint64_t*)keys.request().ptr;
  for (int i = 0; i < uniqkeys.size(); ++i) {
    pkeys[i] = uniqkeys[i];
    uint64_t lb = breaks[i], ub = breaks[i + 1];
    pranges[i] = lb << 32 | ub;
  }

  py::tuple out(3);
  out[0] = order;
  out[1] = keys;
  out[2] = ranges;
  return out;
}

PYBIND11_MODULE(_motif, m) {
  m.def("jagged_bin", &jagged_bin);
  m.def("logsum_bins", &logsum_bins);
}

}  // namespace motif
}  // namespace sicdock
