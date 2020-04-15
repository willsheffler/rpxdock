/*/*cppimport
<%


cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = []

cfg['parallel'] = False


setup_pybind11(cfg)
%>
*/
/** \file */

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <iostream>
#include <limits>

#include "parallel_hashmap/phmap.h"
#include "rpxdock/util/Timer.hpp"
#include "rpxdock/util/pybind_types.hpp"

using namespace pybind11::literals;
using namespace Eigen;

namespace py = pybind11;

namespace rpxdock {
/**
\namespace rpxdock::motif
\brief namespace for rpx ("motif") scoring utils
*/
namespace motif {

using namespace util;
using std::cout;
using std::endl;

typedef uint64_t Key;

template <typename Map, typename K, typename V>
void update_max(Map& map, K k, V v) {
  auto [it, inserted] = map.emplace(k, v);
  if (!inserted) it->second = std::max(it->second, v);
}

template <typename I, typename V>
py::tuple marginal_max_score(Mx<I> lbub, Mx<I> pairs, Vx<V> vals) {
  using Map = phmap::flat_hash_map<I, V>;
  if (lbub.cols() != 2) throw std::runtime_error("lbub must be shape (N,2)");
  if (pairs.cols() != 2) throw std::runtime_error("pairs must be shape (N,2)");
  if (pairs.rows() != vals.rows())
    throw std::runtime_error("pairs and vals must be same len");
  auto lbub1 = std::make_unique<Mx<I>>();
  auto lbub2 = std::make_unique<Mx<I>>();
  auto idx1 = std::make_unique<Vx<I>>();
  auto idx2 = std::make_unique<Vx<I>>();
  auto max1 = std::make_unique<Vx<V>>();
  auto max2 = std::make_unique<Vx<V>>();
  {
    py::gil_scoped_release release;
    std::vector<Map> maps1(lbub.rows()), maps2(lbub.rows());
    lbub1->resize(lbub.rows(), 2);
    lbub2->resize(lbub.rows(), 2);
    int n1 = 0, n2 = 0;
    for (int i = 0; i < lbub.rows(); ++i) {
      I lb = lbub(i, 0), ub = lbub(i, 1);
      (*lbub1)(i, 0) = n1;
      (*lbub2)(i, 0) = n2;
      for (int ipair = lb; ipair < ub; ++ipair) {
        update_max(maps1[i], pairs(ipair, 0), vals[ipair]);
        update_max(maps2[i], pairs(ipair, 1), vals[ipair]);
      }
      n1 += maps1[i].size();
      n2 += maps2[i].size();
      (*lbub1)(i, 1) = n1;
      (*lbub2)(i, 1) = n2;
    }
    idx1->resize(n1, 2);
    idx2->resize(n2, 2);
    max1->resize(n1, 2);
    max2->resize(n2, 2);
    int nn1 = 0, nn2 = 0;
    for (int i = 0; i < lbub.rows(); ++i) {
      for (auto [k, v] : maps1[i]) {
        (*idx1)[nn1] = k;
        (*max1)[nn1++] = v;
      }
      for (auto [k, v] : maps2[i]) {
        (*idx2)[nn2] = k;
        (*max2)[nn2++] = v;
      }
    }
    if (nn1 != n1) throw std::runtime_error("marginal_max_score error");
    if (nn2 != n2) throw std::runtime_error("marginal_max_score error");
  }
  return py::make_tuple(*lbub1, *lbub2, *idx1, *idx2, *max1, *max2);
}

Vx<double> logsum_bins(Vx<uint64_t> lbub, Vx<double> vals) {
  py::gil_scoped_release release;
  Vx<double> out(lbub.size());
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

py::tuple jagged_bin(Vx<Key> keys0) {
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
  Vx<uint64_t> order(pairs.size());
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

  Vx<uint64_t> ranges(uniqkeys.size());
  Vx<Key> keys(uniqkeys.size());
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
  m.def("marginal_max_score", &marginal_max_score<int32_t, double>, "lbub"_c,
        "pairs"_c, "vals"_c);
}

}  // namespace motif
}  // namespace rpxdock
