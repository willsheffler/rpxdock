/*/*cppimport
<%


cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['bcc.hpp']


cfg['parallel'] = False


setup_pybind11(cfg)
%>
*/
/** \file */

#include "rpxdock/geom/bcc.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "rpxdock/util/types.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

namespace rpxdock {
namespace geom {

using namespace Eigen;
using namespace util;

template <typename K>
struct MyIter {
  std::vector<std::pair<K, K>>& vals;
  MyIter(std::vector<std::pair<K, K>>& v) : vals(v) {}
  MyIter<K>& operator++(int) { return *this; }
  MyIter<K>& operator*() { return *this; }
  void operator=(std::pair<K, K> kk) { vals.push_back(kk); }
};

template <typename F, typename K>
Vx<K> BCC_neighbors_6_3(BCC<6, F, K>& bcc, K index, int radius, bool extrahalf,
                        bool oddlast3, bool sphere) {
  py::gil_scoped_release release;
  std::vector<std::pair<K, K>> tmp;
  MyIter<K> iter(tmp);
  bcc.neighbors_6_3(index, iter, radius, extrahalf, oddlast3, sphere);
  Vx<K> out(tmp.size());
  for (int i = 0; i < tmp.size(); ++i) out[i] = tmp[i].first;
  return out;
}

template <typename F, typename K>
py::tuple BCC_neighbors_6_3_dist(BCC<6, F, K>& bcc, K index, int radius,
                                 bool extrahalf, bool oddlast3, bool sphere) {
  auto kout = std::make_unique<Vx<K>>();
  auto dout = std::make_unique<Vx<K>>();
  {
    py::gil_scoped_release release;
    std::vector<std::pair<K, K>> tmp;
    MyIter<K> iter(tmp);
    bcc.neighbors_6_3(index, iter, radius, extrahalf, oddlast3, sphere);
    kout->resize(tmp.size());
    dout->resize(tmp.size());
    for (int i = 0; i < tmp.size(); ++i) {
      (*kout)[i] = tmp[i].first;
      (*dout)[i] = tmp[i].second;
    }
  }
  return py::make_tuple(*kout, *dout);
}

template <typename F, typename K>
Vx<K> BCC_neighbors_3(BCC<3, F, K>& bcc, K index, int radius, bool extrahalf,
                      bool sphere) {
  py::gil_scoped_release release;
  std::vector<std::pair<K, K>> tmp;
  MyIter<K> iter(tmp);
  bcc.neighbors_3(index, iter, radius, extrahalf, sphere);
  Vx<K> out(tmp.size());
  for (int i = 0; i < tmp.size(); ++i) out[i] = tmp[i].first;
  return out;
}
template <typename F, typename K>
py::tuple BCC_neighbors_3_dist(BCC<3, F, K>& bcc, K index, int radius,
                               bool extrahalf, bool sphere) {
  auto kout = std::make_unique<Vx<K>>();
  auto dout = std::make_unique<Vx<K>>();
  {
    py::gil_scoped_release release;
    std::vector<std::pair<K, K>> tmp;
    MyIter<K> iter(tmp);
    bcc.neighbors_3(index, iter, radius, extrahalf, sphere);
    kout->resize(tmp.size());
    dout->resize(tmp.size());
    for (int i = 0; i < tmp.size(); ++i) {
      (*kout)[i] = tmp[i].first;
      (*dout)[i] = tmp[i].second;
    }
  }
  return py::make_tuple(*kout, *dout);
}

template <int DIM, typename F, typename K>
Mx<F> BCC_getvals(BCC<DIM, F, K>& bcc, RefVx<K> keys) {
  py::gil_scoped_release release;
  Mx<F> out(keys.size(), DIM);
  for (int i = 0; i < keys.rows(); ++i) out.row(i) = bcc[keys[i]];
  return out;
}

template <int DIM, typename F, typename K>
Vx<K> BCC_getkeys(BCC<DIM, F, K>& bcc, RefMx<F> vals) {
  py::gil_scoped_release release;
  Vx<K> out(vals.rows());
  for (int i = 0; i < vals.rows(); ++i) out[i] = bcc[vals.row(i)];
  return out;
}

template <int DIM, typename F, typename K>
void bind_bcc(py::module m, std::string name) {
  using BCC = BCC<DIM, F, K>;
  using Sizes = Matrix<K, DIM, 1>;
  using Floats = Matrix<F, DIM, 1>;
  auto cls = py::class_<BCC>(m, name.c_str());
  cls.def(py::init<Sizes, Floats, Floats>());
  cls.def("__len__", &BCC::size);
  cls.def("__getitem__", &BCC_getkeys<DIM, F, K>);
  cls.def("__getitem__", &BCC_getvals<DIM, F, K>);
  cls.def("keys", &BCC_getkeys<DIM, F, K>);
  cls.def("vals", &BCC_getvals<DIM, F, K>);
  cls.def_property_readonly("lower", &BCC::lower);
  cls.def_property_readonly("upper", &BCC::upper);
  cls.def_property_readonly("width", &BCC::width);
  cls.def_property_readonly("nside", &BCC::nside);
  cls.def("neighbor_sphere_radius_square_cut",
          &BCC::neighbor_sphere_radius_square_cut, "radius"_a,
          "extrahalf"_a = false);
  cls.def("neighbor_radius_square_cut", &BCC::neighbor_radius_square_cut,
          "radius"_a, "extrahalf"_a = false);
  if constexpr (DIM == 6)
    cls.def("neighbors_6_3", &BCC_neighbors_6_3<F, K>,
            "get indices of neighboring cells, last3 dims only +-1", "index"_a,
            "radius"_a = 1, "extrahalf"_a = false, "oddlast3"_a = true,
            "sphere"_a = true);
  cls.def("neighbors_6_3_dist", &BCC_neighbors_6_3_dist<F, K>,
          "get indices of neighboring cells, last3 dims only +-1", "index"_a,
          "radius"_a = 1, "extrahalf"_a = false, "oddlast3"_a = true,
          "sphere"_a = true);
  if constexpr (DIM == 3)
    cls.def("neighbors_3", &BCC_neighbors_3<F, K>,
            "get indices of neighboring cells", "index"_a, "radius"_a = 1,
            "extrahalf"_a = false, "sphere"_a = true);
  cls.def("neighbors_3_dist", &BCC_neighbors_3_dist<F, K>,
          "get indices of neighboring cells", "index"_a, "radius"_a = 1,
          "extrahalf"_a = false, "sphere"_a = true);
}

PYBIND11_MODULE(bcc, m) {
  bind_bcc<3, double, uint64_t>(m, "BCC3");
  bind_bcc<6, double, uint64_t>(m, "BCC6");
  bind_bcc<3, float, uint64_t>(m, "BCC3_float");
  bind_bcc<6, float, uint64_t>(m, "BCC6_float");
  /**/
}
}  // namespace geom
}  // namespace rpxdock