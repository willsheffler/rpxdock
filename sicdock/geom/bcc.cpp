/*cppimport
<%
cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-O1']
cfg['dependencies'] = ['bcc.hpp']


setup_pybind11(cfg)
%>
*/

#include "sicdock/geom/bcc.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace sicdock {
namespace geom {

using namespace Eigen;
using namespace util;

template <typename K>
struct MyIter {
  std::vector<K>& vals;
  MyIter(std::vector<K>& v) : vals(v) {}
  MyIter<K>& operator++(int) { return *this; }
  MyIter<K>& operator*() { return *this; }
  void operator=(K k) { vals.push_back(k); }
};

template <typename F, typename K>
py::array_t<K> BCC_neighbors_6_3(BCC<6, F, K>& bcc, K index, int range,
                                 bool do_midpoints, bool odd_last3) {
  std::vector<K> tmp;
  MyIter<K> iter(tmp);
  bcc.neighbors_6_3(index, iter, range, do_midpoints, odd_last3);
  py::array_t<K> out(tmp.size());
  K* kptr = (K*)out.request().ptr;
  std::copy(tmp.begin(), tmp.end(), kptr);
  return out;
}

template <typename F, typename K>
py::array_t<K> BCC_neighbors_3(BCC<3, F, K>& bcc, K index, int range,
                               bool do_midpoints, bool more_midpoints) {
  std::vector<K> tmp;
  MyIter<K> iter(tmp);
  bcc.neighbors_3(index, iter, range, do_midpoints, more_midpoints);
  py::array_t<K> out(tmp.size());
  K* kptr = (K*)out.request().ptr;
  std::copy(tmp.begin(), tmp.end(), kptr);
  return out;
}

template <int DIM, typename F, typename K>
RowMajorX<F> BCC_getvals(BCC<DIM, F, K>& bcc, RefVectorX<K> keys) {
  RowMajorX<F> out(keys.size(), DIM);
  for (int i = 0; i < keys.rows(); ++i) out.row(i) = bcc[keys[i]];
  return out;
}

template <int DIM, typename F, typename K>
VectorX<K> BCC_getkeys(BCC<DIM, F, K>& bcc, RefRowMajorX<F> vals) {
  VectorX<K> out(vals.rows());
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
  if constexpr (DIM == 6) cls.def("neighbors_6_3", &BCC_neighbors_6_3<F, K>);
  if constexpr (DIM == 3) cls.def("neighbors_3", &BCC_neighbors_3<F, K>);
}

PYBIND11_MODULE(bcc, m) {
  bind_bcc<3, double, uint64_t>(m, "BCC3");
  bind_bcc<6, double, uint64_t>(m, "BCC6");
  /**/
}
}  // namespace geom
}  // namespace sicdock