/*cppimport
<%
cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['../geom/bcc.hpp','../util/assertions.hpp',
'../util/global_rng.hpp', 'xbin.hpp', '../util/numeric.hpp']

setup_pybind11(cfg)
%>
*/
#include <iostream>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sicdock/util/Timer.hpp"
#include "sicdock/util/assertions.hpp"
#include "sicdock/util/global_rng.hpp"
#include "sicdock/util/numeric.hpp"
#include "sicdock/util/types.hpp"
#include "sicdock/xbin/xbin.hpp"

using namespace pybind11::literals;
using namespace Eigen;
using namespace sicdock;
using namespace util;
using namespace geom;

namespace py = pybind11;

namespace sicdock {
namespace xbin {

using Key = XformHash_bt24_BCC6<X3f>::Key;

template <typename F>
py::array_t<F> _bincen_of(XformHash_bt24_BCC6<X3<F>> const &binner,
                          py::array_t<Key> keys) {
  py::buffer_info inp = keys.request();
  if (inp.ndim != 1) throw std::runtime_error("Shape must be (N,)");
  Key *ptr = (Key *)inp.ptr;

  std::vector<int> shape{inp.shape[0], 4, 4};
  py::array_t<F> aout(shape);
  py::buffer_info out = aout.request();
  Key *outptr = (Key *)out.ptr;

  for (int i = 0; i < inp.shape[0]; ++i) {
    X3<F> *xp = (X3<F> *)(outptr + 16 * i);
    *xp = binner.get_center(ptr[i]);
  }

  return aout;
}

py::array bincen_of(py::array_t<Key> keys, double rcart, double rori,
                    double mxcart) {
  XformHash_bt24_BCC6<X3d> binner(rcart, rori, mxcart);
  return _bincen_of(binner, keys);
}

template <typename F>
py::array_t<Key> _key_of(XformHash_bt24_BCC6<X3<F>> const &binner,
                         py::array_t<F> xforms) {
  py::buffer_info inp = xforms.request();
  if (inp.ndim != 3 || inp.shape[1] != 4 || inp.shape[2] != 4)
    throw std::runtime_error("Shape must be (N, 4, 4)");
  F *ptr = (F *)inp.ptr;

  py::array_t<Key> aout(inp.shape[0]);
  py::buffer_info out = aout.request();
  Key *outptr = (Key *)out.ptr;

  for (int i = 0; i < inp.shape[0]; ++i) {
    X3<F> x(Eigen::Map<M4<F>>(ptr + 16 * i));
    Key k = binner.get_key(x);
    outptr[i] = k;
  }
  return aout;
}

template <typename F>
py::array_t<Key> key_of_type(py::array_t<F> x, double rcart, double rori,
                             double mxcart) {
  XformHash_bt24_BCC6<X3<F>> binner(rcart, rori, mxcart);
  return _key_of(binner, x);
}

py::array_t<Key> key_of(py::array x, double rcart, double rori, double mxcart) {
  auto buf = pybind11::array::ensure(x);
  if (!buf) throw std::runtime_error("bad array");
  if (buf.ndim() != 3 || buf.shape()[1] != 4 || buf.shape()[2] != 4)
    throw std::runtime_error("array must be shape (N,4,4)");
  if (py::isinstance<py::array_t<double>>(x)) {
    return key_of_type<double>(x, rcart, rori, mxcart);
  } else if (py::isinstance<py::array_t<float>>(x)) {
    return key_of_type<float>(x, rcart, rori, mxcart);
  } else {
    throw std::runtime_error("array dtype must be f4 or f8");
  }
}

template <typename F>
void bind_xbin(py::module m, std::string name) {
  auto cls = py::class_<XformHash_bt24_BCC6<X3<F>>>(m, name.c_str())
                 .def(py::init<F, F, F>())
                 .def("key_of", &_key_of<F>)
                 .def("bincen_of", &_bincen_of<F>);
}

PYBIND11_MODULE(_xbin, m) {
  bind_xbin<double>(m, "XBin");
  bind_xbin<float>(m, "XBin_float");
  m.def("key_of", &key_of);
  m.def("bincen_of", &bincen_of);
}

}  // namespace xbin
}  // namespace sicdock