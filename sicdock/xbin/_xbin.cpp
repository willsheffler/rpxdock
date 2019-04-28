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
#include <string>

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
using XMatrixX = Matrix<X3<F>, Dynamic, 1, RowMajor>;
template <typename F>
using RowMatrixX2 = Matrix<F, Dynamic, 2, RowMajor>;

template <typename I, typename F>
py::array_t<Key> key_of_pairs_type(py::array_t<I> p, py::array_t<F> x1,
                                   py::array_t<F> x2, double rcart, double rori,
                                   double mxcart) {
  I *pp = (I *)p.request().ptr;
  X3<F> *px1 = (X3<F> *)x1.request().ptr;
  X3<F> *px2 = (X3<F> *)x2.request().ptr;
  py::array_t<Key> keys(p.shape()[0]);
  Key *out = (Key *)keys.request().ptr;
  XformHash_bt24_BCC6<X3<F>> binner(rcart, rori, mxcart);
  for (int ip = 0; ip < keys.size(); ++ip) {
    I i1 = pp[2 * ip + 0];
    I i2 = pp[2 * ip + 1];
    X3<F> x = px1[i1].inverse() * (px2[i2]);
    out[ip] = binner.get_key(x);
  }

  return keys;
}

void check_xform_array(py::array a) {
  auto buf = pybind11::array::ensure(a);
  size_t s = buf.itemsize();
  if (!a) throw std::runtime_error("bad array");
  if (a.ndim() != 3 || a.shape()[1] != 4 || a.shape()[2] != 4)
    throw std::runtime_error("array X2 must be shape (N,4,4)");
  if (a.strides()[0] != 16 * s || a.strides()[1] != 4 * s)
    throw std::runtime_error("bad strides, strides not supported");
}

py::array_t<Key> key_of_pairs(py::array xp, py::array x1, py::array x2,
                              double rcart, double rori, double mxcart) {
  check_xform_array(x1);
  check_xform_array(x2);
  pybind11::array::ensure(xp);
  if (!xp) throw std::runtime_error("bad array");
  if (xp.ndim() != 2 || xp.shape()[1] != 2)
    throw std::runtime_error("array must be shape (N,2)");
  size_t sp = xp.itemsize(), sx1 = x1.itemsize(), sx2 = x2.itemsize();
  if (xp.strides()[0] != 2 * sp || xp.strides()[1] != sp)
    throw std::runtime_error("bad strides, strides not supported");
  if (x1.dtype() != x2.dtype())
    throw std::runtime_error("xform arrays must have same dtype");
  if (py::isinstance<py::array_t<int64_t>>(xp) &&
      py::isinstance<py::array_t<double>>(x1) &&
      py::isinstance<py::array_t<double>>(x2)) {
    return key_of_pairs_type<int64_t, double>(xp, x1, x2, rcart, rori, mxcart);
  } else if (py::isinstance<py::array_t<int32_t>>(xp) &&
             py::isinstance<py::array_t<double>>(x1) &&
             py::isinstance<py::array_t<double>>(x2)) {
    return key_of_pairs_type<int64_t, double>(xp, x1, x2, rcart, rori, mxcart);
  } else if (py::isinstance<py::array_t<int64_t>>(xp) &&
             py::isinstance<py::array_t<float>>(x1) &&
             py::isinstance<py::array_t<float>>(x2)) {
    return key_of_pairs_type<int64_t, float>(xp, x1, x2, rcart, rori, mxcart);
  } else if (py::isinstance<py::array_t<int32_t>>(xp) &&
             py::isinstance<py::array_t<float>>(x1) &&
             py::isinstance<py::array_t<float>>(x2)) {
    return key_of_pairs_type<int64_t, float>(xp, x1, x2, rcart, rori, mxcart);
  } else {
    throw std::runtime_error("array dtype must be matching f4 or f8");
  }
}

template <typename I, typename F>
py::array_t<Key> key_of_pairs2_type(py::array_t<I> i1, py::array_t<I> i2,
                                    py::array_t<F> x1, py::array_t<F> x2,
                                    double rcart, double rori, double mxcart) {
  I *i1p = (I *)i1.request().ptr;
  I *i2p = (I *)i2.request().ptr;
  X3<F> *px1 = (X3<F> *)x1.request().ptr;
  X3<F> *px2 = (X3<F> *)x2.request().ptr;
  py::array_t<Key> keys(i1.shape()[0]);
  Key *out = (Key *)keys.request().ptr;
  XformHash_bt24_BCC6<X3<F>> binner(rcart, rori, mxcart);
  for (int i = 0; i < keys.size(); ++i) {
    X3<F> x = px1[i1p[i]].inverse() * (px2[i2p[i]]);
    out[i] = binner.get_key(x);
  }

  return keys;
}

py::array_t<Key> key_of_pairs2(py::array i1, py::array i2, py::array x1,
                               py::array x2, double rcart, double rori,
                               double mxcart) {
  check_xform_array(x1);
  check_xform_array(x2);
  pybind11::array::ensure(i1);
  pybind11::array::ensure(i2);
  size_t sp = i1.itemsize();

  if (!i1) throw std::runtime_error("bad array");
  if (!i2) throw std::runtime_error("bad array");
  if (i1.ndim() != 1 || i2.ndim() != 1 || i1.size() != i2.size())
    throw std::runtime_error("index must be shape (N,) and same length");
  if (i1.dtype() != i2.dtype())
    throw std::runtime_error("index arrays must have same dtype");
  if (x1.dtype() != x2.dtype())
    throw std::runtime_error("xform arrays must have same dtype");
  if (py::isinstance<py::array_t<int64_t>>(i1) &&
      py::isinstance<py::array_t<double>>(x1)) {
    return key_of_pairs2_type<int64_t, double>(i1, i2, x1, x2, rcart, rori,
                                               mxcart);
  } else if (py::isinstance<py::array_t<int32_t>>(i1) &&
             py::isinstance<py::array_t<double>>(x1)) {
    return key_of_pairs2_type<int64_t, double>(i1, i2, x1, x2, rcart, rori,
                                               mxcart);
  } else if (py::isinstance<py::array_t<int64_t>>(i1) &&
             py::isinstance<py::array_t<float>>(x1)) {
    return key_of_pairs2_type<int64_t, float>(i1, i2, x1, x2, rcart, rori,
                                              mxcart);
  } else if (py::isinstance<py::array_t<int32_t>>(i1) &&
             py::isinstance<py::array_t<float>>(x1)) {
    return key_of_pairs2_type<int64_t, float>(i1, i2, x1, x2, rcart, rori,
                                              mxcart);
  } else {
    throw std::runtime_error("array dtype must be matching f4 or f8");
  }
}

template <typename I, typename F>
py::array_t<Key> key_of_pairs2_ss_type(py::array_t<I> i1, py::array_t<I> i2,
                                       py::array_t<I> ss1, py::array_t<I> ss2,
                                       py::array_t<F> x1, py::array_t<F> x2,
                                       double rcart, double rori,
                                       double mxcart) {
  I *i1p = (I *)i1.request().ptr;
  I *i2p = (I *)i2.request().ptr;
  I *ss1p = (I *)ss1.request().ptr;
  I *ss2p = (I *)ss2.request().ptr;
  X3<F> *x1p = (X3<F> *)x1.request().ptr;
  X3<F> *x2p = (X3<F> *)x2.request().ptr;
  py::array_t<Key> keys(i1.shape()[0]);
  Key *out = (Key *)keys.request().ptr;
  XformHash_bt24_BCC6<X3<F>> binner(rcart, rori, mxcart);
  for (int i = 0; i < keys.size(); ++i) {
    X3<F> x = x1p[i1p[i]].inverse() * (x2p[i2p[i]]);
    Key k = binner.get_key(x);
    out[i] = k | ((Key)ss1p[i1p[i]] << 62) | ((Key)ss2p[i2p[i]] << 60);
  }

  return keys;
}

py::array_t<Key> key_of_pairs2_ss(py::array i1, py::array i2, py::array ss1,
                                  py::array ss2, py::array x1, py::array x2,
                                  double rcart, double rori, double mxcart) {
  check_xform_array(x1);
  check_xform_array(x2);
  pybind11::array::ensure(i1);
  pybind11::array::ensure(i2);
  size_t sp = i1.itemsize();

  if (!i1) throw std::runtime_error("bad array");
  if (!i2) throw std::runtime_error("bad array");
  if (i1.ndim() != 1 || i2.ndim() != 1 || i1.size() != i2.size())
    throw std::runtime_error("index must be shape (N,) and same length");
  if (i1.dtype() != i2.dtype())
    throw std::runtime_error("index arrays must have same dtype");
  if (x1.dtype() != x2.dtype())
    throw std::runtime_error("xform arrays must have same dtype");
  if (py::isinstance<py::array_t<int64_t>>(i1) &&
      py::isinstance<py::array_t<double>>(x1)) {
    return key_of_pairs2_ss_type<int64_t, double>(i1, i2, ss1, ss2, x1, x2,
                                                  rcart, rori, mxcart);
  } else if (py::isinstance<py::array_t<int32_t>>(i1) &&
             py::isinstance<py::array_t<double>>(x1)) {
    return key_of_pairs2_ss_type<int64_t, double>(i1, i2, ss1, ss2, x1, x2,
                                                  rcart, rori, mxcart);
  } else if (py::isinstance<py::array_t<int64_t>>(i1) &&
             py::isinstance<py::array_t<float>>(x1)) {
    return key_of_pairs2_ss_type<int64_t, float>(i1, i2, ss1, ss2, x1, x2,
                                                 rcart, rori, mxcart);
  } else if (py::isinstance<py::array_t<int32_t>>(i1) &&
             py::isinstance<py::array_t<float>>(x1)) {
    return key_of_pairs2_ss_type<int64_t, float>(i1, i2, ss1, ss2, x1, x2,
                                                 rcart, rori, mxcart);
  } else {
    throw std::runtime_error("array dtype must be matching f4 or f8");
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
  m.def("key_of", &key_of, "xform"_a, "cart_resl"_a = 1.0, "ori_resl"_a = 20.0,
        "cart_bound"_a = 512.0);
  m.def("bincen_of", &bincen_of);
  m.def("key_of_pairs", &key_of_pairs, "pairs"_a, "xform1"_a, "xform2"_a,
        "cart_resl"_a = 1.0, "ori_resl"_a = 20.0, "cart_bound"_a = 512.0);
  m.def("key_of_pairs2", &key_of_pairs2, "idx1"_a, "idx2"_a, "xform1"_a,
        "xform2"_a, "cart_resl"_a = 1.0, "ori_resl"_a = 20.0,
        "cart_bound"_a = 512.0);
  m.def("key_of_pairs2_ss", &key_of_pairs2_ss, "idx1"_a, "idx2"_a, "ss1"_a,
        "ss2"_a, "xform1"_a, "xform2"_a, "cart_resl"_a = 1.0,
        "ori_resl"_a = 20.0, "cart_bound"_a = 512.0);
}

}  // namespace xbin
}  // namespace sicdock