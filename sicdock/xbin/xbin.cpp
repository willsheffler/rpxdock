/*cppimport
<%
cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-O1']
cfg['dependencies'] = ['../geom/bcc.hpp','../util/assertions.hpp',
'../util/global_rng.hpp', 'xbin.hpp', '../util/numeric.hpp',
'../util/pybind_types.hpp']

setup_pybind11(cfg)
%>
*/

#include <iostream>
#include <string>

#include "sicdock/util/Timer.hpp"
#include "sicdock/util/assertions.hpp"
#include "sicdock/util/global_rng.hpp"
#include "sicdock/util/numeric.hpp"
#include "sicdock/util/pybind_types.hpp"
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

using namespace util;
template <typename F, typename K>
using XBin = XformHash_bt24_BCC6<X3<F>, K>;

template <typename F, typename K>
py::array_t<F> _bincen_of(XBin<F, K> const &binner, VectorX<K> keys) {
  VectorX<X3<F>> out(keys.size());
  for (int i = 0; i < keys.size(); ++i) out[i] = binner.get_center(keys[i]);
  return xform_eigen_to_py(out);
}

template <typename K>
py::array bincen_of(VectorX<K> keys, double rcart, double rori, double mxcart) {
  XformHash_bt24_BCC6<X3d, K> binner(rcart, rori, mxcart);
  return _bincen_of(binner, keys);
}

template <typename F, typename K>
VectorX<K> key_of(XBin<F, K> const &binner, py::array_t<F> _xforms) {
  MapVectorXform<F> xforms = xform_py_to_eigen(_xforms);
  VectorX<K> out(xforms.size());
  for (int i = 0; i < xforms.size(); ++i) {
    K k = binner.get_key(xforms[i]);
    out[i] = k;
  }
  return out;
}

template <typename I, typename F, typename K>
py::array_t<K> kop_impl(XBin<F, K> const &xb, py::array_t<I> p,
                        py::array_t<F> x1, py::array_t<F> x2) {
  I *pp = (I *)p.request().ptr;
  X3<F> *px1 = (X3<F> *)x1.request().ptr;
  X3<F> *px2 = (X3<F> *)x2.request().ptr;
  py::array_t<K> keys(p.shape()[0]);
  K *out = (K *)keys.request().ptr;
  for (int ip = 0; ip < keys.size(); ++ip) {
    I i1 = pp[2 * ip + 0];
    I i2 = pp[2 * ip + 1];
    out[ip] = xb.get_key(px1[i1].inverse() * (px2[i2]));
  }

  return keys;
}

template <typename K, typename F>
py::array_t<K> key_of_pairs(XBin<F, K> const &xb, py::array xp, py::array x1,
                            py::array x2) {
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
  if (py::isinstance<py::array_t<int64_t>>(xp)) {
    return kop_impl<int64_t, F, K>(xb, xp, x1, x2);
  } else if (py::isinstance<py::array_t<int32_t>>(xp)) {
    return kop_impl<int32_t, F, K>(xb, xp, x1, x2);
  } else if (py::isinstance<py::array_t<uint64_t>>(xp)) {
    return kop_impl<uint64_t, F, K>(xb, xp, x1, x2);
  } else if (py::isinstance<py::array_t<uint32_t>>(xp)) {
    return kop_impl<uint32_t, F, K>(xb, xp, x1, x2);
  } else {
    throw std::runtime_error("array dtype must be matching f4 or f8");
  }
}

template <typename I, typename F, typename K>
py::array_t<K> kop2_impl(XBin<F, K> const &xb, py::array_t<I> i1,
                         py::array_t<I> i2, py::array_t<F> x1,
                         py::array_t<F> x2) {
  I *i1p = (I *)i1.request().ptr;
  I *i2p = (I *)i2.request().ptr;
  X3<F> *px1 = (X3<F> *)x1.request().ptr;
  X3<F> *px2 = (X3<F> *)x2.request().ptr;
  py::array_t<K> keys(i1.shape()[0]);
  K *out = (K *)keys.request().ptr;
  for (int i = 0; i < keys.size(); ++i) {
    out[i] = xb.get_key(px1[i1p[i]].inverse() * (px2[i2p[i]]));
  }

  return keys;
}

template <typename K, typename F>
py::array_t<uint64_t> key_of_pairs2(XBin<F, K> const &xb, py::array i1,
                                    py::array i2, py::array x1, py::array x2) {
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
  if (py::isinstance<py::array_t<int64_t>>(i1)) {
    return kop2_impl<int64_t, F, K>(xb, i1, i2, x1, x2);
  } else if (py::isinstance<py::array_t<int32_t>>(i1)) {
    return kop2_impl<int32_t, F, K>(xb, i1, i2, x1, x2);
  } else if (py::isinstance<py::array_t<uint64_t>>(i1)) {
    return kop2_impl<uint64_t, F, K>(xb, i1, i2, x1, x2);
  } else if (py::isinstance<py::array_t<uint32_t>>(i1)) {
    return kop2_impl<uint32_t, F, K>(xb, i1, i2, x1, x2);
  } else {
    throw std::runtime_error("array dtype must be matching f4 or f8");
  }
}

template <typename I, typename F, typename K>
py::array_t<K> kop2ss_impl(XBin<F, K> const &xb, py::array_t<I> i1,
                           py::array_t<I> i2, py::array_t<I> ss1,
                           py::array_t<I> ss2, py::array_t<F> x1,
                           py::array_t<F> x2) {
  I *i1p = (I *)i1.request().ptr;
  I *i2p = (I *)i2.request().ptr;
  I *ss1p = (I *)ss1.request().ptr;
  I *ss2p = (I *)ss2.request().ptr;
  X3<F> *x1p = (X3<F> *)x1.request().ptr;
  X3<F> *x2p = (X3<F> *)x2.request().ptr;
  py::array_t<K> keys(i1.shape()[0]);
  K *out = (K *)keys.request().ptr;
  for (int i = 0; i < keys.size(); ++i) {
    K k = xb.get_key(x1p[i1p[i]].inverse() * (x2p[i2p[i]]));
    out[i] = k | ((K)ss1p[i1p[i]] << 62) | ((K)ss2p[i2p[i]] << 60);
  }

  return keys;
}

template <typename K, typename F>
py::array_t<K> key_of_pairs2_ss(XBin<F, K> const &xb, py::array i1,
                                py::array i2, py::array ss1, py::array ss2,
                                py::array x1, py::array x2) {
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
  if (py::isinstance<py::array_t<int64_t>>(i1)) {
    return kop2ss_impl<int64_t, F, K>(xb, i1, i2, ss1, ss2, x1, x2);
  } else if (py::isinstance<py::array_t<int32_t>>(i1)) {
    return kop2ss_impl<int32_t, F, K>(xb, i1, i2, ss1, ss2, x1, x2);
  } else if (py::isinstance<py::array_t<uint64_t>>(i1)) {
    return kop2ss_impl<uint64_t, F, K>(xb, i1, i2, ss1, ss2, x1, x2);
  } else if (py::isinstance<py::array_t<uint32_t>>(i1)) {
    return kop2ss_impl<uint32_t, F, K>(xb, i1, i2, ss1, ss2, x1, x2);
  } else {
    throw std::runtime_error("array dtype must be matching f4 or f8");
  }
}  // namespace xbin

template <typename K, typename FX>
py::array_t<K> key_of_pairs2_ss_same(XBin<FX, K> const &xb, py::array i1,
                                     py::array i2, py::array ss, py::array x) {
  return key_of_pairs2_ss(xb, i1, i2, ss, ss, x, x);
}

template <typename F, typename K>
py::tuple xform_to_F6(XBin<F, K> const &xbin, py::array_t<F> _xform) {
  auto xform = xform_py_to_eigen(_xform);
  RowMajorX<F> f6(xform.size(), 6);
  VectorX<K> cell(xform.size());
  for (int i = 0; i < xform.size(); ++i)
    f6.row(i) = xbin.xform_to_F6(xform[i], cell[i]);
  return py::make_tuple(f6, cell);
}

template <typename F, typename K>
py::array_t<F> F6_to_xform(XBin<F, K> const &xbin, RowMajorX<F> f6,
                           VectorX<K> cell) {
  if (f6.cols() != 6) throw std::runtime_error("f6 must be shape(N,6)");
  if (f6.rows() != cell.size())
    throw std::runtime_error("f6 and cell must have same length");
  VectorX<X3<F>> out(cell.size());
  for (int i = 0; i < cell.size(); ++i)
    out[i] = xbin.F6_to_xform(f6.row(i), cell[i]);
  return xform_eigen_to_py(out);
}

template <typename F, typename K>
void bind_xbin(py::module m, std::string name) {
  using THIS = XBin<F, K>;
  auto cls =
      py::class_<THIS>(m, name.c_str())
          .def(py::init<F, F, F>(), "cart_resl"_a = 1.0, "ori_resl"_a = 20.0,
               "cart_bound"_a = 512.0)
          .def("__getitem__", &key_of<F, K>)
          .def("__getitem__", &_bincen_of<F, K>)
          .def("key_of", &key_of<F, K>, "key of xform", "xform"_c)
          .def("bincen_of", &_bincen_of<F, K>)
          .def("xform_to_F6", &xform_to_F6<F, K>)
          .def("F6_to_xform", &F6_to_xform<F, K>)
          .def_readonly("grid6", &THIS::grid6_)
          .def_readonly("cart_resl", &THIS::cart_resl_)
          .def_readonly("ori_resl", &THIS::ori_resl_)
          .def_readonly("cart_bound", &THIS::cart_bound_)
          .def_readonly("ori_nside", &THIS::ori_nside_)
          .def("key_of_pairs2_ss", &key_of_pairs2_ss<K, F>, "idx1"_c, "idx2"_c,
               "ss1"_c, "ss2"_c, "xform1"_c, "xform2"_c)
          .def("key_of_pairs2_ss", &key_of_pairs2_ss_same<K, F>, "idx1"_c,
               "idx2"_c, "ss"_c, "xform"_c)
          .def("key_of_pairs2", &key_of_pairs2<K, F>, "idx1"_c, "idx2"_c,
               "xform1"_c, "xform2"_c)
          .def("key_of_pairs", &key_of_pairs<K, F>, "pairs"_c, "xform1"_c,
               "xform2"_c)
          .def(py::pickle(
              [](const THIS &xbin) {  // __getstate__
                return py::make_tuple(xbin.orig_cart_resl_, xbin.ori_nside_,
                                      xbin.cart_bound_);
              },
              [](py::tuple t) {  // __setstate__
                if (t.size() != 3) throw std::runtime_error("Invalid state!");
                return THIS(t[0].cast<F>(), t[1].cast<int>(), t[2].cast<F>());
              }))

      /**/;
}  // namespace xbin

template <typename F, typename K>
XBin<F, K> create_XBin_nside(F cart_resl, int nside, F cart_bound) {
  return XformHash_bt24_BCC6<X3<F>, K>(cart_resl, nside, cart_bound);
}

PYBIND11_MODULE(xbin, m) {
  using K = uint64_t;
  bind_xbin<double, K>(m, "XBin_double");
  bind_xbin<float, K>(m, "XBin_float");
  m.def("create_XBin_nside", &create_XBin_nside<double, K>);
  m.def("create_XBin_nside_float", &create_XBin_nside<float, K>);
}

}  // namespace xbin
}  // namespace sicdock