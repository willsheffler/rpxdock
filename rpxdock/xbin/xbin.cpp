/*/*cppimport
<%


cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['../geom/bcc.hpp','../util/assertions.hpp',
'../util/global_rng.hpp', 'xbin.hpp', '../util/numeric.hpp',
'../util/pybind_types.hpp']

cfg['parallel'] = False


setup_pybind11(cfg)
%>
*/
/** \file */

#include "rpxdock/xbin/xbin.hpp"

#include <iostream>
#include <string>

#include "rpxdock/phmap/phmap.hpp"
#include "rpxdock/util/Timer.hpp"
#include "rpxdock/util/assertions.hpp"
#include "rpxdock/util/global_rng.hpp"
#include "rpxdock/util/numeric.hpp"
#include "rpxdock/util/pybind_types.hpp"
#include "rpxdock/util/types.hpp"

using namespace pybind11::literals;
using namespace Eigen;
using namespace rpxdock;
using namespace util;
using namespace geom;

namespace py = pybind11;

namespace rpxdock {
namespace xbin {

using namespace phmap;

using namespace util;
template <typename F, typename K>
using Xbin = XformHash_bt24_BCC6<X3<F>, K>;

template <typename F, typename K>
py::array_t<F> _bincen_of(Xbin<F, K> const &binner, RefVx<K> keys) {
  auto out = std::make_unique<Vx<X3<F>>>();
  {
    py::gil_scoped_release release;
    out->resize(keys.size());
    for (int i = 0; i < keys.size(); ++i)
      (*out)[i] = binner.get_center(keys[i]);
  }
  return xform_eigen_to_py(*out);
}

template <typename K>
py::array bincen_of(RefVx<K> keys, double rcart, double rori, double mxcart) {
  XformHash_bt24_BCC6<X3d, K> binner(rcart, rori, mxcart);
  return _bincen_of(binner, keys);
}

template <typename F, typename K>
Vx<K> key_of(Xbin<F, K> const &binner, py::array_t<F> _xforms) {
  MapVxX3<F> xforms = xform_py_to_eigen(_xforms);
  py::gil_scoped_release release;
  Vx<K> out(xforms.size());
  for (int i = 0; i < xforms.size(); ++i) {
    K k = binner.get_key(xforms[i]);
    out[i] = k;
  }
  return out;
}
template <typename F, typename K>
Vx<K> ori_cell_of(Xbin<F, K> const &binner, py::array_t<F> _xforms) {
  MapVxX3<F> xforms = xform_py_to_eigen(_xforms);
  py::gil_scoped_release release;
  Vx<K> out(xforms.size());
  for (int i = 0; i < xforms.size(); ++i) {
    K k = binner.cell_index(xforms[i]);
    out[i] = k;
  }
  return out;
}

template <typename I, typename F, typename K>
Vx<K> kop_impl(Xbin<F, K> const &xb, py::array_t<I> p, py::array_t<F> x1,
               py::array_t<F> x2) {
  I *pp = (I *)p.request().ptr;
  X3<F> *px1 = (X3<F> *)x1.request().ptr;
  X3<F> *px2 = (X3<F> *)x2.request().ptr;
  py::gil_scoped_release release;
  Vx<K> keys(p.shape()[0]);
  for (int ip = 0; ip < keys.size(); ++ip) {
    I i1 = pp[2 * ip + 0];
    I i2 = pp[2 * ip + 1];
    keys[ip] = xb.get_key(px1[i1].inverse() * (px2[i2]));
  }

  return keys;
}

template <typename K, typename F>
Vx<K> key_of_pairs(Xbin<F, K> const &xb, py::array xp, py::array x1,
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
  if (x1.dtype().kind() != x2.dtype().kind())
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
Vx<K> kop2_impl(Xbin<F, K> const &xb, py::array_t<I> i1, py::array_t<I> i2,
                py::array_t<F> x1, py::array_t<F> x2) {
  I *i1p = (I *)i1.request().ptr;
  I *i2p = (I *)i2.request().ptr;
  X3<F> *px1 = (X3<F> *)x1.request().ptr;
  X3<F> *px2 = (X3<F> *)x2.request().ptr;
  py::gil_scoped_release release;
  Vx<K> keys(i1.shape()[0]);
  for (int i = 0; i < keys.size(); ++i) {
    keys[i] = xb.get_key(px1[i1p[i]].inverse() * (px2[i2p[i]]));
  }

  return keys;
}

template <typename K, typename F>
Vx<K> key_of_selected_pairs(Xbin<F, K> const &xb, py::array i1, py::array i2,
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
  if (i1.dtype().kind() != i2.dtype().kind())
    throw std::runtime_error("index arrays must have same dtype");
  if (x1.dtype().kind() != x2.dtype().kind())
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

template <typename K, typename F>
Vx<K> key_of_selected_pairs_same(Xbin<F, K> const &xb, py::array i1,
                                 py::array i2, py::array x) {
  return key_of_selected_pairs(xb, i1, i2, x, x);
}

//////////////////////////// N,2 idx array key lookup
///////////////////////////////////////

template <typename I, typename F, typename K>
Vx<K> kop2_onearray_impl(Xbin<F, K> const &xb, py::array_t<I> _idx,
                         py::array_t<F> x1, py::array_t<F> x2) {
  auto idx = py::cast<Mx<I>>(_idx);
  X3<F> *px1 = (X3<F> *)x1.request().ptr;
  X3<F> *px2 = (X3<F> *)x2.request().ptr;
  py::gil_scoped_release release;
  Vx<K> keys(idx.rows());
  for (int i = 0; i < keys.size(); ++i) {
    keys[i] = xb.get_key(px1[idx(i, 0)].inverse() * (px2[idx(i, 1)]));
  }
  return keys;
}

template <typename K, typename F>
Vx<K> key_of_selected_pairs_onearray(Xbin<F, K> const &xb, py::array idx,
                                     py::array x1, py::array x2) {
  check_xform_array(x1);
  check_xform_array(x2);
  pybind11::array::ensure(idx);
  size_t sp = idx.itemsize();

  if (!idx) throw std::runtime_error("bad array");
  if (idx.ndim() != 2 || idx.shape()[1] != 2)
    throw std::runtime_error("index must be shape (N,2)");
  if (x1.dtype().kind() != x2.dtype().kind())
    throw std::runtime_error("xform arrays must have same dtype");
  if (py::isinstance<py::array_t<int64_t>>(idx)) {
    return kop2_onearray_impl<int64_t, F, K>(xb, idx, x1, x2);
  } else if (py::isinstance<py::array_t<int32_t>>(idx)) {
    return kop2_onearray_impl<int32_t, F, K>(xb, idx, x1, x2);
  } else if (py::isinstance<py::array_t<uint64_t>>(idx)) {
    return kop2_onearray_impl<uint64_t, F, K>(xb, idx, x1, x2);
  } else if (py::isinstance<py::array_t<uint32_t>>(idx)) {
    return kop2_onearray_impl<uint32_t, F, K>(xb, idx, x1, x2);
  } else {
    throw std::runtime_error("array dtype must be matching f4 or f8");
  }
}

template <typename K, typename F>
Vx<K> key_of_selected_pairs_onearray_same(Xbin<F, K> const &xb, py::array idx,
                                          py::array x) {
  return key_of_selected_pairs(xb, idx, x, x);
}

template <typename I, typename F, typename K>
Vx<K> kop2ss_impl(Xbin<F, K> const &xb, py::array_t<I> i1, py::array_t<I> i2,
                  py::array_t<I> ss1, py::array_t<I> ss2, py::array_t<F> x1,
                  py::array_t<F> x2) {
  I *i1p = (I *)i1.request().ptr;
  I *i2p = (I *)i2.request().ptr;
  I *ss1p = (I *)ss1.request().ptr;
  I *ss2p = (I *)ss2.request().ptr;
  X3<F> *x1p = (X3<F> *)x1.request().ptr;
  X3<F> *x2p = (X3<F> *)x2.request().ptr;
  py::gil_scoped_release release;
  Vx<K> keys(i1.shape()[0]);
  for (int i = 0; i < keys.size(); ++i) {
    K k = xb.get_key(x1p[i1p[i]].inverse() * (x2p[i2p[i]]));
    keys[i] = k | ((K)ss1p[i1p[i]] << 62) | ((K)ss2p[i2p[i]] << 60);
  }
  return keys;
}

template <typename K, typename F>
Vx<K> sskey_of_selected_pairs(Xbin<F, K> const &xb, py::array i1, py::array i2,
                              py::array ss1, py::array ss2, py::array x1,
                              py::array x2) {
  check_xform_array(x1);
  check_xform_array(x2);
  pybind11::array::ensure(i1);
  pybind11::array::ensure(i2);
  size_t sp = i1.itemsize();

  if (!i1) throw std::runtime_error("bad array");
  if (!i2) throw std::runtime_error("bad array");
  if (i1.ndim() != 1 || i2.ndim() != 1 || i1.size() != i2.size())
    throw std::runtime_error("index must be shape (N,) and same length");
  if (i1.dtype().kind() != i2.dtype().kind())
    throw std::runtime_error("index arrays must have same dtype");
  if (x1.dtype().kind() != x2.dtype().kind())
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
}

template <typename K, typename FX>
Vx<K> sskey_of_selected_pairs_same(Xbin<FX, K> const &xb, py::array i1,
                                   py::array i2, py::array ss, py::array x) {
  return sskey_of_selected_pairs(xb, i1, i2, ss, ss, x, x);
}

template <typename I, typename F, typename K>
Vx<K> kop3ss_impl(Xbin<F, K> const &xb, py::array_t<I> idx, py::array_t<I> ss1,
                  py::array_t<I> ss2, py::array_t<F> x1, py::array_t<F> x2) {
  I *idxp = (I *)idx.request().ptr;
  I *ss1p = (I *)ss1.request().ptr;
  I *ss2p = (I *)ss2.request().ptr;
  X3<F> *x1p = (X3<F> *)x1.request().ptr;
  X3<F> *x2p = (X3<F> *)x2.request().ptr;
  py::gil_scoped_release release;
  Vx<K> keys(idx.shape()[0]);
  for (int i = 0; i < keys.size(); ++i) {
    K k = xb.get_key(x1p[idxp[2 * i]].inverse() * (x2p[idxp[2 * i + 1]]));
    k |= ((K)ss1p[idxp[2 * i]] << 62) | ((K)ss2p[idxp[2 * i + 1]] << 60);
    keys[i] = k;
  }
  return keys;
}

template <typename K, typename F>
Vx<K> sskey_of_selected_pairs_onearray(Xbin<F, K> const &xb, py::array idx,
                                       py::array ss1, py::array ss2,
                                       py::array x1, py::array x2) {
  check_xform_array(x1);
  check_xform_array(x2);
  pybind11::array::ensure(idx);
  size_t sp = idx.itemsize();

  if (!idx) throw std::runtime_error("bad array");
  if (idx.ndim() != 2 || idx.shape()[1] != 2)
    throw std::runtime_error("index must be shape (N,2)");
  if (x1.dtype().kind() != x2.dtype().kind())
    throw std::runtime_error("xform arrays must have same dtype");
  if (py::isinstance<py::array_t<int64_t>>(idx)) {
    return kop3ss_impl<int64_t, F, K>(xb, idx, ss1, ss2, x1, x2);
  } else if (py::isinstance<py::array_t<int32_t>>(idx)) {
    return kop3ss_impl<int32_t, F, K>(xb, idx, ss1, ss2, x1, x2);
  } else if (py::isinstance<py::array_t<uint64_t>>(idx)) {
    return kop3ss_impl<uint64_t, F, K>(xb, idx, ss1, ss2, x1, x2);
  } else if (py::isinstance<py::array_t<uint32_t>>(idx)) {
    return kop3ss_impl<uint32_t, F, K>(xb, idx, ss1, ss2, x1, x2);
  } else {
    throw std::runtime_error("array dtype must be matching f4 or f8");
  }
}

template <typename K, typename FX>
Vx<K> sskey_of_selected_pairs_onearray_same(Xbin<FX, K> const &xb,
                                            py::array idx, py::array ss,
                                            py::array x) {
  return sskey_of_selected_pairs_onearray(xb, idx, ss, ss, x, x);
}

///////////////////////// with ss / maps //////////////////////////

template <typename I, typename F, typename K, typename V>
Vx<V> mapkop3ss_impl(Xbin<F, K> const &xb, PHMap<K, V> const &map,
                     py::array_t<I> idx, py::array_t<I> ss1, py::array_t<I> ss2,
                     py::array_t<F> x1, py::array_t<F> x2, V v0) {
  I *idxp = (I *)idx.request().ptr;
  I *ss1p = (I *)ss1.request().ptr;
  I *ss2p = (I *)ss2.request().ptr;
  X3<F> *x1p = (X3<F> *)x1.request().ptr;
  X3<F> *x2p = (X3<F> *)x2.request().ptr;
  py::gil_scoped_release release;
  Vx<V> vals(idx.shape()[0]);
  for (int i = 0; i < vals.size(); ++i) {
    K k = xb.get_key(x1p[idxp[2 * i]].inverse() * (x2p[idxp[2 * i + 1]]));
    k = k | ((K)ss1p[idxp[2 * i]] << 62) | ((K)ss2p[idxp[2 * i + 1]] << 60);
    auto it = map.phmap_.find(k);
    if (it == map.phmap_.end())
      vals[i] = v0;
    else
      vals[i] = it->second;
  }
  return vals;
}

template <typename K, typename F, typename V>
Vx<V> ssmap_of_selected_pairs_onearray(Xbin<F, K> const &xb,
                                       PHMap<K, V> const &map, py::array idx,
                                       py::array ss1, py::array ss2,
                                       py::array x1, py::array x2, V v0) {
  check_xform_array(x1);
  check_xform_array(x2);
  pybind11::array::ensure(idx);
  size_t sp = idx.itemsize();

  if (!idx) throw std::runtime_error("bad array");
  if (idx.ndim() != 2 || idx.shape()[1] != 2)
    throw std::runtime_error("index must be shape (N,2) and same length");
  if (x1.dtype().kind() != x2.dtype().kind())
    throw std::runtime_error("xform arrays must have same dtype");
  if (py::isinstance<py::array_t<int64_t>>(idx)) {
    return mapkop3ss_impl<int64_t, F, K>(xb, map, idx, ss1, ss2, x1, x2, v0);
  } else if (py::isinstance<py::array_t<int32_t>>(idx)) {
    return mapkop3ss_impl<int32_t, F, K>(xb, map, idx, ss1, ss2, x1, x2, v0);
  } else if (py::isinstance<py::array_t<uint64_t>>(idx)) {
    return mapkop3ss_impl<uint64_t, F, K>(xb, map, idx, ss1, ss2, x1, x2, v0);
  } else if (py::isinstance<py::array_t<uint32_t>>(idx)) {
    return mapkop3ss_impl<uint32_t, F, K>(xb, map, idx, ss1, ss2, x1, x2, v0);
  } else {
    throw std::runtime_error("array dtype must be matching f4 or f8");
  }
}

template <typename K, typename F, typename V>
Vx<V> ssmap_of_selected_pairs_onearray_same(Xbin<F, K> const &xb,
                                            PHMap<K, V> const &map,
                                            py::array idx, py::array ss,
                                            py::array x, V v0) {
  return ssmap_of_selected_pairs_onearray(xb, map, idx, ss, ss, x, x, v0);
}

/////////////////////////// map no ss //////////////////////////////////

template <typename I, typename F, typename K, typename V>
Vx<V> mapkop3_impl(Xbin<F, K> const &xb, PHMap<K, V> const &map,
                   py::array_t<I> idx, py::array_t<F> x1, py::array_t<F> x2,
                   V v0) {
  I *idxp = (I *)idx.request().ptr;
  X3<F> *x1p = (X3<F> *)x1.request().ptr;
  X3<F> *x2p = (X3<F> *)x2.request().ptr;
  py::gil_scoped_release release;
  Vx<V> vals(idx.shape()[0]);
  for (int i = 0; i < vals.size(); ++i) {
    K k = xb.get_key(x1p[idxp[2 * i]].inverse() * (x2p[idxp[2 * i + 1]]));
    auto it = map.phmap_.find(k);
    if (it == map.phmap_.end())
      vals[i] = v0;
    else
      vals[i] = it->second;
  }
  return vals;
}

template <typename K, typename F, typename V>
Vx<V> map_of_selected_pairs_onearray(Xbin<F, K> const &xb,
                                     PHMap<K, V> const &map, py::array idx,
                                     py::array x1, py::array x2, V v0) {
  check_xform_array(x1);
  check_xform_array(x2);
  pybind11::array::ensure(idx);
  size_t sp = idx.itemsize();

  if (!idx) throw std::runtime_error("bad array");
  if (idx.ndim() != 2 || idx.shape()[1] != 2)
    throw std::runtime_error("index must be shape (N,2) and same length");
  if (x1.dtype().kind() != x2.dtype().kind())
    throw std::runtime_error("xform arrays must have same dtype");
  if (py::isinstance<py::array_t<int64_t>>(idx)) {
    return mapkop3_impl<int64_t, F, K>(xb, map, idx, x1, x2, v0);
  } else if (py::isinstance<py::array_t<int32_t>>(idx)) {
    return mapkop3_impl<int32_t, F, K>(xb, map, idx, x1, x2, v0);
  } else if (py::isinstance<py::array_t<uint64_t>>(idx)) {
    return mapkop3_impl<uint64_t, F, K>(xb, map, idx, x1, x2, v0);
  } else if (py::isinstance<py::array_t<uint32_t>>(idx)) {
    return mapkop3_impl<uint32_t, F, K>(xb, map, idx, x1, x2, v0);
  } else {
    throw std::runtime_error("array dtype must be matching f4 or f8");
  }
}

template <typename K, typename F, typename V>
Vx<V> map_of_selected_pairs_onearray_same(Xbin<F, K> const &xb,
                                          PHMap<K, V> const &map, py::array idx,
                                          py::array x, V v0) {
  return map_of_selected_pairs_onearray(xb, map, idx, x, x, v0);
}

//////////////////////////////////////////////////////////////////////////////

template <typename F, typename K>
py::tuple xform_to_F6(Xbin<F, K> const &xbin, py::array_t<F> _xform) {
  auto xform = xform_py_to_eigen(_xform);
  auto f6 = std::make_unique<Mx<F>>();
  auto cell = std::make_unique<Vx<K>>();
  {
    py::gil_scoped_release release;
    f6->resize(xform.size(), 6);
    cell->resize(xform.size());
    for (int i = 0; i < xform.size(); ++i)
      f6->row(i) = xbin.xform_to_F6(xform[i], (*cell)[i]);
  }
  return py::make_tuple(*f6, *cell);
}

template <typename F, typename K>
py::array_t<F> F6_to_xform(Xbin<F, K> const &xbin, Mx<F> f6, Vx<K> cell) {
  if (f6.cols() != 6) throw std::runtime_error("f6 must be shape(N,6)");
  if (f6.rows() != cell.size())
    throw std::runtime_error("f6 and cell must have same length");
  auto out = std::make_unique<Vx<X3<F>>>();
  {
    py::gil_scoped_release release;
    out->resize(cell.size());
    for (int i = 0; i < cell.size(); ++i)
      (*out)[i] = xbin.F6_to_xform(f6.row(i), cell[i]);
  }
  return xform_eigen_to_py(*out);
}

template <typename F, typename K>
void bind_xbin(py::module m, std::string name) {
  using THIS = Xbin<F, K>;
  auto cls =
      py::class_<THIS>(m, name.c_str())
          .def(py::init<F, F, F>(), "cart_resl"_a = 1.0, "ori_resl"_a = 20.0,
               "max_cart"_a = 512.0)
          .def("__getitem__", &key_of<F, K>)
          .def("__getitem__", &_bincen_of<F, K>)
          .def("key_of", &key_of<F, K>, "key of xform", "xform"_c)
          .def("ori_cell_of", &ori_cell_of<F, K>, "key of xform", "xform"_c)
          .def("bincen_of", &_bincen_of<F, K>)
          .def("xform_to_F6", &xform_to_F6<F, K>)
          .def("F6_to_xform", &F6_to_xform<F, K>)
          .def_readonly("grid6", &THIS::grid6_)
          .def_readonly("cart_resl", &THIS::cart_resl_)
          .def_readonly("ori_resl", &THIS::ori_resl_)
          .def_readonly("max_cart", &THIS::cart_bound_)
          .def_readonly("ori_nside", &THIS::ori_nside_)
          .def("__eq__",
               [](THIS const &a, THIS const &b) {
                 return a.cart_resl_ == b.cart_resl_ &&
                        a.ori_nside_ == b.ori_nside_ &&
                        a.cart_bound_ == b.cart_bound_;
               })
          .def(py::pickle(
              [](const THIS &xbin) {  // __getstate__
                return py::make_tuple(xbin.cart_resl_, xbin.ori_nside_,
                                      xbin.cart_bound_);
              },
              [](py::tuple t) {  // __setstate__
                if (t.size() != 3) throw std::runtime_error("Invalid state!");
                return THIS(t[0].cast<F>(), t[1].cast<int>(), t[2].cast<F>());
              }))

      /**/;
}

template <typename F, typename K>
Xbin<F, K> create_Xbin_nside(F cart_resl, int nside, F max_cart) {
  return XformHash_bt24_BCC6<X3<F>, K>(cart_resl, nside, max_cart);
}

PYBIND11_MODULE(xbin, m) {
  using K = uint64_t;
  bind_xbin<double, K>(m, "Xbin_double");
  bind_xbin<float, K>(m, "Xbin_float");
  m.def("create_Xbin_nside_double", &create_Xbin_nside<double, K>);
  m.def("create_Xbin_nside_float", &create_Xbin_nside<float, K>);
}

}  // namespace xbin
}  // namespace rpxdock