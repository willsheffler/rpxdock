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

#include <iostream>
#include <string>

#include "rpxdock/phmap/phmap.hpp"
#include "rpxdock/util/Timer.hpp"
#include "rpxdock/util/assertions.hpp"
#include "rpxdock/util/global_rng.hpp"
#include "rpxdock/util/numeric.hpp"
#include "rpxdock/util/pybind_types.hpp"
#include "rpxdock/util/types.hpp"
#include "rpxdock/xbin/xbin.hpp"

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

template <typename I, typename F, typename K>
Vx<K> kop_impl(Xbin<F, K> const &xb, py::array_t<I> p, py::array_t<F> x1,
               py::array_t<F> x2, M4<F> p1, M4<F> p2) noexcept {
  I *pp = (I *)p.request().ptr;
  X3<F> *px1 = (X3<F> *)x1.request().ptr;
  X3<F> *px2 = (X3<F> *)x2.request().ptr;
  py::gil_scoped_release release;
  X3<F> x21 = X3<F>(p1).inverse() * X3<F>(p2);
  Vx<K> keys(p.shape()[0]);
  for (int ip = 0; ip < keys.size(); ++ip) {
    I i1 = pp[2 * ip + 0];
    I i2 = pp[2 * ip + 1];
    keys[ip] = xb.get_key(px1[i1].inverse() * (x21 * px2[i2]));
  }
  return keys;
}

template <typename K, typename F>
Vx<K> key_of_pairs(Xbin<F, K> const &xb, py::array xp, py::array x1,
                   py::array x2, M4<F> p1, M4<F> p2) {
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
    return kop_impl<int64_t, F, K>(xb, xp, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<int32_t>>(xp)) {
    return kop_impl<int32_t, F, K>(xb, xp, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<uint64_t>>(xp)) {
    return kop_impl<uint64_t, F, K>(xb, xp, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<uint32_t>>(xp)) {
    return kop_impl<uint32_t, F, K>(xb, xp, x1, x2, p1, p2);
  } else {
    throw std::runtime_error("array dtype must be matching f4 or f8");
  }
}

template <typename I, typename F, typename K>
Vx<K> kop2_impl(Xbin<F, K> const &xb, py::array_t<I> i1, py::array_t<I> i2,
                py::array_t<F> x1, py::array_t<F> x2, M4<F> p1,
                M4<F> p2) noexcept {
  I *i1p = (I *)i1.request().ptr;
  I *i2p = (I *)i2.request().ptr;
  X3<F> *px1 = (X3<F> *)x1.request().ptr;
  X3<F> *px2 = (X3<F> *)x2.request().ptr;
  py::gil_scoped_release release;
  X3<F> x21 = X3<F>(p1).inverse() * X3<F>(p2);
  Vx<K> keys(i1.shape()[0]);
  for (int i = 0; i < keys.size(); ++i) {
    keys[i] = xb.get_key(px1[i1p[i]].inverse() * (x21 * px2[i2p[i]]));
  }

  return keys;
}

template <typename K, typename F>
Vx<K> key_of_selected_pairs(Xbin<F, K> const &xb, py::array i1, py::array i2,
                            py::array x1, py::array x2, M4<F> p1, M4<F> p2) {
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
    return kop2_impl<int64_t, F, K>(xb, i1, i2, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<int32_t>>(i1)) {
    return kop2_impl<int32_t, F, K>(xb, i1, i2, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<uint64_t>>(i1)) {
    return kop2_impl<uint64_t, F, K>(xb, i1, i2, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<uint32_t>>(i1)) {
    return kop2_impl<uint32_t, F, K>(xb, i1, i2, x1, x2, p1, p2);
  } else {
    throw std::runtime_error("array dtype must be matching f4 or f8");
  }
}

template <typename K, typename F>
Vx<K> key_of_selected_pairs_same(Xbin<F, K> const &xb, py::array i1,
                                 py::array i2, py::array x, M4<F> p1,
                                 M4<F> p2) {
  return key_of_selected_pairs(xb, i1, i2, x, x, p1, p2);
}

//////////////////////////// N,2 idx array key lookup
///////////////////////////////////////

template <typename I, typename F, typename K>
Vx<K> kop2_onearray_impl(Xbin<F, K> const &xb, py::array_t<I> _idx,
                         py::array_t<F> x1, py::array_t<F> x2, M4<F> p1,
                         M4<F> p2) noexcept {
  auto idx = py::cast<Mx<I>>(_idx);
  X3<F> *px1 = (X3<F> *)x1.request().ptr;
  X3<F> *px2 = (X3<F> *)x2.request().ptr;
  py::gil_scoped_release release;
  X3<F> x21 = X3<F>(p1).inverse() * X3<F>(p2);
  Vx<K> keys(idx.rows());
  for (int i = 0; i < keys.size(); ++i) {
    keys[i] = xb.get_key(px1[idx(i, 0)].inverse() * (x21 * px2[idx(i, 1)]));
  }
  return keys;
}

template <typename K, typename F>
Vx<K> key_of_selected_pairs_onearray(Xbin<F, K> const &xb, py::array idx,
                                     py::array x1, py::array x2, M4<F> p1,
                                     M4<F> p2) {
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
    return kop2_onearray_impl<int64_t, F, K>(xb, idx, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<int32_t>>(idx)) {
    return kop2_onearray_impl<int32_t, F, K>(xb, idx, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<uint64_t>>(idx)) {
    return kop2_onearray_impl<uint64_t, F, K>(xb, idx, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<uint32_t>>(idx)) {
    return kop2_onearray_impl<uint32_t, F, K>(xb, idx, x1, x2, p1, p2);
  } else {
    throw std::runtime_error("array dtype must be matching f4 or f8");
  }
}

template <typename K, typename F>
Vx<K> key_of_selected_pairs_onearray_same(Xbin<F, K> const &xb, py::array idx,
                                          py::array x, M4<F> p1, M4<F> p2) {
  return key_of_selected_pairs(xb, idx, x, x, p1, p2);
}

template <typename I, typename F, typename K>
Vx<K> kop2ss_impl(Xbin<F, K> const &xb, py::array_t<I> i1, py::array_t<I> i2,
                  py::array_t<I> ss1, py::array_t<I> ss2, py::array_t<F> x1,
                  py::array_t<F> x2, M4<F> p1, M4<F> p2) noexcept {
  I *i1p = (I *)i1.request().ptr;
  I *i2p = (I *)i2.request().ptr;
  I *ss1p = (I *)ss1.request().ptr;
  I *ss2p = (I *)ss2.request().ptr;
  X3<F> *x1p = (X3<F> *)x1.request().ptr;
  X3<F> *x2p = (X3<F> *)x2.request().ptr;
  py::gil_scoped_release release;
  X3<F> x21 = X3<F>(p1).inverse() * X3<F>(p2);
  Vx<K> keys(i1.shape()[0]);
  for (int i = 0; i < keys.size(); ++i) {
    K k = xb.get_key(x1p[i1p[i]].inverse() * (x21 * x2p[i2p[i]]));
    keys[i] = k | ((K)ss1p[i1p[i]] << 62) | ((K)ss2p[i2p[i]] << 60);
  }
  return keys;
}

template <typename K, typename F>
Vx<K> sskey_of_selected_pairs(Xbin<F, K> const &xb, py::array i1, py::array i2,
                              py::array ss1, py::array ss2, py::array x1,
                              py::array x2, M4<F> p1, M4<F> p2) {
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
    return kop2ss_impl<int64_t, F, K>(xb, i1, i2, ss1, ss2, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<int32_t>>(i1)) {
    return kop2ss_impl<int32_t, F, K>(xb, i1, i2, ss1, ss2, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<uint64_t>>(i1)) {
    return kop2ss_impl<uint64_t, F, K>(xb, i1, i2, ss1, ss2, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<uint32_t>>(i1)) {
    return kop2ss_impl<uint32_t, F, K>(xb, i1, i2, ss1, ss2, x1, x2, p1, p2);
  } else {
    throw std::runtime_error("array dtype must be matching f4 or f8");
  }
}

template <typename K, typename F>
Vx<K> sskey_of_selected_pairs_same(Xbin<F, K> const &xb, py::array i1,
                                   py::array i2, py::array ss, py::array x,
                                   M4<F> p1, M4<F> p2) {
  return sskey_of_selected_pairs(xb, i1, i2, ss, ss, x, x, p1, p2);
}

template <typename I, typename F, typename K>
Vx<K> kop3ss_impl(Xbin<F, K> const &xb, py::array_t<I> idx, py::array_t<I> ss1,
                  py::array_t<I> ss2, py::array_t<F> x1, py::array_t<F> x2,
                  M4<F> p1, M4<F> p2) noexcept {
  I *idxp = (I *)idx.request().ptr;
  I *ss1p = (I *)ss1.request().ptr;
  I *ss2p = (I *)ss2.request().ptr;
  X3<F> *x1p = (X3<F> *)x1.request().ptr;
  X3<F> *x2p = (X3<F> *)x2.request().ptr;
  py::gil_scoped_release release;
  X3<F> x21 = X3<F>(p1).inverse() * X3<F>(p2);
  Vx<K> keys(idx.shape()[0]);
  for (int i = 0; i < keys.size(); ++i) {
    K k = xb.get_key(x1p[idxp[2 * i]].inverse() * (x21 * x2p[idxp[2 * i + 1]]));
    k |= ((K)ss1p[idxp[2 * i]] << 62) | ((K)ss2p[idxp[2 * i + 1]] << 60);
    keys[i] = k;
  }
  return keys;
}

template <typename K, typename F>
Vx<K> sskey_of_selected_pairs_onearray(Xbin<F, K> const &xb, py::array idx,
                                       py::array ss1, py::array ss2,
                                       py::array x1, py::array x2, M4<F> p1,
                                       M4<F> p2) {
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
    return kop3ss_impl<int64_t, F, K>(xb, idx, ss1, ss2, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<int32_t>>(idx)) {
    return kop3ss_impl<int32_t, F, K>(xb, idx, ss1, ss2, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<uint64_t>>(idx)) {
    return kop3ss_impl<uint64_t, F, K>(xb, idx, ss1, ss2, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<uint32_t>>(idx)) {
    return kop3ss_impl<uint32_t, F, K>(xb, idx, ss1, ss2, x1, x2, p1, p2);
  } else {
    throw std::runtime_error("array dtype must be matching f4 or f8");
  }
}

template <typename K, typename F>
Vx<K> sskey_of_selected_pairs_onearray_same(Xbin<F, K> const &xb, py::array idx,
                                            py::array ss, py::array x, M4<F> p1,
                                            M4<F> p2) {
  return sskey_of_selected_pairs_onearray(xb, idx, ss, ss, x, x, p1, p2);
}

///////////////////////// with ss / maps //////////////////////////

template <typename I, typename F, typename K, typename V>
Vx<V> mapkop3ss_impl(Xbin<F, K> const &xb, PHMap<K, V> const &map,
                     py::array_t<I> idx, py::array_t<I> ss1, py::array_t<I> ss2,
                     py::array_t<F> x1, py::array_t<F> x2, M4<F> p1,
                     M4<F> p2) noexcept {
  I *idxp = (I *)idx.request().ptr;
  I *ss1p = (I *)ss1.request().ptr;
  I *ss2p = (I *)ss2.request().ptr;
  X3<F> *x1p = (X3<F> *)x1.request().ptr;
  X3<F> *x2p = (X3<F> *)x2.request().ptr;
  py::gil_scoped_release release;
  X3<F> x21 = X3<F>(p1).inverse() * X3<F>(p2);
  Vx<V> vals(idx.shape()[0]);
  for (int i = 0; i < vals.size(); ++i) {
    K k = xb.get_key(x1p[idxp[2 * i]].inverse() * (x21 * x2p[idxp[2 * i + 1]]));
    k = k | ((K)ss1p[idxp[2 * i]] << 62) | ((K)ss2p[idxp[2 * i + 1]] << 60);
    vals[i] = map.get_default(k);
  }
  return vals;
}

template <typename K, typename F, typename V>
Vx<V> ssmap_of_selected_pairs_onearray(Xbin<F, K> const &xb,
                                       PHMap<K, V> const &m, py::array idx,
                                       py::array ss1, py::array ss2,
                                       py::array x1, py::array x2, M4<F> p1,
                                       M4<F> p2) {
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
    return mapkop3ss_impl<int64_t, F, K>(xb, m, idx, ss1, ss2, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<int32_t>>(idx)) {
    return mapkop3ss_impl<int32_t, F, K>(xb, m, idx, ss1, ss2, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<uint64_t>>(idx)) {
    return mapkop3ss_impl<uint64_t, F, K>(xb, m, idx, ss1, ss2, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<uint32_t>>(idx)) {
    return mapkop3ss_impl<uint32_t, F, K>(xb, m, idx, ss1, ss2, x1, x2, p1, p2);
  } else {
    throw std::runtime_error("array dtype must be matching f4 or f8");
  }
}

template <typename K, typename F, typename V>
Vx<V> ssmap_of_selected_pairs_onearray_same(Xbin<F, K> const &xb,
                                            PHMap<K, V> const &m, py::array idx,
                                            py::array ss, py::array x, M4<F> p1,
                                            M4<F> p2) {
  return ssmap_of_selected_pairs_onearray(xb, m, idx, ss, ss, x, x, p1, p2);
}

/////////////////////////// map no ss //////////////////////////////////

template <typename I, typename F, typename K, typename V>
Vx<V> mapkop3_impl(Xbin<F, K> const &xb, PHMap<K, V> const &map,
                   py::array_t<I> idx, py::array_t<F> x1, py::array_t<F> x2,
                   M4<F> p1, M4<F> p2) noexcept {
  I *idxp = (I *)idx.request().ptr;
  X3<F> *x1p = (X3<F> *)x1.request().ptr;
  X3<F> *x2p = (X3<F> *)x2.request().ptr;
  py::gil_scoped_release release;
  X3<F> x21 = X3<F>(p1).inverse() * X3<F>(p2);
  Vx<V> vals(idx.shape()[0]);
  for (int i = 0; i < vals.size(); ++i) {
    K k = xb.get_key(x1p[idxp[2 * i]].inverse() * (x21 * x2p[idxp[2 * i + 1]]));
    vals[i] = map.get_default(k);
  }
  return vals;
}

template <typename K, typename F, typename V>
Vx<V> map_of_selected_pairs_onearray(Xbin<F, K> const &xb,
                                     PHMap<K, V> const &map, py::array idx,
                                     py::array x1, py::array x2, M4<F> p1,
                                     M4<F> p2) {
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
    return mapkop3_impl<int64_t, F, K>(xb, map, idx, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<int32_t>>(idx)) {
    return mapkop3_impl<int32_t, F, K>(xb, map, idx, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<uint64_t>>(idx)) {
    return mapkop3_impl<uint64_t, F, K>(xb, map, idx, x1, x2, p1, p2);
  } else if (py::isinstance<py::array_t<uint32_t>>(idx)) {
    return mapkop3_impl<uint32_t, F, K>(xb, map, idx, x1, x2, p1, p2);
  } else {
    throw std::runtime_error("array dtype must be matching f4 or f8");
  }
}

template <typename K, typename F, typename V>
Vx<V> map_of_selected_pairs_onearray_same(Xbin<F, K> const &xb,
                                          PHMap<K, V> const &map, py::array idx,
                                          py::array x, M4<F> p1, M4<F> p2) {
  return map_of_selected_pairs_onearray(xb, map, idx, x, x, p1, p2);
}

///////////////////////// ssmap_pairs_multipos

template <typename K, typename F, typename V>
Vx<V> ssmap_pairs_multipos(Xbin<F, K> const &xb, PHMap<K, V> const &map,
                           Mx<int32_t> pairs, Vx<K> ss1, Vx<K> ss2,
                           py::array_t<F> x1, py::array_t<F> x2,
                           Mx<int32_t> lbub, py::array_t<F> p1,
                           py::array_t<F> p2) {
  auto stub1 = xform_py_to_eigen(x1);
  auto stub2 = xform_py_to_eigen(x2);
  auto pos1 = xform_py_to_eigen(p1);
  auto pos2 = xform_py_to_eigen(p2);
  if (pairs.cols() != 2) throw std::runtime_error("pairs mest be shape (N,2)");
  if (lbub.cols() != 2) throw std::runtime_error("lbub mest be shape (M,2)");
  if (pos1.rows() != lbub.rows() && pos2.rows() != lbub.rows())
    throw std::runtime_error("pos1/2 must match lbub");
  if (pos1.rows() != pos2.rows() && pos1.rows() > 1 && pos2.size() > 1)
    throw std::runtime_error("pos1 / pos2 must be same size or size 1");
  if (ss1.size() != stub1.size() || ss2.size() != stub2.size())
    throw std::runtime_error("ss/stub must be same len");

  py::gil_scoped_release release;

  Vx<V> vals(pairs.rows());
  int ntot = 0;
  for (int ipos = 0; ipos < lbub.rows(); ++ipos) {
    size_t i1 = pos1.rows() == 1 ? 0 : ipos;
    size_t i2 = pos2.rows() == 1 ? 0 : ipos;
    X3<F> x21 = pos1[i1].inverse() * pos2[i2];
    int32_t lb = lbub(ipos, 0), ub = lbub(ipos, 1);
    for (int32_t i = lb; i < ub; ++i) {
      X3<F> x = stub1[pairs(i, 0)].inverse() * x21 * stub2[pairs(i, 1)];
      K k = xb.get_key(x);
      k |= (ss2[pairs(i, 0)] << 62) | (ss2[pairs(i, 1)] << 60);
      vals[ntot++] = map.get_default(k);
    }
  }
  if (ntot != pairs.rows())
    throw std::runtime_error("ssmap_pairs_multipos error");
  return vals;
}

template <typename K, typename F, typename V>
Vx<V> map_pairs_multipos(Xbin<F, K> const &xb, PHMap<K, V> const &map,
                         Mx<int32_t> pairs, py::array_t<F> x1,
                         py::array_t<F> x2, Mx<int32_t> lbub, py::array_t<F> p1,
                         py::array_t<F> p2) {
  auto stub1 = xform_py_to_eigen(x1);
  auto stub2 = xform_py_to_eigen(x2);
  auto pos1 = xform_py_to_eigen(p1);
  auto pos2 = xform_py_to_eigen(p2);
  if (pairs.cols() != 2) throw std::runtime_error("pairs mest be shape (N,2)");
  if (lbub.cols() != 2) throw std::runtime_error("lbub mest be shape (M,2)");
  if (pos1.rows() != lbub.rows() && pos2.rows() != lbub.rows())
    throw std::runtime_error("pos1/2 must match lbub");
  if (pos1.rows() != pos2.rows() && pos1.rows() > 1 && pos2.size() > 1)
    throw std::runtime_error("pos1 / pos2 must be same size or size 1");

  py::gil_scoped_release release;

  Vx<V> vals(pairs.rows());
  int ntot = 0;
  for (int ipos = 0; ipos < lbub.rows(); ++ipos) {
    size_t i1 = pos1.rows() == 1 ? 0 : ipos;
    size_t i2 = pos2.rows() == 1 ? 0 : ipos;
    X3<F> x21 = pos1[i1].inverse() * pos2[i2];
    int32_t lb = lbub(ipos, 0), ub = lbub(ipos, 1);
    for (int32_t i = lb; i < ub; ++i) {
      X3<F> x = stub1[pairs(i, 0)].inverse() * x21 * stub2[pairs(i, 1)];
      K k = xb.get_key(x);
      vals[ntot++] = map.get_default(k);
    }
  }
  if (ntot != pairs.rows())
    throw std::runtime_error("ssmap_pairs_multipos error");
  return vals;
}

//////////////////////////////////////////////////////////////////////////////

template <typename F, typename K>
void bind_xbin_util(py::module m) {
  auto eye4 = M4<F>::Identity();
  m.def("key_of_pairs", &key_of_pairs<K, F>, "xbin"_a, "pairs"_c, "xform1"_c,
        "xform2"_c, "pos1"_a = eye4, "pos2"_a = eye4);

  m.def("key_of_selected_pairs", &key_of_selected_pairs<K, F>, "xbin"_a,
        "idx1"_c, "idx2"_c, "xform1"_c, "xform2"_c, "pos1"_a = eye4,
        "pos2"_a = eye4);
  m.def("key_of_selected_pairs", &key_of_selected_pairs_onearray<K, F>,
        "xbin"_a, "idx"_c, "xform1"_c, "xform2"_c, "pos1"_a = eye4,
        "pos2"_a = eye4);

  m.def("sskey_of_selected_pairs", &sskey_of_selected_pairs<K, F>, "xbin"_a,
        "idx1"_c, "idx2"_c, "ss1"_c, "ss2"_c, "xform1"_c, "xform2"_c,
        "pos1"_a = eye4, "pos2"_a = eye4);
  m.def("sskey_of_selected_pairs", &sskey_of_selected_pairs_onearray<K, F>,
        "xbin"_a, "idx"_c, "ss1"_c, "ss2"_c, "xform1"_c, "xform2"_c,
        "pos1"_a = eye4, "pos2"_a = eye4);
  m.def("sskey_of_selected_pairs", &sskey_of_selected_pairs_same<K, F>,
        "xbin"_a, "idx1"_c, "idx2"_c, "ss"_c, "xform"_c, "pos1"_a = eye4,
        "pos2"_a = eye4);
  m.def("sskey_of_selected_pairs", &sskey_of_selected_pairs_onearray_same<K, F>,
        "xbin"_a, "idx"_c, "ss"_c, "xform"_c, "pos1"_a = eye4, "pos2"_a = eye4);

  m.def("map_of_selected_pairs", &map_of_selected_pairs_onearray<K, F, double>,
        "xbin"_a, "phmap"_a, "idx"_c, "xform1"_c, "xform2"_c, "pos1"_a = eye4,
        "pos2"_a = eye4);
  m.def("map_of_selected_pairs",
        &map_of_selected_pairs_onearray_same<K, F, double>, "xbin"_a, "phmap"_a,
        "idx"_c, "xform"_c, "pos1"_a = eye4, "pos2"_a = eye4);

  m.def("ssmap_of_selected_pairs",
        &ssmap_of_selected_pairs_onearray<K, F, double>, "xbin"_a, "phmap"_a,
        "idx"_c, "ss1"_c, "ss2"_c, "xform1"_c, "xform2"_c, "pos1"_a = eye4,
        "pos2"_a = eye4);
  m.def("ssmap_of_selected_pairs",
        &ssmap_of_selected_pairs_onearray_same<K, F, double>, "xbin"_a,
        "phmap"_a, "idx"_c, "ss"_c, "xform"_c, "pos1"_a = eye4,
        "pos2"_a = eye4);

  m.def("map_pairs_multipos", &map_pairs_multipos<K, F, double>, "xbin"_a,
        "phmap"_a, "idx"_c, "xform1"_c, "xform2"_c, "lbub"_c, "pos1"_a = eye4,
        "pos2"_a = eye4);
  m.def("ssmap_pairs_multipos", &ssmap_pairs_multipos<K, F, double>, "xbin"_a,
        "phmap"_a, "idx"_c, "ss1"_c, "ss2"_c, "xform1"_c, "xform2"_c, "lbub"_c,
        "pos1"_a = eye4, "pos2"_a = eye4);
}

PYBIND11_MODULE(xbin_util, m) {
  bind_xbin_util<double, uint64_t>(m);
  bind_xbin_util<float, uint64_t>(m);
}

}  // namespace xbin
}  // namespace rpxdock