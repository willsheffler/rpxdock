#pragma once
/** \file */

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "rpxdock/util/types.hpp"

namespace rpxdock {
namespace util {

using nogil = pybind11::call_guard<pybind11::gil_scoped_release>;

pybind11::arg operator"" _c(const char *name, size_t) {
  return pybind11::arg(name).noconvert();
}

template <typename F>
X3<F> xform_py_to_X3(pybind11::array_t<F> a) {
  if (!a) throw std::runtime_error("bad array");
  if (a.ndim() != 2) throw std::runtime_error("must be 2D array shape (4,4)");
  if (a.shape()[0] != 4 || a.shape()[1] != 4)
    if (a.ndim() != 2) throw std::runtime_error("must be 2D array shape (4,4)");
  return (X3<F>)*((X3<F> *)(a.request().ptr));
}

template <typename F>
MapVxX3<F> xform_py_to_eigen(pybind11::array_t<F> a) {
  auto buf = pybind11::array::ensure(a);
  size_t s = buf.itemsize();
  if (!a) throw std::runtime_error("bad array");
  if (a.ndim() == 2) {
    if (a.shape()[0] != 4 || a.shape()[1] != 4)
      throw std::runtime_error("2D array must be shape (4,4)");
    if (a.strides()[0] != 4 * s || a.strides()[1] != s)
      throw std::runtime_error("bad strides, strides not supported");
    X3<F> *ptr = (X3<F> *)a.request().ptr;
    return MapVxX3<F>(ptr, 1);
  } else {
    if (a.ndim() != 3 || a.shape()[1] != 4 || a.shape()[2] != 4)
      throw std::runtime_error("3D array must be shape (N,4,4)");
    if (a.strides()[0] != 16 * s || a.strides()[1] != 4 * s)
      throw std::runtime_error("bad strides, strides not supported");
    X3<F> *ptr = (X3<F> *)a.request().ptr;
    return MapVxX3<F>(ptr, a.shape()[0]);
  }
}

template <typename XformArray>
auto xform_eigen_to_py(XformArray xform, int size = -1) {
  using Xform = typename XformArray::Scalar;
  using F = typename Xform::Scalar;
  if (size < 0) size = (int)xform.size();
  F *data = (F *)xform.data();
  std::vector<size_t> shape{size, 4, 4};
  std::vector<size_t> stride{16 * sizeof(F), 4 * sizeof(F), sizeof(F)};
  auto buf = pybind11::buffer_info(data, shape, stride);
  return pybind11::array_t<F>(buf);
}

template <typename XformArray>
auto xform_eigenptr_to_py(std::unique_ptr<XformArray> xform) {
  using Xform = typename XformArray::Scalar;
  using F = typename Xform::Scalar;
  F *data = (F *)xform->data();
  std::vector<size_t> shape{xform->size(), 4, 4};
  std::vector<size_t> stride{16 * sizeof(F), 4 * sizeof(F), sizeof(F)};
  auto buf = pybind11::buffer_info(data, shape, stride);
  return pybind11::array_t<F>(buf);
}

inline void check_xform_array(pybind11::array a) {
  auto buf = pybind11::array::ensure(a);
  size_t s = buf.itemsize();
  if (!a) throw std::runtime_error("bad array");
  if (a.ndim() != 3 || a.shape()[1] != 4 || a.shape()[2] != 4)
    throw std::runtime_error("array X2 must be shape (N,4,4)");
  if (a.strides()[0] != 16 * s || a.strides()[1] != 4 * s)
    throw std::runtime_error("bad strides, strides not supported");
}

}  // namespace util
}  // namespace rpxdock