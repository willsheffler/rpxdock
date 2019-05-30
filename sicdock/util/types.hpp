#pragma once

#include <Eigen/Geometry>
#include <limits>

namespace sicdock {
namespace util {

// RowMajor to match numpy
template <class F>
using V1 = Eigen::Matrix<F, 1, 1>;
template <class F>
using V2 = Eigen::Matrix<F, 2, 1>;
template <class F>
using V3 = Eigen::Matrix<F, 3, 1>;  // todo: should use aligned vector3?
template <class F>
using M3 = Eigen::Matrix<F, 3, 3, Eigen::RowMajor>;
template <class F>
using X3 = Eigen::Transform<F, 3, Eigen::Affine, Eigen::RowMajor>;
template <class F>
using X3C = Eigen::Transform<F, 3, Eigen::AffineCompact, Eigen::RowMajor>;
template <class F>
using V4 = Eigen::Matrix<F, 4, 1>;
template <class F>
using M4 = Eigen::Matrix<F, 4, 4, Eigen::RowMajor>;
template <class F>
using V5 = Eigen::Matrix<F, 5, 1>;
template <class F>
using V6 = Eigen::Matrix<F, 6, 1>;

using V3f = V3<float>;
using V3d = V3<double>;
using X3f = X3<float>;
using X3d = X3<double>;

using M4f = M4<float>;
using M4d = M4<double>;

template <typename F>
using Mx = Eigen::Matrix<F, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <typename F>
using RefMx = Eigen::Ref<Mx<F>>;

template <typename F>
using Vx = Eigen::Matrix<F, Eigen::Dynamic, 1>;
template <typename F>
using RefVx = Eigen::Ref<Vx<F>>;

using Mxd = Mx<double>;
using RefMxd = RefMx<double>;
using Vxd = Vx<double>;
using RefVxd = RefVx<double>;
using Vxi = Vx<int>;
using RefVxi = RefVx<int>;

template <typename F>
using MapVxX3 = Eigen::Map<Vx<X3<F>>>;

template <typename T>
using NL = std::numeric_limits<T>;

template <class F>
F epsilon2() {
  return std::sqrt(std::numeric_limits<F>::epsilon());
}

}  // namespace util
}  // namespace sicdock

namespace Eigen {

template <typename V>
V *begin(sicdock::util::Vx<V> v) {
  return v.data();
}
template <typename V>
V *end(sicdock::util::Vx<V> v) {
  return v.data() + v.size();
}

}  // namespace Eigen
