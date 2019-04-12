#pragma once

#include <Eigen/Geometry>

namespace tcdock {
namespace util {

template <class F>
using V3 = Eigen::Matrix<F, 3, 1>;  // todo: should use aligned vector3?
template <class F>
using M3 = Eigen::Matrix<F, 3, 3, Eigen::RowMajor>;  // to match numpy (??)
template <class F>
using X3 = Eigen::Transform<F, 3, Eigen::Affine, Eigen::RowMajor>;

using V3f = V3<float>;
using V3d = V3<double>;

template <class F>
F epsilon2() {
  return std::sqrt(std::numeric_limits<F>::epsilon());
}

}  // namespace util
}  // namespace tcdock