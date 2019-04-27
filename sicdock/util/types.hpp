#pragma once

#include <Eigen/Geometry>

namespace sicdock {
namespace util {

// RowMajor to match numpy
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

using RowMajorXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RefRowMajorXd = Eigen::Ref<RowMajorXd>;
using VectorXd = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using RefVectorXd = Eigen::Ref<VectorXd>;

template <typename F>
using RowMajorX =
    Eigen::Matrix<F, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <typename F>
using RefRowMajorX = Eigen::Ref<RowMajorX<F>>;

template <class F>
F epsilon2() {
  return std::sqrt(std::numeric_limits<F>::epsilon());
}

}  // namespace util
}  // namespace sicdock