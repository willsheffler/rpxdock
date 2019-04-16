// #include "tcdock/util/types.hpp"
#pragma once

namespace tcdock {
namespace util {

template <class F>
F square(F x) {
  return x * x;
}

///@brief return sum of highest two elements in vector
template <class Vector, class Index>
void max2(Vector vector, typename Vector::Scalar &mx1,
          typename Vector::Scalar &mx2, Index &argmax_1, Index &argmax_2) {
  // TODO: is there a faster way to do this?
  mx1 = vector.maxCoeff(&argmax_1);
  vector[argmax_1] = -std::numeric_limits<typename Vector::Scalar>::max();
  mx2 = vector.maxCoeff(&argmax_2);
}

}  // namespace util
}  // namespace tcdock