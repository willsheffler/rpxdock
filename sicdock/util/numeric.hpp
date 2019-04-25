#pragma once

#include "sicdock/util/SimpleArray.hpp"
#include "sicdock/util/types.hpp"

namespace sicdock {
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

static bool is_not_0(double a) { return fabs(a) > 0.00000001; }
static bool is_not_0(float a) {
  return fabs(a) > 0.0001;
}  // need something higher than numeric_limits::epsilon
template <class Q>
Q to_half_cell(Q const &q) {
  return is_not_0(q.w())
             ? (q.w() > 0 ? q : Q(-q.w(), -q.x(), -q.y(), -q.z()))
             : (
                   // q
                   is_not_0(q.x())
                       ? (q.x() > 0 ? q : Q(-q.w(), -q.x(), -q.y(), -q.z()))
                       : (is_not_0(q.y())
                              ? (q.y() > 0 ? q
                                           : Q(-q.w(), -q.x(), -q.y(), -q.z()))
                              : ((q.z() > 0
                                      ? q
                                      : Q(-q.w(), -q.x(), -q.y(), -q.z())))));
}

template <class F>
static F const *get_raw_48cell_half() {
  static F const r = sqrt(2) / 2;
  static F const h = 0.5;
  static F const raw48[24 * 4] = {
      1,  0,  0,  0,  // 0
      0,  1,  0,  0,  // 1
      0,  0,  1,  0,  // 2
      0,  0,  0,  1,  // 3
      h,  h,  h,  h,  // 8
      -h, h,  h,  h,  // 10
      h,  -h, h,  h,  // 12
      -h, -h, h,  h,  // 14
      h,  h,  -h, h,  // 16
      -h, h,  -h, h,  // 18
      h,  -h, -h, h,  // 20
      -h, -h, -h, h,  // 22
      r,  r,  0,  0,  // 24
      r,  0,  r,  0,  // 25
      r,  0,  0,  r,  // 26
      0,  r,  r,  0,  // 27
      0,  r,  0,  r,  // 28
      0,  0,  r,  r,  // 29
      -r, r,  0,  0,  // 30
      -r, 0,  r,  0,  // 31
      -r, 0,  0,  r,  // 32
      0,  -r, r,  0,  // 33
      0,  -r, 0,  r,  // 34
      0,  0,  -r, r   // 35
  };
  return raw48;
}

template <class V4, class Index>
void get_cell_48cell_half(V4 const &quat, Index &cell) {
  typedef typename V4::Scalar F;
  V4 const quat_pos = quat.cwiseAbs();
  V4 tmpv = quat_pos;

  F hyperface_dist;      // dist to closest face
  Index hyperface_axis;  // closest hyperface-pair
  F edge_dist;           // dist to closest edge
  Index edge_axis_1;     // first axis of closest edge
  Index edge_axis_2;     // second axis of closest edge
  F corner_dist;         // dist to closest corner

  // std::cout << quat_pos.transpose() << std::endl;
  util::max2(quat_pos, hyperface_dist, edge_dist, hyperface_axis, edge_axis_2);
  edge_dist = sqrt(2) / 2 * (hyperface_dist + edge_dist);
  corner_dist = quat_pos.sum() / 2;
  // std::cout << hyperface_axis << " " << edge_axis_2 << std::endl;
  edge_axis_1 = hyperface_axis < edge_axis_2 ? hyperface_axis : edge_axis_2;
  edge_axis_2 = hyperface_axis < edge_axis_2 ? edge_axis_2 : hyperface_axis;
  assert(edge_axis_1 < edge_axis_2);

  // cell if closest if of form 1000 (add 4 if negative)
  Index facecell = hyperface_axis;  // | (quat[hyperface_axis]<0 ? 4 : 0);

  // cell if closest is of form 1111, bitwise by ( < 0)
  Index bit0 = quat[0] < 0;
  Index bit1 = quat[1] < 0;
  Index bit2 = quat[2] < 0;
  Index cornercell = quat[3] > 0 ? bit0 | bit1 << 1 | bit2 << 2
                                 : (!bit0) | (!bit1) << 1 | (!bit2) << 2;

  // cell if closest is of form 1100
  Index perm_shift[3][4] = {{9, 0, 1, 2}, {0, 9, 3, 4}, {1, 3, 9, 5}};
  Index sign_shift = (quat[edge_axis_1] < 0 != quat[edge_axis_2] < 0) * 1 * 6;
  Index edgecell = sign_shift + perm_shift[edge_axis_1][edge_axis_2];

  if (hyperface_dist > corner_dist) {
    if (hyperface_dist > edge_dist) {
      cell = facecell;
    } else {
      cell = edgecell + 12;
    }
  } else {
    if (corner_dist > edge_dist) {
      cell = cornercell + 4;
    } else {
      cell = edgecell + 12;
    }
  }
}

using namespace Eigen;
using std::cout;
using std::endl;

template <class F>
F cell_width() {
  return 2.0 * sqrt(2.0) - 2.0;
}

template <class F, class Index>
Eigen::Map<Eigen::Quaternion<F> const> hbt24_cellcen(Index const &i) {
  // F const * tmp = numeric::get_raw_48cell_half<F>() + 4*i;
  // std::cout << "   raw hbt24_cellcen " << tmp[0] << " " << tmp[1] << " " <<
  // tmp[2] << " " << tmp[3] << std::endl;
  return Eigen::Map<Eigen::Quaternion<F> const>(get_raw_48cell_half<F>() +
                                                4 * i);
}

template <class A>
void clamp01(A &a) {
  for (int i = 0; i < A::N; ++i) {
    a[i] = fmin(1.0, fmax(0.0, a[i]));
  }
}

}  // namespace util
}  // namespace sicdock