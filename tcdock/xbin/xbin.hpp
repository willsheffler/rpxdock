#pragma once

#include "tcdock/geom/bcc.hpp"
#include "tcdock/util/numeric.hpp"
#include "tcdock/util/types.hpp"
// #include "util/SimpleArray.hpp"
// #include "util/assert.hpp"
// #include "util/dilated_int.hpp"

// #include <boost/utility/binary.hpp>

namespace tcdock {
namespace xbin {

using namespace util;

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

template <class Float>
static Float const *get_raw_48cell_half() {
  static Float const r = sqrt(2) / 2;
  static Float const h = 0.5;
  static Float const raw48[24 * 4] = {
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
  typedef typename V4::Scalar Float;
  V4 const quat_pos = quat.cwiseAbs();
  V4 tmpv = quat_pos;

  Float hyperface_dist;  // dist to closest face
  Index hyperface_axis;  // closest hyperface-pair
  Float edge_dist;       // dist to closest edge
  Index edge_axis_1;     // first axis of closest edge
  Index edge_axis_2;     // second axis of closest edge
  Float corner_dist;     // dist to closest corner

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

  // pick case 1000 1111 1100 without if statements
  Index swtch;
  util::SimpleArray<3, Float>(hyperface_dist, corner_dist, edge_dist)
      .maxCoeff(&swtch);
  cell = swtch == 0 ? facecell : (swtch == 1 ? cornercell + 4 : edgecell + 12);
  // this is slower !?!
  // Float mx = std::max(std::max(hyperface_dist,corner_dist),edge_dist);
  // cell2[i] = hyperface_dist==mx ? facecell : (corner_dist==mx ? cornercell+8
  // : edgecell+24);
}

using namespace Eigen;
using std::cout;
using std::endl;

template <class Float>
Float cell_width() {
  return 2.0 * sqrt(2.0) - 2.0;
}

template <class Float, class Index>
Eigen::Map<Eigen::Quaternion<Float> const> hbt24_cellcen(Index const &i) {
  // Float const * tmp = numeric::get_raw_48cell_half<Float>() + 4*i;
  // std::cout << "   raw hbt24_cellcen " << tmp[0] << " " << tmp[1] << " " <<
  // tmp[2] << " " << tmp[3] << std::endl;
  return Eigen::Map<Eigen::Quaternion<Float> const>(
      get_raw_48cell_half<Float>() + 4 * i);
}

template <class A>
void clamp01(A &a) {
  for (int i = 0; i < A::N; ++i) {
    a[i] = fmin(1.0, fmax(0.0, a[i]));
  }
}

// TODO: add bounds check angles version!

template <class _Xform>
struct XformHash_bt24_BCC6 {
  using Xform = _Xform;
  typedef uint64_t Key;
  typedef typename Xform::Scalar Float;
  typedef typename Xform::Scalar Scalar;
  typedef tcdock::geom::BCC<6, Float, uint64_t> Grid;
  typedef tcdock::util::SimpleArray<3, Float> F3;
  typedef tcdock::util::SimpleArray<3, uint64_t> I3;
  typedef tcdock::util::SimpleArray<6, Float> F6;
  typedef tcdock::util::SimpleArray<6, uint64_t> I6;

  Float grid_size_ = -1;
  Float grid_spacing_ = -1;
  Grid grid6_;
  auto grid() const { return grid6_; }
  Float cart_resl_ = -1, ori_resl_ = -1, cart_bound_ = -1;
  Float cart_resl() const { return cart_resl_; }
  Float ori_resl() const { return ori_resl_; }
  Float cart_bound() const { return cart_bound_; }
  int ori_nside_ = -1;
  int ori_nside() const { return ori_nside_; }

  static std::string name() { return "XformHash_bt24_BCC6"; }

  XformHash_bt24_BCC6() {}
  XformHash_bt24_BCC6(Float cart_resl, Float ori_resl,
                      Float cart_bound = 512.0) {
    this->cart_bound_ = cart_bound;
    init(cart_resl, ori_resl, cart_bound);
  }
  XformHash_bt24_BCC6(Float cart_resl, int ori_nside, Float cart_bound) {
    // std::cout << "ori_nside c'tor" << std::endl;
    init2(cart_resl, ori_nside, cart_bound);
  }
  int get_ori_nside(float fudge = 1.45) {
    static float const covrad[64] = {
        49.66580, 25.99805, 17.48845, 13.15078, 10.48384, 8.76800, 7.48210,
        6.56491,  5.84498,  5.27430,  4.78793,  4.35932,  4.04326, 3.76735,
        3.51456,  3.29493,  3.09656,  2.92407,  2.75865,  2.62890, 2.51173,
        2.39665,  2.28840,  2.19235,  2.09949,  2.01564,  1.94154, 1.87351,
        1.80926,  1.75516,  1.69866,  1.64672,  1.59025,  1.54589, 1.50077,
        1.46216,  1.41758,  1.38146,  1.35363,  1.31630,  1.28212, 1.24864,
        1.21919,  1.20169,  1.17003,  1.14951,  1.11853,  1.09436, 1.07381,
        1.05223,  1.02896,  1.00747,  0.99457,  0.97719,  0.95703, 0.93588,
        0.92061,  0.90475,  0.89253,  0.87480,  0.86141,  0.84846, 0.83677,
        0.82164};
    int ori_nside = 1;
    while (covrad[ori_nside - 1] * fudge > ori_resl_ && ori_nside < 62)
      ++ori_nside;  // TODO: HACK multiplier!
    return ori_nside;
  }
  void init(Float cart_resl, Float ori_resl, Float cart_bound = 512.0) {
    this->cart_bound_ = cart_bound;
    this->ori_resl_ = ori_resl;
    init2(cart_resl, get_ori_nside(), cart_bound);
  }
  void init2(Float cart_resl, int ori_nside, Float cart_bound) {
    cart_resl_ = cart_resl / (sqrt(3.0) / 2.0);
    cart_bound_ = cart_bound;
    ori_nside_ = ori_nside;
    F6 lb, ub;
    I6 nside = get_bounds(cart_resl_, ori_nside_, cart_bound_, lb, ub);
    grid6_.init(nside, lb, ub);
  }
  I6 get_bounds(Float cart_resl, int ori_nside, float cart_bound, F6 &lb,
                F6 &ub) {
    I6 nside;
    if (2 * (int)(cart_bound / cart_resl) > 8192) {
      throw std::out_of_range("can have at most 8192 cart cells!");
    }
    nside[0] = nside[1] = nside[2] = 2.0 * cart_bound / cart_resl;
    nside[3] = nside[4] = nside[5] = ori_nside + 1;
    lb[0] = lb[1] = lb[2] = -cart_bound;
    ub[0] = ub[1] = ub[2] = cart_bound;
    lb[3] = lb[4] = lb[5] = -1.0 / ori_nside;
    ub[3] = ub[4] = ub[5] = 1.0;
    return nside;
  }
  F6 xform_to_F6(Xform x, Key &cell_index) const {
    Eigen::Matrix<Float, 3, 3> rotation = x.linear();
    Eigen::Quaternion<Float> q(rotation);
    // std::cout << q.coeffs().transpose() << std::endl;
    get_cell_48cell_half(q.coeffs(), cell_index);
    q = hbt24_cellcen<Float>(cell_index).inverse() * q;
    q = to_half_cell(q);
    F3 params(params[0] = q.x() / q.w() / cell_width<Float>() + 0.5,
              params[1] = q.y() / q.w() / cell_width<Float>() + 0.5,
              params[2] = q.z() / q.w() / cell_width<Float>() + 0.5);
    assert(cell_index < 24);
    clamp01(params);
    F6 params6;
    for (int i = 0; i < 3; ++i) {
      params6[i] = x.translation()[i];
      params6[i + 3] = params[i];
    }
    // std::cout << params6 << std::endl;
    return params6;
  }
  /* compiler totally gets rid of the copies on O1 or better
  F6 xform_to_F6_raw(Float *fp, Key &cell_index) const {
    // strange that this isn'd RowMajor
    Eigen::Map<Matrix<Float, 3, 3>, Unaligned, Stride<1, 4>> rotation(fp);
    Eigen::Quaternion<Float> q(rotation);
    // std::cout << q.coeffs().transpose() << std::endl;
    get_cell_48cell_half(q.coeffs(), cell_index);
    q = hbt24_cellcen<Float>(cell_index).inverse() * q;
    q = to_half_cell(q);
    F3 params(params[0] = q.x() / q.w() / cell_width<Float>() + 0.5,
              params[1] = q.y() / q.w() / cell_width<Float>() + 0.5,
              params[2] = q.z() / q.w() / cell_width<Float>() + 0.5);
    assert(cell_index < 24);
    clamp01(params);
    F6 params6;
    for (int i = 0; i < 3; ++i) {
      params6[i] = fp[4 * i + 3];
      params6[i + 3] = params[i];
    }
    // std::cout << params6 << std::endl;
    return params6;
  }
  Key get_key_raw(Float *fp) const {
    Key cell_index;
    F6 p6 = xform_to_F6_raw(fp, cell_index);
    assert((grid6_[p6] >> 59) == 0);
    return cell_index << 59 | grid6_[p6];
  }
  */
  Xform F6_to_xform(F6 params6, Key cell_index) const {
    F3 params = params6.template last<3>();
    Float const &w(cell_width<Float>());
    clamp01(params);
    params = w * (params - 0.5);  // now |params| < sqrt(2)-1
    Eigen::Quaternion<Float> q(1.0, params[0], params[1], params[2]);
    q.normalize();
    q = hbt24_cellcen<Float>(cell_index) * q;
    Xform center(q.matrix());
    for (int i = 0; i < 3; ++i) center.translation()[i] = params6[i];
    return center;
  }

  Key get_key(Xform x) const {
    Key cell_index;
    F6 p6 = xform_to_F6(x, cell_index);
    assert((grid6_[p6] >> 59) == 0);
    return cell_index << 59 | grid6_[p6];
  }
  Xform get_center(Key key) const {
    Key cell_index = key >> 59;
    F6 params6 = grid6_[key & (((Key)1 << 59) - (Key)1)];
    return F6_to_xform(params6, cell_index);
  }
  Key approx_size() const { return grid6_.size() * 24; }
  Key approx_nori() const {
    static int const nori[18] = {192,   648,   1521,  2855,   4990,   7917,
                                 11682, 16693, 23011, 30471,  39504,  50464,
                                 62849, 77169, 93903, 112604, 133352, 157103};
    return nori[grid6_.nside_[3] - 2];  // -1 for 0-index, -1 for ori_side+1
  }
};

}  // namespace xbin
}  // namespace tcdock
