#pragma once

#include "sicdock/geom/bcc.hpp"
#include "sicdock/util/numeric.hpp"
#include "sicdock/util/types.hpp"
// #include "util/SimpleArray.hpp"
// #include "util/assert.hpp"
// #include "util/dilated_int.hpp"

// #include <boost/utility/binary.hpp>

namespace sicdock {
namespace xbin {

using namespace util;

// TODO: add bounds check angles version!

template <class _Xform>
struct XformHash_bt24_BCC6 {
  using Xform = _Xform;
  typedef uint64_t Key;
  typedef typename Xform::Scalar F;
  typedef typename Xform::Scalar Scalar;
  typedef sicdock::geom::BCC<6, F, uint64_t> Grid;
  typedef sicdock::util::SimpleArray<3, F> F3;
  typedef sicdock::util::SimpleArray<3, uint64_t> I3;
  typedef sicdock::util::SimpleArray<6, F> F6;
  typedef sicdock::util::SimpleArray<6, uint64_t> I6;

  F grid_size_ = -1;
  F grid_spacing_ = -1;
  Grid grid6_;
  auto grid() const { return grid6_; }
  F cart_resl_ = -1, ori_resl_ = -1, cart_bound_ = -1;
  F cart_resl() const { return cart_resl_; }
  F ori_resl() const { return ori_resl_; }
  F cart_bound() const { return cart_bound_; }
  int ori_nside_ = -1;
  int ori_nside() const { return ori_nside_; }

  static std::string name() { return "XformHash_bt24_BCC6"; }

  XformHash_bt24_BCC6() {}
  template <typename Float>
  XformHash_bt24_BCC6(Float cart_resl, Float ori_resl,
                      Float cart_bound = 512.0) {
    this->cart_bound_ = cart_bound;
    init(cart_resl, ori_resl, cart_bound);
  }
  XformHash_bt24_BCC6(F cart_resl, int ori_nside, F cart_bound) {
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
  void init(F cart_resl, F ori_resl, F cart_bound = 512.0) {
    this->cart_bound_ = cart_bound;
    this->ori_resl_ = ori_resl;
    init2(cart_resl, get_ori_nside(), cart_bound);
  }
  void init2(F cart_resl, int ori_nside, F cart_bound) {
    cart_resl_ = cart_resl / (sqrt(3.0) / 2.0);
    cart_bound_ = cart_bound;
    ori_nside_ = ori_nside;
    F6 lb, ub;
    I6 nside = get_bounds(cart_resl_, ori_nside_, cart_bound_, lb, ub);
    grid6_.init(nside, lb, ub);
  }
  I6 get_bounds(F cart_resl, int ori_nside, float cart_bound, F6 &lb, F6 &ub) {
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
    Eigen::Matrix<F, 3, 3> rotation = x.linear();
    Eigen::Quaternion<F> q(rotation);
    // std::cout << q.coeffs().transpose() << std::endl;
    get_cell_48cell_half(q.coeffs(), cell_index);
    q = hbt24_cellcen<F>(cell_index).inverse() * q;
    q = to_half_cell(q);
    F3 params(params[0] = q.x() / q.w() / cell_width<F>() + 0.5,
              params[1] = q.y() / q.w() / cell_width<F>() + 0.5,
              params[2] = q.z() / q.w() / cell_width<F>() + 0.5);
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
  F6 xform_to_F6_raw(F *fp, Key &cell_index) const {
    // strange that this isn'd RowMajor
    Eigen::Map<Matrix<F, 3, 3>, Unaligned, Stride<1, 4>> rotation(fp);
    Eigen::Quaternion<F> q(rotation);
    // std::cout << q.coeffs().transpose() << std::endl;
    get_cell_48cell_half(q.coeffs(), cell_index);
    q = hbt24_cellcen<F>(cell_index).inverse() * q;
    q = to_half_cell(q);
    F3 params(params[0] = q.x() / q.w() / cell_width<F>() + 0.5,
              params[1] = q.y() / q.w() / cell_width<F>() + 0.5,
              params[2] = q.z() / q.w() / cell_width<F>() + 0.5);
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
  Key get_key_raw(F *fp) const {
    Key cell_index;
    F6 p6 = xform_to_F6_raw(fp, cell_index);
    assert((grid6_[p6] >> 55) == 0);
    return cell_index << 55 | grid6_[p6];
  }
  */
  Xform F6_to_xform(F6 params6, Key cell_index) const {
    F3 params = params6.template last<3>();
    F const &w(cell_width<F>());
    clamp01(params);
    params = w * (params - 0.5);  // now |params| < sqrt(2)-1
    Eigen::Quaternion<F> q(1.0, params[0], params[1], params[2]);
    q.normalize();
    q = hbt24_cellcen<F>(cell_index) * q;
    Xform center(q.matrix());
    for (int i = 0; i < 3; ++i) center.translation()[i] = params6[i];
    return center;
  }

  Key get_key(Xform x) const {
    Key cell_index;
    F6 p6 = xform_to_F6(x, cell_index);
#ifdef NDEBUG
#undef NDEBUG
    assert((grid6_[p6] >> 55) == 0);
#define NDEBUG
#else
    assert((grid6_[p6] >> 55) == 0);
#endif

    return cell_index << 55 | grid6_[p6];
  }
  Xform get_center(Key key) const {
    Key cell_index = key >> 55;
    F6 params6 = grid6_[key & (((Key)1 << 55) - (Key)1)];
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
}  // namespace sicdock
