#pragma once

#include "sicdock/util/dilated_int.hpp"
#include "sicdock/util/numeric.hpp"
#include "sicdock/util/types.hpp"

namespace sicdock {
namespace sampling {

using namespace util;

template <int CART_DIM, typename F, typename I>
struct CartHier {
  using Fn = Eigen::Matrix<F, CART_DIM, 1>;
  using In = Eigen::Matrix<I, CART_DIM, 1>;
  static I const ONE = 1;

  Fn cart_lb_, cart_ub_;
  In cart_bs_;
  Fn cart_cell_width_;
  In cart_bs_pref_prod_;
  I cart_ncell_;

  CartHier(Fn cartlb, Fn cartub, In cartbs) {
    for (int i = 0; i < CART_DIM; ++i) {
      this->cart_lb_[i] = cartlb[i];
      this->cart_ub_[i] = cartub[i];
      this->cart_bs_[i] = cartbs[i];
    }
    this->cart_ncell_ = this->cart_bs_.prod();
    this->cart_bs_pref_prod_[0] = 1;
    for (size_t i = 0; i < CART_DIM; ++i) {
      if (i > 0)
        this->cart_bs_pref_prod_[i] =
            this->cart_bs_[i - 1] * this->cart_bs_pref_prod_[i - 1];
      // std::cout << "cart_bs_pref_prod " << this->cart_bs_pref_prod_[i] <<
      // std::endl;
      this->cart_cell_width_[i] =
          (this->cart_ub_[i] - this->cart_lb_[i]) / (F)this->cart_bs_[i];
      assert(this->cart_ub_[i] > this->cart_lb_[i]);
    }
  }
  I size(I resl) const { return cart_ncell_ * ONE << (CART_DIM * resl); }

  bool get_value(I resl, I index, Fn& trans) const {
    assert(resl <= MAX_RESL_ONE_CELL);  // not rigerous check if Ncells > 1
    if (index >= size(resl)) return false;
    I cell_index = index >> (CART_DIM * resl);
    I hier_index = index & ((ONE << (CART_DIM * resl)) - 1);
    F scale = 1.0 / F(ONE << resl);
    Fn params;
    for (size_t i = 0; i < CART_DIM; ++i) {
      I undilated = util::undilate<CART_DIM>(hier_index >> i);
      params[i] = (static_cast<F>(undilated) + 0.5) * scale;
    }
    return this->trans_params_to_value(params, cell_index, resl, trans);
  }

  ///@brief sets value based on cell_index and parameters using geometric bounds
  ///@return false iff invalid parameters
  bool trans_params_to_value(Fn const& params, I cell_index, I resl,
                             Fn& value) const {
    for (size_t i = 0; i < CART_DIM; ++i) {
      assert(this->cart_bs_[i] > 0);
      assert(this->cart_bs_[i] < 100000);
      assert(this->cart_lb_[i] < this->cart_ub_[i]);
      F bi = (cell_index / this->cart_bs_pref_prod_[i]) % this->cart_bs_[i];
      value[i] =
          this->cart_lb_[i] + this->cart_cell_width_[i] * (bi + params[i]);
    }
    return true;
  }
};

template <typename F = double, typename I = uint64_t>
struct OriHier {
  static I const ORI_DIM = 3;
  static I const MAX_RESL_ONE_CELL = sizeof(I) * 8 / ORI_DIM;
  static I const ONE = 1;
  using F3 = V3<F>;
  using I6 = V6<I>;
  using I3 = V3<I>;
  I onside_;
  F recip_nside_;
  I ori_ncell_;
  OriHier(F ori_resl) {
    onside_ = ori_get_nside_for_rot_resl_deg(ori_resl);
    recip_nside_ = 1.0 / (F)onside_;
    ori_ncell_ = 24 * onside_ * onside_ * onside_;
  }
  OriHier(int nside) {
    onside_ = nside;
    recip_nside_ = 1.0 / (F)onside_;
    ori_ncell_ = 24 * onside_ * onside_ * onside_;
  }
  I size(I resl) const { return ori_ncell_ * ONE << (ORI_DIM * resl); }
  I ori_nside() const { return onside_; }

  bool get_value(I resl, I index, M3<F>& ori) const {
    assert(resl <= MAX_RESL_ONE_CELL);  // not rigerous check if Ncells > 1
    if (index >= size(resl)) return false;
    I cell_index = index >> (ORI_DIM * resl);
    I hier_index = index & ((ONE << (ORI_DIM * resl)) - 1);
    F scale = 1.0 / F(ONE << resl);
    F3 params;
    for (size_t i = 0; i < ORI_DIM; ++i) {
      I undilated = util::undilate<ORI_DIM>(hier_index >> i);
      params[i] = (static_cast<F>(undilated) + 0.5) * scale;
    }
    bool valid = this->ori_params_to_value(params, cell_index, resl, ori);
    return valid;
  }

  ///@brief sets value to parameters without change
  ///@return false iff invalid parameters
  bool ori_params_to_value(F3 params, I cell_index, I resl,
                           M3<F>& value) const {
    F3 p = params * recip_nside_;

    I h48_cell_index = cell_index / (onside_ * onside_ * onside_);
    cell_index = cell_index % (onside_ * onside_ * onside_);
    p[0] += recip_nside_ * (F)(cell_index % onside_);
    p[1] += recip_nside_ * (F)(cell_index / onside_ % onside_);
    p[2] += recip_nside_ * (F)(cell_index / (onside_ * onside_) % onside_);

    // if( !( p[0] >= 0.0 && p[0] <= 1.0 ) ) cout << "BAD param val: " << p[0]
    // << endl; if( !( p[1] >= 0.0 && p[1] <= 1.0 ) ) cout << "BAD param val: "
    // << p[1] << endl; if( !( p[2] >= 0.0 && p[2] <= 1.0 ) ) cout << "BAD param
    // val: " << p[2] << endl;

    assert(p[0] >= -0.00001 && p[0] <= 1.00001);
    assert(p[1] >= -0.00001 && p[1] <= 1.00001);
    assert(p[2] >= -0.00001 && p[2] <= 1.00001);
    p[0] = fmax(0.0, p[0]);
    p[1] = fmax(0.0, p[1]);
    p[2] = fmax(0.0, p[2]);
    p[0] = fmin(1.0, p[0]);
    p[1] = fmin(1.0, p[1]);
    p[2] = fmin(1.0, p[2]);

    // std::cout << cell_index << " " << p << " " << p << std::endl;
    // static int count = 0; if( ++count > 30 ) std::exit(-1);

    F w = 2 * (sqrt(2) - 1);
    p = w * (p - F3(0.5, 0.5, 0.5));  // now |p| < sqrt(2)-1
    F corner_dist = fabs(p[0]) + fabs(p[1]) + fabs(p[2]);
    F delta = sqrt(3.0) / 2.0 / w / (F)(1 << resl);

    // TODO make this check more rigerous???
    if (corner_dist - delta > 1.0) return false;

    Eigen::Quaternion<F> q(1.0, p[0], p[1], p[2]);
    // Eigen::Quaternion<F> q(sqrt(1.0 - p.squaredNorm()), p[0], p[1], p[2]);
    // assert(fabs(q.squaredNorm() - 1.0) < 0.000001);

    q.normalize();
    q = hbt24_cellcen<F>(h48_cell_index) * q;

    value = q.matrix();

    return true;
  }

  static F const* ori_get_covrad_data() {
    static F const covrad[15] = {
        92.609,  //  1
        66.065,  //  2
        47.017,  //  3
        37.702,  //  4
        30.643,  //  5
        26.018,  //  6
        22.466,  //  7
        19.543,  //  8
        17.607,  //  9
        15.928,  //  10
        14.282,  //  11
        13.149,  //  12
        12.238,  //  13
        11.405,  //  14
        10.589,  //  15
    };
    return covrad;
  }

  static I ori_get_nside_for_rot_resl_deg(F rot_resl_deg) {
    static F const* covrad = ori_get_covrad_data();
    I nside = 0;
    while (covrad[nside] > rot_resl_deg && nside < 23) {
      // std::cout << nside << " " << covrad[nside] << std::endl;
      ++nside;
    }
    return nside + 1;
  }
};  // namespace sampling

template <typename F = double, typename I = uint64_t>
struct XformHier : public OriHier<F, I>, public CartHier<3, F, I> {
  static I const FULL_DIM = 6;
  static I const ORI_DIM = 3;
  static I const CART_DIM = 3;
  static I const MAX_RESL_ONE_CELL = sizeof(I) * 8 / FULL_DIM;
  static I const ONE = 1;

  using F6 = V6<F>;
  using F3 = V3<F>;
  using I6 = V6<I>;
  using I3 = V3<I>;
  using X = X3<F>;

  I ncell_;

  XformHier(F3 cartlb, F3 cartub, I3 cartbs, F ori_resl)
      : OriHier<F, I>(ori_resl),
        CartHier<CART_DIM, F, I>(cartlb, cartub, cartbs) {
    ncell_ = this->cart_ncell_ * this->ori_ncell_;
    // std::cout << "cart_ncell " << this->cart_ncell_ << std::endl;
  }

  I size(I resl) const { return ncell_ * ONE << (FULL_DIM * resl); }

  bool get_value(I resl, I index, X& xform) const {
    assert(resl <= MAX_RESL_ONE_CELL);  // not rigerous check if Ncells > 1
    if (index >= size(resl)) return false;
    I cell_index = index >> (FULL_DIM * resl);
    I hier_index = index & ((ONE << (FULL_DIM * resl)) - 1);
    F scale = 1.0 / F(ONE << resl);
    F6 params;
    for (size_t i = 0; i < FULL_DIM; ++i) {
      I undilated = util::undilate<FULL_DIM>(hier_index >> i);
      params[i] = (static_cast<F>(undilated) + 0.5) * scale;
    }
    return this->params_to_value(params, cell_index, resl, xform);
  }

  bool params_to_value(F6 params, I cell_index, I resl, X& value) const {
    I cori = cell_index % this->ori_ncell_;
    I ctrans = cell_index / this->ori_ncell_;
    F3 pori, ptrans;
    for (size_t i = 0; i < 3; ++i) {
      pori[i] = params[i];
      ptrans[i] = params[i + 3];
    }
    M3<F> m;
    F3 v;
    bool valid = this->ori_params_to_value(pori, cori, resl, m);
    valid &= this->trans_params_to_value(ptrans, ctrans, resl, v);
    if (!valid) return false;
    value = X(m);
    value.translation()[0] = v[0];
    value.translation()[1] = v[1];
    value.translation()[2] = v[2];
    return true;
  }
};

}  // namespace sampling
}  // namespace sicdock