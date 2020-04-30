#pragma once
/** \file */

#include <iostream>

#include "rpxdock/util/dilated_int.hpp"
#include "rpxdock/util/numeric.hpp"
#include "rpxdock/util/types.hpp"

namespace rpxdock {
/**
\namespace rpxdock::sampling
\brief namespace for hierarchical and other sampling grids
*/

namespace sampling {

using namespace util;

template <int CART_DIM, typename F, typename I>
struct CartHier {
  using Fn = Eigen::Matrix<F, CART_DIM, 1>;
  using In = Eigen::Matrix<I, CART_DIM, 1>;
  static I const FULL_DIM = CART_DIM;
  int dim() const { return FULL_DIM; }
  static I const ONE = 1;
  static I const NEXPAND = (ONE << FULL_DIM);

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
    sanity_check();
  }
  bool sanity_check() const {
    for (int i = 0; i < CART_DIM; ++i) {
      for (int j = 1 + 1; j < CART_DIM; ++j) {
        float r = fabs(cart_cell_width_[i] - cart_cell_width_[j]) /
                  fabs(cart_cell_width_[i] + cart_cell_width_[j]);
        if (r > 0.1) return false;
      }
    }
    return true;
  }

  I size(I resl) const { return cart_ncell_ * ONE << (CART_DIM * resl); }

  auto get_params(I resl, I index) const {
    assert(resl <= MAX_RESL_ONE_CELL);  // not rigerous check if Ncells > 1
    I cell_index = index >> (CART_DIM * resl);
    I hier_index = index & ((ONE << (CART_DIM * resl)) - 1);
    F scale = 1.0 / F(ONE << resl);
    Fn params;
    for (size_t i = 0; i < CART_DIM; ++i) {
      I undilated = util::undilate<CART_DIM>(hier_index >> i);
      params[i] = (static_cast<F>(undilated) + 0.5) * scale;
    }
    return std::make_pair(cell_index, params);
  }

  bool get_value(I resl, I index, Fn& trans) const {
    if (index >= size(resl)) return false;
    auto [cell_index, params] = this->get_params(resl, index);
    return this->trans_params_to_value(params, cell_index, resl, trans);
  }

  bool get_value(I resl, I index, X3<F>& x) const {
    if (index >= size(resl)) return false;
    auto [cell_index, params] = this->get_params(resl, index);
    Fn trans;
    bool success = this->trans_params_to_value(params, cell_index, resl, trans);
    if (!success) return false;
    x = X3<F>::Identity();
    for (int i = 0; i < std::min<int>(3, FULL_DIM); ++i)
      x.translation()[i] = trans[i];
    return true;
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
struct RotHier {
  static I const ORI_DIM = 1;
  static I const FULL_DIM = ORI_DIM;
  int dim() const { return FULL_DIM; }
  static I const MAX_RESL_ONE_CELL = sizeof(I) * 8 / ORI_DIM;
  static I const ONE = 1;
  static I const NEXPAND = (ONE << FULL_DIM);
  using A = V3<F>;

  F rot_lb_ = 0.0, rot_ub_ = 360.0 * torad, rot_cell_width_ = 15.0 * torad;
  I rot_ncell_ = 1;
  A axis_ = A(0, 0, 1);

  RotHier(F lb, F ub, F resl, A axis = A(0, 0, 1))
      : rot_lb_(lb * torad),
        rot_ub_(ub * torad),
        rot_cell_width_(resl * torad),
        axis_(axis) {
    rot_ncell_ = std::ceil((rot_ub_ - rot_lb_) / rot_cell_width_);
    rot_cell_width_ = (rot_ub_ - rot_lb_) / rot_ncell_;
    axis_.normalize();
  }
  RotHier(F lb, F ub, I rot_ncell_, A axis = A(0, 0, 1))
      : rot_lb_(lb * torad),
        rot_ub_(ub * torad),
        rot_ncell_(rot_ncell_),
        axis_(axis) {
    rot_cell_width_ = (rot_ub_ - rot_lb_) / rot_ncell_;
    axis_.normalize();
  }
  bool get_value(I resl, I index, M3<F>& ori) const {
    assert(resl <= MAX_RESL_ONE_CELL);  // not rigerous check if Ncells > 1
    if (index >= size(resl)) return false;
    I cell_index = index >> (ORI_DIM * resl);
    I hier_index = index & ((ONE << (ORI_DIM * resl)) - 1);
    F scale = 1.0 / F(ONE << resl);
    I undilated = util::undilate<ORI_DIM>(hier_index);
    F param = (static_cast<F>(undilated) + 0.5) * scale;
    bool valid = this->ori_params_to_value(param, cell_index, resl, ori);
    return valid;
  }
  bool ori_params_to_value(F param, I cell_index, I resl, M3<F>& value) const {
    F ang = rot_lb_ + (cell_index + param) * rot_cell_width_;
    auto aa = Eigen::AngleAxis(ang, axis_);
    value = aa.toRotationMatrix();
    // auto aa2 = Eigen::AngleAxis<F>(value);
    // std::cout << cell_index << " " << param << " " << ang << " "
    // << ang * 180 / M_PI << " " << aa2.angle() << std::endl;
    return true;
  }
  I size(I resl) const { return rot_ncell_ * ONE << (ORI_DIM * resl); }
  I rot_nside() const { return ONE; }
};

////////////////////////// SphereHier  todo ////////////////////////////

template <typename F = double, typename I = uint64_t>
struct OriHier {
  static I const ORI_DIM = 3;
  static I const FULL_DIM = ORI_DIM;
  int dim() const { return FULL_DIM; }
  static I const MAX_RESL_ONE_CELL = sizeof(I) * 8 / ORI_DIM;
  static I const ONE = 1;
  static I const NEXPAND = (ONE << FULL_DIM);

  using F3 = V3<F>;
  using I6 = V6<I>;
  using I3 = V3<I>;
  I onside_;
  F recip_nside_;
  I ori_ncell_;
  F ori_resl_;
  OriHier(F ori_resl) {
    onside_ = ori_get_nside_for_rot_resl_deg(ori_resl);
    recip_nside_ = 1.0 / (F)onside_;
    ori_ncell_ = 24 * onside_ * onside_ * onside_;
    ori_resl_ = ori_get_covrad_data()[onside_ - 1];
  }
  OriHier(I nside) {
    onside_ = nside;
    recip_nside_ = 1.0 / (F)onside_;
    ori_ncell_ = 24 * onside_ * onside_ * onside_;
    ori_resl_ = ori_get_covrad_data()[onside_ - 1];
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
    // << endl; if( !( p[1] >= 0.0 && p[1] <= 1.0 ) ) cout << "BAD param val:
    // "
    // << p[1] << endl; if( !( p[2] >= 0.0 && p[2] <= 1.0 ) ) cout << "BAD
    // param val: " << p[2] << endl;

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
    while (covrad[nside] > rot_resl_deg - 0.1 && nside < 23) {
      // std::cout << nside << " " << covrad[nside] << std::endl;
      ++nside;
    }
    return nside + 1;
  }
};  // namespace sampling

template <typename F = double, typename I = uint64_t>
struct XformHier : public OriHier<F, I>, public CartHier<3, F, I> {
  static I const FULL_DIM = 6;
  int dim() const { return FULL_DIM; }
  static I const ORI_DIM = 3;
  static I const CART_DIM = 3;
  static I const MAX_RESL_ONE_CELL = sizeof(I) * 8 / FULL_DIM;
  static I const ONE = 1;
  static I const NEXPAND = (ONE << FULL_DIM);

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
  }
  XformHier(F3 cartlb, F3 cartub, I3 cartbs, I ori_nside)
      : OriHier<F, I>(ori_nside),
        CartHier<CART_DIM, F, I>(cartlb, cartub, cartbs) {
    ncell_ = this->cart_ncell_ * this->ori_ncell_;
  }

  bool sanity_check() const {
    bool pass = true;
    pass &= CartHier<CART_DIM, F, I>::sanity_check();
    return pass;
  }

  I size(I resl) const { return ncell_ * ONE << (FULL_DIM * resl); }

  I cell_index_of(I resl, I index) const { return index >> (FULL_DIM * resl); }
  I hier_index_of(I resl, I index) const {
    return index & ((ONE << (FULL_DIM * resl)) - 1);
  }
  I parent_of(I index) const { return index >> FULL_DIM; }
  I child_of_begin(I index) const { return index << FULL_DIM; }
  I child_of_end(I index) const { return (index + 1) << FULL_DIM; }

  bool get_value(I resl, I index, X& xform) const {
    assert(resl <= MAX_RESL_ONE_CELL);  // not rigerous check if Ncells > 1
    if (index >= size(resl)) return false;
    I cell_index = cell_index_of(resl, index);
    I hier_index = hier_index_of(resl, index);
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

template <typename F = double, typename I = uint64_t>
struct OriCart1Hier : public OriHier<F, I>, public CartHier<1, F, I> {
  static I const FULL_DIM = 4;
  int dim() const { return FULL_DIM; }
  static I const ORI_DIM = 3;
  static I const CART_DIM = 1;
  static I const MAX_RESL_ONE_CELL = sizeof(I) * 8 / FULL_DIM;
  static I const ONE = 1;
  static I const NEXPAND = (ONE << FULL_DIM);

  using Params = V4<F>;
  using F3 = V3<F>;
  using F1 = V1<F>;
  using I4 = V4<I>;
  using I3 = V3<I>;
  using I1 = V1<I>;
  using X = X3<F>;

  I ncell_;

  OriCart1Hier(F1 cartlb, F1 cartub, I1 cartbs, F ori_resl)
      : OriHier<F, I>(ori_resl),
        CartHier<CART_DIM, F, I>(cartlb, cartub, cartbs) {
    ncell_ = this->cart_ncell_ * this->ori_ncell_;
  }
  OriCart1Hier(F1 cartlb, F1 cartub, I1 cartbs, I ori_nside)
      : OriHier<F, I>(ori_nside),
        CartHier<CART_DIM, F, I>(cartlb, cartub, cartbs) {
    ncell_ = this->cart_ncell_ * this->ori_ncell_;
  }

  bool sanity_check() const {
    bool pass = true;
    pass &= CartHier<CART_DIM, F, I>::sanity_check();
    return pass;
  }

  I size(I resl) const { return ncell_ * ONE << (FULL_DIM * resl); }

  I cell_index_of(I resl, I index) const { return index >> (FULL_DIM * resl); }
  I hier_index_of(I resl, I index) const {
    return index & ((ONE << (FULL_DIM * resl)) - 1);
  }
  I parent_of(I index) const { return index >> FULL_DIM; }
  I child_of_begin(I index) const { return index << FULL_DIM; }
  I child_of_end(I index) const { return (index + 1) << FULL_DIM; }

  bool get_value(I resl, I index, X& xform) const {
    assert(resl <= MAX_RESL_ONE_CELL);  // not rigerous check if Ncells > 1
    if (index >= size(resl)) return false;
    I cell_index = cell_index_of(resl, index);
    I hier_index = hier_index_of(resl, index);
    F scale = 1.0 / F(ONE << resl);
    Params params;
    for (size_t i = 0; i < FULL_DIM; ++i) {
      I undilated = util::undilate<FULL_DIM>(hier_index >> i);
      params[i] = (static_cast<F>(undilated) + 0.5) * scale;
    }
    return this->params_to_value(params, cell_index, resl, xform);
  }

  bool params_to_value(Params params, I cell_index, I resl, X& value) const {
    I cori = cell_index % this->ori_ncell_;
    I ctrans = cell_index / this->ori_ncell_;
    F3 pori;
    for (size_t i = 0; i < 3; ++i) pori[i] = params[i];
    F1 ptrans;
    for (size_t i = 3; i < FULL_DIM; ++i) ptrans[i - 3] = params[i];
    M3<F> m;
    F1 v;
    bool valid = this->ori_params_to_value(pori, cori, resl, m);
    valid &= this->trans_params_to_value(ptrans, ctrans, resl, v);
    if (!valid) return false;
    value = X(m);
    value.translation()[0] = v[0];
    value.translation()[1] = 0;
    value.translation()[2] = 0;
    return true;
  }
};

template <typename F = double, typename I = uint64_t>
struct RotCart1Hier : public RotHier<F, I>, public CartHier<1, F, I> {
  static I const FULL_DIM = 2;
  int dim() const { return FULL_DIM; }
  static I const ORI_DIM = 1;
  static I const CART_DIM = 1;
  static I const MAX_RESL_ONE_CELL = sizeof(I) * 8 / FULL_DIM;
  static I const ONE = 1;
  static I const NEXPAND = (ONE << FULL_DIM);

  using F2 = V2<F>;
  using F1 = V1<F>;
  using I1 = V1<I>;
  using X = X3<F>;

  I ncell_;

  RotCart1Hier(F cartlb, F cartub, I cartnc, F rotlb, F rotub, I rotnc,
               V3<F> axis = V3<F>(0, 0, 1))
      : RotHier<F, I>(rotlb, rotub, rotnc, axis),
        CartHier<CART_DIM, F, I>(F1(cartlb), F1(cartub), I1(cartnc)) {
    ncell_ = this->cart_ncell_ * this->rot_ncell_;
  }
  bool sanity_check() const {
    bool pass = true;
    pass &= CartHier<CART_DIM, F, I>::sanity_check();
    return pass;
  }

  I size(I resl) const { return ncell_ * ONE << (FULL_DIM * resl); }
  I cell_index_of(I resl, I index) const { return index >> (FULL_DIM * resl); }
  I hier_index_of(I resl, I index) const {
    return index & ((ONE << (FULL_DIM * resl)) - 1);
  }
  I parent_of(I index) const { return index >> FULL_DIM; }
  I child_of_begin(I index) const { return index << FULL_DIM; }
  I child_of_end(I index) const { return (index + 1) << FULL_DIM; }

  bool get_value(I resl, I index, X& xform) const {
    assert(resl <= MAX_RESL_ONE_CELL);  // not rigerous check if Ncells > 1
    if (index >= size(resl)) return false;
    I cell_index = cell_index_of(resl, index);
    I hier_index = hier_index_of(resl, index);
    F scale = 1.0 / F(ONE << resl);
    F2 params;
    for (size_t i = 0; i < FULL_DIM; ++i) {
      I undilated = util::undilate<FULL_DIM>(hier_index >> i);
      params[i] = (static_cast<F>(undilated) + 0.5) * scale;
    }
    return this->params_to_value(params, cell_index, resl, xform);
  }
  bool params_to_value(F2 params, I cell_index, I resl, X& value) const {
    I cori = cell_index % this->rot_ncell_;
    I ctrans = cell_index / this->rot_ncell_;
    M3<F> m;
    F1 v;
    bool valid = this->ori_params_to_value(params[0], cori, resl, m);
    valid &= this->trans_params_to_value(F1(params[1]), ctrans, resl, v);
    if (!valid) return false;
    value = X(m);
    for (int i = 0; i < 3; ++i) value.translation()[i] = v[0] * this->axis_[i];
    return true;
  }
};

}  // namespace sampling
}  // namespace rpxdock