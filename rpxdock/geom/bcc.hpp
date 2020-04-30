#pragma once
/** \file */

#include "rpxdock/util/numeric.hpp"
#include "rpxdock/util/str.hpp"
// #include "util/template_loop.hpp"

#include <iostream>

namespace rpxdock {
/**
\namespace rpxdock::geom
\brief namespace for geometry utils like spheres and bcc grids
*/
namespace geom {

using util::mod;
using util::sqr;

// TODO: add bounds checking .at methods!

/**
 * @brief      n-dimensional BCC lattice
 *
 * @tparam     DIM     { description }
 * @tparam     _Float  { description }
 * @detail     will hit lower bound but not upper... for example
 *      if range is 0..1 and 2 cells, x values could be 0, 0.25, 0.5 and 0.75
 * ((not 1.0!!))
 */
template <int _DIM, typename _Float = double, typename _Index = uint64_t>
struct BCC {
  static const int DIM = _DIM;
  using F = _Float;
  using I = _Index;
  using In = Eigen::Array<I, DIM, 1>;
  using Fn = Eigen::Array<F, DIM, 1>;

  static_assert((DIM > 2));

  In nside_, nside_prefsum_;
  Fn lower_, width_, lower_cen_, half_width_;
  In nside() const { return nside_; }
  Fn lower() const { return lower_; }
  Fn width() const { return width_; }
  Fn upper() const { return lower_ + width_ * (nside().template cast<F>()); }

  bool operator!=(BCC<DIM, F, I> o) const { return !(*this == o); }
  bool operator==(BCC<DIM, F, I> o) const {
    return nside_ == o.nside_ && lower_ == o.lower_ && width_ == o.width_;
  }

  BCC() {}

  BCC(In sizes, Fn lower = Fn(0), Fn upper = Fn(1)) {
    init(sizes, lower, upper);
  }

  void init(In sizes, Fn lower, Fn upper) {
    nside_ = sizes;
    lower_ = lower;
    nside_prefsum_[0] = 1;
    for (size_t i = 1; i < DIM; ++i)
      nside_prefsum_[i] = nside_prefsum_[i - 1] * nside_[i - 1];
    width_ = (upper - lower_) / nside_.template cast<F>();
    half_width_ = width_ / 2.0;
    lower_cen_ = lower_ + half_width_;
    __int128 totsize = 2;
    for (size_t i = 0; i < DIM; ++i) {
      if (sizes[i] > 18446744073709551)
        throw std::invalid_argument(
            "I Type is too narrow, are you passing negative vals?");
      totsize *= __int128(nside_[i]);
      if (totsize > __int128(std::numeric_limits<I>::max())) {
        for (int i = 0; i < DIM; ++i)
          std::cout << i << " " << nside_[i] << std::endl;
        throw std::invalid_argument("I Type is too narrow");
      }
    }
    // std::cout << lower_ << std::endl;
    // std::cout << lower_ + nside_.template cast<F>() * width_ <<
    // std::endl;
  }

  I size() const noexcept { return nside_.prod() * 2; }
  int dim() const noexcept { return DIM; }

  Fn get_value(I index) const noexcept {
    bool odd = index & 1;
    // std::cout << "bcc get_floats " << index << std::endl;
    In indices;
    for (int i = 0; i < DIM; ++i)
      indices[i] = ((index >> 1) / nside_prefsum_[i]) % nside_[i];
    // std::cout << "bcc get_floats " << indices << " " << odd << std::endl;
    return this->get_center(indices, odd);
  }
  Fn operator[](I index) const noexcept { return get_value(index); }

  Fn get_center(In indices, bool odd) const noexcept {
    return lower_cen_ + width_ * indices.template cast<F>() +
           (odd ? half_width_ : Fn(0));
  }

  In get_indices(Fn value, bool &odd) const noexcept {
    value = (value - lower_) / width_;
    // std::cout << "bcc::get_indices lower_ " << lower_ << std::endl;
    // std::cout << "bcc::get_indices width_ " << width_ << std::endl;
    // std::cout << "bcc::get_indices " << value << std::endl;
    In const indices = value.template cast<I>();
    // std::cout << "bcc::get_indices " << indices << std::endl;
    value = value - indices.template cast<F>() - 0.5;
    // std::cout << "bcc::get_indices " << value << std::endl;
    In const corner_indices = indices - (value < 0).template cast<I>();
    // std::cout << "bcc::get_indices " << corner_indices << std::endl;
    odd = (0.25 * DIM) < fabs((value.sign() * value).sum());
    return odd ? corner_indices : indices;
  }

  template <typename Ary>
  I get_index(Ary value) const noexcept {
    bool odd;
    In indices = get_indices(value, odd);
    // std::cout << "bcc get_idx " << indices << " " << odd << std::endl;
    I index = (nside_prefsum_ * indices).sum();
    index = (index << 1) + odd;
    // std::cout << "bcc get idx " << index << std::endl;
    return index;
  }
  template <typename Ary>
  I operator[](Ary value) const noexcept {
    return get_index(value);
  }

  int neighbor_sphere_radius_square_cut(int rad, bool exhalf) const noexcept {
    return (sqr(2 * rad + exhalf) + sqr(2 * (rad + 1) + exhalf)) / 2;
  }
  int neighbor_radius_square_cut(int rad, bool exhalf) const noexcept {
    return 3 * sqr(2 * rad + exhalf) + 1;
  }

  // template <typename Iiter>
  // std::enable_if_t<DIM != 6> neighbors_6_3(I index, Iiter &iter, int rad,
  //                                          bool exhalf, bool oddlast3,
  //                                          bool sphere) const noexcept {}

  template <typename Iiter, typename = std::enable_if_t<DIM == 6>>
  void neighbors_6_3(I index, Iiter &iter, int rad, bool exhalf, bool oddlast3,
                     bool sphere) const noexcept {
    bool odd = index & 1;
    In idx0 = mod((In)((index >> 1) / nside_prefsum_), nside_);
    In idx;
    int lb = -rad - (exhalf && !odd);
    int ub = +rad + (exhalf && odd);
    int rcut = neighbor_sphere_radius_square_cut(rad, exhalf);
    // if (sphere) std::cout << "SPHCUT " << rad << " " << rcut <<
    // std::endl;
    int eh = odd ? -1 : 0;
    int oh = odd ? 0 : 1;
    bool oddex = (odd != exhalf);
    int l3shift = oddlast3 ? -(!odd) : 0;
    int last3ub = oddlast3 ? 1 : 0;

    for (int i5 = 0; i5 <= last3ub; ++i5) {
      I key5 = nside_prefsum_[5] * (idx0[5] + i5 + l3shift);
      for (int i4 = 0; i4 <= last3ub; ++i4) {
        I key4 = key5 + nside_prefsum_[4] * (idx0[4] + i4 + l3shift);
        for (int i3 = 0; i3 <= last3ub; ++i3) {
          I key3 = key4 + nside_prefsum_[3] * (idx0[3] + i3 + l3shift);

          for (int i2 = lb; i2 <= ub; ++i2) {
            I key2 = key3 + nside_prefsum_[2] * (idx0[2] + i2);
            bool edge2 = oddex ? i2 == lb : i2 == ub;
            for (int i1 = lb; i1 <= ub; ++i1) {
              I key1 = key2 + nside_prefsum_[1] * (idx0[1] + i1);
              bool edge1 = edge2 || ((oddex ? i1 == lb : i1 == ub));
              for (int i0 = lb; i0 <= ub; ++i0) {
                I key0 = key1 + nside_prefsum_[0] * (idx0[0] + i0);
                I key = key0 << 1;
                bool edge = edge1 || (oddex ? i0 == lb : i0 == ub);

                bool inoddlast3 = i5 + l3shift || i4 + l3shift || i3 + l3shift;
                bool skip0 = inoddlast3 && !odd;
                bool skip1 = inoddlast3 && odd;

                if (!skip0) {
                  int erad = sqr(2 * i2 + eh) + sqr(2 * i1 + eh) +
                             sqr(2 * i0 + eh) /*+ sqr(i5 + l3shift) +
                             sqr(i4 + l3shift) + sqr(i3 + l3shift)*/
                      ;
                  if ((!sphere || erad < rcut) && (!oddex || !edge))
                    *iter++ = std::make_pair(key | 0, erad);
                }
                if (!skip1) {
                  int orad = sqr(2 * i2 + oh) + sqr(2 * i1 + oh) +
                             sqr(2 * i0 + oh) /* + sqr(i5 + l3shift) +
                              sqr(i4 + l3shift) + sqr(i3 + l3shift)*/
                      ;
                  if ((!sphere || orad < rcut) && (oddex || !edge))
                    *iter++ = std::make_pair(key | 1, orad);
                }
              }
            }
          }
        }
      }
    }
  }
  // template <typename Iiter>
  // std::enable_if_t<DIM != 3> neighbors_3(I index, Iiter &iter, int const rad,
  //                                        bool const exhalf,
  //                                        bool const sphere) const noexcept {}
  // template <typename Iiter, typename = std::enable_if_t<DIM != 3>>
  // void neighbors_3(I index, Iiter &iter, int const rad, bool const exhalf,
  //                  bool const sphere) const noexcept {}

  template <typename Iiter, typename = std::enable_if_t<DIM == 3>>
  void neighbors_3(I index, Iiter &iter, int const rad, bool const exhalf,
                   bool const sphere) const noexcept {
    bool odd = index & 1;
    In idx0 = mod((In)((index >> 1) / nside_prefsum_), nside_);
    In idx;
    int lb = -rad - (exhalf && !odd);
    int ub = +rad + (exhalf && odd);
    int rcut = neighbor_sphere_radius_square_cut(rad, exhalf);
    // if (sphere) std::cout << "SPHCUT " << rad << " " << rcut <<
    // std::endl;
    int eh = odd ? -1 : 0;
    int oh = odd ? 0 : 1;
    bool oddex = (odd != exhalf);
    for (int i2 = lb; i2 <= ub; ++i2) {
      I key2 = nside_prefsum_[2] * (idx0[2] + i2);
      bool edge2 = oddex ? i2 == lb : i2 == ub;
      for (int i1 = lb; i1 <= ub; ++i1) {
        I key1 = key2 + nside_prefsum_[1] * (idx0[1] + i1);
        bool edge1 = edge2 || ((oddex ? i1 == lb : i1 == ub));
        for (int i0 = lb; i0 <= ub; ++i0) {
          I key0 = key1 + nside_prefsum_[0] * (idx0[0] + i0);
          I key = key0 << 1;
          bool edge = edge1 || (oddex ? i0 == lb : i0 == ub);
          int erad = sqr(2 * i2 + eh) + sqr(2 * i1 + eh) + sqr(2 * i0 + eh);
          int orad = sqr(2 * i2 + oh) + sqr(2 * i1 + oh) + sqr(2 * i0 + oh);
          if ((!sphere || erad < rcut) && (!oddex || !edge))
            *iter++ = std::make_pair(key | 0, erad);
          if ((!sphere || orad < rcut) && (oddex || !edge))
            *iter++ = std::make_pair(key | 1, orad);
        }
      }
    }
  }

  template <typename Iiter>
  void neighbors_old(I index, Iiter iter, bool edges = false) const noexcept {
    *iter++ = index;
    bool odd = index & 1;
    In indices = ((index >> 1) / nside_prefsum_) % nside_;
    // std::cout << indices << std::endl;
    for (I i = 0; i < DIM; ++i) {
      indices[i] += 1;
      // std::cout << indices << " " << i1 << std::endl;
      if ((indices < nside_).sum() == DIM)
        *iter++ = (nside_prefsum_ * indices).sum() << 1 | odd;
      indices[i] -= 2;
      // std::cout << indices << " " << i2 << std::endl;
      if ((indices < nside_).sum() == DIM)
        *iter++ = (nside_prefsum_ * indices).sum() << 1 | odd;
      indices[i] += 1;  // reset
    }
    odd = !odd;
    I sodd = odd ? -1 : 1;
    for (I i = 0; i < (1 << DIM); ++i) {
      In corner(indices);
      for (int d = 0; d < DIM; ++d) corner[d] += ((i >> d) & 1) ? sodd : 0;
      // std::cout << corner << std::endl;
      if ((corner < nside_).sum() == DIM)
        *iter++ = (nside_prefsum_ * corner).sum() << 1 | odd;
    }
    if (edges) {
      odd = !odd;
      for (I i = 0; i < DIM - 1; ++i) {
        for (I j = i + 1; j < DIM; ++j) {
          indices[i] += 1;
          indices[j] += 1;  // +1,+1
          // std::cout << indices << " " << i1 << std::endl;
          if ((indices < nside_).sum() == DIM)
            *iter++ = (nside_prefsum_ * indices).sum() << 1 | odd;
          indices[i] -= 2;  // -1,+1
          // std::cout << indices << " " << i2 << std::endl;
          if ((indices < nside_).sum() == DIM)
            *iter++ = (nside_prefsum_ * indices).sum() << 1 | odd;
          indices[j] -= 2;  // -1,-1
          // std::cout << indices << " " << i2 << std::endl;
          if ((indices < nside_).sum() == DIM)
            *iter++ = (nside_prefsum_ * indices).sum() << 1 | odd;
          indices[i] += 2;  // +1,-1
          // std::cout << indices << " " << i2 << std::endl;
          if ((indices < nside_).sum() == DIM)
            *iter++ = (nside_prefsum_ * indices).sum() << 1 | odd;
          // reset
          indices[i] -= 1;
          indices[j] += 1;
        }
      }
    }
  }
};

template <int DIM, typename F, typename I>
std::ostream &operator<<(std::ostream &out, BCC<DIM, F, I> bcc) {
  using namespace util;
  std::string name = "BCC" + str(DIM) + short_str<F>() + short_str<I>();
  return out << name << "(lb=[" << bcc.lower_ << "], ub=["
             << (bcc.width_ * bcc.nside_.template cast<F>() + bcc.lower_)
             << "], width=[" << bcc.width_ << "], nside=[" << bcc.nside_
             << "])";
}

template <int DIM, typename F, typename I = uint64_t>
struct Cubic {
  typedef Eigen::Array<I, DIM, 1> In;
  typedef Eigen::Array<F, DIM, 1> Fn;
  static_assert((DIM > 2));

  In nside_, nside_prefsum_;
  Fn lower_, width_, lower_cen_, half_width_;

  Cubic() {}

  template <typename Sizes>
  Cubic(Sizes sizes, Fn lower = Fn(0), Fn upper = Fn(1)) {
    init(sizes, lower, upper);
  }

  template <typename Sizes>
  void init(Sizes sizes, Fn lower = Fn(0), Fn upper = Fn(1)) {
    nside_ = sizes;
    lower_ = lower;
    for (size_t i = 0; i < DIM; ++i) nside_prefsum_[i] = nside_.prod(i);
    width_ = (upper - lower_) / nside_.template cast<F>();
    half_width_ = width_ / 2.0;
    lower_cen_ = lower_ + half_width_;
  }

  I size() const noexcept { return nside_.prod(); }

  Fn operator[](I index) const noexcept {
    In indices = (index / nside_prefsum_) % nside_;
    return get_center(indices);
  }

  Fn get_center(In indices) const noexcept {
    return lower_cen_ + width_ * indices.template cast<F>();
  }

  In get_indices(Fn value) const noexcept {
    value = (value - lower_) / width_;
    return value.template cast<I>();
  }

  I operator[](Fn value) const noexcept {
    In indices = get_indices(value);
    return (nside_prefsum_ * indices).sum();
  }

  template <typename Iiter>
  void neighbors(I index, Iiter iter, bool = false) const noexcept {
    In idx0 = (index / nside_prefsum_) % nside_;
    In threes(1);
    for (int d = 1; d < DIM; ++d) threes[d] = 3 * threes[d - 1];
    for (int i = 0; i < threes[DIM - 1] * 3; ++i) {
      In idx(idx0);
      for (int d = 0; d < DIM; ++d) idx[d] += ((i / threes[d]) % 3) - 1;
      // std::cout << i << " " << (idx-idx0).template cast<int>()+1 <<
      // std::endl;
      if ((idx < nside_).sum() == DIM) *iter++ = (nside_prefsum_ * idx).sum();
    }
  }
};
}  // namespace geom
}  // namespace rpxdock
