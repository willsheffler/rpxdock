#pragma once

#include "sicdock/util/SimpleArray.hpp"
#include "sicdock/util/str.hpp"
// #include "util/template_loop.hpp"

#include <iostream>

namespace sicdock {
namespace geom {

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
template <int _DIM, class _Float = double, class _Index = uint64_t>
struct BCC {
  static const int DIM = _DIM;
  using Float = _Float;
  using Index = _Index;
  using Indices = util::SimpleArray<DIM, Index>;
  using Floats = util::SimpleArray<DIM, Float>;

  static_assert((DIM > 2));

  Indices nside_, nside_prefsum_;
  Floats lower_, width_, lower_cen_, half_width_;
  Indices nside() const { return nside_; }
  Floats lower() const { return lower_; }
  Floats width() const { return width_; }
  Floats upper() const {
    return lower_ + width_ * (nside().template cast<Float>());
  }

  bool operator!=(BCC<DIM, Float, Index> o) const { return !(*this == o); }
  bool operator==(BCC<DIM, Float, Index> o) const {
    return nside_ == o.nside_ && lower_ == o.lower_ && width_ == o.width_;
  }

  BCC() {}

  template <class Sizes>
  BCC(Sizes sizes, Floats lower = Floats(0), Floats upper = Floats(1)) {
    init(sizes, lower, upper);
  }

  template <class Sizes>
  void init(Sizes sizes, Floats lower, Floats upper) {
    nside_ = sizes;
    lower_ = lower;
    for (size_t i = 0; i < DIM; ++i) nside_prefsum_[i] = nside_.prod(i);
    width_ = (upper - lower_) / nside_.template cast<Float>();
    half_width_ = width_ / 2.0;
    lower_cen_ = lower_ + half_width_;
    __int128 totsize = 2;
    for (size_t i = 0; i < DIM; ++i) {
      if (totsize * nside_[i] < totsize)
        throw std::invalid_argument("Index Type is too narrow");
      totsize *= __int128(nside_[i]);
      if (totsize > __int128(std::numeric_limits<Index>::max()))
        throw std::invalid_argument("Index Type is too narrow");
    }
    // std::cout << lower_ << std::endl;
    // std::cout << lower_ + nside_.template cast<Float>() * width_ <<
    // std::endl;
  }

  Index size() const noexcept { return nside_.prod() * 2; }
  int dim() const noexcept { return DIM; }

  Floats operator[](Index index) const noexcept {
    bool odd = index & 1;
    // std::cout << "bcc get_floats " << index << std::endl;
    Indices indices = ((index >> 1) / nside_prefsum_) % nside_;
    // std::cout << "bcc get_floats " << indices << " " << odd << std::endl;
    return this->get_center(indices, odd);
  }

  Floats get_center(Indices indices, bool odd) const noexcept {
    return lower_cen_ + width_ * indices.template cast<Float>() +
           (odd ? half_width_ : 0);
  }

  Indices get_indices(Floats value, bool &odd) const noexcept {
    value = (value - lower_) / width_;
    // std::cout << "bcc::get_indices lower_ " << lower_ << std::endl;
    // std::cout << "bcc::get_indices width_ " << width_ << std::endl;
    // std::cout << "bcc::get_indices " << value << std::endl;
    Indices const indices = value.template cast<Index>();
    // std::cout << "bcc::get_indices " << indices << std::endl;
    value = value - indices.template cast<Float>() - 0.5;
    // std::cout << "bcc::get_indices " << value << std::endl;
    Indices const corner_indices = indices - (value < 0).template cast<Index>();
    // std::cout << "bcc::get_indices " << corner_indices << std::endl;
    odd = (0.25 * DIM) < fabs((value.sign() * value).sum());
    return odd ? corner_indices : indices;
  }

  Index operator[](Floats value) const noexcept {
    bool odd;
    Indices indices = get_indices(value, odd);
    // std::cout << "bcc get_idx " << indices << " " << odd << std::endl;
    Index index = (nside_prefsum_ * indices).sum();
    index = (index << 1) + odd;
    // std::cout << "bcc get idx " << index << std::endl;
    return index;
  }

  template <class Iiter>
  void neighbors(Index index, Iiter iter, bool edges = false,
                 bool edges2 = false) const noexcept {
    *iter++ = index;
    bool odd = index & 1;
    Indices indices = ((index >> 1) / nside_prefsum_) % nside_;
    // std::cout << indices << std::endl;
    for (Index i = 0; i < DIM; ++i) {
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
    Index sodd = odd ? -1 : 1;
    for (Index i = 0; i < (1 << DIM); ++i) {
      Indices corner(indices);
      for (int d = 0; d < DIM; ++d) corner[d] += ((i >> d) & 1) ? sodd : 0;
      // std::cout << corner << std::endl;
      if ((corner < nside_).sum() == DIM)
        *iter++ = (nside_prefsum_ * corner).sum() << 1 | odd;
    }
    if (edges) {
      odd = !odd;
      for (Index i = 0; i < DIM - 1; ++i) {
        for (Index j = i + 1; j < DIM; ++j) {
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

template <int DIM, class Float, class Index>
std::ostream &operator<<(std::ostream &out, BCC<DIM, Float, Index> bcc) {
  using namespace util;
  std::string name = "BCC" + str(DIM) + short_str<Float>() + short_str<Index>();
  return out << name << "(lb=[" << bcc.lower_ << "], ub=["
             << (bcc.width_ * bcc.nside_.template cast<Float>() + bcc.lower_)
             << "], width=[" << bcc.width_ << "], nside=[" << bcc.nside_
             << "])";
}

template <int DIM, class Float, class Index = uint64_t>
struct Cubic {
  typedef util::SimpleArray<DIM, Index> Indices;
  typedef util::SimpleArray<DIM, Float> Floats;
  static_assert((DIM > 2));

  Indices nside_, nside_prefsum_;
  Floats lower_, width_, lower_cen_, half_width_;

  Cubic() {}

  template <class Sizes>
  Cubic(Sizes sizes, Floats lower = Floats(0), Floats upper = Floats(1)) {
    init(sizes, lower, upper);
  }

  template <class Sizes>
  void init(Sizes sizes, Floats lower = Floats(0), Floats upper = Floats(1)) {
    nside_ = sizes;
    lower_ = lower;
    for (size_t i = 0; i < DIM; ++i) nside_prefsum_[i] = nside_.prod(i);
    width_ = (upper - lower_) / nside_.template cast<Float>();
    half_width_ = width_ / 2.0;
    lower_cen_ = lower_ + half_width_;
  }

  Index size() const noexcept { return nside_.prod(); }

  Floats operator[](Index index) const noexcept {
    Indices indices = (index / nside_prefsum_) % nside_;
    return get_center(indices);
  }

  Floats get_center(Indices indices) const noexcept {
    return lower_cen_ + width_ * indices.template cast<Float>();
  }

  Indices get_indices(Floats value) const noexcept {
    value = (value - lower_) / width_;
    return value.template cast<Index>();
  }

  Index operator[](Floats value) const noexcept {
    Indices indices = get_indices(value);
    return (nside_prefsum_ * indices).sum();
  }

  template <class Iiter>
  void neighbors(Index index, Iiter iter, bool = false) const noexcept {
    Indices idx0 = (index / nside_prefsum_) % nside_;
    Indices threes(1);
    for (int d = 1; d < DIM; ++d) threes[d] = 3 * threes[d - 1];
    for (int i = 0; i < threes[DIM - 1] * 3; ++i) {
      Indices idx(idx0);
      for (int d = 0; d < DIM; ++d) idx[d] += ((i / threes[d]) % 3) - 1;
      // std::cout << i << " " << (idx-idx0).template cast<int>()+1 <<
      // std::endl;
      if ((idx < nside_).sum() == DIM) *iter++ = (nside_prefsum_ * idx).sum();
    }
  }
};
}  // namespace geom
}  // namespace sicdock
