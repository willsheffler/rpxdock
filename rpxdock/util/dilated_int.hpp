/// @file   util/dilated_int.hpp
/// @brief  Z-order or Morton style indexing utilities for arbitrary dimension
/// @author will sheffler

// inspired by code from here: http://www.marcusbannerman.co.uk/dynamo
// see "Converting to and from Dilated Integers"(doi:10.1109/TC.2007.70814)
// maybe redo with modern code?

#pragma once

#include "rpxdock/util/template_math.hpp"
// #include <iostream>
#ifdef DEBUG
#include <cstddef>
#include <stdexcept>
#endif

namespace rpxdock {
namespace util {

namespace impl {

template <uint64_t D>
struct MAXBITS {
  static const uint64_t VAL = (8 * sizeof(uint64_t)) / D;
};
template <uint64_t D>
struct DILATION_ROUNDS {
  static const uint64_t VAL = CEIL_LOG<MAXBITS<D>::VAL, D - 1>::VAL;
};
template <>
struct DILATION_ROUNDS<2> {
  static const uint64_t VAL = CEIL_LOG<MAXBITS<2>::VAL, 2>::VAL;
};
template <uint64_t D>
struct UNDILATION_ROUNDS {
  static const uint64_t VAL = CEIL_LOG<MAXBITS<D>::VAL, D>::VAL;
};

template <uint64_t P, uint64_t Q>
struct X_P_Q {
  template <uint64_t l, uint64_t MOCK>
  struct XPQ_IMPL {
    static const uint64_t VAL =
        XPQ_IMPL<l - 1, MOCK>::VAL + SAFE_LSHIFT<uint64_t, 1, l * Q>::VAL;
  };
  template <uint64_t MOCK>
  struct XPQ_IMPL<0, MOCK> {
    static const uint64_t VAL = 1;
  };
  static const uint64_t VAL = XPQ_IMPL<P - 1, 0>::VAL;
};

template <uint64_t I_PLUS_1, uint64_t D>
struct UMULT_I_D {
  static const uint64_t VAL =
      X_P_Q<D, (D - 1) * POW<D, I_PLUS_1 - 1>::VAL>::VAL;
};
template <uint64_t ITER, uint64_t D>
struct DMULT_I_D {
  static const uint64_t VAL =
      X_P_Q<D, POW<D - 1, DILATION_ROUNDS<D>::VAL - ITER + 1>::VAL>::VAL;
};
template <uint64_t N>
struct N_BITS_LO {
  static const uint64_t VAL = (SAFE_LSHIFT<uint64_t, 1, N>::VAL - 1);
};
template <uint64_t D>
struct MAX_DILATABLE {
  static const uint64_t VAL = N_BITS_LO<MAXBITS<D>::VAL>::VAL;
};

template <uint64_t ITER, uint64_t D>
struct UMASK_I_D {
  static const uint64_t _di = POW<D, ITER>::VAL;
  static const uint64_t BITCOUNT =
      (_di < MAXBITS<D>::VAL) ? _di : MAXBITS<D>::VAL;
  template <uint64_t ITER_2, uint64_t MOCK>
  struct ZDI_IMPL {
    static const uint64_t VAL = ZDI_IMPL<ITER_2 - 1, MOCK>::VAL |
                                SAFE_RSHIFT<uint64_t, ZDI_IMPL<0, MOCK>::VAL,
                                            ((_di * D) * ITER_2)>::VAL;
  };
  template <uint64_t MOCK>
  struct ZDI_IMPL<0, MOCK> {
    static const uint64_t VAL =
        SAFE_LSHIFT<uint64_t,
                    N_BITS_LO<BITCOUNT>::VAL & N_BITS_LO<MAXBITS<D>::VAL>::VAL,
                    D*(MAXBITS<D>::VAL - 1) + 1 - BITCOUNT>::VAL;
  };
  static const uint64_t VAL =
      ZDI_IMPL<MAXBITS<D>::VAL / POW<D, ITER>::VAL, 0>::VAL;
};

// bit mask used after each round of the dilation algorithm.
template <uint64_t ITER, uint64_t D>
struct DMASK_I_D {
  static const uint64_t BITCOUNT =
      POW<D - 1, DILATION_ROUNDS<D>::VAL - ITER>::VAL;
  static const uint64_t BITSEP =
      POW<D - 1, DILATION_ROUNDS<D>::VAL - ITER + 1>::VAL + BITCOUNT;
  template <uint64_t ITER_2, uint64_t MOCK>
  struct YDI_IMPL {
    static const uint64_t VAL =
        YDI_IMPL<ITER_2 - 1, MOCK>::VAL |
        SAFE_LSHIFT<uint64_t, N_BITS_LO<BITCOUNT>::VAL, BITSEP * ITER_2>::VAL;
  };
  template <uint64_t MOCK>
  struct YDI_IMPL<0, MOCK> {
    static const uint64_t VAL = N_BITS_LO<BITCOUNT>::VAL;
  };
  static const uint64_t VAL = YDI_IMPL<MAXBITS<D>::VAL / BITCOUNT, 0>::VAL;
};
template <uint64_t ITER>
struct DMASK_I_D<ITER, 2> {
  static const uint64_t BITCOUNT = MAXBITS<2>::VAL / (1 << ITER);
  static const uint64_t BITSEP = 2 * BITCOUNT;
  template <uint64_t ITER_2, uint64_t MOCK>
  struct YDI_IMPL {
    static const uint64_t VAL =
        YDI_IMPL<ITER_2 - 1, MOCK>::VAL |
        SAFE_LSHIFT<uint64_t, N_BITS_LO<BITCOUNT>::VAL, BITSEP * ITER_2>::VAL;
  };
  template <uint64_t MOCK>
  struct YDI_IMPL<0, MOCK> {
    static const uint64_t VAL = N_BITS_LO<BITCOUNT>::VAL;
  };
  static const uint64_t VAL = YDI_IMPL<MAXBITS<2>::VAL / BITCOUNT, 0>::VAL;
};

template <uint64_t D>
struct MAX_DILATED {
  static const uint64_t VAL = DMASK_I_D<DILATION_ROUNDS<D>::VAL, D>::VAL;
};

template <uint64_t D>
struct UNDILATE {
  template <uint64_t ITER, uint64_t MOCK>
  struct UNDILATE_IMPL {
    static inline uint64_t eval(uint64_t val) {
      return (UNDILATE_IMPL<ITER - 1, MOCK>::eval(val) *
              UMULT_I_D<ITER, D>::VAL) &
             UMASK_I_D<ITER, D>::VAL;
    }
  };
  template <uint64_t MOCK>
  struct UNDILATE_IMPL<1, MOCK> {
    static inline uint64_t eval(uint64_t val) {
      return (val * UMULT_I_D<1, D>::VAL) & UMASK_I_D<1, D>::VAL;
    }
  };
  static inline uint64_t eval(uint64_t val) {
    static const uint64_t undilate_shift =
        (D * (MAXBITS<D>::VAL - 1) + 1 - MAXBITS<D>::VAL);
    return UNDILATE_IMPL<UNDILATION_ROUNDS<D>::VAL, 0>::eval(val) >>
           undilate_shift;
  };
};

// see "Converting to and from Dilated Integers"(doi:10.1109/TC.2007.70814)
template <uint64_t D>
struct DILATE {
  template <uint64_t ITER, uint64_t MOCK>
  struct DILATE_IMPL {
    static inline uint64_t eval(uint64_t val) {
      return (DILATE_IMPL<ITER - 1, D>::eval(val) * DMULT_I_D<ITER, D>::VAL) &
             DMASK_I_D<ITER, D>::VAL;
    }
  };
  template <uint64_t MOCK>
  struct DILATE_IMPL<1, MOCK> {
    static inline uint64_t eval(uint64_t val) {
      return (val * DMULT_I_D<1, D>::VAL) & DMASK_I_D<1, D>::VAL;
    }
  };
  static inline uint64_t eval(uint64_t val) {
    return DILATE_IMPL<DILATION_ROUNDS<D>::VAL, 0>::eval(val);
  };
};

// for 2, must use Shift-Or rather than multiply, see paper
// see "Converting to and from Dilated Integers"(doi:10.1109/TC.2007.70814)
template <>
struct DILATE<2> {
  template <uint64_t ITER>
  struct shiftval {
    static const uint64_t VAL =
        SAFE_LSHIFT<uint64_t, 1, DILATION_ROUNDS<2>::VAL - ITER>::VAL;
  };
  template <uint64_t ITER, uint64_t MOCK>
  struct DILATE_IMPL {
    static inline uint64_t eval(uint64_t val) {
      const uint64_t val2 = DILATE_IMPL<ITER - 1, 2>::eval(val);
      return (val2 | (val2 << shiftval<ITER>::VAL)) & DMASK_I_D<ITER, 2>::VAL;
    }
  };
  template <uint64_t MOCK>
  struct DILATE_IMPL<1, MOCK> {
    static inline uint64_t eval(uint64_t val) {
      return (val | (val << shiftval<1>::VAL)) & DMASK_I_D<1, 2>::VAL;
    }
  };
  static inline uint64_t eval(uint64_t val) {
    return DILATE_IMPL<DILATION_ROUNDS<2>::VAL, 0>::eval(val);
  };
};
}  // namespace impl

template <uint64_t D>
inline uint64_t dilate(uint64_t val) {
#ifdef DEBUG
  if (val > impl::MAX_DILATABLE<D>::VAL)
    throw std::out_of_range("too big to dilate");
#endif
  return impl::DILATE<D>::eval(val);
}
template <uint64_t D>
inline uint64_t undilate(uint64_t val) {
#ifdef DEBUG
  if (val > impl::MAX_DILATED<D>::VAL)
    throw std::out_of_range("too big to undilate");
#endif
  val = val & impl::MAX_DILATED<D>::VAL;
  return impl::UNDILATE<D>::eval(val);
}
template <>
inline uint64_t dilate<1>(uint64_t val) {
  return val;
}
template <>
inline uint64_t undilate<1>(uint64_t val) {
  return val;
}

template <class Index>
uint64_t undilate(uint64_t dim, Index val) {
  switch (dim) {
    case 1:
      return undilate<1>(val);
    case 2:
      return undilate<2>(val);
    case 3:
      return undilate<3>(val);
    case 4:
      return undilate<4>(val);
    case 5:
      return undilate<5>(val);
    case 6:
      return undilate<6>(val);
    case 7:
      return undilate<7>(val);
    case 8:
      return undilate<8>(val);
    case 9:
      return undilate<9>(val);
    case 10:
      return undilate<10>(val);
    case 11:
      return undilate<11>(val);
    case 12:
      return undilate<12>(val);
    case 13:
      return undilate<13>(val);
    case 14:
      return undilate<14>(val);
    case 15:
      return undilate<15>(val);
    case 16:
      return undilate<16>(val);
    case 17:
      return undilate<17>(val);
    case 18:
      return undilate<18>(val);
    case 19:
      return undilate<19>(val);
    case 20:
      return undilate<20>(val);
    case 21:
      return undilate<21>(val);
    case 22:
      return undilate<22>(val);
    case 23:
      return undilate<23>(val);
    case 24:
      return undilate<24>(val);
    case 25:
      return undilate<25>(val);
    case 26:
      return undilate<26>(val);
    case 27:
      return undilate<27>(val);
    case 28:
      return undilate<28>(val);
    case 29:
      return undilate<29>(val);
    case 30:
      return undilate<30>(val);
    case 31:
      return undilate<31>(val);
    case 32:
      return undilate<32>(val);
  }
  return 0;
}

template <class Index>
uint64_t dilate(uint64_t dim, Index val) {
  switch (dim) {
    case 1:
      return dilate<1>(val);
    case 2:
      return dilate<2>(val);
    case 3:
      return dilate<3>(val);
    case 4:
      return dilate<4>(val);
    case 5:
      return dilate<5>(val);
    case 6:
      return dilate<6>(val);
    case 7:
      return dilate<7>(val);
    case 8:
      return dilate<8>(val);
    case 9:
      return dilate<9>(val);
    case 10:
      return dilate<10>(val);
    case 11:
      return dilate<11>(val);
    case 12:
      return dilate<12>(val);
    case 13:
      return dilate<13>(val);
    case 14:
      return dilate<14>(val);
    case 15:
      return dilate<15>(val);
    case 16:
      return dilate<16>(val);
    case 17:
      return dilate<17>(val);
    case 18:
      return dilate<18>(val);
    case 19:
      return dilate<19>(val);
    case 20:
      return dilate<20>(val);
    case 21:
      return dilate<21>(val);
    case 22:
      return dilate<22>(val);
    case 23:
      return dilate<23>(val);
    case 24:
      return dilate<24>(val);
    case 25:
      return dilate<25>(val);
    case 26:
      return dilate<26>(val);
    case 27:
      return dilate<27>(val);
    case 28:
      return dilate<28>(val);
    case 29:
      return dilate<29>(val);
    case 30:
      return dilate<30>(val);
    case 31:
      return dilate<31>(val);
    case 32:
      return dilate<32>(val);
  }
  return 0;
}

}  // namespace util
}  // namespace rpxdock
