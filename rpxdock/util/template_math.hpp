/// @file   util/template_math.hpp
/// @brief  a place for some template math to live
/// @author will sheffler

#pragma once
/** \file */

#include <stdint.h>

namespace rpxdock {
namespace util {

template <int X, int Y>
struct POW {
  static const int VAL = X * POW<X, Y - 1>::VAL;
};
template <int X /*1*/>
struct POW<X, 1> {
  static const int VAL = X;
};
template <int X /*0*/>
struct POW<X, 0> {
  static const int VAL = 1;
};

namespace LOG_IMPL {
template <uint64_t val, uint64_t base>
struct FLOOR_LOG_IMPL {
  static const uint64_t VAL = FLOOR_LOG_IMPL<val / base, base>::VAL + 1ul;
};
template <uint64_t base>
struct FLOOR_LOG_IMPL<0ul, base> {
  static const uint64_t VAL = 0ul;
};
template <uint64_t val, uint64_t base, uint64_t remainder>
struct CEIL_LOG_TEST {
  static const uint64_t VAL = 1ul;
};
template <uint64_t base>
struct CEIL_LOG_TEST<1ul, base, 0ul> {
  static const uint64_t VAL = 0ul;
};
template <uint64_t base>
struct CEIL_LOG_TEST<0ul, base, 0ul> {
  static const uint64_t VAL = 0ul;
};
template <uint64_t val, uint64_t base>
struct CEIL_LOG_TEST<val, base, 0ul> {
  static const uint64_t VAL = CEIL_LOG_TEST<val / base, base, val % base>::VAL;
};
}  // namespace LOG_IMPL
template <uint64_t val, uint64_t base>
struct FLOOR_LOG {
  static const uint64_t VAL = LOG_IMPL::FLOOR_LOG_IMPL<val / base, base>::VAL;
};
template <uint64_t val, uint64_t base>
struct CEIL_LOG {
  static const uint64_t VAL =
      FLOOR_LOG<val, base>::VAL +
      LOG_IMPL::CEIL_LOG_TEST<val / base, base, val % base>::VAL;
};
template <uint64_t base>
struct CEIL_LOG<1ul, base> {
  static const uint64_t VAL = 0ul;
};
template <uint64_t base>
struct FLOOR_LOG<0ul, base> {};  // ERROR 0ul
template <uint64_t base>
struct CEIL_LOG<0ul, base> {};
template <uint64_t val>
struct FLOOR_LOG<val, 1ul> {};  // ERROR base 1ul
template <uint64_t val>
struct CEIL_LOG<val, 1ul> {};
template <uint64_t val>
struct FLOOR_LOG<val, 0ul> {};  // ERROR base 0ul
template <uint64_t val>
struct CEIL_LOG<val, 0ul> {};

// G prevent compiler from complaining about out-of-bound shifts in SAFE_*SHIFT
// leave at bottom for safety I guess....
#pragma GCC system_header
template <class T, T val, uint64_t shift>
struct SAFE_LSHIFT {
  static const T VAL = (shift >= (sizeof(T) * 8)) ? 0 : (val << shift);
};
template <class T, T val, uint64_t shift>
struct SAFE_RSHIFT {
  static const T VAL = (shift >= (sizeof(T) * 8)) ? 0 : (val >> shift);
};
}  // namespace util
}  // namespace rpxdock
