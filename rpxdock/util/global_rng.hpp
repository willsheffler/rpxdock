#pragma once
/** \file */

#include <random>

namespace rpxdock {
namespace util {

static std::mt19937& global_rng() {
  static std::mt19937 rng((unsigned int)time(0) + 750374);
  return rng;
}
}  // namespace util
}  // namespace rpxdock
