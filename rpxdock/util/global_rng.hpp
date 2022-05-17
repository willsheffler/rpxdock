#pragma once
/** \file */

#include <random>

namespace rpxdock {
namespace util {

static std::mt19937& global_rng() {
  std::random_device rd;
  static std::mt19937 rng(rd());
  return rng;
}
}  // namespace util
}  // namespace rpxdock
