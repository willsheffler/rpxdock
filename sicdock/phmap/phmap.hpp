#pragma once

#include <parallel_hashmap/phmap.h>

namespace sicdock {
namespace phmap {

template <typename K, typename V>
struct PHMap {
  using Map = ::phmap::parallel_flat_hash_map<K, V>;
  size_t size() const { return phmap_.size(); }
  Map phmap_;
};

}  // namespace phmap
}  // namespace sicdock