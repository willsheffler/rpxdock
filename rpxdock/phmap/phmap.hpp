#pragma once

#include <parallel_hashmap/phmap.h>

namespace rpxdock {
/**
\namespace rpxdock::phmap
\brief namespace for phmap wrapper and related utils
*/
namespace phmap {

template <typename K, typename V>
struct PHMap {
  using Map =
      ::phmap::parallel_flat_hash_map<K, V, ::phmap::Hash<K>, std::equal_to<K>>;
  PHMap() {}
  PHMap(V d) : default_(d) {}
  size_t size() const { return phmap_.size(); }
  Map phmap_;
  V default_ = 0;
  void set_default(V new_default) noexcept { default_ = new_default; }
  V get_default(V new_default) noexcept { return default_; }
  V get_default(K k) const noexcept {
    auto it = phmap_.find(k);
    return (it == phmap_.end()) ? default_ : it->second;
  }
  bool has(K k) const noexcept { return phmap_.find(k) != phmap_.end(); }
};

}  // namespace phmap
}  // namespace rpxdock