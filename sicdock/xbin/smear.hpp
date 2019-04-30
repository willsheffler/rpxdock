#pragma once

#include "sicdock/phmap/phmap.hpp"
#include "sicdock/util/types.hpp"
#include "sicdock/xbin/xbin.hpp"

#include <set>

namespace sicdock {
namespace xbin {

using std::cout;
using std::endl;

template <typename F, typename K>
using XBin = XformHash_bt24_BCC6<X3<F>, K>;
using phmap::PHMap;

template <typename F, typename K, typename V>
struct PHMapUpdateMax {
  typename PHMap<K, V>::Map& map;
  XBin<F, K>& bin;
  K cell_index;
  V val0;
  PHMapUpdateMax(PHMap<K, V>& m, XBin<F, K>& b, K c, V v)
      : map(m.phmap_), bin(b), cell_index(c), val0(v) {}
  PHMapUpdateMax<F, K, V>& operator++(int) { return *this; }
  PHMapUpdateMax<F, K, V>& operator*() { return *this; }
  void operator=(K grid_key) {
    K key = bin.combine_cell_grid_index(cell_index, grid_key);
    auto it = map.find(key);
    if (it == map.end() || it->second < val0) map.emplace(key, val0);
  }
};

template <typename F, typename K, typename V>
PHMap<K, V> smear(XBin<F, K>& xbin, PHMap<K, V>& phmap, int range,
                  bool do_midpoints, bool odd_last3) {
  PHMap<K, V> out;
  for (auto [key, val] : phmap.phmap_) {
    auto updater = PHMapUpdateMax(out, xbin, xbin.cell_index(key), val);
    xbin.grid().neighbors_6_3(key, updater, range, do_midpoints, odd_last3);
  }
  return out;
}

}  // namespace xbin
}  // namespace sicdock