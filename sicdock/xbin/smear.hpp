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
  XBin<F, K>& xbin;
  K cell_index;
  V val0;
  K key0;
  PHMapUpdateMax(PHMap<K, V>& m, XBin<F, K>& b, K c, V v, K k0)
      : map(m.phmap_), xbin(b), cell_index(c), val0(v), key0(k0) {}
  PHMapUpdateMax<F, K, V>& operator++(int) { return *this; }
  PHMapUpdateMax<F, K, V>& operator*() { return *this; }
  void operator=(K bcc_key) {
    K key = xbin.combine_cell_grid_index(cell_index, bcc_key);

    // if (xbin.bad_grid_key(bcc_key)) {
    //   std::cout << "BAD GRID KEY" << std::endl;
    // }
    // if (xbin.cell_index(key) != cell_index) {
    //   std::cout << "BAD CELL INDEX " << cell_index << " "
    //             << xbin.cell_index(key) << std::endl;
    // }

    // if (key == key0) {
    //   std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" <<
    //   std::endl; std::cout << "KEY0 " << (map.find(key) != map.end()) << " "
    //   << &map
    //             << std::endl;
    //   std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" <<
    //   std::endl;
    // }

    auto [it, inserted] = map.emplace(key, val0);
    if (!inserted) {
      F val = std::max(it->second, val0);
      it->second = val;
      // std::cout << "modified " << bcc_key << " " << key << " " << val
      // << std::endl;
    } else {
      // std::cout << "emplaced " << bcc_key << " " << key << " " << val0
      // << std::endl;
    }
  }
};

template <typename F, typename K, typename V>
std::unique_ptr<PHMap<K, V>> smear(XBin<F, K>& xbin, PHMap<K, V>& phmap,
                                   int radius = 1, bool do_midpoints = true,
                                   bool odd_last3 = true) {
  auto out = std::make_unique<PHMap<K, V>>();
  // std::cout << "MAP LOC " << &out->phmap_ << std::endl;
  for (auto [key, val] : phmap.phmap_) {
    K bcc_key = xbin.grid_key(key);
    K cell_key = xbin.cell_index(key);
    auto updater = PHMapUpdateMax(*out, xbin, cell_key, val, key);
    xbin.grid().neighbors_6_3(bcc_key, updater, radius, do_midpoints,
                              odd_last3);
  }
  // std::cout << "smear out size " << out->size() << std::endl;
  return out;
}

}  // namespace xbin
}  // namespace sicdock