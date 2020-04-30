#pragma once
/** \file */

#include <set>

#include "rpxdock/phmap/phmap.hpp"
#include "rpxdock/util/types.hpp"
#include "rpxdock/xbin/xbin.hpp"

namespace rpxdock {
namespace xbin {

using std::cout;
using std::endl;

template <typename F, typename K>
using Xbin = XformHash_bt24_BCC6<X3<F>, K>;
using phmap::PHMap;

template <typename F, typename K, typename V>
struct PHMapUpdateMax {
  typename PHMap<K, V>::Map& map;
  Xbin<F, K> const& xbin;
  K cell_index;
  V val0;
  K key0;
  Vx<V> const& kernel;
  PHMapUpdateMax(PHMap<K, V>& m, Xbin<F, K>& b, K c, V v, K k0, Vx<V>& ker)
      : map(m.phmap_), xbin(b), cell_index(c), val0(v), key0(k0), kernel(ker) {}
  PHMapUpdateMax<F, K, V>& operator++(int) noexcept { return *this; }
  PHMapUpdateMax<F, K, V>& operator*() noexcept { return *this; }
  void operator=(std::pair<K, K> key_rad) noexcept {
    K bcc_key = key_rad.first;
    K radius = key_rad.second;
    K key = xbin.combine_cell_grid_index(cell_index, bcc_key);
    V val_feather = val0 * kernel[radius];
    auto [it, inserted] = map.emplace(key, val_feather);
    if (!inserted) {
      F val = std::max(it->second, val_feather);
      it->second = val;
    }
  }
};

template <typename F, typename K, typename V>
std::unique_ptr<PHMap<K, V>> smear(Xbin<F, K>& xbin, PHMap<K, V>& phmap,
                                   int radius = 1, bool exhalf = false,
                                   bool oddlast3 = true, bool sphere = true,
                                   Vx<V> kernel = Vx<V>()) {
  int r = xbin.grid().neighbor_radius_square_cut(radius, exhalf);
  if (sphere) r = xbin.grid().neighbor_sphere_radius_square_cut(radius, exhalf);
  if (kernel.size() == 0) {
    kernel = Vx<V>(r);
    kernel.fill(1);
  }
  auto out = std::make_unique<PHMap<K, V>>();
  // std::cout << "MAP LOC " << &out->phmap_ << std::endl;
  for (auto [key, val] : phmap.phmap_) {
    K bcc_key = xbin.grid_key(key);
    K cell_key = xbin.cell_index(key);
    auto updater = PHMapUpdateMax(*out, xbin, cell_key, val, key, kernel);
    xbin.grid().neighbors_6_3(bcc_key, updater, radius, exhalf, oddlast3,
                              sphere);
  }
  // std::cout << "smear out size " << out->size() << std::endl;
  return out;
}

}  // namespace xbin
}  // namespace rpxdock