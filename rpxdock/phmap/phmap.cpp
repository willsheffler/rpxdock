/*/*cppimport
<%


cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast', '--verbose']
cfg['dependencies'] = ['phmap.hpp']

cfg['parallel'] = False


setup_pybind11(cfg)
%>
*/
/** \file */

#include "rpxdock/phmap/phmap.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>

#include "rpxdock/util/types.hpp"
namespace py = pybind11;
using namespace pybind11::literals;

namespace rpxdock {
namespace phmap {
using namespace util;

template <typename K, typename V>
void test_mod_phmap_inplace(PHMap<K, V> &phmap) {
  for (auto &[k, v] : phmap.phmap_) {
    v = v * 2;
  }
  phmap.phmap_.insert_or_assign(12345, 12345);
}

template <typename K, typename V>
Vx<V> PHMap_get(PHMap<K, V> const &phmap, RefVx<K> keys) {
  py::gil_scoped_release release;
  Vx<V> out(keys.size());
  for (size_t i = 0; i < keys.size(); i++) out[i] = phmap.get_default(keys[i]);
  return out;
}
template <typename K, typename V>
V PHMap_get_single(PHMap<K, V> const &phmap, K key) {
  return phmap.get_default(key);
}

template <typename K, typename V>
void PHMap_set(PHMap<K, V> &phmap, RefVx<K> keys, RefVx<V> vals) {
  if (keys.size() != vals.size())
    throw std::runtime_error("Size of first dimension must match.");
  py::gil_scoped_release release;
  for (size_t idx = 0; idx < keys.size(); idx++)
    phmap.phmap_.insert_or_assign(keys[idx], vals[idx]);
}
template <typename K, typename V>
void PHMap_set_single(PHMap<K, V> &phmap, RefVx<K> keys, V val) {
  py::gil_scoped_release release;
  for (size_t idx = 0; idx < keys.size(); idx++)
    phmap.phmap_.insert_or_assign(keys[idx], val);
}
template <typename K, typename V>
void PHMap_set_single_single(PHMap<K, V> &phmap, K key, V val) {
  phmap.phmap_.insert_or_assign(key, val);
}

template <typename K, typename V>
void PHMap_del(PHMap<K, V> &phmap, RefVx<K> keys) {
  py::gil_scoped_release release;
  for (size_t idx = 0; idx < keys.size(); idx++) phmap.phmap_.erase(keys[idx]);
}

template <typename K, typename V>
Vx<bool> PHMap_has(PHMap<K, V> &phmap, RefVx<K> keys) {
  py::gil_scoped_release release;
  Vx<bool> out(keys.size());
  for (size_t i = 0; i < keys.size(); i++) out[i] = phmap.has(keys[i]);

  return out;
}

template <typename K, typename V>
bool PHMap_contains(PHMap<K, V> &phmap, Vx<K> keys) {
  py::gil_scoped_release release;
  for (size_t i = 0; i < keys.size(); i++)
    if (!phmap.has(keys[i])) return false;
  return true;
}
template <typename K, typename V>
bool PHMap_contains_single(PHMap<K, V> &phmap, K key) {
  auto search = phmap.phmap_.find(key);
  return search != phmap.phmap_.end();
}

template <typename K, typename V>
py::tuple PHMap_items_array(PHMap<K, V> const &phmap, int n = -1) {
  auto keys = std::make_unique<Vx<K>>();
  auto vals = std::make_unique<Vx<V>>();
  {
    py::gil_scoped_release release;
    if (n < 0) n = phmap.size();
    n = std::min<int>(n, phmap.size());

    keys->resize(n);
    vals->resize(n);
    int i = 0;
    for (auto [k, v] : phmap.phmap_) {
      (*keys)[i] = k;
      (*vals)[i] = v;
      if (++i == n) break;
    }
  }
  return py::make_tuple(*keys, *vals);
}
template <typename K, typename V>
Vx<K> PHMap_keys(PHMap<K, V> const &phmap, int nkey = -1) {
  py::gil_scoped_release release;
  if (nkey < 0) nkey = phmap.size();
  nkey = std::min<int>(nkey, phmap.size());
  Vx<K> keys(nkey);
  int i = 0;
  for (auto [k, v] : phmap.phmap_) {
    keys[i] = k;
    if (++i == nkey) break;
  }
  return keys;
}

template <typename K, typename V>
bool PHMap_eq(PHMap<K, V> const &a, PHMap<K, V> const &b) {
  py::gil_scoped_release release;
  if (a.size() != b.size()) return false;
  if (a.default_ != b.default_) return false;
  for (auto [k, v] : a.phmap_) {
    auto it = b.phmap_.find(k);
    if (it == b.phmap_.end()) return false;
    if (it->second != v) return false;
  }
  return true;
}

template <typename K, typename V>
void bind_phmap(const py::module &m, std::string name) {
  using THIS = PHMap<K, V>;

  py::class_<THIS>(m, name.c_str())
      .def(py::init<>())
      .def("__len__", &THIS::size)
      .def("__getitem__", &PHMap_get<K, V>, "getitem", "keys"_a)
      .def("__getitem__", &PHMap_get_single<K, V>, "getitem", "key"_a)
      .def("__setitem__", &PHMap_set<K, V>, "keys"_a, "vals"_a)
      .def("__setitem__", &PHMap_set_single<K, V>, "keys"_a, "val"_a)
      .def("__setitem__", &PHMap_set_single_single<K, V>, "key"_a, "val"_a)
      .def("__delitem__", &PHMap_del<K, V>)
      .def("has", &PHMap_has<K, V>)
      .def("__contains__", &PHMap_contains<K, V>)
      .def("__contains__", &PHMap_contains_single<K, V>)
      .def("keys", &PHMap_keys<K, V>, "num"_a = -1)
      .def("items_array", &PHMap_items_array<K, V>, "num"_a = -1)
      .def("__eq__", &PHMap_eq<K, V>)
      .def_readwrite("default", &THIS::default_)
      .def(
          "items",
          [](THIS const &c) {
            return py::make_iterator(c.phmap_.begin(), c.phmap_.end());
          },
          py::keep_alive<0, 1>())
      .def(py::pickle(
          [](THIS const &map) {  // __getstate__
            py::tuple tup = PHMap_items_array(map);
            auto keys = tup[0].cast<Vx<K>>();
            auto vals = tup[1].cast<Vx<V>>();
            return py::make_tuple(keys, vals, map.default_);
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 2 && t.size() != 3)
              throw std::runtime_error("Invalid state!");
            V v0 = (t.size() == 3) ? t[2].cast<V>() : 0;
            auto map = std::make_unique<THIS>(v0);
            auto keys = t[0].cast<Vx<K>>();
            auto vals = t[1].cast<Vx<V>>();
            PHMap_set<K, V>(*map, keys, vals);
            return map;
          }))

      /**/;
}

PYBIND11_MODULE(phmap, m) {
  bind_phmap<uint32_t, float>(m, "PHMap_u4f4");
  bind_phmap<uint64_t, float>(m, "PHMap_u8f4");
  bind_phmap<uint64_t, double>(m, "PHMap_u8f8");
  bind_phmap<uint64_t, uint64_t>(m, "PHMap_u8u8");

  m.def("test_mod_phmap_inplace", &test_mod_phmap_inplace<uint64_t, uint64_t>);
}

}  // namespace phmap
}  // namespace rpxdock