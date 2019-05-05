/*cppimport
<%
cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['phmap.hpp']

setup_pybind11(cfg)
%>
*/

#include "sicdock/phmap/phmap.hpp"
// #include <parallel_hashmap/phmap_utils.h>

#include <iostream>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;

namespace sicdock {
namespace phmap {

template <typename K, typename V>
void test_mod_phmap_inplace(PHMap<K, V> &phmap) {
  for (auto &[k, v] : phmap.phmap_) {
    v = v * 2;
  }
  phmap.phmap_.insert_or_assign(12345, 12345);
}

template <typename K, typename V>
py::array_t<V> PHMap_get(PHMap<K, V> const &phmap, py::array_t<K> keys,
                         V default_val = 0) {
  py::buffer_info kbuf = keys.request();
  auto *kptr = (K *)kbuf.ptr;
  auto out = py::array_t<V>(keys.size());
  py::buffer_info obuf = out.request();
  auto *optr = (V *)obuf.ptr;
  for (size_t idx = 0; idx < keys.size(); idx++) {
    auto search = phmap.phmap_.find(kptr[idx]);

    if (search != phmap.phmap_.end()) {
      optr[idx] = search->second;
    } else {
      optr[idx] = default_val;
    }
  }
  return out;
}
// template <typename K, typename V>
// V PHMap_get_single(PHMap<K, V> const &phmap, K key, V default_val = 0) {
//   auto search = phmap.phmap_.find(key);
//   if (search != phmap.phmap_.end()) {
//     return search->second;
//   } else {
//     return default_val;
//   }
// }

template <typename K, typename V>
void PHMap_set(PHMap<K, V> &phmap, py::array_t<K> keys, py::array_t<V> vals) {
  if (keys.size() != vals.size() && vals.size() != 1)
    throw std::runtime_error("Size of first dimension must match.");
  py::buffer_info kbuf = keys.request();
  auto *kptr = (K *)kbuf.ptr;
  py::buffer_info vbuf = vals.request();
  auto *vptr = (V *)vbuf.ptr;
  if (vals.size() == 1)
    for (size_t idx = 0; idx < keys.size(); idx++)
      phmap.phmap_.insert_or_assign(kptr[idx], vptr[0]);
  else
    for (size_t idx = 0; idx < keys.size(); idx++)
      phmap.phmap_.insert_or_assign(kptr[idx], vptr[idx]);
}
// template <typename K, typename V>
// void PHMap_set_single(PHMap<K, V> &phmap, K key, V val) {
//   phmap.phmap_.insert_or_assign(key, val);
// }

template <typename K, typename V>
void PHMap_del(PHMap<K, V> &phmap, py::array_t<K> keys) {
  py::buffer_info kbuf = keys.request();
  auto *kptr = (K *)kbuf.ptr;
  for (size_t idx = 0; idx < keys.size(); idx++) phmap.phmap_.erase(kptr[idx]);
}
// template <typename K, typename V>
// void PHMap_del_single(PHMap<K, V> &phmap, K key) {
//   phmap.phmap_.erase(key);
// }

template <typename K, typename V>
py::array_t<bool> PHMap_has(PHMap<K, V> &phmap, py::array_t<K> keys) {
  py::buffer_info kbuf = keys.request();
  auto *kptr = (K *)kbuf.ptr;
  auto out = py::array_t<bool>(keys.size());
  py::buffer_info obuf = out.request();
  auto *optr = (bool *)obuf.ptr;
  for (size_t idx = 0; idx < keys.size(); idx++) {
    auto search = phmap.phmap_.find(kptr[idx]);
    if (search != phmap.phmap_.end()) {
      optr[idx] = true;
    } else {
      optr[idx] = false;
    }
  }
  return out;
}
template <typename K, typename V>
bool PHMap_contains(PHMap<K, V> &phmap, py::array_t<K> keys) {
  py::buffer_info kbuf = keys.request();
  auto *kptr = (K *)kbuf.ptr;
  auto out = py::array_t<bool>(keys.size());
  for (size_t idx = 0; idx < keys.size(); idx++) {
    auto search = phmap.phmap_.find(kptr[idx]);
    if (search == phmap.phmap_.end()) {
      return false;
    }
  }
  return true;
}

template <typename K, typename V>
py::tuple PHMap_items_array(PHMap<K, V> const &phmap, int n = -1) {
  if (n < 0) n = phmap.size();
  n = std::min<int>(n, phmap.size());

  py::array_t<K> keys(n);
  py::array_t<V> vals(n);
  K *kptr = (K *)keys.request().ptr;
  V *vptr = (V *)vals.request().ptr;
  int i = 0;
  for (auto [k, v] : phmap.phmap_) {
    kptr[i] = k;
    vptr[i] = v;
    if (++i == n) break;
  }
  return py::make_tuple(keys, vals);
}
template <typename K, typename V>
py::array_t<K> PHMap_keys(PHMap<K, V> const &phmap, int nkey = -1) {
  if (nkey < 0) nkey = phmap.size();
  nkey = std::min<int>(nkey, phmap.size());
  py::array_t<K> keys(nkey);
  K *kptr = (K *)keys.request().ptr;
  int i = 0;
  for (auto [k, v] : phmap.phmap_) {
    kptr[i] = k;
    if (++i == nkey) break;
  }
  return keys;
}

template <typename K, typename V>
bool PHMap_eq(PHMap<K, V> const &a, PHMap<K, V> const &b) {
  if (a.size() != b.size()) return false;
  for (auto [k, v] : a.phmap_) {
    auto it = b.phmap_.find(k);
    if (it == b.phmap_.end()) return false;
    if (it->second != v) return false;
  }
  return true;
}

template <typename K, typename V>
void bind_phmap(const py::module &m, std::string name) {
  using PHMap = PHMap<K, V>;

  py::class_<PHMap>(m, name.c_str())
      .def(py::init<>())
      .def("__len__", &PHMap::size)
      .def("__getitem__", &PHMap_get<K, V>, "getitem", "keys"_a,
           "default"_a = 0)
      .def("__setitem__", &PHMap_set<K, V>)
      .def("__delitem__", &PHMap_del<K, V>)
      .def("has", &PHMap_has<K, V>)
      .def("__contains__", &PHMap_contains<K, V>)
      .def("keys", &PHMap_keys<K, V>, "num"_a = -1)
      .def("items_array", &PHMap_items_array<K, V>, "num"_a = -1)
      .def("__eq__", &PHMap_eq<K, V>)
      .def("items",
           [](PHMap const &c) {
             return py::make_iterator(c.phmap_.begin(), c.phmap_.end());
           },
           py::keep_alive<0, 1>())
      .def(py::pickle(
          [](PHMap const &map) {  // __getstate__
            return PHMap_items_array(map);
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 2) throw std::runtime_error("Invalid state!");
            auto map = std::make_unique<PHMap>();
            auto keys = t[0].cast<py::array_t<K>>();
            auto vals = t[1].cast<py::array_t<V>>();
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
}  // namespace sicdock