/*cppimport
<%
cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['../util/assertions.hpp', 'xbin.hpp']
setup_pybind11(cfg)
%>
*/

#include <parallel_hashmap/phmap.h>
#include <chrono>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>

#include "sicdock/util/assertions.hpp"
#include "sicdock/util/types.hpp"
#include "sicdock/xbin/xbin.hpp"

using namespace pybind11::literals;
using namespace Eigen;
using namespace sicdock;
using namespace util;
using namespace geom;

namespace py = pybind11;

namespace sicdock {
namespace xbin {

namespace patch {
template <typename T>
std::string to_string(const T& n) {
  std::ostringstream stm;
  stm << n;
  return stm.str();
}
}  // namespace patch

template <typename T>
using milliseconds = std::chrono::duration<T, std::milli>;

class custom_type {
  std::string one = "one";
  std::string two = "two";
  std::uint32_t three = 3;
  std::uint64_t four = 4;
  std::uint64_t five = 5;

 public:
  custom_type() = default;
  // Make object movable and non-copyable
  custom_type(custom_type&&) = default;
  custom_type& operator=(custom_type&&) = default;
  // should be automatically deleted per
  // http://www.slideshare.net/ripplelabs/howard-hinnant-accu2014
  // custom_type(custom_type const&) = delete;
  // custom_type& operator=(custom_type const&) = delete;
};

bool test_phmap2(std::size_t iterations, std::size_t container_size) {
  std::clog << "bench: iterations: " << iterations
            << " / container_size: " << container_size << "\n";
  {
    std::size_t count = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
      std::unordered_map<std::string, custom_type> m;
      m.reserve(container_size);
      for (std::size_t j = 0; j < container_size; ++j)
        m.emplace(patch::to_string(j), custom_type());
      count += m.size();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto elapsed = milliseconds<double>(t2 - t1).count();
    if (count != iterations * container_size)
      std::clog << "  invalid count: " << count << "\n";
    std::clog << "  std::unordered_map:     " << std::fixed << int(elapsed)
              << " ms\n";
  }

  {
    std::size_t count = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
      std::map<std::string, custom_type> m;
      for (std::size_t j = 0; j < container_size; ++j)
        m.emplace(patch::to_string(j), custom_type());
      count += m.size();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto elapsed = milliseconds<double>(t2 - t1).count();
    if (count != iterations * container_size)
      std::clog << "  invalid count: " << count << "\n";
    std::clog << "  std::map:               " << std::fixed << int(elapsed)
              << " ms\n";
  }

  {
    std::size_t count = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
      std::vector<std::pair<std::string, custom_type>> m;
      m.reserve(container_size);
      for (std::size_t j = 0; j < container_size; ++j)
        m.emplace_back(patch::to_string(j), custom_type());
      count += m.size();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto elapsed = milliseconds<double>(t2 - t1).count();
    if (count != iterations * container_size)
      std::clog << "  invalid count: " << count << "\n";
    std::clog << "  std::vector<std::pair>: " << std::fixed << int(elapsed)
              << " ms\n";
  }

  {
    std::size_t count = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
      phmap::flat_hash_map<std::string, custom_type> m;
      m.reserve(container_size);
      for (std::size_t j = 0; j < container_size; ++j)
        m.emplace(patch::to_string(j), custom_type());
      count += m.size();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto elapsed = milliseconds<double>(t2 - t1).count();
    if (count != iterations * container_size)
      std::clog << "  invalid count: " << count << "\n";
    std::clog << "  phmap::flat_hash_map:   " << std::fixed << int(elapsed)
              << " ms\n";
  }
  return true;
}

bool test_phmap() {
  using phmap::flat_hash_map;
  // Create an unordered_map of three strings (that map to strings)
  flat_hash_map<std::string, std::string> email = {
      {"tom", "tom@gmail.com"},
      {"jeff", "jk@gmail.com"},
      {"jim", "jimg@microsoft.com"}};

  // Iterate and print keys and values
  // for (const auto &n : email)
  // std::cout << n.first << "'s email is: " << n.second << "\n";

  // Add a new entry
  email["bill"] = "bg@whatever.com";

  // and print it
  // std::cout << "bill's email is: " << email["bill"] << "\n";

  return true;
}

PYBIND11_MODULE(xmap, m) {
  m.def("test_phmap", &test_phmap);
  m.def("test_phmap2", &test_phmap2);
}

}  // namespace xbin
}  // namespace sicdock