/*cppimport
<%
cfg['include_dirs'] = ['../..']
cfg['compiler_args'] = ['-std=c++17', '-w']
cfg['dependencies'] = ['SimpleArray.hpp']

setup_pybind11(cfg)
%>
*/

#include "sicdock/util/SimpleArray.hpp"
#include <pybind11/pybind11.h>
#include "sicdock/util/assertions.hpp"
namespace py = pybind11;

namespace sicdock {
namespace util {

using std::cout;
using std::endl;

bool TEST_SimpleArray_bounds_check() {
  SimpleArray<3, int> a;
  a[3];  // non-bounds checked
         // #ifndef NDEBUG
         // #ifndef CXX14
  // ASSERT_DEATH(a.at(3), ".*");
  // #endif
  // #endif
  return true;
}

bool TEST_SimpleArray_iteration() {
  SimpleArray<3, int> a;
  int v;
  v = 0;
  for (int &i : a) i = ++v;
  v = 0;
  for (int i : a) ASSERT_EQ(++v, i);
  v = 0;
  for (int i : a) ASSERT_EQ(++v, i);
  SimpleArray<3, int> const &r = a;
  v = 0;
  for (int i : r) ASSERT_EQ(++v, i);
  v = 0;
  for (int const &i : r) ASSERT_EQ(++v, i);
  return true;
}

PYBIND11_MODULE(SimpleArray_test, m) {
  m.def("TEST_SimpleArray_iteration", &TEST_SimpleArray_iteration);
  m.def("TEST_SimpleArray_bounds_check", &TEST_SimpleArray_bounds_check);
}
}  // namespace util
}  // namespace sicdock
