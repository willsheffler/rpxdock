/*/*cppimport
<%


cfg['include_dirs'] = ['../..']
cfg['compiler_args'] = ['-std=c++17', '-w']
cfg['dependencies'] = ['dilated_int.hpp']

cfg['parallel'] = False


setup_pybind11(cfg)
%>
*/
/** \file */

/// @brief  Z-order or Morton style indexing utilities for arbitrary dimension
/// @author will sheffler

// inspired by code from here: http://www.marcusbannerman.co.uk/dynamo
// see "Converting to and from Dilated Integers"(doi:10.1109/TC.2007.70814)

#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "rpxdock/util/assertions.hpp"
#include "rpxdock/util/dilated_int.hpp"

namespace rpxdock {
namespace util {

template <uint64_t D>
bool dilated_int_test() {
  uint64_t maxval = util::impl::MAX_DILATABLE<D>::VAL;
  maxval = std::min(maxval, uint64_t(2097151));
  for (uint64_t i(0); i < maxval; ++i) {
    uint64_t dilated = util::dilate<D>(i);
    uint64_t undilated = util::undilate<D>(dilated);
    ASSERT_EQ(undilated, i);
  }
  return true;
}

bool TEST_dilated_int_64bit() {
  bool pass = true;
  pass &= util::dilated_int_test<1>();
  pass &= util::dilated_int_test<2>();
  pass &= util::dilated_int_test<3>();
  pass &= util::dilated_int_test<4>();
  pass &= util::dilated_int_test<5>();
  pass &= util::dilated_int_test<6>();
  pass &= util::dilated_int_test<7>();
  pass &= util::dilated_int_test<8>();
  pass &= util::dilated_int_test<9>();
  pass &= util::dilated_int_test<10>();
  pass &= util::dilated_int_test<11>();
  pass &= util::dilated_int_test<12>();
  pass &= util::dilated_int_test<13>();
  pass &= util::dilated_int_test<14>();
  pass &= util::dilated_int_test<15>();
  pass &= util::dilated_int_test<16>();
  pass &= util::dilated_int_test<17>();
  pass &= util::dilated_int_test<18>();
  pass &= util::dilated_int_test<19>();
  pass &= util::dilated_int_test<20>();
  pass &= util::dilated_int_test<21>();
  pass &= util::dilated_int_test<22>();
  pass &= util::dilated_int_test<23>();
  pass &= util::dilated_int_test<24>();
  pass &= util::dilated_int_test<25>();
  pass &= util::dilated_int_test<26>();
  pass &= util::dilated_int_test<27>();
  pass &= util::dilated_int_test<28>();
  pass &= util::dilated_int_test<29>();
  pass &= util::dilated_int_test<30>();
  pass &= util::dilated_int_test<31>();
  pass &= util::dilated_int_test<32>();
  // above 32 makes no sense for 64 bit
  return pass;
}

PYBIND11_MODULE(dilated_int_test, m) {
  m.def("TEST_dilated_int_64bit", &TEST_dilated_int_64bit);
}
}  // namespace util
}  // namespace rpxdock
