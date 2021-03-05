/*/*cppimport
<%
cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-O1','--verbose']

cfg['parallel'] = False

setup_pybind11(cfg)
%>
*/
/** \file */

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>

#include "rpxdock/pack/HackPack.hh"
// #include "rpxdock/pack/TwoBodyTable.hh"
#include "rpxdock/util/numeric.hpp"
#include "rpxdock/util/pybind_types.hpp"
#include "rpxdock/util/types.hpp"

namespace rpxdock {
namespace pack {

namespace py = pybind11;
using namespace py::literals;

using std::cout;
using std::endl;

using RotID = std::pair<int32_t, int32_t>;

void foo() {
  cout << "create TwoBodyTable" << endl;
  auto pairtable = std::make_shared<TwoBodyTable<float>>(13, 7);

  // init onebody

  cout << "create init_onebody_filter" << endl;
  pairtable->init_onebody_filter(1);

  // init twobody

  HackPackOpts opts;
  cout << "create packer" << endl;
  HackPack packer(opts, 0);

  cout << "reinitialize packer" << endl;
  packer.reinitialize(pairtable);

  cout << "packer.pack" << endl;
  auto resultrots = std::vector<RotID>();
  packer.pack(resultrots);

  cout << "result rots " << resultrots.size() << endl;
  for (auto rotid : resultrots) {
    cout << "HackPack: " << rotid.first << ' ' << rotid.second << std::endl;
  }
}

PYBIND11_MODULE(_annealer, m) { m.def("foo", &foo); }

}  // namespace pack
}  // namespace rpxdock