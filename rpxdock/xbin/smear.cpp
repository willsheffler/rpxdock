/*/*cppimport
<%


cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['xbin.hpp', '../phmap/phmap.hpp', 'smear.hpp',
'../geom/bcc.hpp']

cfg['parallel'] = False


setup_pybind11(cfg)
%>
*/
/** \file */

#include "rpxdock/xbin/smear.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;

namespace rpxdock {
namespace xbin {

template <typename F, typename K, typename V>
void bind_smear(py::module &m, std::string name) {
  m.def("smear", &smear<F, K, V>, "smear out xmap into neighbor cells",
        "xbin"_a, "phmap"_a, "radius"_a = 1, "extrahalf"_a = false,
        "oddlast3"_a = true, "sphere"_a = true, "kernel"_a = Vx<V>(),
        py::call_guard<py::gil_scoped_release>());
}

PYBIND11_MODULE(smear, m) {
  bind_smear<double, uint64_t, double>(m, "smear");
  bind_smear<float, uint64_t, double>(m, "smear");
  bind_smear<double, uint64_t, float>(m, "smear");
  bind_smear<float, uint64_t, float>(m, "smear");
}

}  // namespace xbin
}  // namespace rpxdock