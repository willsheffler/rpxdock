/*cppimport
<%
cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-O1']
cfg['dependencies'] = ['xbin.hpp', '../phmap/phmap.hpp', 'smear.hpp',
'../geom/bcc.hpp']

setup_pybind11(cfg)
%>
*/

#include "sicdock/xbin/smear.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;

namespace sicdock {
namespace xbin {

PYBIND11_MODULE(smear, m) { m.def("smear", &smear<double, uint64_t, double>); }

}  // namespace xbin
}  // namespace sicdock