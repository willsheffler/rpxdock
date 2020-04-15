/*/*cppimport
<%


cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w']
cfg['dependencies'] = ['primitive.hpp','../util/assertions.hpp',
'../util/global_rng.hpp','../util/types.hpp']

cfg['parallel'] = False


setup_pybind11(cfg)
%>
*/
/** \file */

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "rpxdock/geom/primitive.hpp"
#include "rpxdock/util/assertions.hpp"
#include "rpxdock/util/global_rng.hpp"

namespace rif_geom_prim_test {
using namespace rpxdock;
using namespace rpxdock::geom;
using namespace rpxdock::util;
using rpxdock::util::global_rng;

template <class F>
bool geom_primitive_sphere_test(int niter) {
  using Sph = Sphere<F>;
  using Vec = V3<F>;
  using S = Sph;
  using V = Vec;
  ASSERT_TRUE(S(V(0, 0, 0), 10).contains(S(V(0, 0, 0), 9.99)));
  ASSERT_TRUE(S(V(0, 0, 0), 10).contains(S(V(8, 0, 0), 1.99)));
  ASSERT_FALSE(S(V(0, 0, 0), 10).contains(S(V(8, 0, 0), 2.01)));
  ASSERT_EQ(S(V(0, 0, 0), 1).merged(S(V(2, 0, 0), 1)), S(V(1, 0, 0), 2));
  ASSERT_EQ(S(V(0, 0, 0), 5).merged(S(V(2, 0, 0), 3)), S(V(0, 0, 0), 5));
  ASSERT_EQ(S(V(0, 0, 0), 5).merged(S(V(5, 0, 0), 2)), S(V(1, 0, 0), 6));
  ASSERT_EQ(S(V(3, 0, 0), 5).merged(S(V(3, 0, 0), 3)), S(V(3, 0, 0), 5));
  ASSERT_EQ(S(V(-2, 0, 0), 2).merged(S(V(-2, 0, 0), 3)), S(V(-2, 0, 0), 3));
  srand(global_rng()());
  for (int iter = 0; iter < niter; ++iter) {
    Vec p1 = Vec::Random();
    Vec p2 = Vec::Random();
    Vec p3 = Vec::Random();
    Vec p4 = Vec::Random();
    F const eps = epsilon2<F>() * 10;
    Sph s1(p1);
    Sph s2(p1, p2);
    Sph s3(p1, p2, p3);
    Sph s4(p1, p2, p3, p4);
    s1.rad -= eps;
    s2.rad -= eps;
    s3.rad -= eps;
    s4.rad -= eps;

    // check on boundary
    ASSERT_LE(fabs(s1.signdis(p1)), eps);

    ASSERT_LE(fabs(s2.signdis(p1)), eps);
    ASSERT_LE(fabs(s2.signdis(p2)), eps);

    ASSERT_LE(fabs(s3.signdis(p1)), eps);
    ASSERT_LE(fabs(s3.signdis(p2)), eps);
    ASSERT_LE(fabs(s3.signdis(p3)), eps);

    ASSERT_LE(fabs(s4.signdis(p1)), (sizeof(F) == 4 ? 100 : 2) * eps);
    ASSERT_LE(fabs(s4.signdis(p2)), (sizeof(F) == 4 ? 100 : 2) * eps);
    ASSERT_LE(fabs(s4.signdis(p3)), (sizeof(F) == 4 ? 100 : 2) * eps);
    ASSERT_LE(fabs(s4.signdis(p4)), (sizeof(F) == 4 ? 100 : 2) * eps);

    if (sizeof(F) > 4) {
      auto sm = s4.merged(s3);
      ASSERT_EQ(s3.merged(s4), s4.merged(s3));
      ASSERT_TRUE(sm.contains(s3));
      ASSERT_TRUE(sm.contains(s4));
      V dir = (s4.cen - s3.cen).normalized();
      ASSERT_FALSE(sm.contains(s4.cen + dir * (s4.rad + 0.01)));
      ASSERT_FALSE(sm.contains(s3.cen - dir * (s4.rad + 0.01)));
    }
  }
  return true;
}

bool TEST_geom_primitive_sphere() {
  bool success = true;
  success &= geom_primitive_sphere_test<float>(10000);
  success &= geom_primitive_sphere_test<double>(10000);
  return success;
}

template <class F>
bool geom_primitive_welzl_test(size_t Ntest, size_t Npoints) {
  using Sph = Sphere<F>;
  using Vec = V3<F>;
  srand(global_rng()());
  F const eps = std::sqrt(std::numeric_limits<F>::epsilon());
  F avgvolratio = 0;
  for (size_t i = 0; i < Ntest; ++i) {
    std::vector<Vec> points;
    for (size_t k = 0; k < Npoints; ++k) points.push_back(10 * Vec::Random());
    Sph aprox_bv = central_bounding_sphere(points);
    Sph welzl_bv = welzl_bounding_sphere(points);
    ASSERT_GE(aprox_bv.rad, welzl_bv.rad);
    F ratio = aprox_bv.rad / welzl_bv.rad;
    avgvolratio += ratio * ratio * ratio;
    for (size_t k = 0; k < points.size(); ++k) {
      ASSERT_LE(welzl_bv.signdis2(points[k]), eps);
      ASSERT_LE(aprox_bv.signdis2(points[k]), eps);
    }
  }
  std::cout << "    average welzl sphere volume improvement over COM: "
            << avgvolratio / Ntest << " Ntest= " << Ntest << std::endl;
  return true;
}

bool TEST_geom_primitive_welzl_bounding_sphere() {
  bool success = true;
  success &= geom_primitive_welzl_test<float>(10, 10);
  success &= geom_primitive_welzl_test<double>(100, 100);
  return success;
}

PYBIND11_MODULE(primitive_test, m) {
  m.def("TEST_geom_primitive_sphere", &TEST_geom_primitive_sphere);
  m.def("TEST_geom_primitive_welzl_bounding_sphere",
        &TEST_geom_primitive_welzl_bounding_sphere);
}

}  // namespace rif_geom_prim_test
