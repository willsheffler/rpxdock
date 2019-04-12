/*cppimport
<%
cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17']
cfg['dependencies'] = ['../geom/primitive.hpp','../util/assertions.hpp',
'../util/global_rng.hpp']

setup_pybind11(cfg)
%>
*/

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <iostream>
#include <random>

#include "tcdock/bvh/bvh.hpp"
#include "tcdock/util/Timer.hpp"
#include "tcdock/util/assertions.hpp"
#include "tcdock/util/global_rng.hpp"

using namespace Eigen;
using namespace tcdock;
using namespace util;
using namespace geom;
using namespace bvh;

using F = double;

namespace Eigen {
auto bounding_vol(V3<float> v) { return Sphere<float>(v); }
auto bounding_vol(V3<double> v) { return Sphere<double>(v); }
}  // namespace Eigen

namespace rif_geom_bvh_test {

template <class F, int M, int O>
void rand_xform(std::mt19937& rng, Eigen::Transform<F, 3, M, O>& x,
                float cart_bound = 512.0f) {
  std::uniform_real_distribution<F> runif;
  std::normal_distribution<F> rnorm;
  Eigen::Quaternion<F> qrand(rnorm(rng), rnorm(rng), rnorm(rng), rnorm(rng));
  qrand.normalize();
  x.linear() = qrand.matrix();
  x.translation() = V3<F>(runif(rng) * cart_bound - cart_bound / 2.0,
                          runif(rng) * cart_bound - cart_bound / 2.0,
                          runif(rng) * cart_bound - cart_bound / 2.0);
}

template <class F>
X3<F> rand_xform(F cart_bound = 512.0) {
  X3<F> x;
  rand_xform(global_rng(), x, cart_bound);
  return x;
}

struct PPMin {
  typedef F Scalar;
  using Xform = X3<F>;
  PPMin(Xform x = Xform::Identity()) : bXa(x) {}
  Scalar minimumOnVolumeVolume(Sphere<Scalar> r1, Sphere<Scalar> r2) {
    ++calls;
    return r1.signdis(bXa * r2);
  }
  Scalar minimumOnVolumeObject(Sphere<Scalar> r, V3<F> v) {
    ++calls;
    return r.signdis(bXa * v);
  }
  Scalar minimumOnObjectVolume(V3<F> v, Sphere<Scalar> r) {
    ++calls;
    return (bXa * r).signdis(v);
  }
  Scalar minimumOnObjectObject(V3<F> v1, V3<F> v2) {
    ++calls;
    return (v1 - bXa * v2).norm();
  }
  int calls = 0;
  Xform bXa = Xform::Identity();
  void reset() { calls = 0; }
};

bool test_bvh_test_min() {
  typedef std::vector<V3<F>, aligned_allocator<V3<F>>> StdVectorOfVector3d;
  StdVectorOfVector3d ptsA, ptsB;
  std::uniform_real_distribution<> r(0, 1);
  std::mt19937& g(global_rng());
  for (F dx = 0.91; dx < 1.1; dx += 0.02) {
    StdVectorOfVector3d ptsA, ptsB;
    for (int i = 0; i < 100; ++i) {
      ptsA.push_back(V3<F>(r(g), r(g), r(g)));
      ptsB.push_back(V3<F>(r(g), r(g), r(g)) + V3<F>(dx, 0, 0));
    }

    // brute force
    PPMin minimizer;
    auto tbrute = Timer("tb");
    F brutemin = std::numeric_limits<F>::max();
    // brute force to find closest red-blue pair
    for (int i = 0; i < (int)ptsA.size(); ++i)
      for (int j = 0; j < (int)ptsB.size(); ++j)
        brutemin = std::min(brutemin,
                            minimizer.minimumOnObjectObject(ptsA[i], ptsB[j]));
    tbrute.stop();
    int brutecalls = minimizer.calls;

    // bvh
    // move Pa by random X, set bXa in minimizer
    auto X = rand_xform(F(999));
    for (auto& p : ptsA) p = X * p;
    minimizer.bXa = X;

    minimizer.reset();
    auto tcreate = Timer("tc");
    WelzlBVH<F, V3<F>> bvhA(ptsA.begin(), ptsA.end()),
        bvhB(ptsB.begin(), ptsB.end());  // construct the trees
    tcreate.stop();
    auto tbvh = Timer("tbvh");
    F bvhmin = BVMinimize(bvhA, bvhB, minimizer);
    tbvh.stop();
    int bvhcalls = minimizer.calls;

    ASSERT_FLOAT_EQ(brutemin, bvhmin);

    float ratio = 1. * brutecalls / bvhcalls;
    std::cout << "    min Brute/BVH " << dx << " " << ratio << " " << brutemin
              << " " << bvhmin << " " << brutecalls << " " << bvhcalls << " "
              << tbrute << " " << tcreate << " " << tbvh << " "
              << tbrute.elapsed() / tbvh.elapsed() << std::endl;
  }
  return true;
}

struct PPIsect {
  using Scalar = F;
  using Xform = X3<F>;
  PPIsect(F r, Xform x = Xform::Identity()) : radius(r), bXa(x) {}
  bool intersectVolumeVolume(Sphere<Scalar> r1, Sphere<Scalar> r2) {
    ++calls;
    return r1.signdis(bXa * r2) < radius;
  }
  bool intersectVolumeObject(Sphere<Scalar> r, V3<F> v) {
    ++calls;
    return r.signdis(bXa * v) < radius;
  }
  bool intersectObjectVolume(V3<F> v, Sphere<Scalar> r) {
    ++calls;
    return (bXa * r).signdis(v) < radius;
  }
  bool intersectObjectObject(V3<F> v1, V3<F> v2) {
    ++calls;
    bool isect = (v1 - bXa * v2).norm() < radius;
    result |= isect;
    return isect;
  }
  void reset() {
    calls = 0;
    result = false;
  }
  int calls = 0;
  F radius = 0.0;
  bool result = false;
  Xform bXa = Xform::Identity();
};

bool test_bvh_test_isect() {
  typedef std::vector<V3<F>, aligned_allocator<V3<F>>> StdVectorOfVector3d;
  std::uniform_real_distribution<> r(0, 1);
  std::mt19937& g(global_rng());
  F avg_ratio = 0.0;
  int niter = 0;
  for (F dx = 0.001 + 0.95; dx < 1.05; dx += 0.005) {
    ++niter;

    StdVectorOfVector3d ptsA, ptsB;
    for (int i = 0; i < 100; ++i) {
      ptsA.push_back(V3<F>(r(g), r(g), r(g)));
      ptsB.push_back(V3<F>(r(g), r(g), r(g)) + V3<F>(dx, 0, 0));
    }
    PPIsect query(0.1);

    // brute force
    bool bruteisect = false;
    // brute force to find closest red-blue pair
    auto tbrute = Timer("tb");
    for (int i = 0; i < (int)ptsA.size(); ++i) {
      for (int j = 0; j < (int)ptsB.size(); ++j) {
        if (query.intersectObjectObject(ptsA[i], ptsB[j])) {
          bruteisect = true;
          break;
        }
      }
      if (bruteisect) break;
    }
    int brutecalls = query.calls;
    tbrute.stop();

    // bvh

    query.reset();

    auto X = rand_xform(F(999));
    for (auto& p : ptsA) p = X * p;
    query.bXa = X;  // commenting this out should fail

    auto tcreate = Timer("tc");
    WelzlBVH<F, V3<F>> bvhA(ptsA.begin(), ptsA.end());
    WelzlBVH<F, V3<F>> bvhB(ptsB.begin(), ptsB.end());
    tcreate.stop();
    // std::cout << bvhA.vols[0] << std::endl;
    // std::cout << bvhB.vols[0] << std::endl;

    auto tbvh = Timer("tbvh");
    BVIntersect(bvhA, bvhB, query);
    tbvh.stop();
    bool bvhisect = query.result;
    int bvhcalls = query.calls;

    ASSERT_EQ(bruteisect, bvhisect);

    float ratio = 1. * brutecalls / bvhcalls;
    avg_ratio += ratio;
    std::cout << "    isect Brute/BVH " << dx << " " << ratio << " "
              << bruteisect << " " << bvhisect << " " << brutecalls << " "
              << bvhcalls << " " << tbrute << " " << tcreate << " " << tbvh
              << " " << tbrute.elapsed() / tbvh.elapsed() << std::endl;
  }
  avg_ratio /= niter;
  std::cout << "avg Brute/BVH " << avg_ratio << std::endl;
  return true;
}

PYBIND11_MODULE(bvh_test, m) {
  m.def("test_bvh_test_min", &test_bvh_test_min);
  m.def("test_bvh_test_isect", &test_bvh_test_isect);
}

}  // namespace rif_geom_bvh_test
