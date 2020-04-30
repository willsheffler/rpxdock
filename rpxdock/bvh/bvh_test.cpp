/*/*cppimport
<%


cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w']
cfg['dependencies'] = ['../geom/primitive.hpp','../util/assertions.hpp',
'../util/global_rng.hpp']

cfg['parallel'] = False


setup_pybind11(cfg)
%>
*/
/** \file */

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <iostream>
#include <random>

#include "rpxdock/bvh/bvh.hpp"
#include "rpxdock/util/Timer.hpp"
#include "rpxdock/util/assertions.hpp"
#include "rpxdock/util/global_rng.hpp"

using namespace Eigen;
using namespace rpxdock;
using namespace util;
using namespace geom;
using namespace bvh;

using F = double;

namespace Eigen {
template <class F>
struct PtIdx {
  PtIdx() : pos(0), idx(0) {}
  PtIdx(V3<F> v, int i = 0) : pos(v), idx(i) {}
  V3<F> pos;
  int idx;
};
auto bounding_vol(V3<float> v) { return Sphere<float>(v); }
auto bounding_vol(V3<double> v) { return Sphere<double>(v); }
auto bounding_vol(PtIdx<float> v) { return Sphere<float>(v.pos); }
auto bounding_vol(PtIdx<double> v) { return Sphere<double>(v.pos); }
}  // namespace Eigen

namespace rpxdock_geom_bvh_test {
using PtIdxF = PtIdx<F>;

template <class F, int M, int O>
void rand_xform(std::mt19937& rng, Eigen::Transform<F, 3, M, O>& x,
                float max_cart = 512.0f) {
  std::uniform_real_distribution<F> runif;
  std::normal_distribution<F> rnorm;
  Eigen::Quaternion<F> qrand(rnorm(rng), rnorm(rng), rnorm(rng), rnorm(rng));
  qrand.normalize();
  x.linear() = qrand.matrix();
  x.translation() = V3<F>(runif(rng) * max_cart - max_cart / 2.0,
                          runif(rng) * max_cart - max_cart / 2.0,
                          runif(rng) * max_cart - max_cart / 2.0);
}

template <class F>
X3<F> rand_xform(F max_cart = 512.0) {
  X3<F> x;
  rand_xform(global_rng(), x, max_cart);
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
  Scalar minimumOnVolumeObject(Sphere<Scalar> r, PtIdxF v) {
    ++calls;
    return r.signdis(bXa * v.pos);
  }
  Scalar minimumOnObjectVolume(PtIdxF v, Sphere<Scalar> r) {
    ++calls;
    return (bXa * r).signdis(v.pos);
  }
  Scalar minimumOnObjectObject(PtIdxF a, PtIdxF b) {
    ++calls;
    return (a.pos - bXa * b.pos).norm();
  }
  int calls = 0;
  Xform bXa = Xform::Identity();
  void reset() { calls = 0; }
};

bool TEST_bvh_test_min() {
  typedef std::vector<PtIdxF, aligned_allocator<PtIdxF>> StdVectorOfVector3d;
  StdVectorOfVector3d ptsA, ptsB;
  std::uniform_real_distribution<> r(0, 1);
  std::mt19937& g(global_rng());
  for (F dx = 0.91; dx < 1.1; dx += 0.02) {
    StdVectorOfVector3d ptsA, ptsB;
    for (int i = 0; i < 100; ++i) {
      ptsA.push_back(PtIdxF(V3<F>(r(g), r(g), r(g)), i));
      ptsB.push_back(PtIdxF(V3<F>(r(g), r(g), r(g)) + V3<F>(dx, 0, 0), i));
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
    for (auto& p : ptsA) p.pos = X * p.pos;
    minimizer.bXa = X;

    minimizer.reset();
    auto tcreate = Timer("tc");
    SphereBVH<F, PtIdxF> bvhA(ptsA.begin(), ptsA.end());
    SphereBVH<F, PtIdxF> bvhB(ptsB.begin(), ptsB.end());
    tcreate.stop();
    auto tbvh = Timer("tbvh");
    F bvhmin = BVMinimize(bvhA, bvhB, minimizer);
    tbvh.stop();
    int bvhcalls = minimizer.calls;

    ASSERT_FLOAT_EQ(brutemin, bvhmin);

    float ratio = 1. * brutecalls / bvhcalls;
    // std::cout << "    min Brute/BVH " << dx << " " << ratio << " " <<
    // brutemin
    //           << " " << bvhmin << " " << brutecalls << " " << bvhcalls << " "
    //           << tbrute << " " << tcreate << " " << tbvh << " "
    //           << tbrute.elapsed() / tbvh.elapsed() << std::endl;
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
  bool intersectVolumeObject(Sphere<Scalar> r, PtIdxF v) {
    ++calls;
    return r.signdis(bXa * v.pos) < radius;
  }
  bool intersectObjectVolume(PtIdxF v, Sphere<Scalar> r) {
    ++calls;
    return (bXa * r).signdis(v.pos) < radius;
  }
  bool intersectObjectObject(PtIdxF v1, PtIdxF v2) {
    ++calls;
    bool isect = (v1.pos - bXa * v2.pos).norm() < radius;
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

bool TEST_bvh_test_isect() {
  typedef std::vector<PtIdxF, aligned_allocator<PtIdxF>> StdVectorOfVector3d;
  std::uniform_real_distribution<> r(0, 1);
  std::mt19937& g(global_rng());
  F avg_ratio = 0.0;
  int niter = 0;
  for (F dx = 0.001 + 0.95; dx < 1.05; dx += 0.005) {
    ++niter;

    StdVectorOfVector3d ptsA, ptsB;
    for (int i = 0; i < 100; ++i) {
      ptsA.push_back(PtIdxF(V3<F>(r(g), r(g), r(g)), i));
      ptsB.push_back(PtIdxF(V3<F>(r(g), r(g), r(g)) + V3<F>(dx, 0, 0), i));
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
    for (auto& p : ptsA) p.pos = X * p.pos;
    query.bXa = X;  // commenting this out should fail

    auto tcreate = Timer("tc");
    SphereBVH<F, PtIdxF> bvhA(ptsA.begin(), ptsA.end());
    SphereBVH<F, PtIdxF> bvhB(ptsB.begin(), ptsB.end());
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
    // std::cout << "    isect Brute/BVH " << dx << " " << ratio << " "
    //           << bruteisect << " " << bvhisect << " " << brutecalls << " "
    //           << bvhcalls << " " << tbrute << " " << tcreate << " " << tbvh
    //           << " " << tbrute.elapsed() / tbvh.elapsed() << std::endl;
  }
  avg_ratio /= niter;
  std::cout << "avg Brute/BVH " << avg_ratio << std::endl;
  return true;
}

PYBIND11_MODULE(bvh_test, m) {
  m.def("TEST_bvh_test_min", &TEST_bvh_test_min);
  m.def("TEST_bvh_test_isect", &TEST_bvh_test_isect);
}

}  // namespace rpxdock_geom_bvh_test
