/*cppimport
<%
cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w']
cfg['dependencies'] = ['../geom/primitive.hpp','../util/assertions.hpp',
'../util/global_rng.hpp', 'bvh.hpp', 'bvh_algo.hpp']

setup_pybind11(cfg)
%>
*/

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "iostream"

#include "tcdock/bvh/bvh.hpp"
#include "tcdock/util/Timer.hpp"
#include "tcdock/util/assertions.hpp"
#include "tcdock/util/global_rng.hpp"
#include "tcdock/util/types.hpp"

using namespace pybind11::literals;
using namespace Eigen;
using namespace tcdock;
using namespace util;
using namespace geom;
using namespace bvh;

namespace py = pybind11;

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

using PyBVH = tcdock::bvh::WelzlBVH<float, PtIdx<float>>;

namespace tcdock {

PyBVH make_bvh(py::array_t<float> xyz) {
  py::buffer_info buf = xyz.request();
  if (buf.ndim != 2 or buf.shape[1] != 3)
    throw std::runtime_error("Shape must be (N, 3)");
  float *ptr = (float *)buf.ptr;

  typedef std::vector<PtIdx<float>, aligned_allocator<PtIdx<float>>> Holder;
  Holder holder;
  for (int i = 0; i < buf.shape[0]; ++i) {
    float x = ptr[3 * i + 0];
    float y = ptr[3 * i + 1];
    float z = ptr[3 * i + 2];
    // std::cout << "add point " << x << " " << y << " " << z << std::endl;
    holder.push_back(PtIdx<float>(V3<float>(x, y, z), i));
  }
  return PyBVH(holder.begin(), holder.end());
}

template <typename F>
struct PPMin {
  using Scalar = F;
  using Xform = X3<F>;
  int idx1 = -1, idx2 = -1;
  Xform bXa = Xform::Identity();
  float minval = 9e9;
  PPMin(Xform x = Xform::Identity()) : bXa(x) {}
  F minimumOnVolumeVolume(Sphere<F> r1, Sphere<F> r2) {
    return r1.signdis(bXa * r2);
  }
  F minimumOnVolumeObject(Sphere<F> r, PtIdx<F> v) {
    return r.signdis(bXa * v.pos);
  }
  F minimumOnObjectVolume(PtIdx<F> v, Sphere<F> r) {
    return (bXa * r).signdis(v.pos);
  }
  F minimumOnObjectObject(PtIdx<F> a, PtIdx<F> b) {
    F v = (a.pos - bXa * b.pos).norm();
    if (v < minval) {
      // std::cout << v << a.pos.transpose() << " " << b.pos.transpose()
      // << std::endl;
      minval = v;
      idx1 = a.idx;
      idx2 = b.idx;
    }
    return v;
  }
};

auto min_dist_bvh(PyBVH &bvh1, PyBVH &bvh2) {
  PPMin<float> minimizer;
  auto result = tcdock::bvh::BVMinimize(bvh1, bvh2, minimizer);
  return py::make_tuple(result, minimizer.idx1, minimizer.idx2);
}
auto min_dist_bvh_pos(PyBVH &bvh1, PyBVH &bvh2, M44f pos1, M44f pos2) {
  X3f x1(pos1), x2(pos2);
  PPMin<float> minimizer(x1.inverse() * x2);
  auto result = tcdock::bvh::BVMinimize(bvh1, bvh2, minimizer);
  return py::make_tuple(result, minimizer.idx1, minimizer.idx2);
}

auto min_dist_naive(PyBVH &bvh1, PyBVH &bvh2) {
  double mind2 = 9e9;
  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      mind2 = std::min<double>(mind2, (o1.pos - o2.pos).squaredNorm());
    }
  }
  return std::sqrt(mind2);
}
auto min_dist_naive_pos(PyBVH &bvh1, PyBVH &bvh2, M44f pos1, M44f pos2) {
  X3f x1(pos1), x2(pos2);
  X3f pos = x1.inverse() * x2;
  double mind2 = 9e9;
  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      mind2 = std::min<double>(mind2, (o1.pos - pos * o2.pos).squaredNorm());
    }
  }
  return std::sqrt(mind2);
}

template <typename F>
struct PPIsect {
  using Scalar = F;
  using Xform = X3<F>;
  PPIsect(F r, Xform x = Xform::Identity())
      : radius(r), radius2(r * r), bXa(x) {}
  bool intersectVolumeVolume(Sphere<F> r1, Sphere<F> r2) {
    return r1.signdis(bXa * r2) < radius;
  }
  bool intersectVolumeObject(Sphere<F> r, PtIdx<F> v) {
    return r.signdis(bXa * v.pos) < radius;
  }
  bool intersectObjectVolume(PtIdx<F> v, Sphere<F> r) {
    return (bXa * r).signdis(v.pos) < radius;
  }
  bool intersectObjectObject(PtIdx<F> v1, PtIdx<F> v2) {
    bool isect = (v1.pos - bXa * v2.pos).squaredNorm() < radius2;
    // bool isect = (v1.pos - bXa * v2.pos).norm() < radius;
    result |= isect;
    return isect;
  }
  F radius = 0, radius2 = 0;
  bool result = false;
  Xform bXa = Xform::Identity();
};

auto isect_bvh(PyBVH &bvh1, PyBVH &bvh2, double thresh) {
  PPIsect<float> query(thresh);
  tcdock::bvh::BVIntersect(bvh1, bvh2, query);
  return query.result;
}
auto isect_naive(PyBVH &bvh1, PyBVH &bvh2, double thresh) {
  double dist2 = thresh * thresh;
  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      auto d2 = (o1.pos - o2.pos).squaredNorm();
      if (d2 <= dist2) return true;
    }
  }
  return false;
}
auto isect_bvh_pos(PyBVH &bvh1, PyBVH &bvh2, double thresh, M44f pos1,
                   M44f pos2) {
  X3f x1(pos1), x2(pos2);
  PPIsect<float> query(thresh, x1.inverse() * x2);
  tcdock::bvh::BVIntersect(bvh1, bvh2, query);
  return query.result;
}
auto isect_naive_pos(PyBVH &bvh1, PyBVH &bvh2, double thresh, M44f pos1,
                     M44f pos2) {
  X3f x1(pos1), x2(pos2);
  X3f pos = x1.inverse() * x2;
  double dist2 = thresh * thresh;
  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      auto d2 = (o1.pos - pos * o2.pos).squaredNorm();
      if (d2 <= dist2) return true;
    }
  }
  return false;
}

template <typename F>
struct PPCollect {
  using Scalar = F;
  using Xform = X3<F>;
  PPCollect(F r, Xform x = Xform::Identity()) : radius(r), bXa(x) {}
  bool intersectVolumeVolume(Sphere<F> r1, Sphere<F> r2) {
    return r1.signdis(bXa * r2) < radius;
  }
  bool intersectVolumeObject(Sphere<F> r, PtIdx<F> v) {
    return r.signdis(bXa * v.pos) < radius;
  }
  bool intersectObjectVolume(PtIdx<F> v, Sphere<F> r) {
    return (bXa * r).signdis(v.pos) < radius;
  }
  bool intersectObjectObject(PtIdx<F> v1, PtIdx<F> v2) {
    bool isect = (v1.pos - bXa * v2.pos).norm() < radius;
    return false;
  }
  F radius = 0.0;
  bool result = false;
  Xform bXa = Xform::Identity();
};

PYBIND11_MODULE(bvh, m) {
  py::class_<PyBVH>(m, "WelzlBVH", py::module_local());
  m.def("make_bvh", &make_bvh);
  m.def("min_dist_bvh", &min_dist_bvh);
  m.def("min_dist_bvh_pos", &min_dist_bvh_pos);
  m.def("min_dist_naive", &min_dist_naive);
  m.def("min_dist_naive_pos", &min_dist_naive_pos);
  m.def("isect_bvh", &isect_bvh);
  m.def("isect_naive", &isect_naive);
  m.def("isect_bvh_pos", &isect_bvh_pos);
  m.def("isect_naive_pos", &isect_naive_pos);
}

}  // namespace tcdock