/*cppimport
<%
cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-O1']
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

PyBVH bvh_create(py::array_t<float> xyz) {
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

auto bvh_min_dist_fixed(PyBVH &bvh1, PyBVH &bvh2) {
  PPMin<float> minimizer;
  auto result = tcdock::bvh::BVMinimize(bvh1, bvh2, minimizer);
  return py::make_tuple(result, minimizer.idx1, minimizer.idx2);
}
auto bvh_min_dist(PyBVH &bvh1, PyBVH &bvh2, M44f pos1, M44f pos2) {
  X3f x1(pos1), x2(pos2);
  PPMin<float> minimizer(x1.inverse() * x2);
  auto result = tcdock::bvh::BVMinimize(bvh1, bvh2, minimizer);
  return py::make_tuple(result, minimizer.idx1, minimizer.idx2);
}

auto naive_min_dist_fixed(PyBVH &bvh1, PyBVH &bvh2) {
  double mind2 = 9e9;
  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      mind2 = std::min<double>(mind2, (o1.pos - o2.pos).squaredNorm());
    }
  }
  return std::sqrt(mind2);
}
auto naive_min_dist(PyBVH &bvh1, PyBVH &bvh2, M44f pos1, M44f pos2) {
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

auto bvh_isect_fixed(PyBVH &bvh1, PyBVH &bvh2, double thresh) {
  PPIsect<float> query(thresh);
  tcdock::bvh::BVIntersect(bvh1, bvh2, query);
  return query.result;
}
auto naive_isect_fixed(PyBVH &bvh1, PyBVH &bvh2, double thresh) {
  double dist2 = thresh * thresh;
  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      auto d2 = (o1.pos - o2.pos).squaredNorm();
      if (d2 < dist2) return true;
    }
  }
  return false;
}
auto bvh_isect(PyBVH &bvh1, PyBVH &bvh2, M44f pos1, M44f pos2, double mindist) {
  X3f x1(pos1), x2(pos2);
  PPIsect<float> query(mindist, x1.inverse() * x2);
  tcdock::bvh::BVIntersect(bvh1, bvh2, query);
  return query.result;
}
auto naive_isect(PyBVH &bvh1, PyBVH &bvh2, M44f pos1, M44f pos2,
                 double mindist) {
  X3f x1(pos1), x2(pos2);
  X3f pos = x1.inverse() * x2;
  double dist2 = mindist * mindist;
  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      auto d2 = (o1.pos - pos * o2.pos).squaredNorm();
      if (d2 < dist2) return true;
    }
  }
  return false;
}

template <typename F>
struct BVMinAxis {
  using Scalar = F;
  using Xform = X3<F>;
  Xform bXa = Xform::Identity();
  F minval = 9e9;
  F radius;
  V3f direction;
  BVMinAxis(V3f d, Xform x, F r) : direction(d), bXa(x), radius(r) {
    assert(r > 0);
    direction.normalize();
  }
  F minimumOnVolumeVolume(Sphere<F> r1, Sphere<F> r2) {
    return get_slide(r1.center, bXa * r2.center, r1.radius + radius,
                     r2.radius + radius);
  }
  F minimumOnVolumeObject(Sphere<F> r, PtIdx<F> v) {
    return get_slide(r.center, bXa * v.pos, r.radius + radius, radius);
  }
  F minimumOnObjectVolume(PtIdx<F> v, Sphere<F> r) {
    return get_slide(v.pos, bXa * r.center, radius, r.radius + radius);
  }
  F minimumOnObjectObject(PtIdx<F> a, PtIdx<F> b) {
    // std::cout << "Obj Obj" << std::endl;
    return get_slide(a.pos, bXa * b.pos, radius, radius);
  }
  F get_slide(V3f c1, V3f c2, F r1, F r2) {
    V3f delta = c2 - c1;
    F d0 = delta.dot(direction);
    F dperp2 = delta.squaredNorm() - d0 * d0;
    F target_d2 = (r1 + r2) * (r1 + r2);
    if (target_d2 < dperp2) return 9e9;
    F dpar = std::sqrt(target_d2 - dperp2);
    F moveby = d0 - dpar;
    V3f cnew = c1 + moveby * direction;
    F dnew = (cnew - c2).norm();
    return moveby;
  }
};

auto bvh_slide(PyBVH &bvh1, PyBVH &bvh2, M44f pos1, M44f pos2, double radius,
               V3f direction) {
  X3f x1(pos1), x2(pos2);
  X3f x1inv = x1.inverse();
  X3f pos = x1inv * x2;
  V3f local_dir = x1inv.rotation() * direction;
  BVMinAxis<float> query(local_dir, pos, radius);
  double result = tcdock::bvh::BVMinimize(bvh1, bvh2, query);
  return result;
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
  m.def("bvh_create", &bvh_create);

  m.def("bvh_min_dist", &bvh_min_dist);
  m.def("bvh_min_dist_fixed", &bvh_min_dist_fixed);
  m.def("naive_min_dist", &naive_min_dist);
  m.def("naive_min_dist_fixed", &naive_min_dist_fixed);

  m.def("bvh_isect", &bvh_isect, "intersction test", "bvh1"_a, "bvh2"_a,
        "pos1"_a, "pos2"_a, "mindist"_a);
  m.def("bvh_isect_fixed", &bvh_isect_fixed);

  m.def("naive_isect", &naive_isect);
  m.def("naive_isect_fixed", &naive_isect_fixed);

  m.def("bvh_slide", &bvh_slide, "slide into contact", "bvh1"_a, "bvh2"_a,
        "pos1"_a, "pos2"_a, "radius"_a, "direction"_a);
}

}  // namespace tcdock