/*cppimport
<%
cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17']
cfg['dependencies'] = ['../geom/primitive.hpp','../util/assertions.hpp',
'../util/global_rng.hpp', 'bvh.hpp', 'bvh_algo.hpp']

setup_pybind11(cfg)
%>
*/

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "iostream"

#include "tcdock/bvh/bvh.hpp"
#include "tcdock/util/Timer.hpp"
#include "tcdock/util/assertions.hpp"
#include "tcdock/util/global_rng.hpp"

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
  int idx1, idx2;
  float minval;
  PPMin(Xform x = Xform::Identity()) : bXa(x), minval(9e9) {}
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
  Xform bXa = Xform::Identity();
};

auto bvh_min_dist(PyBVH &bvh1, PyBVH &bvh2) {
  PPMin<float> minimizer;
  auto result = tcdock::bvh::BVMinimize(bvh1, bvh2, minimizer);
  return py::make_tuple(result, minimizer.idx1, minimizer.idx2);
}

PYBIND11_MODULE(bvh, m) {
  py::class_<PyBVH>(m, "WelzlBVH");
  m.def("make_bvh", &make_bvh);
  m.def("bvh_min_dist", &bvh_min_dist);
}

}  // namespace tcdock