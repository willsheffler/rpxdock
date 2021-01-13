/*/*cppimport
<%
cfg['include_dirs'] = ['../..', '../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = []
cfg['parallel'] = False
setup_pybind11(cfg)
%>
*/
/** \file */

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "parallel_hashmap/phmap.h"
#include "rpxdock/util/global_rng.hpp"
#include "rpxdock/util/numeric.hpp"
#include "rpxdock/util/pybind_types.hpp"
#include "rpxdock/xbin/xbin.hpp"

namespace py = pybind11;
using namespace py::literals;
using namespace Eigen;

#include <iostream>
#include <unordered_set>

using std::cout;
using std::endl;

namespace rpxdock {
namespace geom {

using namespace util;
using namespace Eigen;

template <typename F>
py::tuple expand_xforms_rand(std::vector<py::array_t<F>> gen_in, int depth,
                             int ntrials, F radius, V3<F> cen,
                             F reset_radius_ratio) {
  std::uniform_int_distribution<> randidx(0, gen_in.size() - 1);

  using X = X3<F>;
  std::vector<X> gen;
  for (auto g : gen_in) gen.push_back(xform_py_to_X3(g));
  xbin::XformHash_bt24_BCC6<X, uint64_t> binner(0.1654, 1.74597, 107.745);
  ::phmap::parallel_flat_hash_map<uint64_t, uint64_t> uniq_keys;
  Mx<X> frames(depth, ntrials);

  for (int idepth = 0; idepth < depth; ++idepth) {
    for (int itrial = 0; itrial < ntrials; ++itrial) {
      // compute new frame
      X xdelta = gen[randidx(global_rng())];
      X& x = frames(idepth, itrial);
      x = idepth == 0 ? xdelta : xdelta * frames(idepth - 1, itrial);
      F dist2cen = (x.translation() - cen).squaredNorm();
      if (dist2cen > sqr(reset_radius_ratio) * sqr(radius)) {
        x = frames(idepth / 2, itrial);  // reset to prior location in bounds
        continue;
      } else if (dist2cen > sqr(radius)) {
        continue;  // out of bounds, don't record
      }
      // record frame
      int id = idepth * ntrials + itrial;
      uniq_keys.insert_or_assign(binner.get_key(x), id);
    }
  }
  std::vector<X> frames_uniq_key;
  for (auto& [k, v] : uniq_keys) {
    int idepth = v / ntrials;
    int itrial = v % ntrials;
    frames_uniq_key.push_back(frames(idepth, itrial));
  }
  std::vector<X> frames_uniq;
  for (int i = 0; i < frames_uniq_key.size(); ++i) {
    bool uniq = true;
    auto x = frames_uniq_key[i];
    for (int j = 0; j < i; ++j) {
      auto y = frames_uniq_key[j];
      auto y2x = x * y.inverse();
      Matrix<F, 3, 3> m = y2x.linear();
      if (y2x.translation().squaredNorm() > 0.0001) continue;
      if (AngleAxis<F>(m).angle() < 0.0001) uniq = false;
    }
    if (uniq) frames_uniq.push_back(x);
  }
  Vx<X> ret(frames_uniq.size());
  for (int i = 0; i < frames_uniq.size(); ++i) ret[i] = frames_uniq[i];
  return py::make_tuple(xform_eigen_to_py(ret), 0);
}

// ZOMG...
template <typename F>
py::tuple expand_xforms_2gen_loop(X3<F> G1, X3<F> G2, int N = 5, F maxrad = 9e9,
                                  F maxrad_intermediate = 9e9) {
  // auto xbuf = Vx<X3<F>>(10000);  // reasonable maximum?
  auto xbuf = std::vector<X3<F>>();

  xbin::XformHash_bt24_BCC6<X3<F>, uint64_t> binner(0.979837589843, 10.0,
                                                    1000.0);
  // V4<F> representative(3.78593478659, 7.72308475, 13.234075748, 1);
  X3<F> I = X3<F>::Identity();

  std::unordered_set<uint64_t> seenit;

  int count = 0;
  for (int i0a = 0; i0a < 2; ++i0a) {
    X3<F> x0a((i0a == 1 ? G1 : I) * I);
    for (int i0b = 0; i0b < 2; ++i0b) {
      X3<F> x0b((i0b == 1 ? G2 : I) * x0a);
      for (int i1a = 0; i1a < 2; ++i1a) {
        X3<F> x1a((i1a == 1 ? G1 : I) * x0b);
        for (int i1b = 0; i1b < 2; ++i1b) {
          X3<F> x1b((i1b == 1 ? G2 : I) * x1a);
          for (int i2a = 0; i2a < 2; ++i2a) {
            X3<F> x2a((i2a == 1 ? G1 : I) * x1b);
            for (int i2b = 0; i2b < 2; ++i2b) {
              X3<F> x2b((i2b == 1 ? G2 : I) * x2a);
              for (int i3a = 0; i3a < 2; ++i3a) {
                X3<F> x3a((i3a == 1 ? G1 : I) * x2b);
                for (int i3b = 0; i3b < 2; ++i3b) {
                  X3<F> x3b((i3b == 1 ? G2 : I) * x3a);
                  for (int i4a = 0; i4a < 2; ++i4a) {
                    X3<F> x4a((i4a == 1 ? G1 : I) * x3b);
                    for (int i4b = 0; i4b < 2; ++i4b) {
                      X3<F> x4b((i4b == 1 ? G2 : I) * x4a);
                      for (int i5a = 0; i5a < 2; ++i5a) {
                        X3<F> x5a((i5a == 1 ? G1 : I) * x4b);
                        for (int i5b = 0; i5b < 2; ++i5b) {
                          X3<F> x5b((i5b == 1 ? G2 : I) * x5a);
                          for (int i6a = 0; i6a < 2; ++i6a) {
                            X3<F> x6a((i6a == 1 ? G1 : I) * x5b);
                            for (int i6b = 0; i6b < 2; ++i6b) {
                              X3<F> x6b((i6b == 1 ? G2 : I) * x6a);
                              for (int i7a = 0; i7a < 2; ++i7a) {
                                X3<F> x7a((i7a == 1 ? G1 : I) * x6b);
                                for (int i7b = 0; i7b < 2; ++i7b) {
                                  X3<F> x7b((i7b == 1 ? G2 : I) * x7a);
                                  for (int i8a = 0; i8a < 2; ++i8a) {
                                    X3<F> x8a((i8a == 1 ? G1 : I) * x7b);
                                    for (int i8b = 0; i8b < 2; ++i8b) {
                                      X3<F> x8b((i8b == 1 ? G2 : I) * x8a);
                                      for (int i9a = 0; i9a < 2; ++i9a) {
                                        X3<F> x9a((i9a == 1 ? G1 : I) * x8b);
                                        for (int i9b = 0; i9b < 2; ++i9b) {
                                          X3<F> x9b((i9b == 1 ? G2 : I) * x9a);
                                          for (int iAa = 0; iAa < 2; ++iAa) {
                                            X3<F> xAa((i7a == 1 ? G1 : I) *
                                                      x9b);
                                            for (int iAb = 0; iAb < 2; ++iAb) {
                                              X3<F> xAb((i7b == 1 ? G2 : I) *
                                                        xAa);
                                              for (int iBa = 0; iBa < 2;
                                                   ++iBa) {
                                                X3<F> xBa((i8a == 1 ? G1 : I) *
                                                          xAb);
                                                for (int iBb = 0; iBb < 2;
                                                     ++iBb) {
                                                  X3<F> xBb(
                                                      (i8b == 1 ? G2 : I) *
                                                      xBa);
                                                  for (int iCa = 0; iCa < 2;
                                                       ++iCa) {
                                                    X3<F> xCa(
                                                        (i9a == 1 ? G1 : I) *
                                                        xBb);
                                                    for (int iCb = 0; iCb < 2;
                                                         ++iCb) {
                                                      X3<F> xCb(
                                                          (i9b == 1 ? G2 : I) *
                                                          xCa);
                                                      X3<F> x(xCb);
                                                      if (x.translation()
                                                              .squaredNorm() >
                                                          maxrad_intermediate *
                                                              maxrad_intermediate)
                                                        continue;
                                                      // V4<F> rep =
                                                      // x * representative;
                                                      count++;

                                                      // offest to aviod cell
                                                      // boundary crap (?)
                                                      auto xtmp = x;
                                                      xtmp.translation() +=
                                                          V3<F>(0.001312,
                                                                0.0097834,
                                                                0.0047847529);
                                                      uint64_t bin =
                                                          binner.get_key(xtmp);
                                                      if (seenit.find(bin) ==
                                                          seenit.end()) {
                                                        // if (true) {
                                                        // print(depth, seenit,
                                                        // "new bin ", 0);
                                                        // cout << bin << endl;
                                                        xbuf.push_back(x);
                                                        seenit.insert(bin);
                                                        // print(depth, seenit,
                                                        // "recurse pos2");b
                                                      }
                                                      // bool
                                                      // redundant(false); for
                                                      // (auto frorigin :
                                                      // seenit) {
                                                      // if ((frorigin - rep)
                                                      // .squaredNorm() <
                                                      // 0.0001)
                                                      // redundant = true;
                                                      // }
                                                      // cout << x(0, 0) << '
                                                      // '
                                                      // << x(0, 1) << ' '
                                                      // << seenit.size()
                                                      // << endl;
                                                      // if (redundant)
                                                      // continue;
                                                    }
                                                  }
                                                  if (1 >= N) goto DONE;
                                                }
                                              }
                                              if (2 >= N) goto DONE;
                                            }
                                          }
                                          if (3 >= N) goto DONE;
                                        }
                                      }
                                      if (4 >= N) goto DONE;
                                    }
                                  }
                                  if (5 >= N) goto DONE;
                                }
                              }
                              if (6 >= N) goto DONE;
                            }
                          }
                          if (7 >= N) goto DONE;
                        }
                      }
                      if (8 >= N) goto DONE;
                    }
                  }
                  if (9 >= N) goto DONE;
                }
              }
              if (10 >= N) goto DONE;
            }
          }
          if (11 >= N) goto DONE;
        }
      }
      if (12 >= N) goto DONE;
    }
  }
DONE:

  Vx<X3<F>> xuniq = Vx<X3<F>>(10000);  // reasonable maximum?
  size_t n = 0;
  for (auto x : xbuf) {
    if (x.translation().squaredNorm() > maxrad * maxrad) continue;
    bool redundant(false);
    for (size_t i = 0; i < n; ++i) {
      auto x2 = xuniq[i];
      auto cartdis = (x2.translation() - x.translation()).squaredNorm();
      auto oridis = (x2.linear() - x.linear()).squaredNorm();
      if (cartdis < 1e-6 && oridis < 1e-6) redundant = true;
    }
    if (!redundant) {
      xuniq[n] = x;
      ++n;
    }
  }
  // if (seenit.size() != n)
  // cout << "CPP  N=" << N << " seenit.size() " << seenit.size() << " uniq " <<
  // n
  // << endl;
  assert(seenit.size() >= n);

  auto ret = xform_eigen_to_py(xuniq, n);

  return py::make_tuple(ret, py::make_tuple((int)seenit.size(), count));
}  // namespace rpxdock

template <typename F>
py::tuple expand_xforms_2gen_loop_pyarray(py::array_t<F> G1, py::array_t<F> G2,
                                          int N = 5, F maxrad = 9e9,
                                          F maxrad_intermediate = 9e9) {
  auto g1 = xform_py_to_X3(G1);
  auto g2 = xform_py_to_X3(G2);
  return expand_xforms_2gen_loop(g1, g2, N, maxrad, maxrad_intermediate);
}

void print(int depth, std::unordered_set<uint64_t> const& seenit,
           std::string msg, bool newline = 1) {
  for (int i = 0; i < 6 - depth; ++i) cout << "  ";
  cout << seenit.size() << " " << msg;
  if (newline) cout << endl;
}

template <typename F, typename Binner>
void expx(std::vector<X3<F>> generators, Binner& binner, F maxrad_sq, X3<F> pos,
          int depth, Vx<X3<F>>& xbuf, std::unordered_set<uint64_t>& seenit,
          int& count) {
  if (!depth) {
    // print(depth, seenit, "depth==0");
    return;
  }
  ++count;

  // print(depth, seenit, "recurse nochange");
  expx(generators, binner, maxrad_sq, pos, depth - 1, xbuf, seenit, count);

  auto g = generators[depth % generators.size()];
  auto pos2 = pos * g;
  if (pos2.translation().squaredNorm() > maxrad_sq) {
    print(depth, seenit, "out of bounds");
    return;
  }
  uint64_t bin = binner.get_key(pos2);
  if (seenit.find(bin) == seenit.end()) {
    print(depth, seenit, "new bin ", 0);
    cout << bin << endl;
    xbuf[seenit.size()] = pos2;
    seenit.insert(bin);
    // print(depth, seenit, "recurse pos2");b
    expx(generators, binner, maxrad_sq, pos2, depth - 1, xbuf, seenit, count);
  } else {
    print(depth, seenit, "old bin");
  }
}

// template <typename F>
// py::array_t<F> expand_xforms(py::array_t<F> G1in, py::array_t<F> G2in,
// int N = 5, F maxrad = 9e9) {
// X3<F> G1 = xform_py_to_X3(G1in);
// X3<F> G2 = xform_py_to_X3(G2in);
// std::vector<X3<F>> G = std::vector<X3<F>>{G1, G2};
//
// Vx<X3<F>> xbuf = Vx<X3<F>>(10000);  // reasonable maximum?
//
// xbin::XformHash_bt24_BCC6<X3<F>, uint64_t> binner(0.979837589843, 10.0,
// 1000.0);
// F maxrad_sq = maxrad * maxrad;
// int count;
// std::unordered_set<uint64_t> seenit;
//
// print(N, seenit, "initial recurse");
// expx(G, binner, maxrad_sq, X3<F>::Identity(), N, xbuf, seenit, count);
//
// cout << "seenit " << seenit.size() << endl;
//
// Vx<M4<F>> xout = Vx<M4<F>>(seenit.size());
// std::vector<V3<F>> seenit2;
// for (int i = 0; i < seenit.size(); ++i) {
// X3<F> x(xbuf[i]);
// bool redundant(false);
// for (auto frorigin : seenit2) {
// if ((frorigin - x.translation()).squaredNorm() < 0.0001) {
// redundant = true;
// break;
// }
// }
// if (redundant) continue;
// xout[seenit2.size()] = x.matrix();
// seenit2.push_back(x.translation());
// }
//
// auto ret = xform_eigen_to_py(xout, seenit.size());
// return ret;
// }

template <typename F>
py::tuple expand_xforms(py::array_t<F> generators, int N = 5, F maxrad = 9e9,
                        F maxrad_intermediate = 9e9) {
  auto gen = xform_py_to_eigen(generators);

  // cout << "GEN " << gen.size() << endl;
  // cout << gen[0].translation() << endl;
  // cout << gen[1].translation() << endl;

  if (gen.size() == 2) {
    return expand_xforms_2gen_loop(gen[0], gen[1], N, maxrad,
                                   maxrad_intermediate);
  } else {
    cout << "only support 2 generators currently" << endl;
  }
}

PYBIND11_MODULE(expand_xforms, m) {
  // m.def("expand_xforms_loop", &expand_xforms_loop<double>);

  m.def("expand_xforms", &expand_xforms<double>, "", "generators"_a, "depth"_a,
        "radius"_a = 9e9, "maxrad_intermediate"_a = 9e9);

  m.def("expand_xforms_2gen_loop_pyarray",
        &expand_xforms_2gen_loop_pyarray<double>, "", "g1"_a, "g2"_a, "depth"_a,
        "radius"_a = 9e9, "maxrad_intermediate"_a = 9e9);

  m.def("expand_xforms_rand", &expand_xforms_rand<double>, "", "generators"_a,
        "depth"_a = 1000, "trials"_a = 1000, "radius"_a = 9e9,
        "cen"_a = Vector3d(0, 0, 0), "reset_radius_ratio"_a = 9e9);
}

}  // namespace geom
}  // namespace rpxdock
