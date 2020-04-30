/*/*cppimport
<%


cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w']
cfg['dependencies'] = ['xbin.hpp', '../util/assertions.hpp',
'../util/global_rng.hpp']

cfg['parallel'] = False


setup_pybind11(cfg)
%>
*/
/** \file */

#include "rpxdock/xbin/xbin.hpp"

#include <pybind11/pybind11.h>

#include <Eigen/Geometry>
#include <random>
#include <unordered_set>

#include "rpxdock/util/Timer.hpp"
#include "rpxdock/util/assertions.hpp"
#include "rpxdock/util/global_rng.hpp"
#include "rpxdock/util/types.hpp"
namespace py = pybind11;

namespace rpxdock {
using namespace util;

namespace xbin {
namespace test {

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

using std::cout;
using std::endl;

typedef Eigen::Transform<double, 3, Eigen::AffineCompact> Xform;
// typedef Eigen::Affine3d Xform;

template <template <class X> class XformHash>
int get_num_ori_cells(int ori_nside, double& xcov) {
  std::mt19937 rng((unsigned int)time(0) + 7693487);
  XformHash<Xform> xh(1.0, ori_nside, 512.0);
  int n_ori_bins;
  {
    std::unordered_set<size_t> idx_seen;
    // idx_seen.set_empty_key(std::numeric_limits<uint64_t>::max());

    int NSAMP = std::max(1000000, 500 * ori_nside * ori_nside * ori_nside);
    Xform x;
    for (int i = 0; i < NSAMP; ++i) {
      rand_xform(rng, x, 512.0);
      x.translation()[0] = x.translation()[1] = x.translation()[2] = 0;
      idx_seen.insert(xh.get_key(x));
    }
    n_ori_bins = (int)idx_seen.size();
    xcov = (double)NSAMP / n_ori_bins;
  }
  return n_ori_bins;
}

template <template <class X> class XformHash>
bool xform_hash_perf_test(double cart_resl, double ang_resl,
                          int const N2 = 100 * 1000, unsigned int seed = 0) {
  std::mt19937 rng((unsigned int)time(0) + seed);

  double time_key = 0.0, time_cen = 0.0;

  XformHash<Xform> xh(cart_resl, ang_resl, 512.0);
  ang_resl = xh.ori_resl();
  double cart_resl2 = cart_resl * cart_resl;
  double ang_resl2 = ang_resl * ang_resl;

  std::vector<Xform> samples(N2), centers(N2);

  for (int i = 0; i < N2; ++i) rand_xform(rng, samples[i], 512.0);

  util::Timer tk;
  std::vector<uint64_t> keys(N2);
  for (int i = 0; i < N2; ++i) {
    keys[i] = xh.get_key(samples[i]);
    // centers[i] = xh.get_center( keys[i] );
    // cout << endl;
  }
  time_key += (double)tk.elapsed_nano();

  util::Timer tc;
  for (int i = 0; i < N2; ++i) centers[i] = xh.get_center(keys[i]);
  time_cen += (double)tc.elapsed_nano();

  std::unordered_set<size_t> idx_seen;
  // idx_seen.set_empty_key(std::numeric_limits<uint64_t>::max());
  for (int i = 0; i < N2; ++i) idx_seen.insert(keys[i]);

  double covrad = 0, max_dt = 0, max_da = 0;
  for (int i = 0; i < N2; ++i) {
    Xform l = centers[i].inverse() * samples[i];
    double dt = l.translation().norm();
    Eigen::Matrix3d m;
    for (int k = 0; k < 9; ++k) m.data()[k] = l.data()[k];
    // cout << m << endl;
    // cout << l.rotation() << endl;
    double da = Eigen::AngleAxisd(m).angle() * 180.0 / M_PI;
    // double da = Eigen::AngleAxisd(l.rotation()).angle()*180.0/M_PI;
    double err = sqrt(da * da / ang_resl2 * cart_resl2 + dt * dt);
    covrad = fmax(covrad, err);
    max_dt = fmax(max_dt, dt);
    max_da = fmax(max_da, da);
  }
  if (max_dt > cart_resl * 1.1 || max_dt < cart_resl * 0.8)
    std::cout << "TEST FAIL cart: " << cart_resl << " " << max_dt << std::endl;
  if (max_da > ang_resl * 1.1 || max_da < ang_resl * 0.8)
    std::cout << "TEST FAIL ang: " << ang_resl << " " << max_da << " "
              << xh.get_ori_resl(xh.ori_nside_) << std::endl;
  ASSERT_GT(max_dt, cart_resl * 0.8)
  ASSERT_LT(max_dt, cart_resl * 1.1)
  ASSERT_GT(max_da, ang_resl * 0.8)
  ASSERT_LT(max_da, ang_resl * 1.1)

  double tot_cell_vol = covrad * covrad * covrad * covrad * covrad * covrad *
                        xh.approx_nori() / (cart_resl * cart_resl * cart_resl);
  // printf(
  // " %5.3f/%5.1f cr %5.3f dt %5.3f da %6.3f x2k: %7.3fns k2x: %7.3fns "
  // "%9.3f "
  // "%7lu\n",
  // cart_resl, ang_resl, covrad, max_dt / cart_resl, max_da / ang_resl,
  // time_key / N2, time_cen / N2, tot_cell_vol, xh.approx_nori());

  // cout << " rate " << N1*N2/time_key << "  " << N1*N2/time_cen << endl;
  return true;
}

bool TEST_XformHash_XformHash_bt24_BCC6() {
  unsigned int s = 0;
  int N = 10 * 1000;
  bool pass = true;
  // cout << "  bt24_BCC6";
  pass &= xform_hash_perf_test<XformHash_bt24_BCC6>(4.00, 30.0, N, ++s);
  // cout << "  bt24_BCC6";
  pass &= xform_hash_perf_test<XformHash_bt24_BCC6>(2.00, 20.0, N, ++s);
  // cout << "  bt24_BCC6";
  pass &= xform_hash_perf_test<XformHash_bt24_BCC6>(1.00, 15.0, N, ++s);
  // cout << "  bt24_BCC6";
  pass &= xform_hash_perf_test<XformHash_bt24_BCC6>(0.50, 10.0, N, ++s);
  // cout << "  bt24_BCC6";
  pass &= xform_hash_perf_test<XformHash_bt24_BCC6>(0.25, 5.0, N, ++s);
  // cout << "  bt24_BCC6";
  pass &= xform_hash_perf_test<XformHash_bt24_BCC6>(0.11, 3.3, N, ++s);
  return pass;
}

PYBIND11_MODULE(xbin_test, m) {
  m.def("TEST_XformHash_XformHash_bt24_BCC6",
        &TEST_XformHash_XformHash_bt24_BCC6);
}

}  // namespace test
}  // namespace xbin
}  // namespace rpxdock
