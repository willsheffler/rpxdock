/*/*cppimport
<%


cfg['include_dirs'] = ['../..', '../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['../extern/miniball/Seb.h',
'../extern/miniball/Seb-inl.h', '../util/Timer.hpp']

cfg['parallel'] = False


setup_pybind11(cfg)
%>
*/
/** \file */

// Original Authors: Martin Kutz <kutz@math.fu-berlin.de>,
//                   Kaspar Fischer <kf@iaeth.ch>

#include "rpxdock/geom/miniball.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "rpxdock/util/Timer.hpp"
#include "rpxdock/util/assertions.hpp"

namespace py = pybind11;
using namespace py::literals;
using namespace Eigen;

#include <cstdio>
#include <iostream>

using std::cout;
using std::endl;

namespace rpxdock {
namespace geom {
namespace miniball_test {
using namespace util;

bool miniball_test(int n = 1000, int d = 7, bool on_boundary = false) {
  typedef double F;
  typedef Seb::Point<F> Point;
  typedef std::vector<Point> PointVector;
  typedef Seb::Smallest_enclosing_ball<F> Miniball;

  using std::cout;
  using std::endl;
  using std::vector;

  // Construct n random points in dimension d
  PointVector S;
  vector<double> coords(d);
  srand(clock());
  for (int i = 0; i < n; ++i) {
    // Generate coordindates in [-1,1]
    double len = 0;
    for (int j = 0; j < d; ++j) {
      coords[j] = static_cast<F>(2.0 * rand() / RAND_MAX - 1.0);
      len += coords[j] * coords[j];
    }

    // Normalize length to "almost" 1 (makes it harder for the algorithm)
    if (on_boundary) {
      const double Wiggle = 1e-2;
      len = 1 / (std::sqrt(len) + Wiggle * rand() / RAND_MAX);
      for (int j = 0; j < d; ++j) coords[j] *= len;
    }
    S.push_back(Point(d, coords.begin()));
  }
  util::Timer t;
  Miniball mb(d, S);
  t.stop();
  // Output
  F rad = mb.radius();
  F rad_squared = mb.squared_radius();
  cout << "Running time: " << t.elapsed() << "s" << endl
       << "Radius = " << rad << " (squared: " << rad_squared << ")" << endl;
  // << "Center:" << endl;
  Miniball::Coordinate_iterator center_it = mb.center_begin();
  // for (int j = 0; j < d; ++j) cout << "  " << center_it[j] << endl;
  cout << "=====================================================" << endl;
  using MiniballHack =
      Seb::Smallest_enclosing_ball<F, double *, EigenPointAccessor>;
  Mxd ecrd(n, d);
  EigenPointAccessor pa(ecrd);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < d; ++j) pa[i][j] = S[i][j];
  util::Timer t2;
  MiniballHack mbhack(d, pa);
  t2.stop();
  F rad2 = mbhack.radius();
  F rad2_squared = mbhack.squared_radius();
  // cout << "Running time: " << t2.elapsed() << "s" << endl
  //      << "Radius = " << rad << " (squared: " << rad_squared << ")" << endl
  //      << "Center:" << endl;
  Miniball::Coordinate_iterator center2_it = mbhack.center_begin();
  // for (int j = 0; j < d; ++j) cout << "  " << center2_it[j] << endl;
  // cout << "=====================================================" << endl;

  ASSERT_DOUBLE_EQ(rad, rad2)
  for (int j = 0; j < d; ++j) ASSERT_DOUBLE_EQ(center_it[j], center2_it[j])

  cout << "Running time ratio: " << t.elapsed() / t2.elapsed() << endl;

  return mb.verify(false) && mbhack.verify(false);
}

PYBIND11_MODULE(miniball, m) {
  m.def("miniball_test", &miniball_test);
  m.def("miniball", &miniball, py::call_guard<py::gil_scoped_release>());
}

}  // namespace miniball_test
}  // namespace geom
}  // namespace rpxdock