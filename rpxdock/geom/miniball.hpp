#pragma once
/** \file */

#include "miniball/Seb.h"
#include "rpxdock/util/types.hpp"

namespace rpxdock {
namespace geom {
using namespace util;
struct EigenPointAccessor {
  RefMxd data;
  EigenPointAccessor(RefMxd d) : data(d) {}
  double *operator[](size_t i) const { return (double *)data.row(i).data(); }
  double *operator[](size_t i) { return data.row(i).data(); }
  size_t size() const { return data.rows(); }
};

Vxd miniball(RefMxd coords) {
  using Miniball =
      Seb::Smallest_enclosing_ball<double, double *, EigenPointAccessor>;
  Miniball mb(coords.cols(), EigenPointAccessor(coords));
  Miniball::Coordinate_iterator center_it = mb.center_begin();
  Vxd out(coords.cols() + 1);
  out[0] = mb.radius();
  for (int i = 0; i < coords.cols(); ++i) out[i + 1] = center_it[i];
  return out;
}

}  // namespace geom
}  // namespace rpxdock