#pragma once

#include <stdint.h>

// #ifndef CXX14
// 	#include <boost/static_assert.hpp>
// 	BOOST_STATIC_ASSERT_MSG( false, "c++14 required for rif stuff");
// #endif

#include <memory>

namespace rpxdock {
namespace pack {

using std::enable_shared_from_this;
using std::make_shared;
using std::shared_ptr;
using std::unique_ptr;
using std::weak_ptr;

template <typename T>
struct Bounds {
  T lo;
  T hi;
};

}  // namespace pack
}  // namespace rpxdock
