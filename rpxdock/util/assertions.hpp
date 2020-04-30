#pragma once
/** \file */

#include <iostream>

#define _TEST(cond)                                                    \
  if (!(cond)) {                                                       \
    std::cerr << "ERROR " << __FILE__ << ":" << __LINE__ << std::endl; \
    return false;                                                      \
  }

#define ASSERT_LT(x, y) _TEST(x < y);
#define ASSERT_GT(x, y) _TEST(x > y);
#define ASSERT_LE(x, y) _TEST(x <= y);
#define ASSERT_GE(x, y) _TEST(x >= y);
#define ASSERT_EQ(x, y) _TEST(x == y);
#define ASSERT_TRUE(x) _TEST(x);
#define ASSERT_FALSE(x) _TEST(!x);
#define ASSERT_FLOAT_EQ(x, y) _TEST(fabs(x - y) < 0.0001f);
#define ASSERT_DOUBLE_EQ(x, y) _TEST(fabs(x - y) < 0.00000001);
#define ASSERT_NEAR(x, y, t) _TEST(fabs(x - y) < t);
