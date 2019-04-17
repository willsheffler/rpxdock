#pragma once

#define _TEST(cond)                          \
  if (!(cond)) {                             \
    std::cerr << "some error!" << std::endl; \
    return false;                            \
  }

#define ASSERT_LT(x, y) _TEST(x < y);
#define ASSERT_GT(x, y) _TEST(x > y);
#define ASSERT_LE(x, y) _TEST(x <= y);
#define ASSERT_GE(x, y) _TEST(x >= y);
#define ASSERT_EQ(x, y) _TEST(x == y);
#define ASSERT_TRUE(x) _TEST(x);
#define ASSERT_FALSE(x) _TEST(!x);
#define ASSERT_FLOAT_EQ(x, y) _TEST(fabs(x - y) < 0.0001);
