#include <gtest/gtest.h>
#include <template.hpp>

TEST(test, add_function) {
  EXPECT_EQ(add(2, 3), 5);
  EXPECT_EQ(add(-1, 1), 0);
  EXPECT_EQ(add(0, 0), 0);
  EXPECT_EQ(add(-5, -5), -10);
}
