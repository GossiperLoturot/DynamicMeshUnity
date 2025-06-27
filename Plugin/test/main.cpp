#include <dynamic_mesh.hpp>
#include <gtest/gtest.h>

TEST(test, add_function) {
  Eigen::Vector3f *v;
  int *i;
  float sdf[1] = {0.0};
  naive_surface_nets(v, i, sdf, 1);
}
