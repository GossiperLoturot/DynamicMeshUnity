#include <dynamic_mesh.hpp>
#include <gtest/gtest.h>

TEST(test, add_function) {
  size_t size = 10;
  VertexPosition v[size * size * size];
  VertexId i[size * size * size * 18];
  SDFValue sdf[size * size * size];

  for (size_t j = 0; j < size * size * size; j++) {
    sdf[j] = SDFValue(j % 3 - 1);
  }

  naive_surface_nets(v, i, sdf, size);
}
