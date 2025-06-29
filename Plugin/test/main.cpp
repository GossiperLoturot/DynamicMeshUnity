#include <stdio.h>
#include <dynmesh.hpp>
#include <gtest/gtest.h>

TEST(test, add_function) {
  const int size = 4;

  auto vertices = std::array<VertexPosition, size * size * size>();
  auto triangles = std::array<VertexId, size * size * size * 18>();
  auto sdf = std::array<SDFValue, size * size * size>();

  for (int i = 0; i < size * size * size; ++i) {
    sdf[i] = static_cast<float>(i % 3) - 1.0f;
  }

  naive_surface_nets(vertices.data(), triangles.data(), sdf.data(), size);
}
