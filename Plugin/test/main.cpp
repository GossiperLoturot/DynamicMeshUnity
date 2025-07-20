#include <dynmesh.hpp>
#include <gtest/gtest.h>
#include <stdio.h>

TEST(test, surface_nets) {
  const int size = 4;

  auto vertices = std::array<VertexPosition, size * size * size>();
  auto triangles = std::array<VertexId, size * size * size * 18>();
  auto normals = std::array<VertexPosition, size * size * size>();
  auto sdf = std::array<SDFValue, size * size * size>();

  for (int i = 0; i < size * size * size; ++i) {
    sdf[i] = static_cast<float>(i % 3) - 1.0f;
  }

  naive_surface_nets(vertices.data(), triangles.data(), normals.data(), sdf.data(), size);
}
