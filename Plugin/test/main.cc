#include <stdint.h>

#include <vector>

#include <gtest/gtest.h>

#include "dynmesh.h"

namespace dynmesh {

using namespace std;

TEST(test, plane) {
  const int32_t size = 16;
  const float plane_y = (size - 1) / 2.0f;

  vector<float> sdf(size * size * size);
  for (int32_t z = 0; z < size; ++z) {
    for (int32_t y = 0; y < size; ++y) {
      for (int32_t x = 0; x < size; ++x) {
        sdf[x + y * size + z * size * size] = static_cast<float>(y) - plane_y;
      }
    }
  }

  vector<float> vertices(size * size * size * 3, 0.0f);
  vector<float> normals(size * size * size * 3, 0.0f);
  vector<float> tangents(size * size * size * 4, 0.0f);
  vector<int32_t> triangles(size * size * size * 6, 0);

  float bounds[6];

  naive_surface_nets(sdf.data(), size, vertices.data(), normals.data(),
                     tangents.data(), triangles.data(), bounds);
}

TEST(test, empty_space) {
  const int32_t size = 16;

  vector<float> sdf(size * size * size, 1.0f);

  vector<float> vertices(size * size * size * 3, 0.0f);
  vector<float> normals(size * size * size * 3, 0.0f);
  vector<float> tangents(size * size * size * 4, 0.0f);
  vector<int32_t> triangles(size * size * size * 6, 0);

  float bounds[6];

  naive_surface_nets(sdf.data(), size, vertices.data(), normals.data(),
                     tangents.data(), triangles.data(), bounds);
}

TEST(test, full_space) {
  const int32_t size = 16;

  vector<float> sdf(size * size * size, -1.0f);

  vector<float> vertices(size * size * size * 3, 0.0f);
  vector<float> normals(size * size * size * 3, 0.0f);
  vector<float> tangents(size * size * size * 4, 0.0f);
  vector<int32_t> triangles(size * size * size * 6, 0);

  float bounds[6];

  naive_surface_nets(sdf.data(), size, vertices.data(), normals.data(),
                     tangents.data(), triangles.data(), bounds);
}

} // namespace dynmesh
