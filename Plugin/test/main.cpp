#include <dynmesh.hpp>
#include <gtest/gtest.h>
#include <vector>

float plane_sdf(float y, float plane_y) { return y - plane_y; }

TEST(test, plane) {
  const int size = 16;

  // Create a plane SDF
  std::vector<SDFValue> sdf(size * size * size);
  const float plane_y = (size - 1) / 2.0f;
  for (int z = 0; z < size; ++z) {
    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        sdf[x + y * size + z * size * size] =
            plane_sdf(static_cast<float>(y), plane_y);
      }
    }
  }

  std::vector<VertexPosition> vertices(size * size * size,
                                       VertexPosition::Zero());
  std::vector<VertexId> triangles(size * size * size * 6, VertexId(0));
  std::vector<VertexPosition> normals(size * size * size,
                                      VertexPosition::Zero());
  naive_surface_nets(vertices.data(), triangles.data(), normals.data(),
                     sdf.data(), size);

  for (const auto &v : vertices) {
    if (v.norm() < 1e-6)
      continue; // Skip zero-initialized vertices

    // All vertices should lie on the plane
    ASSERT_NEAR(v.y(), plane_y, 1e-5);
  }
  for (const auto &n : normals) {
    if (n.norm() < 1e-6)
      continue; // Skip zero-initialized normals

    // All normals should be pointing up (positive y)
    ASSERT_NEAR(n.x(), 0.0f, 1e-5);
    ASSERT_NEAR(n.y(), 1.0f, 1e-5);
    ASSERT_NEAR(n.z(), 0.0f, 1e-5);
  }
  bool has_triangles = false;
  for (size_t i = 0; i < triangles.size(); i += 3) {
    if (triangles[i] != 0 || triangles[i + 1] != 0 || triangles[i + 2] != 0) {
      has_triangles = true;
      break;
    }
  }
  ASSERT_TRUE(has_triangles);
}

TEST(test, empty_space) {
  const int size = 16;

  // Create a empty space SDF
  std::vector<SDFValue> sdf(size * size * size, 1.0f); // All positive

  std::vector<VertexPosition> vertices(size * size * size,
                                       VertexPosition::Zero());
  std::vector<VertexId> triangles(size * size * size * 6, VertexId(0));
  std::vector<VertexPosition> normals(size * size * size,
                                      VertexPosition::Zero());
  naive_surface_nets(vertices.data(), triangles.data(), normals.data(),
                     sdf.data(), size);

  for (const auto &v : vertices) {
    ASSERT_TRUE(v.norm() < 1e-6);
  }
  for (const auto &t : triangles) {
    ASSERT_EQ(t, 0);
  }
}

TEST(test, full_space) {
  const int size = 16;

  // Create a full space SDF
  std::vector<SDFValue> sdf(size * size * size, -1.0f); // All negative

  std::vector<VertexPosition> vertices(size * size * size,
                                       VertexPosition::Zero());
  std::vector<VertexId> triangles(size * size * size * 6, VertexId(0));
  std::vector<VertexPosition> normals(size * size * size,
                                      VertexPosition::Zero());
  naive_surface_nets(vertices.data(), triangles.data(), normals.data(),
                     sdf.data(), size);

  for (const auto &v : vertices) {
    ASSERT_TRUE(v.norm() < 1e-6);
  }
  for (const auto &t : triangles) {
    ASSERT_EQ(t, 0);
  }
}
