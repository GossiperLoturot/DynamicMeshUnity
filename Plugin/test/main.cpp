#include <cmath>
#include <dynmesh.hpp>
#include <gtest/gtest.h>
#include <vector>

// A simple sphere SDF for testing
float sphere_sdf(float x, float y, float z, float r) {
  return std::sqrt(x * x + y * y + z * z) - r;
}

// A simple plane SDF for testing
float plane_sdf(float y, float plane_y) { return y - plane_y; }

// A simple box SDF for testing
float box_sdf(float x, float y, float z, const Eigen::Vector3f &b) {
  float dx = std::abs(x) - b.x();
  float dy = std::abs(y) - b.y();
  float dz = std::abs(z) - b.z();
  Eigen::Vector3f d(std::max(dx, 0.0f), std::max(dy, 0.0f), std::max(dz, 0.0f));
  return d.norm() + std::min(std::max(dx, std::max(dy, dz)), 0.0f);
}

TEST(test, surface_nets_optimization) {
  const int size = 32;

  std::vector<VertexPosition> vertices(size * size * size);
  std::vector<VertexId> triangles(size * size * size * 6);
  std::vector<VertexPosition> normals(size * size * size);
  std::vector<SDFValue> sdf(size * size * size);

  // Create a sphere SDF
  const float radius = (size - 1) / 3.0f;
  const float center = (size - 1) / 2.0f;
  for (int z = 0; z < size; ++z) {
    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        sdf[x + y * size + z * size * size] =
            sphere_sdf(x - center, y - center, z - center, radius);
      }
    }
  }

  naive_surface_nets(vertices.data(), triangles.data(), normals.data(),
                     sdf.data(), size);

  // Basic validation: check if some vertices and triangles were generated.
  // A more thorough test would check the mesh topology and vertex positions.
  bool has_vertices = false;
  for (const auto &v : vertices) {
    if (v.norm() > 1e-6) {
      has_vertices = true;
      break;
    }
  }
  ASSERT_TRUE(has_vertices);

  bool has_triangles = false;
  for (size_t i = 0; i < triangles.size(); i += 3) {
    if (triangles[i] != 0 || triangles[i + 1] != 0 || triangles[i + 2] != 0) {
      has_triangles = true;
      break;
    }
  }
  ASSERT_TRUE(has_triangles);
}

TEST(test, plane) {
  const int size = 16;
  const float plane_y = (size - 1) / 2.0f;

  std::vector<VertexPosition> vertices(size * size * size,
                                       VertexPosition::Zero());
  std::vector<VertexId> triangles(size * size * size * 6, 0);
  std::vector<VertexPosition> normals(size * size * size,
                                      VertexPosition::Zero());
  std::vector<SDFValue> sdf(size * size * size);

  // Create a plane SDF
  for (int z = 0; z < size; ++z) {
    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        sdf[x + y * size + z * size * size] =
            plane_sdf(static_cast<float>(y), plane_y);
      }
    }
  }

  naive_surface_nets(vertices.data(), triangles.data(), normals.data(),
                     sdf.data(), size);

  int vertex_count = 0;
  for (const auto &v : vertices) {
    if (v.norm() < 1e-6)
      continue; // Skip zero-initialized vertices
    vertex_count++;
    // All vertices should lie on the plane
    ASSERT_NEAR(v.y(), plane_y, 1e-5);
  }

  int normal_count = 0;
  for (const auto &n : normals) {
    if (n.norm() < 1e-6)
      continue; // Skip zero-initialized normals
    normal_count++;
    // All normals should be pointing up (positive y)
    ASSERT_NEAR(n.x(), 0.0f, 1e-5);
    ASSERT_NEAR(n.y(), 1.0f, 1e-5);
    ASSERT_NEAR(n.z(), 0.0f, 1e-5);
  }

  ASSERT_GT(vertex_count, 0);
  ASSERT_EQ(vertex_count, normal_count);

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
  std::vector<VertexPosition> vertices(size * size * size,
                                       VertexPosition::Zero());
  std::vector<VertexId> triangles(size * size * size * 6, 0);
  std::vector<VertexPosition> normals(size * size * size,
                                      VertexPosition::Zero());
  std::vector<SDFValue> sdf(size * size * size, 1.0f); // All positive

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
  std::vector<VertexPosition> vertices(size * size * size,
                                       VertexPosition::Zero());
  std::vector<VertexId> triangles(size * size * size * 6, 0);
  std::vector<VertexPosition> normals(size * size * size,
                                      VertexPosition::Zero());
  std::vector<SDFValue> sdf(size * size * size, -1.0f); // All negative

  naive_surface_nets(vertices.data(), triangles.data(), normals.data(),
                     sdf.data(), size);

  for (const auto &v : vertices) {
    ASSERT_TRUE(v.norm() < 1e-6);
  }
  for (const auto &t : triangles) {
    ASSERT_EQ(t, 0);
  }
}

TEST(test, cube) {
  const int size = 20;
  std::vector<VertexPosition> vertices(size * size * size,
                                       VertexPosition::Zero());
  std::vector<VertexId> triangles(size * size * size * 6, 0);
  std::vector<VertexPosition> normals(size * size * size,
                                      VertexPosition::Zero());
  std::vector<SDFValue> sdf(size * size * size);

  const Eigen::Vector3f half_extents((size - 1) / 4.0f, (size - 1) / 4.0f,
                                     (size - 1) / 4.0f);
  const float center = (size - 1) / 2.0f;

  for (int z = 0; z < size; ++z) {
    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        sdf[x + y * size + z * size * size] =
            box_sdf(x - center, y - center, z - center, half_extents);
      }
    }
  }

  naive_surface_nets(vertices.data(), triangles.data(), normals.data(),
                     sdf.data(), size);

  bool has_vertices = false;
  for (const auto &v : vertices) {
    if (v.norm() > 1e-6) {
      has_vertices = true;
      break;
    }
  }
  ASSERT_TRUE(has_vertices);

  bool has_triangles = false;
  for (size_t i = 0; i < triangles.size(); i += 3) {
    if (triangles[i] != 0 || triangles[i + 1] != 0 || triangles[i + 2] != 0) {
      has_triangles = true;
      break;
    }
  }
  ASSERT_TRUE(has_triangles);
}
