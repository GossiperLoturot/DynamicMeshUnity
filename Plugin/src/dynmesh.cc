#include "dynmesh.h"

#include <cstdint>
#include <vector>

#include "IUnityInterface.h"
#include <Eigen/Dense>

namespace dynmesh {

using namespace std;
using namespace Eigen;

namespace {

Vector3i neighbors[8] = {
    Vector3i(0, 0, 0), Vector3i(1, 0, 0), Vector3i(1, 0, 1), Vector3i(0, 0, 1),
    Vector3i(0, 1, 0), Vector3i(1, 1, 0), Vector3i(1, 1, 1), Vector3i(0, 1, 1),
};

Vector2i edges[12] = {
    Vector2i(0, 1), Vector2i(1, 2), Vector2i(2, 3), Vector2i(3, 0),
    Vector2i(4, 5), Vector2i(5, 6), Vector2i(6, 7), Vector2i(7, 4),
    Vector2i(0, 4), Vector2i(1, 5), Vector2i(2, 6), Vector2i(3, 7),
};

float lerp(const float a, const float b, const float t) {
  return a * (1.0f - t) + b * t;
}

const Vector3f lerp(const Vector3f &a, const Vector3f &b, const float t) {
  return a * (1.0f - t) + b * t;
}

int32_t get_id(const Vector3i &p, const int32_t size) {
  return p[0] + p[1] * size + p[2] * size * size;
}

Vector3i sample_neighbor_p(const Vector3i &p, const int32_t neighbor_id) {
  return p + neighbors[neighbor_id];
}

Vector3i sample_neighbor_n(const Vector3i &p, const int32_t neighbor_id) {
  return p + neighbors[neighbor_id];
}

float sample_sdf(const Vector3f &p, const float *sdf, const int32_t size) {
  const int32_t x0 = static_cast<int32_t>(p.x());
  const int32_t y0 = static_cast<int32_t>(p.y());
  const int32_t z0 = static_cast<int32_t>(p.z());
  const int32_t x1 = x0 + 1;
  const int32_t y1 = y0 + 1;
  const int32_t z1 = z0 + 1;

  const float xd = p.x() - static_cast<float>(x0);
  const float yd = p.y() - static_cast<float>(y0);
  const float zd = p.z() - static_cast<float>(z0);

  const float c00 = lerp(sdf[get_id(Vector3i(x0, y0, z0), size)],
                         +sdf[get_id(Vector3i(x1, y0, z0), size)], xd);
  const float c01 = lerp(sdf[get_id(Vector3i(x0, y0, z1), size)],
                         sdf[get_id(Vector3i(x1, y0, z1), size)], xd);
  const float c10 = lerp(sdf[get_id(Vector3i(x0, y1, z0), size)],
                         sdf[get_id(Vector3i(x1, y1, z0), size)], xd);
  const float c11 = lerp(sdf[get_id(Vector3i(x0, y1, z1), size)],
                         sdf[get_id(Vector3i(x1, y1, z1), size)], xd);
  const float c0 = lerp(c00, c10, yd);
  const float c1 = lerp(c01, c11, yd);
  return lerp(c0, c1, zd);
}

Vector3f sample_normal(const Vector3f &p, const float *sdf,
                       const int32_t size) {
  constexpr float h = 0.01f;

  const float dx = sample_sdf(p + Vector3f(h, 0.0f, 0.0f), sdf, size) -
                   sample_sdf(p - Vector3f(h, 0.0f, 0.0f), sdf, size);
  const float dy = sample_sdf(p + Vector3f(0.0f, h, 0.0f), sdf, size) -
                   sample_sdf(p - Vector3f(0.0f, h, 0.0f), sdf, size);
  const float dz = sample_sdf(p + Vector3f(0.0f, 0.0f, h), sdf, size) -
                   sample_sdf(p - Vector3f(0.0f, 0.0f, h), sdf, size);
  return Vector3f(dx, dy, dz).normalized();
}

Vector3f sample_tangent(const Vector3f &n) {
  Vector3f tangent_candidate = n.cross(Vector3f(0.0f, 1.0f, 0.0f));
  if (tangent_candidate.norm() < 1e-6) {
    tangent_candidate = n.cross(Vector3f(0.0f, 0.0f, 1.0f));
  }
  return tangent_candidate.normalized();
}

void make_face(int32_t *triangles, int32_t *triangles_count, const int32_t id0,
               const int32_t id1, const int32_t id2, const int32_t id3,
               const bool outside) {
  if (outside) {
    triangles[(*triangles_count)++] = id0;
    triangles[(*triangles_count)++] = id3;
    triangles[(*triangles_count)++] = id2;
    triangles[(*triangles_count)++] = id2;
    triangles[(*triangles_count)++] = id1;
    triangles[(*triangles_count)++] = id0;
  } else {
    triangles[(*triangles_count)++] = id0;
    triangles[(*triangles_count)++] = id1;
    triangles[(*triangles_count)++] = id2;
    triangles[(*triangles_count)++] = id2;
    triangles[(*triangles_count)++] = id3;
    triangles[(*triangles_count)++] = id0;
  }
}

} // namespace

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API naive_surface_nets(
    const float *sdf, const int32_t size, float *vertices, float *normals,
    float *tangents, int32_t *triangles, float bounds[6]) {
  Vector3f *vertices_repr = reinterpret_cast<Vector3f *>(vertices);
  Vector3f *normals_repr = reinterpret_cast<Vector3f *>(normals);
  Vector4f *tangents_repr = reinterpret_cast<Vector4f *>(tangents);

  Vector3f min_p = Vector3f::Zero();
  Vector3f max_p = Vector3f::Zero();

  int32_t vertex_count = 0;
  int32_t triangle_count = 0;
  vector<int32_t> indices = vector<int32_t>(size * size * size);

  for (int32_t x = 0; x < size - 1; ++x) {
    for (int32_t y = 0; y < size - 1; ++y) {
      for (int32_t z = 0; z < size - 1; ++z) {
        const Vector3i p = Vector3i(x, y, z);

        int32_t mask = 0;
        if (0.0f > sdf[get_id(sample_neighbor_p(p, 0), size)])
          mask |= 1 << 0;
        if (0.0f > sdf[get_id(sample_neighbor_p(p, 1), size)])
          mask |= 1 << 1;
        if (0.0f > sdf[get_id(sample_neighbor_p(p, 2), size)])
          mask |= 1 << 2;
        if (0.0f > sdf[get_id(sample_neighbor_p(p, 3), size)])
          mask |= 1 << 3;
        if (0.0f > sdf[get_id(sample_neighbor_p(p, 4), size)])
          mask |= 1 << 4;
        if (0.0f > sdf[get_id(sample_neighbor_p(p, 5), size)])
          mask |= 1 << 5;
        if (0.0f > sdf[get_id(sample_neighbor_p(p, 6), size)])
          mask |= 1 << 6;
        if (0.0f > sdf[get_id(sample_neighbor_p(p, 7), size)])
          mask |= 1 << 7;

        if (mask == 0 || mask == 255)
          continue;

        min_p = min_p.cwiseMin(p.cast<float>());
        max_p = max_p.cwiseMax(p.cast<float>());

        Vector3f vertex = Vector3f::Zero();
        int32_t crossing_edge_count = 0;

        for (int32_t i = 0; i < 12; ++i) {
          const int32_t nid0 = edges[i][0];
          const int32_t nid1 = edges[i][1];

          if ((mask >> nid0 & 1) == (mask >> nid1 & 1))
            continue;

          const Vector3i p0 = sample_neighbor_p(p, nid0);
          const Vector3i p1 = sample_neighbor_p(p, nid1);
          const float sd0 = sdf[get_id(p0, size)];
          const float sd1 = sdf[get_id(p1, size)];
          vertex += lerp(p0.cast<float>(), p1.cast<float>(),
                         (0.0f - sd0) / (sd1 - sd0));
          ++crossing_edge_count;
        }

        vertex /= static_cast<float>(crossing_edge_count);
        const Vector3f normal = sample_normal(vertex, sdf, size);
        const Vector3f tangent = sample_tangent(normal);

        vertices_repr[vertex_count] = vertex;
        normals_repr[vertex_count] = normal;
        tangents_repr[vertex_count] =
            Vector4f(tangent.x(), tangent.y(), tangent.z(), 0.0f);

        indices[get_id(p, size)] = vertex_count;
        ++vertex_count;

        if (x == 0 || y == 0 || z == 0)
          continue;

        const bool outside = (mask & 1) != 0;
        const int32_t id0 = indices[get_id(sample_neighbor_n(p, 0), size)];
        const int32_t id1 = indices[get_id(sample_neighbor_n(p, 1), size)];
        const int32_t id2 = indices[get_id(sample_neighbor_n(p, 2), size)];
        const int32_t id3 = indices[get_id(sample_neighbor_n(p, 3), size)];
        const int32_t id4 = indices[get_id(sample_neighbor_n(p, 4), size)];
        const int32_t id5 = indices[get_id(sample_neighbor_n(p, 5), size)];
        const int32_t id7 = indices[get_id(sample_neighbor_n(p, 7), size)];
        if ((mask >> 1 & 1) != 0 != outside)
          make_face(triangles, &triangle_count, id0, id3, id7, id4, outside);
        if ((mask >> 3 & 1) != 0 != outside)
          make_face(triangles, &triangle_count, id0, id4, id5, id1, outside);
        if ((mask >> 4 & 1) != 0 != outside)
          make_face(triangles, &triangle_count, id0, id1, id2, id3, outside);
      }
    }
  }

  const Vector3f center = (min_p + max_p) * 0.5f;
  bounds[0] = center.x();
  bounds[1] = center.y();
  bounds[2] = center.z();

  const Vector3f extent = (max_p - min_p) * 0.5f;
  bounds[3] = extent.x();
  bounds[4] = extent.y();
  bounds[5] = extent.z();
}

} // namespace dynmesh
