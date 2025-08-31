#include "dynmesh.h"

#include <cstdint>
#include <vector>

#include "IUnityInterface.h"
#include "glm/geometric.hpp"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"

namespace dynmesh {

using namespace std;
using namespace glm;

namespace {

ivec3 neighbors[8] = {
    ivec3(0, 0, 0), ivec3(1, 0, 0), ivec3(1, 0, 1), ivec3(0, 0, 1),
    ivec3(0, 1, 0), ivec3(1, 1, 0), ivec3(1, 1, 1), ivec3(0, 1, 1),
};

ivec2 edges[12] = {
    ivec2(0, 1), ivec2(1, 2), ivec2(2, 3), ivec2(3, 0),
    ivec2(4, 5), ivec2(5, 6), ivec2(6, 7), ivec2(7, 4),
    ivec2(0, 4), ivec2(1, 5), ivec2(2, 6), ivec2(3, 7),
};

int32_t get_id(const ivec3 &p, const int32_t size) {
  return p.x + p.y * size + p.z * size * size;
}

float lerp(const float &a, const float &b, const float &t) {
  return a + (b - a) * t;
}

vec3 lerp(const vec3 &a, const vec3 &b, const float &t) {
  return a + (b - a) * t;
}

ivec3 sample_neighbor_p(const ivec3 &p, const int32_t neighbor_id) {
  return p + neighbors[neighbor_id];
}

ivec3 sample_neighbor_n(const ivec3 &p, const int32_t neighbor_id) {
  return p - neighbors[neighbor_id];
}

float sample_sdf(const vec3 &p, const float *sdf, const int32_t size) {
  const int32_t x0 = int32_t(p.x);
  const int32_t y0 = int32_t(p.y);
  const int32_t z0 = int32_t(p.z);
  const int32_t x1 = x0 + 1;
  const int32_t y1 = y0 + 1;
  const int32_t z1 = z0 + 1;

  const float xd = p.x - float(x0);
  const float yd = p.y - float(y0);
  const float zd = p.z - float(z0);

  const float c00 = lerp(sdf[get_id(ivec3(x0, y0, z0), size)],
                         +sdf[get_id(ivec3(x1, y0, z0), size)], xd);
  const float c01 = lerp(sdf[get_id(ivec3(x0, y0, z1), size)],
                         sdf[get_id(ivec3(x1, y0, z1), size)], xd);
  const float c10 = lerp(sdf[get_id(ivec3(x0, y1, z0), size)],
                         sdf[get_id(ivec3(x1, y1, z0), size)], xd);
  const float c11 = lerp(sdf[get_id(ivec3(x0, y1, z1), size)],
                         sdf[get_id(ivec3(x1, y1, z1), size)], xd);
  const float c0 = lerp(c00, c10, yd);
  const float c1 = lerp(c01, c11, yd);
  return lerp(c0, c1, zd);
}

vec3 sample_normal(const vec3 &p, const float *sdf, const int32_t size) {
  constexpr float h = 0.01f;

  const float dx = sample_sdf(p + vec3(h, 0.0f, 0.0f), sdf, size) -
                   sample_sdf(p - vec3(h, 0.0f, 0.0f), sdf, size);
  const float dy = sample_sdf(p + vec3(0.0f, h, 0.0f), sdf, size) -
                   sample_sdf(p - vec3(0.0f, h, 0.0f), sdf, size);
  const float dz = sample_sdf(p + vec3(0.0f, 0.0f, h), sdf, size) -
                   sample_sdf(p - vec3(0.0f, 0.0f, h), sdf, size);
  return normalize(vec3(dx, dy, dz));
}

vec3 sample_tangent(const vec3 &n) {
  vec3 tangent_candidate = cross(n, vec3(0.0f, 1.0f, 0.0f));
  if (dot(tangent_candidate, tangent_candidate) < 1e-6) {
    tangent_candidate = cross(n, vec3(0.0f, 0.0f, 1.0f));
  }
  return normalize(tangent_candidate);
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
  vec3 *vertices_repr = reinterpret_cast<vec3 *>(vertices);
  vec3 *normals_repr = reinterpret_cast<vec3 *>(normals);
  vec4 *tangents_repr = reinterpret_cast<vec4 *>(tangents);

  vec3 min_bound = vec3(0.0f, 0.0f, 0.0f);
  vec3 max_bound = vec3(0.0f, 0.0f, 0.0f);

  int32_t vertex_count = 0;
  int32_t triangle_count = 0;
  vector<int32_t> indices = vector<int32_t>(size * size * size);

  for (int32_t x = 0; x < size - 1; ++x) {
    for (int32_t y = 0; y < size - 1; ++y) {
      for (int32_t z = 0; z < size - 1; ++z) {
        const ivec3 p = ivec3(x, y, z);

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

        min_bound.x = glm::min(min_bound.x, float(p.x));
        min_bound.y = glm::min(min_bound.y, float(p.y));
        min_bound.z = glm::min(min_bound.z, float(p.z));

        max_bound.x = glm::max(max_bound.x, float(p.x));
        max_bound.y = glm::max(max_bound.y, float(p.y));
        max_bound.z = glm::max(max_bound.z, float(p.z));

        vec3 vertex = vec3(0.0f, 0.0f, 0.0f);
        int32_t crossing_edge_count = 0;

        for (int32_t i = 0; i < 12; ++i) {
          const int32_t nid0 = edges[i].x;
          const int32_t nid1 = edges[i].y;

          if ((mask >> nid0 & 1) == (mask >> nid1 & 1))
            continue;

          const ivec3 p0 = sample_neighbor_p(p, nid0);
          const ivec3 p1 = sample_neighbor_p(p, nid1);
          const float sd0 = sdf[get_id(p0, size)];
          const float sd1 = sdf[get_id(p1, size)];
          vertex += lerp(vec3(p0), vec3(p1), (0.0f - sd0) / (sd1 - sd0));
          ++crossing_edge_count;
        }

        vertex /= float(crossing_edge_count);
        const vec3 normal = sample_normal(vertex, sdf, size);
        const vec3 tangent = sample_tangent(normal);

        vertices_repr[vertex_count] = vertex;
        normals_repr[vertex_count] = normal;
        tangents_repr[vertex_count] = vec4(tangent, 0.0f);

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

  const vec3 center = (min_bound + max_bound) * 0.5f;
  bounds[0] = center.x;
  bounds[1] = center.y;
  bounds[2] = center.z;

  const vec3 extent = (max_bound - min_bound) * 0.5f;
  bounds[3] = extent.x;
  bounds[4] = extent.y;
  bounds[5] = extent.z;
}

} // namespace dynmesh
