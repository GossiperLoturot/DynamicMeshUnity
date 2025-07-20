#include <Eigen/Dense>
#include <IUnityInterface.h>
#include <algorithm>
#include <dynmesh.hpp>
#include <vector>

IVec3 neighbors[8] = {
    IVec3(0, 0, 0), IVec3(1, 0, 0), IVec3(1, 0, 1), IVec3(0, 0, 1),
    IVec3(0, 1, 0), IVec3(1, 1, 0), IVec3(1, 1, 1), IVec3(0, 1, 1),
};

IVec2 edges[12] = {
    IVec2(0, 1), IVec2(1, 2), IVec2(2, 3), IVec2(3, 0),
    IVec2(4, 5), IVec2(5, 6), IVec2(6, 7), IVec2(7, 4),
    IVec2(0, 4), IVec2(1, 5), IVec2(2, 6), IVec2(3, 7),
};

float lerp(float a, float b, float t) { return a * (1.0 - t) + b * t; }

Vec3 lerp(const Vec3 &a, const Vec3 &b, float t) {
  return a * (1.0 - t) + b * t;
}

int get_id(const IVec3 &position, int size) {
  return position[0] + position[1] * size + position[2] * size * size;
}

IVec3 sample_neighbor(const IVec3 &position, int neighbor_id,
                      bool negative = false) {
  if (!negative) {
    return position + neighbors[neighbor_id];
  } else {
    return position - neighbors[neighbor_id];
  }
}

float sample_sdf(const Vec3 &p, const float *sdf, int size) {
  int x0 = static_cast<int>(p.x());
  int y0 = static_cast<int>(p.y());
  int z0 = static_cast<int>(p.z());
  int x1 = x0 + 1;
  int y1 = y0 + 1;
  int z1 = z0 + 1;

  float xd = p.x() - x0;
  float yd = p.y() - y0;
  float zd = p.z() - z0;

  float c00 = lerp(sdf[get_id(IVec3(x0, y0, z0), size)],
                   +sdf[get_id(IVec3(x1, y0, z0), size)], xd);
  float c01 = lerp(sdf[get_id(IVec3(x0, y0, z1), size)],
                   sdf[get_id(IVec3(x1, y0, z1), size)], xd);
  float c10 = lerp(sdf[get_id(IVec3(x0, y1, z0), size)],
                   sdf[get_id(IVec3(x1, y1, z0), size)], xd);
  float c11 = lerp(sdf[get_id(IVec3(x0, y1, z1), size)],
                   sdf[get_id(IVec3(x1, y1, z1), size)], xd);
  float c0 = lerp(c00, c10, yd);
  float c1 = lerp(c01, c11, yd);
  return lerp(c0, c1, zd);
}

Vec3 sample_normal(const Vec3 &p, const float *sdf, int size) {
  float h = 0.01f;
  float dx = sample_sdf(p + Vec3(h, 0, 0), sdf, size) -
             sample_sdf(p - Vec3(h, 0, 0), sdf, size);
  float dy = sample_sdf(p + Vec3(0, h, 0), sdf, size) -
             sample_sdf(p - Vec3(0, h, 0), sdf, size);
  float dz = sample_sdf(p + Vec3(0, 0, h), sdf, size) -
             sample_sdf(p - Vec3(0, 0, h), sdf, size);
  return Vec3(dx, dy, dz).normalized();
}

Vec3 sample_tangent(const Vec3 &n) {
  Vec3 tangent_candidate = n.cross(Vec3(0.0f, 1.0f, 0.0f));
  if (tangent_candidate.norm() < 1e-6) {
    tangent_candidate = n.cross(Vec3(0.0f, 0.0f, 1.0f));
  }
  return tangent_candidate.normalized();
}

void make_face(int *triangles, int &triangles_count, int id0, int id1, int id2,
               int id3, bool outside) {
  if (outside) {
    triangles[triangles_count++] = id0;
    triangles[triangles_count++] = id3;
    triangles[triangles_count++] = id2;
    triangles[triangles_count++] = id2;
    triangles[triangles_count++] = id1;
    triangles[triangles_count++] = id0;
  } else {
    triangles[triangles_count++] = id0;
    triangles[triangles_count++] = id1;
    triangles[triangles_count++] = id2;
    triangles[triangles_count++] = id2;
    triangles[triangles_count++] = id3;
    triangles[triangles_count++] = id0;
  }
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
naive_surface_nets(Vec3 *vertices, Vec3 *normals, Vec4 *tangents,
                   int *triangles, Bounds *bounds, const float *sdf, int size) {

  Vec3 min_p = Vec3::Zero();
  Vec3 max_p = Vec3::Zero();

  int vertex_count = 0;
  int triangle_count = 0;
  auto indices = std::vector<int>(size * size * size);

  for (int x = 0; x < size - 1; ++x) {
    for (int y = 0; y < size - 1; ++y) {
      for (int z = 0; z < size - 1; ++z) {
        auto p = IVec3(x, y, z);

        int mask = 0;
        if (0.0f > sdf[get_id(sample_neighbor(p, 0), size)])
          mask |= (1 << 0);
        if (0.0f > sdf[get_id(sample_neighbor(p, 1), size)])
          mask |= (1 << 1);
        if (0.0f > sdf[get_id(sample_neighbor(p, 2), size)])
          mask |= (1 << 2);
        if (0.0f > sdf[get_id(sample_neighbor(p, 3), size)])
          mask |= (1 << 3);
        if (0.0f > sdf[get_id(sample_neighbor(p, 4), size)])
          mask |= (1 << 4);
        if (0.0f > sdf[get_id(sample_neighbor(p, 5), size)])
          mask |= (1 << 5);
        if (0.0f > sdf[get_id(sample_neighbor(p, 6), size)])
          mask |= (1 << 6);
        if (0.0f > sdf[get_id(sample_neighbor(p, 7), size)])
          mask |= (1 << 7);

        if (mask == 0 || mask == 255)
          continue;

        min_p = min_p.cwiseMin(p.cast<float>());
        max_p = max_p.cwiseMax(p.cast<float>());

        Vec3 vertex = Vec3::Zero();
        int crossing_edge_count = 0;

        for (int i = 0; i < 12; ++i) {
          auto nid0 = edges[i][0];
          auto nid1 = edges[i][1];

          if (((mask >> nid0) & 1) == ((mask >> nid1) & 1))
            continue;

          auto p0 = sample_neighbor(p, nid0);
          auto p1 = sample_neighbor(p, nid1);
          auto sd0 = sdf[get_id(p0, size)];
          auto sd1 = sdf[get_id(p1, size)];
          vertex += lerp(p0.cast<float>(), p1.cast<float>(),
                         (0.0f - sd0) / (sd1 - sd0));
          ++crossing_edge_count;
        }

        vertex /= static_cast<float>(crossing_edge_count);
        Vec3 normal = sample_normal(vertex, sdf, size);
        Vec3 tangent = sample_tangent(normals[vertex_count]);

        vertices[vertex_count] = vertex;
        normals[vertex_count] = normal;
        tangents[vertex_count] =
            Vec4(tangent.x(), tangent.y(), tangent.z(), 0.0f);

        indices[get_id(p, size)] = vertex_count;
        ++vertex_count;

        if (x == 0 || y == 0 || z == 0)
          continue;

        bool outside = (mask & 1) != 0;
        auto id0 = indices[get_id(sample_neighbor(p, 0, true), size)];
        auto id1 = indices[get_id(sample_neighbor(p, 1, true), size)];
        auto id2 = indices[get_id(sample_neighbor(p, 2, true), size)];
        auto id3 = indices[get_id(sample_neighbor(p, 3, true), size)];
        auto id4 = indices[get_id(sample_neighbor(p, 4, true), size)];
        auto id5 = indices[get_id(sample_neighbor(p, 5, true), size)];
        auto id7 = indices[get_id(sample_neighbor(p, 7, true), size)];
        if ((((mask >> 1) & 1) != 0) != outside)
          make_face(triangles, triangle_count, id0, id3, id7, id4, outside);
        if ((((mask >> 3) & 1) != 0) != outside)
          make_face(triangles, triangle_count, id0, id4, id5, id1, outside);
        if ((((mask >> 4) & 1) != 0) != outside)
          make_face(triangles, triangle_count, id0, id1, id2, id3, outside);
      }
    }
  }

  Vec3 center = (min_p + max_p) * 0.5f;
  Vec3 extent = (max_p - min_p) * 0.5f;
  *bounds = Bounds(center.x(), center.y(), center.z(), extent.x(), extent.y(),
                   extent.z());
}
