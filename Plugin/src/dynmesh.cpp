#include <Eigen/Dense>
#include <IUnityInterface.h>
#include <dynmesh.hpp>
#include <vector>

Position neighbors[8] = {
    Position(0, 0, 0), Position(1, 0, 0), Position(1, 0, 1), Position(0, 0, 1),
    Position(0, 1, 0), Position(1, 1, 0), Position(1, 1, 1), Position(0, 1, 1),
};

Edge edges[12] = {
    Edge(0, 1), Edge(1, 2), Edge(2, 3), Edge(3, 0), Edge(4, 5), Edge(5, 6),
    Edge(6, 7), Edge(7, 4), Edge(0, 4), Edge(1, 5), Edge(2, 6), Edge(3, 7),
};

VertexPosition lerp(VertexPosition a, VertexPosition b, float t) {
  return a * (1.0 - t) + b * t;
}

Position get_neighbor(Position position, int neighbor_id,
                      bool negative = false) {
  if (!negative) {
    return position + neighbors[neighbor_id];
  } else {
    return position - neighbors[neighbor_id];
  }
}

Id get_id(Position position, int size) {
  return position[0] + position[1] * size + position[2] * size * size;
}

void make_face(VertexId *triangles, int *triangles_count, VertexId id0,
               VertexId id1, VertexId id2, VertexId id3, bool outside) {
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

float interpolate_sdf(VertexPosition p, SDFValue *sdf, int size) {
  int x0 = static_cast<int>(p.x());
  int y0 = static_cast<int>(p.y());
  int z0 = static_cast<int>(p.z());
  int x1 = x0 + 1;
  int y1 = y0 + 1;
  int z1 = z0 + 1;

  float xd = p.x() - x0;
  float yd = p.y() - y0;
  float zd = p.z() - z0;

  float c00 = sdf[get_id(Position(x0, y0, z0), size)] * (1 - xd) +
              sdf[get_id(Position(x1, y0, z0), size)] * xd;
  float c01 = sdf[get_id(Position(x0, y0, z1), size)] * (1 - xd) +
              sdf[get_id(Position(x1, y0, z1), size)] * xd;
  float c10 = sdf[get_id(Position(x0, y1, z0), size)] * (1 - xd) +
              sdf[get_id(Position(x1, y1, z0), size)] * xd;
  float c11 = sdf[get_id(Position(x0, y1, z1), size)] * (1 - xd) +
              sdf[get_id(Position(x1, y1, z1), size)] * xd;

  float c0 = c00 * (1 - yd) + c10 * yd;
  float c1 = c01 * (1 - yd) + c11 * yd;

  return c0 * (1 - zd) + c1 * zd;
}

VertexPosition compute_normal(VertexPosition p, SDFValue *sdf, int size) {
  float h = 0.01f;
  float dx = interpolate_sdf(p + VertexPosition(h, 0, 0), sdf, size) -
             interpolate_sdf(p - VertexPosition(h, 0, 0), sdf, size);
  float dy = interpolate_sdf(p + VertexPosition(0, h, 0), sdf, size) -
             interpolate_sdf(p - VertexPosition(0, h, 0), sdf, size);
  float dz = interpolate_sdf(p + VertexPosition(0, 0, h), sdf, size) -
             interpolate_sdf(p - VertexPosition(0, 0, h), sdf, size);
  return VertexPosition(dx, dy, dz).normalized();
}

// This function implements the naive surface nets algorithm.
// vertices: array of VertexPosition, array length must be at least size^3.
// triangles: array of VertexId, array length must be at least size^3 * 18.
// sdf: array of SDFValue, array length must be at least size^3.
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
naive_surface_nets(VertexPosition *vertices, VertexId *triangles,
                   VertexPosition *normals, SDFValue *sdf, int size) {

  int vertex_count = 0;
  int triangle_count = 0;
  auto indices = std::vector<VertexId>(size * size * size);

  for (int x = 0; x < size - 1; ++x) {
    for (int y = 0; y < size - 1; ++y) {
      for (int z = 0; z < size - 1; ++z) {
        auto p = Position(x, y, z);

        int kind = 0;
        if (0.0f > sdf[get_id(get_neighbor(p, 0), size)])
          kind |= (1 << 0);
        if (0.0f > sdf[get_id(get_neighbor(p, 1), size)])
          kind |= (1 << 1);
        if (0.0f > sdf[get_id(get_neighbor(p, 2), size)])
          kind |= (1 << 2);
        if (0.0f > sdf[get_id(get_neighbor(p, 3), size)])
          kind |= (1 << 3);
        if (0.0f > sdf[get_id(get_neighbor(p, 4), size)])
          kind |= (1 << 4);
        if (0.0f > sdf[get_id(get_neighbor(p, 5), size)])
          kind |= (1 << 5);
        if (0.0f > sdf[get_id(get_neighbor(p, 6), size)])
          kind |= (1 << 6);
        if (0.0f > sdf[get_id(get_neighbor(p, 7), size)])
          kind |= (1 << 7);

        if (kind == 0 || kind == 255)
          continue;

        VertexPosition vertex = VertexPosition::Zero();
        int mix_size = 0;

        for (int i = 0; i < 12; ++i) {
          auto nid0 = edges[i][0];
          auto nid1 = edges[i][1];

          if (((kind >> nid0) & 1) == ((kind >> nid1) & 1))
            continue;

          auto p0 = get_neighbor(p, nid0);
          auto p1 = get_neighbor(p, nid1);
          auto sd0 = sdf[get_id(p0, size)];
          auto sd1 = sdf[get_id(p1, size)];
          vertex += lerp(p0.cast<float>(), p1.cast<float>(),
                         (0.0f - sd0) / (sd1 - sd0));
          ++mix_size;
        }

        vertex /= static_cast<float>(mix_size);

        vertices[vertex_count] = vertex;
        normals[vertex_count] = compute_normal(vertex, sdf, size);
        indices[get_id(p, size)] = vertex_count;
        ++vertex_count;

        if (x == 0 || y == 0 || z == 0)
          continue;

        bool outside = (kind & 1) != 0;
        auto id0 = indices[get_id(get_neighbor(p, 0, true), size)];
        auto id1 = indices[get_id(get_neighbor(p, 1, true), size)];
        auto id2 = indices[get_id(get_neighbor(p, 2, true), size)];
        auto id3 = indices[get_id(get_neighbor(p, 3, true), size)];
        auto id4 = indices[get_id(get_neighbor(p, 4, true), size)];
        auto id5 = indices[get_id(get_neighbor(p, 5, true), size)];
        auto id7 = indices[get_id(get_neighbor(p, 7, true), size)];
        if ((((kind >> 1) & 1) != 0) != outside)
          make_face(triangles, &triangle_count, id0, id3, id7, id4, outside);
        if ((((kind >> 3) & 1) != 0) != outside)
          make_face(triangles, &triangle_count, id0, id4, id5, id1, outside);
        if ((((kind >> 4) & 1) != 0) != outside)
          make_face(triangles, &triangle_count, id0, id1, id2, id3, outside);
      }
    }
  }
}
