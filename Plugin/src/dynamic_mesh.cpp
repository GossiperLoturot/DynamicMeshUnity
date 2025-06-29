#include <Eigen/Dense>
#include <IUnityInterface.h>
#include <dynamic_mesh.hpp>

Position neighbors[8] = {
    Position(0, 0, 0), Position(1, 0, 0), Position(1, 0, 1), Position(0, 0, 1),
    Position(0, 1, 0), Position(1, 1, 0), Position(1, 1, 1), Position(0, 1, 1),
};

Edge edges[12] = {
    Edge(0, 1), Edge(1, 2), Edge(2, 3), Edge(3, 0), Edge(4, 5), Edge(5, 6),
    Edge(6, 7), Edge(7, 4), Edge(0, 4), Edge(1, 5), Edge(2, 6), Edge(3, 7),
};

Position get_neighbor(Position position, size_t neighbor_id,
                      bool negative = false) {
  if (!negative) {
    return position + neighbors[neighbor_id];
  } else {
    return position - neighbors[neighbor_id];
  }
}

Id get_id(Position position, size_t size) {
  return position.x() + position.y() * size + position.z() * size * size;
}

VertexPosition lerp(VertexPosition a, VertexPosition b, float t) {
  return a + (b - a) * t;
}

void make_face(VertexId *triangles, size_t *triangles_count, VertexId id0,
               VertexId id1, VertexId id2, VertexId id3, bool outside) {
  if (outside) {
    triangles[*triangles_count++] = id0;
    triangles[*triangles_count++] = id3;
    triangles[*triangles_count++] = id2;
    triangles[*triangles_count++] = id2;
    triangles[*triangles_count++] = id1;
    triangles[*triangles_count++] = id0;
  } else {
    triangles[*triangles_count++] = id0;
    triangles[*triangles_count++] = id1;
    triangles[*triangles_count++] = id2;
    triangles[*triangles_count++] = id2;
    triangles[*triangles_count++] = id3;
    triangles[*triangles_count++] = id0;
  }
}

// This function implements the naive surface nets algorithm.
// vertices: array of VertexPosition, array length must be at least size^3.
// triangles: array of VertexId, array length must be at least size^3 * 18.
// sdf: array of SDFValue, array length must be at least size^3.
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API naive_surface_nets(
    VertexPosition *vertices, VertexId *triangles, SDFValue *sdf, size_t size) {

  size_t vertex_count = 0;
  size_t triangle_count = 0;
  VertexId indices[size * size * size];

  for (size_t x = 0; x < size - 1; x++) {
    for (size_t y = 0; y < size - 1; y++) {
      for (size_t z = 0; z < size - 1; z++) {
        auto p = Position(x, y, z);

        size_t kind = 0;
        if (0 > sdf[get_id(get_neighbor(p, 0), size)])
          kind |= 1 << 0;
        if (0 > sdf[get_id(get_neighbor(p, 1), size)])
          kind |= 1 << 1;
        if (0 > sdf[get_id(get_neighbor(p, 2), size)])
          kind |= 1 << 2;
        if (0 > sdf[get_id(get_neighbor(p, 3), size)])
          kind |= 1 << 3;
        if (0 > sdf[get_id(get_neighbor(p, 4), size)])
          kind |= 1 << 4;
        if (0 > sdf[get_id(get_neighbor(p, 5), size)])
          kind |= 1 << 5;
        if (0 > sdf[get_id(get_neighbor(p, 6), size)])
          kind |= 1 << 6;
        if (0 > sdf[get_id(get_neighbor(p, 7), size)])
          kind |= 1 << 7;

        if (kind == 0 || kind == 255)
          continue;

        VertexPosition vertex;
        size_t mix_size = 0;

        for (int i = 0; i < 12; i++) {
          auto id0 = edges[i].x();
          auto id1 = edges[i].y();

          if ((kind >> id0 & 1) == (kind >> id1 & 1))
            continue;

          auto sd0 = sdf[get_id(get_neighbor(p, id0), size)];
          auto sd1 = sdf[get_id(get_neighbor(p, id1), size)];
          auto p0 = get_neighbor(p, id0, size);
          auto p1 = get_neighbor(p, id1, size);
          vertex +=
              lerp(p0.cast<float>(), p1.cast<float>(), (0 - sd0) / (sd1 - sd0));
          mix_size++;
        }

        vertex = vertex / static_cast<float>(mix_size);

        vertices[vertex_count] = vertex;
        indices[get_id(get_neighbor(p, 0), size)] = vertex_count;
        vertex_count++;

        if (p == Position::Zero())
          continue;

        auto outside = (kind & 1) != 0;
        auto id0 = indices[get_id(get_neighbor(p, 0, true), size)];
        auto id1 = indices[get_id(get_neighbor(p, 1, true), size)];
        auto id2 = indices[get_id(get_neighbor(p, 2, true), size)];
        auto id3 = indices[get_id(get_neighbor(p, 3, true), size)];
        auto id4 = indices[get_id(get_neighbor(p, 4, true), size)];
        auto id5 = indices[get_id(get_neighbor(p, 5, true), size)];
        auto id6 = indices[get_id(get_neighbor(p, 7, true), size)];
        if ((kind >> 1 & 1) != 0 != outside)
          make_face(triangles, &triangle_count, id0, id3, id6, id4, outside);
        if ((kind >> 3 & 1) != 0 != outside)
          make_face(triangles, &triangle_count, id0, id4, id5, id1, outside);
        if ((kind >> 4 & 1) != 0 != outside)
          make_face(triangles, &triangle_count, id0, id1, id2, id3, outside);
      }
    }
  }
}
