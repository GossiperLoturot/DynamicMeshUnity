#include <Eigen/Dense>
#include <IUnityInterface.h>
#include <dynamic_mesh.hpp>

Eigen::Vector3f neighbor_table[8] = {
    Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 0, 0),
    Eigen::Vector3f(1, 0, 1), Eigen::Vector3f(0, 0, 1),
    Eigen::Vector3f(0, 1, 0), Eigen::Vector3f(1, 1, 0),
    Eigen::Vector3f(1, 1, 1), Eigen::Vector3f(0, 1, 1),
};

int edge_table[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
    {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
};

int get_pos_idx(Eigen::Vector3i p, int neighbor, int size) {
  p += neighbor_table[neighbor];
  return p.x() + p.y() * size + p.z() * size * size;
}

int get_neg_idx(Eigen::Vector3i p, int neighbor, int size) {
  p -= neighbor_table[neighbor];
  return p.x() + p.y() * size + p.z() * size * size;
}

Eigen::Vector3f get_vec(Eigen::Vector3i p, int neighbor) {
  p += neighbor_table[neighbor];
  return p;
}

void make_face(int *i_buf, int *i_cnt, int v0, int v1, int v2, int v3,
               bool outside) {
  if (outside) {
    i_buf[*i_cnt++] = v0;
    i_buf[*i_cnt++] = v3;
    i_buf[*i_cnt++] = v2;
    i_buf[*i_cnt++] = v2;
    i_buf[*i_cnt++] = v1;
    i_buf[*i_cnt++] = v0;
  } else {
    i_buf[*i_cnt++] = v0;
    i_buf[*i_cnt++] = v1;
    i_buf[*i_cnt++] = v2;
    i_buf[*i_cnt++] = v2;
    i_buf[*i_cnt++] = v3;
    i_buf[*i_cnt++] = v0;
  }
}

Eigen::Vector3f lerp(Eigen::Vector3f a, Eigen::Vector3f b, float t) {
  return a + (b - a) * t;
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API naive_surface_nets(
    Eigen::Vector3f *vertices, int *indices, float *sdf, int size) {
  Eigen::Vector3f v_buf[size * size * size];
  int i_buf[size * size * size];
  int v_cnt = 0;
  int i_cnt = 0;

  for (int x = 0; x < size; x++) {
    for (int y = 0; y < size; y++) {
      for (int z = 0; z < size; z++) {
        Eigen::Vector3i p(x, y, z);

        int kind = 0;
        if (0 > sdf[get_pos_idx(p, 0, size)])
          kind |= 1 << 0;
        if (0 > sdf[get_pos_idx(p, 1, size)])
          kind |= 1 << 1;
        if (0 > sdf[get_pos_idx(p, 2, size)])
          kind |= 1 << 2;
        if (0 > sdf[get_pos_idx(p, 3, size)])
          kind |= 1 << 3;
        if (0 > sdf[get_pos_idx(p, 4, size)])
          kind |= 1 << 4;
        if (0 > sdf[get_pos_idx(p, 5, size)])
          kind |= 1 << 5;
        if (0 > sdf[get_pos_idx(p, 6, size)])
          kind |= 1 << 6;
        if (0 > sdf[get_pos_idx(p, 7, size)])
          kind |= 1 << 7;

        if (kind == 0 || kind == 255)
          continue;

        Eigen::Vector3f v;
        int cross = 0;

        for (int i = 0; i < 12; i++) {
          int p0 = edge_table[i][0];
          int p1 = edge_table[i][1];

          if ((kind >> p0 & 1) == (kind >> p1 & 1))
            continue;

          int val0 = sdf[get_pos_idx(p, p0, size)];
          int val1 = sdf[get_pos_idx(p, p1, size)];

          v += lerp(get_vec(p, p0), get_vec(p, p1), (0 - val0) / (val1 - val0));
          cross++;
        }

        v /= cross;

        v_buf[v_cnt] = v;
        i_buf[get_pos_idx(p, 0, size)] = v_cnt;
        v_cnt++;

        if (x == 0 || y == 0 || z == 0)
          continue;

        int outside = (kind & 1) != 0;

        int v0 = i_buf[get_neg_idx(p, 0, size)];
        int v1 = i_buf[get_neg_idx(p, 1, size)];
        int v2 = i_buf[get_neg_idx(p, 2, size)];
        int v3 = i_buf[get_neg_idx(p, 3, size)];
        int v4 = i_buf[get_neg_idx(p, 4, size)];
        int v5 = i_buf[get_neg_idx(p, 5, size)];
        int v7 = i_buf[get_neg_idx(p, 7, size)];

        if ((kind >> 1 & 1) != 0 != outside)
          make_face(i_buf, &i_cnt, v0, v3, v7, v4, outside);
        if ((kind >> 3 & 1) != 0 != outside)
          make_face(i_buf, &i_cnt, v0, v4, v5, v1, outside);
        if ((kind >> 4 & 1) != 0 != outside)
          make_face(i_buf, &i_cnt, v0, v1, v2, v3, outside);
      }
    }
  }
}
