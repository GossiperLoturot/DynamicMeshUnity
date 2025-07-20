#pragma once

#include <Eigen/Dense>
#include <IUnityInterface.h>

using IVec2 = Eigen::Vector2i;
using IVec3 = Eigen::Vector3i;
using Vec2 = Eigen::Vector2f;
using Vec3 = Eigen::Vector3f;
using Vec4 = Eigen::Vector4f;
using Bounds = Eigen::Vector<float, 6>;

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
naive_surface_nets(Vec3 *vertices, Vec3 *normals, Vec4 *tangents,
                   int *triangles, Bounds *bounds, const float *sdf, int size);
