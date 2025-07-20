#pragma once

#include <Eigen/Dense>
#include <IUnityInterface.h>

using SDFValue = float;

using Position = Eigen::Vector3i;
using Edge = Eigen::Vector2i;
using Id = int;

using VertexPosition = Eigen::Vector3f;
using VertexId = int;

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
naive_surface_nets(VertexPosition *vertices, VertexId *triangles,
                   VertexPosition *normals, const SDFValue *sdf, int size);
