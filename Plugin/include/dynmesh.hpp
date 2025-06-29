#pragma once

#include <Eigen/Dense>
#include <IUnityInterface.h>

typedef float SDFValue;

typedef Eigen::Vector3i Position;
typedef Eigen::Vector2i Edge;
typedef int Id;

typedef Eigen::Vector3f VertexPosition;
typedef int VertexId;

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API naive_surface_nets(
    VertexPosition *vertices, VertexId *triangles, SDFValue *sdf, int size);
