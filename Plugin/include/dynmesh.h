#pragma once

#include <stdint.h>

#include "IUnityInterface.h"
#include <Eigen/Dense>

namespace dynmesh {

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API naive_surface_nets(
    const float *sdf, int32_t size, float *vertices, float *normals,
    float *tangents, int32_t *triangles, float bounds[6]);

} // namespace dynmesh
