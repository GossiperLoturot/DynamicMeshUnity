#pragma once

#include <Eigen/Dense>
#include <IUnityInterface.h>

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API naive_surface_nets(
    Eigen::Vector3f *vertices, int *indices, float *sdf, int size);
