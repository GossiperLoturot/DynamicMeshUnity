#include <IUnityInterface.h>
#include <dynamic_mesh.hpp>

int add(int a, int b) { return a + b; }

extern "C" int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API add_unity(int a,
                                                                    int b) {
  return add(a, b);
}
