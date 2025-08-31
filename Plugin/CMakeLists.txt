cmake_minimum_required(VERSION 3.20)
project(dynmesh VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # ccls

# googletest
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(third_party/googletest)

# library
add_library(dynmesh SHARED src/dynmesh.cc)

target_include_directories(dynmesh PUBLIC include)
target_include_directories(dynmesh PRIVATE third_party/glm)

# test
add_executable(dynmesh_test test/main.cc)

target_link_libraries(dynmesh_test dynmesh gtest_main)
target_include_directories(dynmesh_test PRIVATE third_party/googletest/googletest/include)

enable_testing()
add_test(NAME test COMMAND dynmesh_test)

# for Windows static runtime
if (WIN32)
  target_link_options(dynmesh PRIVATE -static)
  target_link_options(dynmesh_test PRIVATE -static)
endif()

