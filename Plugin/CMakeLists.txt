cmake_minimum_required(VERSION 3.10)
project(dynmesh VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # ccls

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP ON
)

FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP ON
)

FetchContent_MakeAvailable(googletest eigen)

# library
add_library(dynmesh SHARED src/dynmesh.cpp)
target_link_libraries(dynmesh -static Eigen3::Eigen)
target_include_directories(dynmesh PUBLIC include)

# test
enable_testing()
add_executable(dynmesh_test test/main.cpp)
target_link_libraries(dynmesh_test GTest::gtest_main dynmesh)
add_test(NAME test COMMAND dynmesh_test)
