cmake_minimum_required(VERSION 3.10)
project(dynamic_mesh VERSION 0.1.0 LANGUAGES CXX)

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

set(BOOST_ENABLE_CMAKE ON)
set(BOOST_INCLUDE_LIBRARIES geometry)
FetchContent_Declare(
  boost
  URL https://github.com/boostorg/boost/releases/download/boost-1.88.0/boost-1.88.0-cmake.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP ON
)
FetchContent_MakeAvailable(googletest boost eigen)

# library
add_library(dynamic_mesh_lib SHARED src/dynamic_mesh.cpp)
target_link_libraries(dynamic_mesh_lib Boost::geometry Eigen3::Eigen)
target_include_directories(dynamic_mesh_lib PUBLIC include)

# test
enable_testing()
add_executable(dynamic_mesh_test test/main.cpp)
target_link_libraries(dynamic_mesh_test GTest::gtest_main dynamic_mesh_lib)
add_test(NAME test COMMAND dynamic_mesh_test)

