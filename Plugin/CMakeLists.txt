cmake_minimum_required(VERSION 3.14)
project(dynmesh VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # ccls

# mold linker
find_program(HAS_MOLD mold)
if (HAS_MOLD)
  set(CMAKE_LINKER_FLAGS "-fuse-ld=mold")
  message(STATUS "Using mold linker")
else()
  message(STATUS "mold linker not found, using default linker")
endif()

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
add_library(dynmesh SHARED src/dynmesh.cc)
target_link_libraries(dynmesh Eigen3::Eigen)
target_include_directories(dynmesh PUBLIC include)
# link for Windows
if (WIN32)
  target_link_options(dynmesh PRIVATE -static)
endif()

# test
enable_testing()
add_executable(dynmesh_test test/main.cc)
target_link_libraries(dynmesh_test GTest::gtest_main dynmesh)
add_test(NAME test COMMAND dynmesh_test)
