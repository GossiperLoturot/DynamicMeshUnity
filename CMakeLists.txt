cmake_minimum_required(VERSION 4.0)
project(template VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # ccls

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
)

set(BOOST_ENABLE_CMAKE ON)
set(BOOST_INCLUDE_LIBRARIES geometry)
FetchContent_Declare(
  boost
  URL https://github.com/boostorg/boost/releases/download/boost-1.88.0/boost-1.88.0-cmake.tar.gz
)
FetchContent_MakeAvailable(googletest boost)

# library
add_library(template_lib SHARED src/template.cpp)
target_link_libraries(template_lib Boost::geometry)
target_include_directories(template_lib PUBLIC include)

# test
enable_testing()
add_executable(template_test test/main.cpp)
target_link_libraries(template_test GTest::gtest_main template_lib)
add_test(NAME test COMMAND template_test)

