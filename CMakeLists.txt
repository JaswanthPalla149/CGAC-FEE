cmake_minimum_required(VERSION 3.16)
project(cgac_slam LANGUAGES C CXX)

find_package(catkin QUIET)

include(CTest)
add_subdirectory(cpp)
