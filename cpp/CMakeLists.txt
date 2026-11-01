set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs video calib3d)

add_library(adaptive_slam_controller src/controller.c)
target_include_directories(adaptive_slam_controller PUBLIC include)

add_library(adaptive_slam
    src/controller_cpp.cpp
    src/slam.cpp
    src/evaluation.cpp
    src/uav_hybrid_slam.cpp
)
target_include_directories(adaptive_slam PUBLIC include)
target_link_libraries(adaptive_slam PUBLIC adaptive_slam_controller ${OpenCV_LIBS})
if(WIN32)
    target_link_libraries(adaptive_slam PUBLIC Psapi)
endif()
target_compile_options(adaptive_slam PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)

add_executable(adaptive_slam_benchmark src/main.cpp)
target_link_libraries(adaptive_slam_benchmark PRIVATE adaptive_slam)

include(CTest)
if(BUILD_TESTING)
    add_executable(controller_test tests/controller_test.cpp)
    target_link_libraries(controller_test PRIVATE adaptive_slam)
    add_test(NAME controller_test COMMAND controller_test)

    add_executable(controller_c_test tests/controller_c_test.c)
    target_link_libraries(controller_c_test PRIVATE adaptive_slam_controller)
    add_test(NAME controller_c_test COMMAND controller_c_test)

    add_executable(hbst_matcher_test tests/hbst_matcher_test.cpp)
    target_link_libraries(hbst_matcher_test PRIVATE adaptive_slam)
    add_test(NAME hbst_matcher_test COMMAND hbst_matcher_test)

    add_executable(uav_hybrid_slam_test tests/uav_hybrid_slam_test.cpp)
    target_link_libraries(uav_hybrid_slam_test PRIVATE adaptive_slam)
    add_test(NAME uav_hybrid_slam_test COMMAND uav_hybrid_slam_test)
endif()

