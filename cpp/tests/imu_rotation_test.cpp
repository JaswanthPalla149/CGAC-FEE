#include "adaptive_slam/slam.hpp"
#include "test_check.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
    using namespace adaptive_slam;

    const std::vector<GyroscopeSample> samples{
        {0, {0.0, 0.0, 1.0}},
        {500000000, {0.0, 0.0, 1.0}},
        {1000000000, {0.0, 0.0, 1.0}},
    };
    const auto rotation = integrateGyroscopeRotation(
        samples, 0, 1000000000, cv::Matx33d::eye());
    CHECK(rotation.has_value());
    CHECK(std::abs((*rotation)(0, 0) - std::cos(1.0)) < 1e-8);
    CHECK(std::abs((*rotation)(0, 1) - std::sin(1.0)) < 1e-8);
    CHECK(std::abs((*rotation)(1, 0) + std::sin(1.0)) < 1e-8);
    CHECK(std::abs((*rotation)(1, 1) - std::cos(1.0)) < 1e-8);
    CHECK(std::abs(cv::determinant(cv::Mat(*rotation)) - 1.0) < 1e-8);

    CHECK(!integrateGyroscopeRotation(
               samples, -1, 500000000, cv::Matx33d::eye())
               .has_value());
    CHECK(!integrateGyroscopeRotation(
               samples, 500000000, 500000000, cv::Matx33d::eye())
               .has_value());

    const auto tempRoot = std::filesystem::temp_directory_path() /
                          ("slam_imu_test_" + std::to_string(
                               std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count()));
    const auto cameraDir = tempRoot / "mav0" / "cam0";
    const auto imuDir = tempRoot / "mav0" / "imu0";
    std::filesystem::create_directories(cameraDir / "data");
    std::filesystem::create_directories(imuDir);
    {
        std::ofstream cameraCalibration(cameraDir / "sensor.yaml");
        cameraCalibration
            << "T_BS:\n"
               "  data:\n"
               "    - [1, 0, 0, 0]\n"
               "    - [0, 1, 0, 0]\n"
               "    - [0, 0, 1, 0]\n"
               "intrinsics: [100, 100, 50, 50]\n"
               "distortion_coefficients: [-0.1, 0.01, 0, 0]\n";
        std::ofstream imuCalibration(imuDir / "sensor.yaml");
        imuCalibration
            << "T_BS:\n"
               "  data:\n"
               "    - [0, -1, 0, 0]\n"
               "    - [1, 0, 0, 0]\n"
               "    - [0, 0, 1, 0]\n";
        std::ofstream frameIndex(cameraDir / "data.csv");
        frameIndex << "#timestamp,filename\n"
                      "50000000,frame0.png\n"
                      "150000000,frame1.png\n";
        std::ofstream(cameraDir / "data" / "frame0.png").put('\0');
        std::ofstream(cameraDir / "data" / "frame1.png").put('\0');
        std::ofstream imuData(imuDir / "data.csv");
        imuData << "#timestamp,wx,wy,wz,ax,ay,az\n"
                   "0,1,0,0,0,0,9.81\n"
                   "100000000,1,0,0,0,0,9.81\n"
                   "200000000,1,0,0,0,0,9.81\n";
    }
    const EuRoCSequence sequence = loadEuRoC(tempRoot);
    CHECK(sequence.frames.size() == 2);
    CHECK(sequence.gyroscope.size() == 3);
    CHECK(std::abs(sequence.camera.distortion[0] + 0.1) < 1e-12);
    CHECK(std::abs(sequence.cameraFromImu(0, 1) + 1.0) < 1e-12);
    CHECK(std::abs(sequence.cameraFromImu(1, 0) - 1.0) < 1e-12);
    const auto extrinsicRotation = integrateGyroscopeRotation(
        sequence.gyroscope, 50000000, 150000000, sequence.cameraFromImu);
    CHECK(extrinsicRotation.has_value());
    CHECK(std::abs((*extrinsicRotation)(0, 2) + std::sin(0.1)) < 1e-8);
    CHECK(std::abs((*extrinsicRotation)(2, 0) - std::sin(0.1)) < 1e-8);
    std::error_code cleanupError;
    std::filesystem::remove_all(tempRoot, cleanupError);
    CHECK(!cleanupError);

    std::cout << "Gyroscope rotation integration tests passed.\n";
    return 0;
}
