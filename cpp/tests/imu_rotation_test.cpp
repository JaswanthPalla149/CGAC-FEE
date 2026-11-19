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

    std::vector<GyroscopeSample> acceleratingSamples;
    for (std::int64_t timestamp = 0; timestamp <= 100000000;
         timestamp += 10000000) {
        acceleratingSamples.push_back(
            {timestamp, {0.0, 0.0, 0.0}, {1.0, 0.0, 9.81}});
    }
    const auto preintegration = integrateInertialTranslation(
        acceleratingSamples, 0, 100000000, cv::Matx33d::eye(),
        cv::Matx33d::eye(), {0.0, 0.0, -9.81}, {0.0, 0.0, 0.0});
    CHECK(preintegration.has_value());
    CHECK(std::abs(preintegration->displacement[0] - 0.005) < 1e-9);
    CHECK(std::abs(preintegration->displacement[2]) < 1e-9);
    CHECK(std::abs(preintegration->velocity[0] - 0.1) < 1e-9);
    CHECK(!integrateInertialTranslation(
               acceleratingSamples, 0, 300000000, cv::Matx33d::eye(),
               cv::Matx33d::eye(), {0.0, 0.0, -9.81}, {0.0, 0.0, 0.0})
               .has_value());

    const cv::Mat blankImage = cv::Mat::zeros(120, 160, CV_8UC1);
    FeatureSLAM fallbackSlam({100.0, 100.0, 80.0, 60.0});
    CHECK(fallbackSlam
              .process(blankImage, 0, TrackingMode::VisualInertial)
              .trackingSucceeded);
    const FrameResult fallbackFrame = fallbackSlam.process(
        blankImage, 50000000, TrackingMode::VisualInertial, std::nullopt,
        cv::Vec3d(0.005, 0.0, 0.0));
    CHECK(!fallbackFrame.trackingSucceeded);
    CHECK(fallbackFrame.inertialFallback);
    CHECK(std::abs(fallbackFrame.position[0] - 0.005) < 1e-12);
    CHECK(std::abs(fallbackSlam.correctedFramePositions()[1][0]) < 1e-12);

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
    CHECK(std::abs(sequence.gyroscope[1].linearAcceleration[2] - 9.81) <
          1e-12);
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
