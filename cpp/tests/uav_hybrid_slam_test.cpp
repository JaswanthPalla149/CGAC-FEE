#include "adaptive_slam/uav_hybrid_slam.hpp"

#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    using namespace adaptive_slam;

    // 1. Test MultiSensorController 3-Tier Transitions
    MultiSensorController controller(0.48, 0.20, 0.68);

    // Initial low-cost mode
    auto mode = controller.choose(0.85, 0.10, HybridPowerMode::LowCost);
    assert(mode == HybridPowerMode::LowCost);

    // Degradation to recovery
    mode = controller.choose(0.35, 0.10, HybridPowerMode::LowCost);
    assert(mode == HybridPowerMode::Recovery);

    // Hysteresis keeps it in recovery
    mode = controller.choose(0.55, 0.10, HybridPowerMode::Recovery);
    assert(mode == HybridPowerMode::Recovery);

    // Extreme sudden light collapse -> VisualBypassLIO
    mode = controller.choose(0.08, 0.10, HybridPowerMode::Recovery);
    assert(mode == HybridPowerMode::VisualBypassLIO);

    // Stays in VisualBypassLIO if confidence is still low
    mode = controller.choose(0.15, 0.10, HybridPowerMode::VisualBypassLIO);
    assert(mode == HybridPowerMode::VisualBypassLIO);

    // Recovers to recovery when light returns
    mode = controller.choose(0.52, 0.10, HybridPowerMode::VisualBypassLIO);
    assert(mode == HybridPowerMode::Recovery);

    // Recovers to low-cost when confidence is high
    mode = controller.choose(0.75, 0.10, HybridPowerMode::Recovery);
    assert(mode == HybridPowerMode::LowCost);

    // 2. Test UAVHybridSLAM High-Rate IMU Pre-integration
    CameraModel cam{458.65, 457.29, 367.21, 248.13};
    HybridSlamConfig config;
    config.gravity = {0.0, 0.0, 0.0}; // Disable gravity for pure acceleration test
    UAVHybridSLAM slam(cam, config);

    // Feed 10 IMU steps with 1.0 m/s^2 forward acceleration in X
    std::int64_t t = 1000000000LL;
    for (int i = 0; i < 10; ++i) {
        ImuMeasurement imu;
        imu.timestampNs = t;
        imu.linearAccel = {1.0, 0.0, 0.0};
        imu.angularVel = {0.0, 0.0, 0.0};
        slam.processImu(imu);
        t += 10000000LL; // 10 ms (100 Hz)
    }
    const auto imuState = slam.currentState();
    // After 0.09s at 1.0 m/s^2: v = a*t = 0.09 m/s, x = 0.5*a*t^2 = ~0.004 m
    assert(imuState.velocityWorld[0] > 0.05);
    assert(imuState.positionWorld[0] > 0.002);

    // 3. Test Camera Frame Processing & Sudden Illumination Shock Survival
    cv::Mat normalFrame(480, 640, CV_8UC1);
    cv::RNG rng(42);
    rng.fill(normalFrame, cv::RNG::UNIFORM, 50, 200);

    // Nominal frame
    UAVState s1 = slam.processFrame(normalFrame, t);
    assert(s1.mode != HybridPowerMode::VisualBypassLIO);
    t += 33000000LL;

    // Sudden 0-Lux Blackout (e.g. Tunnel entry or night drop)
    cv::Mat blackout = applyIlluminationShock(normalFrame, "blackout");
    UAVState sBlackout = slam.processFrame(blackout, t);

    // System must detect optical failure and shift to VisualBypassLIO
    assert(sBlackout.mode == HybridPowerMode::VisualBypassLIO);
    assert(sBlackout.visualInliers == 0);
    // Visual MACs should be 0 in bypass mode
    assert(sBlackout.estimatedMacCount == 0);

    // 4. Test LiDAR Metric Integration During Optical Blackout
    LidarScan scan;
    scan.timestampNs = t;
    for (int i = 0; i < 100; ++i) {
        scan.points.push_back({static_cast<float>(i * 0.1), 2.0f, 1.5f, 50.0f});
    }
    t += 33000000LL;
    UAVState sLidar = slam.processFrame(blackout, t, scan);
    assert(sLidar.mode == HybridPowerMode::VisualBypassLIO);
    assert(sLidar.lidarInliers > 0);

    // 5. Test Light Restoration
    t += 33000000LL;
    UAVState sRecovered = slam.processFrame(normalFrame, t);
    // Once light returns, it exits visual bypass mode
    assert(sRecovered.mode == HybridPowerMode::Recovery || sRecovered.mode == HybridPowerMode::LowCost);

    // Verify energy and trajectory metrics
    assert(slam.totalMacOperations() > 0);
    assert(slam.totalEnergyJoules() > 0.0);
    assert(slam.trajectory().size() >= 4);

    std::cout << "All UAV Hybrid SLAM & Illumination Shock tests PASSED successfully!\n";
    return 0;
}
