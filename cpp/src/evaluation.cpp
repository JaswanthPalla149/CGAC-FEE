#include "adaptive_slam/slam.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string_view>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace adaptive_slam {
namespace {

const std::vector<std::string> kModes{"fixed_low", "fixed_high", "adaptive"};
const std::vector<std::string> kDegradations{"none", "blur", "low_light", "noise"};

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> values;
    std::stringstream stream(line);
    std::string value;
    while (std::getline(stream, value, ',')) {
        const auto first = value.find_first_not_of(" \t\r\n");
        const auto last = value.find_last_not_of(" \t\r\n");
        values.push_back(first == std::string::npos
                             ? std::string{}
                             : value.substr(first, last - first + 1));
    }
    return values;
}

bool isCommentOrEmpty(const std::string& line) {
    const auto first = line.find_first_not_of(" \t\r\n");
    return first == std::string::npos || line[first] == '#';
}

std::int64_t processRssBytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        return static_cast<std::int64_t>(counters.WorkingSetSize);
    }
#elif defined(__linux__)
    std::ifstream status("/proc/self/statm");
    std::int64_t totalPages = 0;
    std::int64_t residentPages = 0;
    if (status >> totalPages >> residentPages) {
        const long pageSize = sysconf(_SC_PAGESIZE);
        if (pageSize > 0) {
            return residentPages * static_cast<std::int64_t>(pageSize);
        }
    }
#endif
    return -1;
}

std::vector<cv::Vec3d> interpolateGroundTruth(
    const std::vector<std::int64_t>& frameTimestamps,
    const GroundTruth& groundTruth) {
    std::vector<cv::Vec3d> values;
    values.reserve(frameTimestamps.size());
    for (std::int64_t timestamp : frameTimestamps) {
        auto upper = std::lower_bound(groundTruth.timestampsNs.begin(),
                                      groundTruth.timestampsNs.end(), timestamp);
        if (upper == groundTruth.timestampsNs.begin()) {
            values.push_back(groundTruth.positions.front());
            continue;
        }
        if (upper == groundTruth.timestampsNs.end()) {
            values.push_back(groundTruth.positions.back());
            continue;
        }
        const std::size_t right =
            static_cast<std::size_t>(upper - groundTruth.timestampsNs.begin());
        const std::size_t left = right - 1;
        const double fraction =
            static_cast<double>(timestamp - groundTruth.timestampsNs[left]) /
            static_cast<double>(groundTruth.timestampsNs[right] -
                               groundTruth.timestampsNs[left]);
        values.push_back(groundTruth.positions[left] * (1.0 - fraction) +
                         groundTruth.positions[right] * fraction);
    }
    return values;
}

double alignedAte(const std::vector<cv::Vec3d>& estimate,
                  const std::vector<cv::Vec3d>& target) {
    if (estimate.size() != target.size() || estimate.size() < 3) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    cv::Vec3d estimateMean(0.0, 0.0, 0.0);
    cv::Vec3d targetMean(0.0, 0.0, 0.0);
    for (std::size_t i = 0; i < estimate.size(); ++i) {
        estimateMean += estimate[i];
        targetMean += target[i];
    }
    estimateMean *= 1.0 / static_cast<double>(estimate.size());
    targetMean *= 1.0 / static_cast<double>(target.size());

    cv::Mat covariance = cv::Mat::zeros(3, 3, CV_64F);
    double variance = 0.0;
    for (std::size_t i = 0; i < estimate.size(); ++i) {
        const cv::Vec3d source = estimate[i] - estimateMean;
        const cv::Vec3d destination = target[i] - targetMean;
        variance += source.dot(source);
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                covariance.at<double>(row, column) +=
                    destination[row] * source[column] /
                    static_cast<double>(estimate.size());
            }
        }
    }
    variance /= static_cast<double>(estimate.size());
    cv::SVD decomposition(covariance, cv::SVD::FULL_UV);
    cv::Mat correction = cv::Mat::eye(3, 3, CV_64F);
    const cv::Mat product = decomposition.u * decomposition.vt;
    correction.at<double>(2, 2) = cv::determinant(product) < 0.0 ? -1.0 : 1.0;
    const cv::Mat rotation = decomposition.u * correction * decomposition.vt;
    const double scale =
        (decomposition.w.at<double>(0) + decomposition.w.at<double>(1) +
         decomposition.w.at<double>(2) * correction.at<double>(2, 2)) /
        std::max(variance, 1e-12);

    double squaredError = 0.0;
    for (std::size_t i = 0; i < estimate.size(); ++i) {
        cv::Mat source = (cv::Mat_<double>(3, 1) << estimate[i][0] - estimateMean[0],
                          estimate[i][1] - estimateMean[1],
                          estimate[i][2] - estimateMean[2]);
        cv::Mat aligned = scale * rotation * source;
        const cv::Vec3d residual(
            aligned.at<double>(0) + targetMean[0] - target[i][0],
            aligned.at<double>(1) + targetMean[1] - target[i][1],
            aligned.at<double>(2) + targetMean[2] - target[i][2]);
        squaredError += residual.dot(residual);
    }
    return std::sqrt(squaredError / static_cast<double>(estimate.size()));
}

void writeFrameRows(const std::filesystem::path& path,
                    const std::vector<FrameResult>& frames) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write " + path.string());
    }
    output << "timestamp_ns,x,y,z,confidence,tracked_ratio,inlier_ratio,"
              "reprojection_error_px,active_features,inliers,map_inliers,"
              "elapsed_ms,mode\n"
           << std::setprecision(10);
    for (const FrameResult& frame : frames) {
        output << frame.timestampNs << ',' << frame.position[0] << ','
               << frame.position[1] << ',' << frame.position[2] << ','
               << frame.confidence << ',' << frame.trackedRatio << ','
               << frame.inlierRatio << ',' << frame.reprojectionErrorPx << ','
               << frame.activeFeatures << ',' << frame.inliers << ','
               << frame.mapInliers << ',' << frame.elapsedMs << ','
               << modeName(frame.mode) << '\n';
    }
}

void writeTrajectory(const std::filesystem::path& path,
                     const std::vector<FrameResult>& frames) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write " + path.string());
    }
    output << "timestamp_ns,x,y,z\n" << std::setprecision(10);
    for (const FrameResult& frame : frames) {
        output << frame.timestampNs << ',' << frame.position[0] << ','
               << frame.position[1] << ',' << frame.position[2] << '\n';
    }
}

void writeMap(const std::filesystem::path& path, const FeatureSLAM& slam) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write " + path.string());
    }
    output << "# Monocular map; coordinates use arbitrary initialized scale\n"
           << "[landmarks]\nid,x,y,z\n"
           << std::setprecision(10);
    const auto& landmarks = slam.landmarks();
    for (std::size_t i = 0; i < landmarks.size(); ++i) {
        output << i << ',' << landmarks[i].x << ',' << landmarks[i].y << ','
               << landmarks[i].z << '\n';
    }
    output << "\n[keyframes]\ntimestamp_ns,x,y,z,r00,r01,r02,r10,r11,r12,r20,r21,r22\n";
    for (const Keyframe& keyframe : slam.keyframes()) {
        output << keyframe.timestampNs << ',' << keyframe.positionWorld[0] << ','
               << keyframe.positionWorld[1] << ',' << keyframe.positionWorld[2];
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                output << ',' << keyframe.rotationWorldCamera(row, column);
            }
        }
        output << '\n';
    }
}

}  // namespace

EuRoCSequence loadEuRoC(const std::filesystem::path& sequencePath) {
    const std::filesystem::path cameraPath =
        sequencePath / "mav0" / "cam0";
    const std::filesystem::path calibrationPath = cameraPath / "sensor.yaml";
    std::ifstream calibration(calibrationPath);
    if (!calibration) {
        throw std::runtime_error("cannot read calibration " +
                                 calibrationPath.string());
    }
    const std::regex intrinsicsPattern(
        R"(intrinsics\s*:\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\])");
    std::smatch match;
    std::string line;
    CameraModel camera;
    bool foundIntrinsics = false;
    while (std::getline(calibration, line)) {
        if (std::regex_search(line, match, intrinsicsPattern)) {
            camera = {std::stod(match[1]), std::stod(match[2]),
                      std::stod(match[3]), std::stod(match[4])};
            foundIntrinsics = true;
            break;
        }
    }
    if (!foundIntrinsics || camera.fx <= 0.0 || camera.fy <= 0.0) {
        throw std::runtime_error("invalid or missing camera intrinsics in " +
                                 calibrationPath.string());
    }

    EuRoCSequence sequence;
    sequence.camera = camera;
    const std::filesystem::path frameCsv = cameraPath / "data.csv";
    std::ifstream frames(frameCsv);
    if (!frames) {
        throw std::runtime_error("cannot read frame index " + frameCsv.string());
    }
    const std::filesystem::path imageDirectory = cameraPath / "data";
    while (std::getline(frames, line)) {
        if (isCommentOrEmpty(line)) {
            continue;
        }
        const auto columns = splitCsvLine(line);
        if (columns.size() < 2) {
            continue;
        }
        const auto imagePath = imageDirectory / columns[1];
        if (std::filesystem::is_regular_file(imagePath)) {
            sequence.frames.push_back({std::stoll(columns[0]), imagePath});
        }
    }
    if (sequence.frames.empty()) {
        throw std::runtime_error("no images found in " + imageDirectory.string());
    }

    const std::filesystem::path truthPath =
        sequencePath / "mav0" / "state_groundtruth_estimate0" / "data.csv";
    std::ifstream truth(truthPath);
    if (truth) {
        GroundTruth groundTruth;
        while (std::getline(truth, line)) {
            if (isCommentOrEmpty(line)) {
                continue;
            }
            const auto columns = splitCsvLine(line);
            if (columns.size() < 4) {
                continue;
            }
            try {
                groundTruth.timestampsNs.push_back(std::stoll(columns[0]));
                groundTruth.positions.emplace_back(
                    std::stod(columns[1]), std::stod(columns[2]),
                    std::stod(columns[3]));
            } catch (const std::invalid_argument&) {
                continue;
            }
        }
        if (!groundTruth.timestampsNs.empty()) {
            sequence.groundTruth = std::move(groundTruth);
        }
    }
    return sequence;
}

cv::Mat applyDegradation(const cv::Mat& image, const std::string& degradation,
                         std::uint32_t seed) {
    if (degradation == "none") {
        return image;
    }
    if (degradation == "blur") {
        cv::Mat output;
        cv::GaussianBlur(image, output, cv::Size(7, 7), 1.6);
        return output;
    }
    if (degradation == "low_light") {
        cv::Mat output;
        image.convertTo(output, image.type(), 0.45, -18.0);
        return output;
    }
    if (degradation == "noise") {
        cv::Mat noise(image.size(), CV_16SC1);
        cv::RNG random(seed);
        random.fill(noise, cv::RNG::NORMAL, 0, 12);
        cv::Mat source16;
        image.convertTo(source16, CV_16SC1);
        cv::Mat output16 = source16 + noise;
        cv::Mat output;
        output16.convertTo(output, image.type());
        return output;
    }
    throw std::invalid_argument("unsupported degradation: " + degradation);
}

RunSummary runConfiguration(const std::filesystem::path& sequencePath,
                            const std::filesystem::path& outputPath,
                            const std::string& configuration,
                            const std::string& degradation,
                            std::uint32_t seed) {
    if (std::find(kModes.begin(), kModes.end(), configuration) == kModes.end()) {
        throw std::invalid_argument("unsupported configuration: " + configuration);
    }
    if (std::find(kDegradations.begin(), kDegradations.end(), degradation) ==
        kDegradations.end()) {
        throw std::invalid_argument("unsupported degradation: " + degradation);
    }

    const EuRoCSequence sequence = loadEuRoC(sequencePath);
    FeatureSLAM slam(sequence.camera);
    AdaptiveController controller;
    TrackingMode mode = configuration == "fixed_high"
                            ? TrackingMode::Recovery
                            : TrackingMode::LowCost;
    std::vector<FrameResult> results;
    results.reserve(sequence.frames.size());
    std::size_t transitions = 0;
    const std::int64_t rssBefore = processRssBytes();
    const std::clock_t cpuStart = std::clock();
    const auto wallStart = std::chrono::steady_clock::now();

    for (std::size_t i = 0; i < sequence.frames.size(); ++i) {
        const EuRoCFrame& frame = sequence.frames[i];
        cv::Mat image = cv::imread(frame.imagePath.string(), cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            throw std::runtime_error("failed to load image " + frame.imagePath.string());
        }
        image = applyDegradation(image, degradation,
                                 seed + static_cast<std::uint32_t>(i));
        FrameResult result = slam.process(image, frame.timestampNs, mode);
        results.push_back(result);
        if (configuration == "adaptive") {
            const TrackingMode nextMode = controller.choose(result.confidence, mode);
            transitions += nextMode != mode ? 1U : 0U;
            mode = nextMode;
        }
    }

    const double wallSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart)
            .count();
    const double cpuSeconds =
        static_cast<double>(std::clock() - cpuStart) / CLOCKS_PER_SEC;
    const std::int64_t rssAfter = processRssBytes();

    RunSummary summary;
    summary.sequence = sequencePath.filename().string();
    summary.configuration = configuration;
    summary.degradation = degradation;
    summary.frames = results.size();
    summary.landmarks = slam.landmarks().size();
    summary.keyframes = slam.keyframes().size();
    summary.wallTimeSeconds = wallSeconds;
    summary.cpuTimeSeconds = cpuSeconds;
    summary.rssAfterBytes = rssAfter;
    summary.rssDeltaBytes =
        rssBefore >= 0 && rssAfter >= 0 ? rssAfter - rssBefore : -1;
    summary.modeTransitions = transitions;

    std::vector<double> latencies;
    std::vector<cv::Vec3d> positions;
    std::vector<std::int64_t> timestamps;
    std::size_t recoveryFrames = 0;
    for (std::size_t i = 0; i < results.size(); ++i) {
        const FrameResult& result = results[i];
        latencies.push_back(result.elapsedMs);
        positions.push_back(result.position);
        timestamps.push_back(result.timestampNs);
        recoveryFrames += result.mode == TrackingMode::Recovery ? 1U : 0U;
        if (i > 0 && result.inliers < 12) {
            ++summary.trackingFailures;
        }
        summary.framesWithMapPose += result.mapInliers >= 12 ? 1U : 0U;
    }
    std::sort(latencies.begin(), latencies.end());
    if (!latencies.empty()) {
        summary.meanLatencyMs =
            std::accumulate(latencies.begin(), latencies.end(), 0.0) /
            static_cast<double>(latencies.size());
        const std::size_t p95Index = static_cast<std::size_t>(
            std::ceil(0.95 * static_cast<double>(latencies.size())) - 1.0);
        summary.p95LatencyMs = latencies[std::min(p95Index, latencies.size() - 1)];
    }
    summary.failureRate =
        static_cast<double>(summary.trackingFailures) /
        static_cast<double>(std::max<std::size_t>(1, results.size() - 1));
    summary.recoveryFraction =
        static_cast<double>(recoveryFrames) /
        static_cast<double>(std::max<std::size_t>(1, results.size()));
    if (sequence.groundTruth.has_value()) {
        summary.ateRmse =
            alignedAte(positions, interpolateGroundTruth(timestamps,
                                                          *sequence.groundTruth));
    }

    std::filesystem::create_directories(outputPath);
    const std::string stem = configuration + "_" + degradation;
    writeFrameRows(outputPath / (stem + "_frames.csv"), results);
    writeTrajectory(outputPath / (stem + "_trajectory.csv"), results);
    writeMap(outputPath / (stem + "_map.csv"), slam);
    return summary;
}

std::vector<RunSummary> runBenchmark(
    const std::filesystem::path& sequencePath,
    const std::filesystem::path& outputPath,
    const std::vector<std::string>& degradations) {
    for (const auto& degradation : degradations) {
        if (std::find(kDegradations.begin(), kDegradations.end(), degradation) ==
            kDegradations.end()) {
            throw std::invalid_argument("unsupported degradation: " + degradation);
        }
    }
    std::vector<RunSummary> summaries;
    summaries.reserve(degradations.size() * kModes.size());
    for (const auto& degradation : degradations) {
        for (const auto& configuration : kModes) {
            summaries.push_back(runConfiguration(sequencePath, outputPath,
                                                  configuration, degradation));
        }
    }
    writeSummaries(outputPath, summaries);
    return summaries;
}

void writeSummaries(const std::filesystem::path& outputPath,
                    const std::vector<RunSummary>& summaries) {
    std::filesystem::create_directories(outputPath);
    std::ofstream output(outputPath / "summary.csv");
    if (!output) {
        throw std::runtime_error("cannot write summary.csv");
    }
    output << "sequence,configuration,degradation,frames,map_landmarks,keyframes,"
              "frames_with_map_pose,ate_rmse_similarity_aligned,"
              "tracking_failure_frames,tracking_failure_rate,mean_latency_ms,"
              "p95_latency_ms,wall_time_s,cpu_time_s,rss_delta_bytes,"
              "rss_after_bytes,energy_joules,recovery_mode_fraction,"
              "mode_transitions,monocular_scale\n"
           << std::setprecision(10);
    for (const RunSummary& summary : summaries) {
        output << summary.sequence << ',' << summary.configuration << ','
               << summary.degradation << ',' << summary.frames << ','
               << summary.landmarks << ',' << summary.keyframes << ','
               << summary.framesWithMapPose << ',' << summary.ateRmse << ','
               << summary.trackingFailures << ',' << summary.failureRate << ','
               << summary.meanLatencyMs << ',' << summary.p95LatencyMs << ','
               << summary.wallTimeSeconds << ',' << summary.cpuTimeSeconds << ','
               << summary.rssDeltaBytes << ',' << summary.rssAfterBytes
               << ",not_measured," << summary.recoveryFraction << ','
               << summary.modeTransitions
               << ",unknown_similarity_aligned\n";
    }
}

}  // namespace adaptive_slam
