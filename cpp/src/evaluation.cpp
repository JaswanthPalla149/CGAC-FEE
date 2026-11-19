#include "adaptive_slam/slam.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <memory>
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
#include <pdh.h>
#include <pdhmsg.h>
#include <psapi.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace adaptive_slam {
namespace {

const std::vector<std::string> kModes{"fixed_low", "fixed_high", "adaptive"};
const std::vector<std::string> kDegradations{
    "none", "blur", "low_light", "noise", "sudden_dim"};

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

cv::Matx33d bodyFromSensorRotation(const std::filesystem::path& sensorPath) {
    std::ifstream sensor(sensorPath);
    if (!sensor) {
        throw std::runtime_error("cannot read sensor calibration " +
                                 sensorPath.string());
    }
    const std::regex numberPattern(
        R"([-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)");
    cv::Matx33d rotation = cv::Matx33d::eye();
    std::string line;
    bool inTransform = false;
    int row = 0;
    while (std::getline(sensor, line)) {
        if (line.find("T_BS:") != std::string::npos) {
            inTransform = true;
            continue;
        }
        if (!inTransform) {
            continue;
        }
        std::vector<double> values;
        for (std::sregex_iterator value(line.begin(), line.end(), numberPattern),
             end;
             value != end; ++value) {
            values.push_back(std::stod(value->str()));
        }
        if (values.size() < 4) {
            continue;
        }
        for (int column = 0; column < 3; ++column) {
            rotation(row, column) = values[static_cast<std::size_t>(column)];
        }
        if (++row == 3) {
            return rotation;
        }
    }
    throw std::runtime_error("missing T_BS rotation in " + sensorPath.string());
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

struct EnergyMeasurement {
    double joules{std::numeric_limits<double>::quiet_NaN()};
    std::string method{"not_measured"};
    std::string scope{"not_available"};
    std::string status{"unsupported_platform"};
};

class PackageEnergyMeter {
public:
    PackageEnergyMeter() {
#ifdef _WIN32
        PDH_STATUS status = PdhOpenQueryW(nullptr, 0, &query_);
        if (status != ERROR_SUCCESS) {
            status_ = statusText("open_query", status);
            return;
        }
        status = PdhAddEnglishCounterW(
            query_, L"\\Energy Meter(RAPL_Package0_PKG)\\Energy", 0,
            &counter_);
        if (status != ERROR_SUCCESS) {
            status_ = statusText("add_rapl_energy_counter", status);
            return;
        }
        status_ = "ready";
#elif defined(__linux__)
        std::ifstream raplFile("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj");
        if (raplFile.good()) {
            status_ = "ready";
            raplPath_ = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj";
        } else {
            status_ = "powercap_not_available";
        }
#endif
    }

    ~PackageEnergyMeter() {
#ifdef _WIN32
        if (query_ != nullptr) {
            PdhCloseQuery(query_);
        }
#endif
    }

    PackageEnergyMeter(const PackageEnergyMeter&) = delete;
    PackageEnergyMeter& operator=(const PackageEnergyMeter&) = delete;

    void begin() {
#ifdef _WIN32
        if (counter_ == nullptr) {
            return;
        }
        if (!readPicoWattHours(startPicoWattHours_)) {
            return;
        }
        started_ = true;
        status_ = "started";
#elif defined(__linux__)
        if (!raplPath_.empty()) {
            std::ifstream file(raplPath_);
            if (file >> startMicroJoules_) {
                started_ = true;
                status_ = "started";
            }
        }
#endif
    }

    EnergyMeasurement finish() {
#ifdef _WIN32
        if (!started_) {
            return {std::numeric_limits<double>::quiet_NaN(), "not_measured",
                    "not_available", status_};
        }
        double endPicoWattHours = 0.0;
        if (!readPicoWattHours(endPicoWattHours)) {
            return {std::numeric_limits<double>::quiet_NaN(), "not_measured",
                    "not_available", status_};
        }
        const double deltaPicoWattHours =
            endPicoWattHours - startPicoWattHours_;
        if (deltaPicoWattHours < 0.0) {
            return {std::numeric_limits<double>::quiet_NaN(), "not_measured",
                    "not_available", "counter_decreased"};
        }
        return {deltaPicoWattHours * 3.6e-9,
                "windows_pdh_rapl_package_energy", "cpu_package_shared",
                "measured"};
#elif defined(__linux__)
        if (!started_ || raplPath_.empty()) {
            return {std::numeric_limits<double>::quiet_NaN(), "not_measured",
                    "not_available", status_};
        }
        std::ifstream file(raplPath_);
        unsigned long long endMicroJoules = 0;
        if (!(file >> endMicroJoules) || endMicroJoules < startMicroJoules_) {
            return {std::numeric_limits<double>::quiet_NaN(), "not_measured",
                    "not_available", "counter_error"};
        }
        const double deltaJoules = static_cast<double>(endMicroJoules - startMicroJoules_) * 1e-6;
        return {deltaJoules, "linux_sysfs_rapl_package_energy", "cpu_package_shared", "measured"};
#else
        return {};
#endif
    }

private:
#ifdef _WIN32
    static std::string statusText(const char* operation, PDH_STATUS status) {
        std::ostringstream message;
        message << operation << "_pdh_status_0x" << std::hex
                << static_cast<unsigned long>(status);
        return message.str();
    }

    bool readPicoWattHours(double& value) {
        PDH_STATUS status = PdhCollectQueryData(query_);
        if (status != ERROR_SUCCESS) {
            status_ = statusText("collect_energy_counter", status);
            return false;
        }
        PDH_FMT_COUNTERVALUE formatted{};
        DWORD counterType = 0;
        status = PdhGetFormattedCounterValue(
            counter_, PDH_FMT_DOUBLE, &counterType, &formatted);
        if (status != ERROR_SUCCESS) {
            status_ = statusText("format_energy_counter", status);
            return false;
        }
        if (formatted.CStatus != PDH_CSTATUS_VALID_DATA &&
            formatted.CStatus != PDH_CSTATUS_NEW_DATA) {
            status_ = statusText("invalid_energy_counter_data",
                                 formatted.CStatus);
            return false;
        }
        if (!std::isfinite(formatted.doubleValue)) {
            status_ = "non_finite_energy_counter_value";
            return false;
        }
        value = formatted.doubleValue;
        return true;
    }

    PDH_HQUERY query_{};
    PDH_HCOUNTER counter_{};
    double startPicoWattHours_{};
    bool started_{};
    std::string status_{"counter_unavailable"};
#elif defined(__linux__)
    std::string raplPath_{};
    unsigned long long startMicroJoules_{};
    bool started_{};
    std::string status_{"counter_unavailable"};
#endif
};

std::optional<cv::Vec3d> interpolateGroundTruth(
    std::int64_t timestamp,
    const GroundTruth& groundTruth) {
    if (groundTruth.timestampsNs.empty() ||
        timestamp < groundTruth.timestampsNs.front() ||
        timestamp > groundTruth.timestampsNs.back()) {
        return std::nullopt;
    }
    auto upper = std::lower_bound(groundTruth.timestampsNs.begin(),
                                  groundTruth.timestampsNs.end(), timestamp);
    if (upper == groundTruth.timestampsNs.end()) {
        return groundTruth.positions.back();
    }
    const std::size_t right =
        static_cast<std::size_t>(upper - groundTruth.timestampsNs.begin());
    if (*upper == timestamp || right == 0) {
        return groundTruth.positions[right];
    }
    const std::size_t left = right - 1;
    const double fraction =
        static_cast<double>(timestamp - groundTruth.timestampsNs[left]) /
        static_cast<double>(groundTruth.timestampsNs[right] -
                           groundTruth.timestampsNs[left]);
    return groundTruth.positions[left] * (1.0 - fraction) +
           groundTruth.positions[right] * fraction;
}

struct SimilarityAlignment {
    double scale{};
    cv::Matx33d rotation{};
    cv::Vec3d estimateMean{};
    cv::Vec3d targetMean{};
};

std::optional<SimilarityAlignment> estimateSimilarityAlignment(
    const std::vector<cv::Vec3d>& estimate,
    const std::vector<cv::Vec3d>& target) {
    if (estimate.size() != target.size() || estimate.size() < 3) {
        return std::nullopt;
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
    if (variance <= 1e-12) {
        return std::nullopt;
    }
    cv::SVD decomposition(covariance, cv::SVD::FULL_UV);
    cv::Mat correction = cv::Mat::eye(3, 3, CV_64F);
    const cv::Mat product = decomposition.u * decomposition.vt;
    correction.at<double>(2, 2) = cv::determinant(product) < 0.0 ? -1.0 : 1.0;
    const cv::Mat rotation = decomposition.u * correction * decomposition.vt;
    const double scale =
        (decomposition.w.at<double>(0) + decomposition.w.at<double>(1) +
         decomposition.w.at<double>(2) * correction.at<double>(2, 2)) /
        variance;
    cv::Matx33d rotationMatrix;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            rotationMatrix(row, column) = rotation.at<double>(row, column);
        }
    }
    return SimilarityAlignment{scale, rotationMatrix, estimateMean, targetMean};
}

cv::Vec3d applyAlignment(const cv::Vec3d& position,
                         const SimilarityAlignment& alignment) {
    return alignment.scale * alignment.rotation *
               (position - alignment.estimateMean) +
           alignment.targetMean;
}

double alignedAte(const std::vector<cv::Vec3d>& estimate,
                  const std::vector<cv::Vec3d>& target,
                  const SimilarityAlignment& alignment) {
    if (estimate.size() != target.size() || estimate.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double squaredError = 0.0;
    for (std::size_t i = 0; i < estimate.size(); ++i) {
        const cv::Vec3d residual = applyAlignment(estimate[i], alignment) - target[i];
        squaredError += residual.dot(residual);
    }
    return std::sqrt(squaredError / static_cast<double>(estimate.size()));
}

double alignedRpeTranslationRmse(
    const std::vector<FrameResult>& frames,
    const std::vector<std::optional<cv::Vec3d>>& groundTruth,
    const SimilarityAlignment& alignment) {
    double squaredError = 0.0;
    std::size_t validPairs = 0;
    for (std::size_t i = 1; i < frames.size(); ++i) {
        if (!frames[i - 1].trackingSucceeded || !frames[i].trackingSucceeded ||
            !groundTruth[i - 1].has_value() || !groundTruth[i].has_value()) {
            continue;
        }
        const cv::Vec3d estimateDelta =
            alignment.scale * alignment.rotation *
            (frames[i].position - frames[i - 1].position);
        const cv::Vec3d targetDelta = *groundTruth[i] - *groundTruth[i - 1];
        const cv::Vec3d residual = estimateDelta - targetDelta;
        squaredError += residual.dot(residual);
        ++validPairs;
    }
    if (validPairs == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::sqrt(squaredError / static_cast<double>(validPairs));
}

void writeFrameRows(const std::filesystem::path& path,
                    const std::vector<FrameResult>& frames) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write " + path.string());
    }
    output << "timestamp_ns,x,y,z,confidence,tracked_ratio,inlier_ratio,"
              "reprojection_error_px,active_features,inliers,map_inliers,"
              "elapsed_ms,mode,tracking_succeeded,loop_closed,"
              "inertial_fallback,"
              "loop_candidate_keyframe,"
              "loop_drift_before_optimization_estimate_units\n"
           << std::setprecision(10);
    for (const FrameResult& frame : frames) {
        output << frame.timestampNs << ',' << frame.position[0] << ','
               << frame.position[1] << ',' << frame.position[2] << ','
               << frame.confidence << ',' << frame.trackedRatio << ','
               << frame.inlierRatio << ',' << frame.reprojectionErrorPx << ','
               << frame.activeFeatures << ',' << frame.inliers << ','
               << frame.mapInliers << ',' << frame.elapsedMs << ','
               << modeName(frame.mode) << ','
               << (frame.trackingSucceeded ? 1 : 0) << ','
               << (frame.loopClosed ? 1 : 0) << ','
               << (frame.inertialFallback ? 1 : 0) << ','
               << frame.loopCandidateId << ','
               << frame.loopDriftEstimateUnits << '\n';
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
    const std::regex distortionPattern(
        R"(distortion_coefficients\s*:\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\])");
    std::smatch match;
    std::string line;
    CameraModel camera;
    bool foundIntrinsics = false;
    while (std::getline(calibration, line)) {
        if (std::regex_search(line, match, intrinsicsPattern)) {
            camera = {std::stod(match[1]), std::stod(match[2]),
                      std::stod(match[3]), std::stod(match[4])};
            foundIntrinsics = true;
        } else if (std::regex_search(line, match, distortionPattern)) {
            camera.distortion = {
                std::stod(match[1]), std::stod(match[2]),
                std::stod(match[3]), std::stod(match[4])};
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

    const std::filesystem::path imuDirectory =
        sequencePath / "mav0" / "imu0";
    const std::filesystem::path imuPath = imuDirectory / "data.csv";
    if (std::filesystem::is_regular_file(imuPath)) {
        const std::filesystem::path imuCalibrationPath =
            imuDirectory / "sensor.yaml";
        if (std::filesystem::is_regular_file(imuCalibrationPath)) {
            const cv::Matx33d bodyFromCamera =
                bodyFromSensorRotation(calibrationPath);
            const cv::Matx33d bodyFromImu =
                bodyFromSensorRotation(imuCalibrationPath);
            sequence.cameraFromImu = bodyFromCamera.t() * bodyFromImu;
        }
        std::ifstream imu(imuPath);
        std::int64_t previousImuTimestamp = -1;
        while (std::getline(imu, line)) {
            if (isCommentOrEmpty(line)) {
                continue;
            }
            const auto columns = splitCsvLine(line);
            if (columns.size() < 7) {
                throw std::runtime_error(
                    "invalid IMU CSV row in " + imuPath.string());
            }
            try {
                GyroscopeSample sample;
                sample.timestampNs = std::stoll(columns[0]);
                sample.angularVelocity = cv::Vec3d(
                    std::stod(columns[1]), std::stod(columns[2]),
                    std::stod(columns[3]));
                sample.linearAcceleration = cv::Vec3d(
                    std::stod(columns[4]), std::stod(columns[5]),
                    std::stod(columns[6]));
                if (sample.timestampNs <= previousImuTimestamp ||
                    !std::isfinite(sample.angularVelocity[0]) ||
                    !std::isfinite(sample.angularVelocity[1]) ||
                    !std::isfinite(sample.angularVelocity[2]) ||
                    !std::isfinite(sample.linearAcceleration[0]) ||
                    !std::isfinite(sample.linearAcceleration[1]) ||
                    !std::isfinite(sample.linearAcceleration[2])) {
                    throw std::runtime_error(
                        "invalid or unordered IMU sample in " + imuPath.string());
                }
                previousImuTimestamp = sample.timestampNs;
                sequence.gyroscope.push_back(sample);
            } catch (const std::invalid_argument&) {
                throw std::runtime_error(
                    "invalid IMU value in " + imuPath.string());
            } catch (const std::out_of_range&) {
                throw std::runtime_error(
                    "out-of-range IMU value in " + imuPath.string());
            }
        }
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

std::optional<cv::Matx33d> integrateGyroscopeRotation(
    const std::vector<GyroscopeSample>& samples,
    std::int64_t startTimestampNs, std::int64_t endTimestampNs,
    const cv::Matx33d& cameraFromImu) {
    if (samples.size() < 2 || endTimestampNs <= startTimestampNs ||
        startTimestampNs < samples.front().timestampNs ||
        endTimestampNs > samples.back().timestampNs) {
        return std::nullopt;
    }

    const auto gyroAt = [&samples](std::int64_t timestamp)
        -> std::optional<cv::Vec3d> {
        const auto upper = std::lower_bound(
            samples.begin(), samples.end(), timestamp,
            [](const GyroscopeSample& sample, std::int64_t value) {
                return sample.timestampNs < value;
            });
        if (upper == samples.end()) {
            return std::nullopt;
        }
        if (upper->timestampNs == timestamp) {
            return upper->angularVelocity;
        }
        if (upper == samples.begin()) {
            return std::nullopt;
        }
        const GyroscopeSample& left = *(upper - 1);
        const double fraction =
            static_cast<double>(timestamp - left.timestampNs) /
            static_cast<double>(upper->timestampNs - left.timestampNs);
        return left.angularVelocity * (1.0 - fraction) +
               upper->angularVelocity * fraction;
    };

    const auto startRate = gyroAt(startTimestampNs);
    const auto endRate = gyroAt(endTimestampNs);
    if (!startRate.has_value() || !endRate.has_value()) {
        return std::nullopt;
    }
    std::vector<GyroscopeSample> interval;
    interval.push_back({startTimestampNs, *startRate});
    for (const GyroscopeSample& sample : samples) {
        if (sample.timestampNs > startTimestampNs &&
            sample.timestampNs < endTimestampNs) {
            interval.push_back(sample);
        }
    }
    interval.push_back({endTimestampNs, *endRate});

    cv::Matx33d relativeRotation = cv::Matx33d::eye();
    for (std::size_t i = 1; i < interval.size(); ++i) {
        const double dt =
            static_cast<double>(interval[i].timestampNs -
                                interval[i - 1].timestampNs) *
            1e-9;
        const cv::Vec3d angularVelocity =
            cameraFromImu *
            ((interval[i - 1].angularVelocity + interval[i].angularVelocity) *
             0.5);
        const cv::Vec3d rotationVector = -angularVelocity * dt;
        const double angle = cv::norm(rotationVector);
        const cv::Matx33d skew(
            0.0, -rotationVector[2], rotationVector[1],
            rotationVector[2], 0.0, -rotationVector[0],
            -rotationVector[1], rotationVector[0], 0.0);
        const double firstScale =
            angle < 1e-8 ? 1.0 : std::sin(angle) / angle;
        const double secondScale =
            angle < 1e-8 ? 0.5 : (1.0 - std::cos(angle)) / (angle * angle);
        const cv::Matx33d increment =
            cv::Matx33d::eye() + firstScale * skew +
            secondScale * (skew * skew);
        relativeRotation = increment * relativeRotation;
    }
    return relativeRotation;
}

std::optional<InertialPreintegration> integrateInertialTranslation(
    const std::vector<GyroscopeSample>& samples,
    std::int64_t startTimestampNs, std::int64_t endTimestampNs,
    const cv::Matx33d& rotationWorldCamera,
    const cv::Matx33d& cameraFromImu, const cv::Vec3d& gravityWorld,
    const cv::Vec3d& initialVelocity) {
    if (samples.size() < 2 || endTimestampNs <= startTimestampNs ||
        startTimestampNs < samples.front().timestampNs ||
        endTimestampNs > samples.back().timestampNs ||
        endTimestampNs - startTimestampNs > 100000000 ||
        !std::isfinite(cv::norm(gravityWorld)) ||
        !std::isfinite(cv::norm(initialVelocity))) {
        return std::nullopt;
    }

    const auto sampleAt = [&samples](std::int64_t timestamp)
        -> std::optional<GyroscopeSample> {
        const auto upper = std::lower_bound(
            samples.begin(), samples.end(), timestamp,
            [](const GyroscopeSample& sample, std::int64_t value) {
                return sample.timestampNs < value;
            });
        if (upper == samples.end()) {
            return std::nullopt;
        }
        if (upper->timestampNs == timestamp) {
            return *upper;
        }
        if (upper == samples.begin()) {
            return std::nullopt;
        }
        const GyroscopeSample& left = *(upper - 1);
        const double fraction =
            static_cast<double>(timestamp - left.timestampNs) /
            static_cast<double>(upper->timestampNs - left.timestampNs);
        return GyroscopeSample{
            timestamp,
            left.angularVelocity * (1.0 - fraction) +
                upper->angularVelocity * fraction,
            left.linearAcceleration * (1.0 - fraction) +
                upper->linearAcceleration * fraction};
    };

    std::vector<std::int64_t> timestamps{startTimestampNs};
    for (const GyroscopeSample& sample : samples) {
        if (sample.timestampNs > startTimestampNs &&
            sample.timestampNs < endTimestampNs) {
            timestamps.push_back(sample.timestampNs);
        }
    }
    timestamps.push_back(endTimestampNs);

    cv::Vec3d position(0.0, 0.0, 0.0);
    cv::Vec3d velocity = initialVelocity;
    cv::Matx33d relativeRotation = cv::Matx33d::eye();
    for (std::size_t i = 1; i < timestamps.size(); ++i) {
        const std::int64_t leftTimestamp = timestamps[i - 1];
        const std::int64_t rightTimestamp = timestamps[i];
        const double dt = static_cast<double>(rightTimestamp - leftTimestamp) *
                          1e-9;
        if (!(dt > 0.0 && dt <= 0.02)) {
            return std::nullopt;
        }
        const auto left = sampleAt(leftTimestamp);
        const auto right = sampleAt(rightTimestamp);
        if (!left.has_value() || !right.has_value()) {
            return std::nullopt;
        }
        const cv::Vec3d angularVelocity =
            cameraFromImu *
            ((left->angularVelocity + right->angularVelocity) * 0.5);
        if (!std::isfinite(cv::norm(angularVelocity)) ||
            cv::norm(angularVelocity) > 20.0) {
            return std::nullopt;
        }
        const cv::Vec3d halfRotationVector = -angularVelocity * (0.5 * dt);
        const cv::Vec3d rotationVector = halfRotationVector * 2.0;
        if (!std::isfinite(cv::norm(left->linearAcceleration)) ||
            !std::isfinite(cv::norm(right->linearAcceleration)) ||
            cv::norm((left->linearAcceleration + right->linearAcceleration) *
                     0.5) > 30.0) {
            return std::nullopt;
        }
        const auto rotationIncrement = [](const cv::Vec3d& vector) {
            const double angle = cv::norm(vector);
            const cv::Matx33d skew(
                0.0, -vector[2], vector[1], vector[2], 0.0, -vector[0],
                -vector[1], vector[0], 0.0);
            const double firstScale =
                angle < 1e-8 ? 1.0 : std::sin(angle) / angle;
            const double secondScale =
                angle < 1e-8 ? 0.5 : (1.0 - std::cos(angle)) / (angle * angle);
            return cv::Matx33d::eye() + firstScale * skew +
                   secondScale * (skew * skew);
        };
        const cv::Matx33d midpointRotation =
            rotationIncrement(halfRotationVector) * relativeRotation;
        const cv::Matx33d rotationWorldCameraAtMidpoint =
            rotationWorldCamera * midpointRotation.t();
        const cv::Vec3d specificForceCamera =
            cameraFromImu *
            ((left->linearAcceleration + right->linearAcceleration) * 0.5);
        const cv::Vec3d accelerationWorld =
            rotationWorldCameraAtMidpoint * specificForceCamera + gravityWorld;
        position += velocity * dt + accelerationWorld * (0.5 * dt * dt);
        velocity += accelerationWorld * dt;
        relativeRotation =
            rotationIncrement(rotationVector) * relativeRotation;
    }
    return InertialPreintegration{
        position, velocity, rotationWorldCamera * relativeRotation.t()};
}

namespace {

class BoundedInertialFallback {
public:
    BoundedInertialFallback(const std::vector<GyroscopeSample>& samples,
                            const cv::Matx33d& cameraFromImu)
        : samples_(samples), cameraFromImu_(cameraFromImu) {
        if (samples_.size() < 20 ||
            samples_.back().timestampNs - samples_.front().timestampNs <
                500000000) {
            return;
        }
        const std::int64_t calibrationEnd =
            samples_.front().timestampNs + 500000000;
        cv::Vec3d meanAcceleration(0.0, 0.0, 0.0);
        std::size_t count = 0;
        for (const GyroscopeSample& sample : samples_) {
            if (sample.timestampNs > calibrationEnd) {
                break;
            }
            meanAcceleration += cameraFromImu_ * sample.linearAcceleration;
            ++count;
        }
        if (count < 20) {
            return;
        }
        meanAcceleration *= 1.0 / static_cast<double>(count);
        double squaredDeviation = 0.0;
        double squaredAngularRate = 0.0;
        for (const GyroscopeSample& sample : samples_) {
            if (sample.timestampNs > calibrationEnd) {
                break;
            }
            const cv::Vec3d delta =
                cameraFromImu_ * sample.linearAcceleration - meanAcceleration;
            squaredDeviation += delta.dot(delta);
            squaredAngularRate += sample.angularVelocity.dot(
                sample.angularVelocity);
        }
        const double rmsDeviation =
            std::sqrt(squaredDeviation / static_cast<double>(count));
        const double rmsAngularRate =
            std::sqrt(squaredAngularRate / static_cast<double>(count));
        const double gravityMagnitude = cv::norm(meanAcceleration);
        if (!std::isfinite(rmsDeviation) || !std::isfinite(rmsAngularRate) ||
            rmsDeviation > 1.0 || rmsAngularRate > 0.2 ||
            gravityMagnitude < 8.5 || gravityMagnitude > 11.2) {
            return;
        }
        gravityWorld_ = -meanAcceleration;
        calibrated_ = true;
    }

    bool available() const noexcept { return calibrated_; }

    std::optional<cv::Vec3d> predict(std::int64_t startTimestampNs,
                                     std::int64_t endTimestampNs,
                                     const cv::Matx33d& rotationWorldCamera) {
        if (!calibrated_ || endTimestampNs <= startTimestampNs) {
            reset();
            return std::nullopt;
        }
        const double dt =
            static_cast<double>(endTimestampNs - startTimestampNs) * 1e-9;
        if (fallbackDurationSeconds_ + dt > 0.25) {
            reset();
            return std::nullopt;
        }
        if (!active_) {
            fallbackRotationWorldCamera_ = rotationWorldCamera;
            active_ = true;
        }
        const auto integrated = integrateInertialTranslation(
            samples_, startTimestampNs, endTimestampNs,
            fallbackRotationWorldCamera_,
            cameraFromImu_, gravityWorld_, velocity_);
        if (!integrated.has_value()) {
            reset();
            return std::nullopt;
        }
        velocity_ = integrated->velocity;
        fallbackRotationWorldCamera_ = integrated->rotationWorldCamera;
        fallbackDisplacementWorld_ += integrated->displacement;
        fallbackDurationSeconds_ += dt;
        return fallbackDisplacementWorld_;
    }

    void reset() noexcept {
        velocity_ = cv::Vec3d(0.0, 0.0, 0.0);
        fallbackDisplacementWorld_ = cv::Vec3d(0.0, 0.0, 0.0);
        fallbackDurationSeconds_ = 0.0;
        active_ = false;
    }

private:
    const std::vector<GyroscopeSample>& samples_;
    cv::Matx33d cameraFromImu_;
    cv::Vec3d gravityWorld_{0.0, 0.0, 0.0};
    cv::Vec3d velocity_{0.0, 0.0, 0.0};
    cv::Vec3d fallbackDisplacementWorld_{0.0, 0.0, 0.0};
    cv::Matx33d fallbackRotationWorldCamera_{cv::Matx33d::eye()};
    double fallbackDurationSeconds_{};
    bool calibrated_{};
    bool active_{};
};

void showBenchmarkFrame(const std::string& windowName, const cv::Mat& image,
                        const FrameResult& result,
                        const std::vector<cv::Point2f>& features) {
    cv::Mat display;
    cv::cvtColor(image, display, cv::COLOR_GRAY2BGR);
    const cv::Scalar featureColor =
        result.inertialFallback ? cv::Scalar(0, 165, 255)
                                : cv::Scalar(0, 220, 0);
    for (const cv::Point2f& feature : features) {
        cv::circle(display, feature, 2, featureColor, -1, cv::LINE_AA);
    }

    std::ostringstream status;
    status << "Mode: " << modeName(result.mode) << "  confidence: "
           << std::fixed << std::setprecision(3) << result.confidence;
    cv::putText(display, status.str(), cv::Point(16, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(20, 20, 20), 4,
                cv::LINE_AA);
    cv::putText(display, status.str(), cv::Point(16, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1,
                cv::LINE_AA);

    const std::string trackingStatus =
        result.inertialFallback
            ? "IMU bridge (short horizon)"
            : result.trackingSucceeded ? "Visual pose" : "Visual tracking lost";
    cv::putText(display, trackingStatus, cv::Point(16, 56),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                result.inertialFallback ? cv::Scalar(0, 165, 255)
                                        : cv::Scalar(255, 255, 255),
                2, cv::LINE_AA);
    cv::imshow(windowName, display);
}

void updateMapWindow(cv::viz::Viz3d& window,
                     const std::vector<cv::Point3d>& landmarks,
                     const std::vector<cv::Vec3d>& trajectory,
                     const FrameResult& result, bool& hasCloud,
                     bool& hasTrajectory, bool& hasCamera,
                     bool& hasBridge, bool& hasStatus) {
    std::vector<cv::Point3f> cloud;
    const std::size_t stride =
        std::max<std::size_t>(1, (landmarks.size() + 19999) / 20000);
    cloud.reserve((landmarks.size() + stride - 1) / stride);
    double sceneScale = 1.0;
    for (std::size_t i = 0; i < landmarks.size(); i += stride) {
        const cv::Point3d& point = landmarks[i];
        cloud.emplace_back(static_cast<float>(point.x),
                           static_cast<float>(point.y),
                           static_cast<float>(point.z));
        sceneScale = std::max(sceneScale, cv::norm(point));
    }
    if (!cloud.empty()) {
        if (hasCloud) {
            window.removeWidget("landmarks");
        }
        window.showWidget("landmarks",
                          cv::viz::WCloud(cloud, cv::viz::Color::yellow()));
        window.setRenderingProperty("landmarks", cv::viz::POINT_SIZE, 3);
        if (!hasCloud) {
            window.resetCameraViewpoint("landmarks");
        }
        hasCloud = true;
    }

    std::vector<cv::Point3f> path;
    path.reserve(trajectory.size());
    for (const cv::Vec3d& pose : trajectory) {
        path.emplace_back(static_cast<float>(pose[0]),
                          static_cast<float>(pose[1]),
                          static_cast<float>(pose[2]));
        sceneScale = std::max(sceneScale, cv::norm(pose));
    }
    if (path.size() > 1) {
        if (hasTrajectory) {
            window.removeWidget("trajectory");
        }
        window.showWidget("trajectory",
                          cv::viz::WPolyLine(path, cv::viz::Color::orange()));
        hasTrajectory = true;
    }

    if (!trajectory.empty()) {
        if (hasCamera) {
            window.removeWidget("camera");
        }
        const cv::Vec3d& position = trajectory.back();
        window.showWidget(
            "camera",
            cv::viz::WSphere(cv::Point3d(position[0], position[1], position[2]),
                             std::max(0.05, sceneScale * 0.01), 12,
                             cv::viz::Color::green()));
        hasCamera = true;
        if (hasBridge) {
            window.removeWidget("imu_bridge");
            hasBridge = false;
        }
        if (result.inertialFallback) {
            const cv::Point3d visualPosition(position[0], position[1],
                                             position[2]);
            const cv::Point3d inertialPosition(
                result.position[0], result.position[1], result.position[2]);
            if (cv::norm(inertialPosition - visualPosition) > 1e-6) {
                window.showWidget(
                    "imu_bridge",
                    cv::viz::WArrow(
                        visualPosition, inertialPosition,
                        std::max(0.01, sceneScale * 0.005),
                        cv::viz::Color::orange_red()));
                hasBridge = true;
            }
        }
    }

    if (hasStatus) {
        window.removeWidget("status");
    }
    std::ostringstream status;
    status << "Mode: " << modeName(result.mode) << "  confidence: "
           << std::fixed << std::setprecision(3) << result.confidence
           << "  landmarks: " << landmarks.size();
    window.showWidget(
        "status",
        cv::viz::WText(status.str(), cv::Point(10, 25), 16,
                       cv::viz::Color::white()));
    hasStatus = true;
}

}  // namespace

cv::Mat applyDegradation(const cv::Mat& image, const std::string& degradation,
                         std::uint32_t seed, std::size_t frameIndex,
                         std::size_t frameCount) {
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
    if (degradation == "sudden_dim") {
        if (frameIndex < frameCount / 2) {
            return image;
        }
        cv::Mat output;
        image.convertTo(output, image.type(), 0.30, 0.0);
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
                            std::uint32_t seed,
                            double enterRecoveryBelow,
                            double exitRecoveryAbove,
                            double enterVisualInertialBelow,
                            double exitVisualInertialAbove,
                            std::size_t maxFrames,
                            bool visualize) {
    if (std::find(kModes.begin(), kModes.end(), configuration) == kModes.end()) {
        throw std::invalid_argument("unsupported configuration: " + configuration);
    }
    if (std::find(kDegradations.begin(), kDegradations.end(), degradation) ==
        kDegradations.end()) {
        throw std::invalid_argument("unsupported degradation: " + degradation);
    }

    const EuRoCSequence sequence = loadEuRoC(sequencePath);
    FeatureSLAM slam(sequence.camera);
    AdaptiveController controller(enterRecoveryBelow, exitRecoveryAbove,
                                  enterVisualInertialBelow,
                                  exitVisualInertialAbove);
    BoundedInertialFallback inertialFallback(sequence.gyroscope,
                                              sequence.cameraFromImu);
    TrackingMode mode = configuration == "fixed_high"
                            ? TrackingMode::Recovery
                            : TrackingMode::LowCost;
    const std::size_t frameCount =
        maxFrames == 0 ? sequence.frames.size()
                       : std::min(maxFrames, sequence.frames.size());
    std::vector<FrameResult> results;
    results.reserve(frameCount);
    std::size_t transitions = 0;
    const std::int64_t rssBefore = processRssBytes();
    std::int64_t peakRss = rssBefore;
    PackageEnergyMeter energyMeter;
    energyMeter.begin();
    const std::clock_t cpuStart = std::clock();
    const auto wallStart = std::chrono::steady_clock::now();
    const std::string windowName =
        "EuRoC adaptive SLAM - " + sequencePath.filename().string() + " (" +
        configuration + ")";
    const std::string mapWindowName =
        "3D reconstruction - " + sequencePath.filename().string() + " (" +
        configuration + ")";
    bool guiActive = visualize;
    std::unique_ptr<cv::viz::Viz3d> mapWindow;
    bool hasCloud = false;
    bool hasTrajectory = false;
    bool hasCamera = false;
    bool hasBridge = false;
    bool hasStatus = false;
    if (guiActive) {
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        mapWindow = std::make_unique<cv::viz::Viz3d>(mapWindowName);
        mapWindow->setWindowSize(cv::Size(1000, 750));
        mapWindow->setBackgroundColor(cv::viz::Color(30, 30, 36));
        mapWindow->showWidget("axes", cv::viz::WCoordinateSystem(0.5));
        mapWindow->showWidget(
            "grid", cv::viz::WGrid(cv::Vec2i(12, 12), cv::Vec2d(0.5, 0.5),
                                   cv::viz::Color(70, 70, 70)));
    }

    for (std::size_t i = 0; i < frameCount; ++i) {
        const EuRoCFrame& frame = sequence.frames[i];
        cv::Mat image = cv::imread(frame.imagePath.string(), cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            throw std::runtime_error("failed to load image " + frame.imagePath.string());
        }
        image = applyDegradation(image, degradation,
                                 seed + static_cast<std::uint32_t>(i), i,
                                 frameCount);
        std::optional<cv::Matx33d> gyroRotationPrior;
        if (i > 0) {
            gyroRotationPrior = integrateGyroscopeRotation(
                sequence.gyroscope, sequence.frames[i - 1].timestampNs,
                frame.timestampNs, sequence.cameraFromImu);
        }
        std::optional<cv::Vec3d> inertialTranslation;
        if (mode == TrackingMode::VisualInertial && i > 0) {
            inertialTranslation = inertialFallback.predict(
                sequence.frames[i - 1].timestampNs, frame.timestampNs,
                slam.rotationWorldCamera());
        } else if (mode != TrackingMode::VisualInertial) {
            inertialFallback.reset();
        }
        FrameResult result =
            slam.process(image, frame.timestampNs, mode, gyroRotationPrior,
                         inertialTranslation);
        if (guiActive) {
            showBenchmarkFrame(windowName, image, result,
                               slam.currentFeatures());
            updateMapWindow(*mapWindow, slam.landmarks(),
                            slam.correctedFramePositions(), result, hasCloud,
                            hasTrajectory, hasCamera, hasBridge, hasStatus);
            mapWindow->spinOnce(1, true);
            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                cv::destroyWindow(windowName);
                mapWindow->close();
                guiActive = false;
            } else if (mapWindow->wasStopped()) {
                cv::destroyWindow(windowName);
                guiActive = false;
            }
        }
        if (result.trackingSucceeded) {
            inertialFallback.reset();
        }
        results.push_back(result);
        const std::int64_t sampledRss = processRssBytes();
        if (sampledRss >= 0) {
            peakRss = std::max(peakRss, sampledRss);
        }
        if (configuration == "adaptive") {
            const TrackingMode nextMode = controller.choose(
                result.confidence, mode, inertialFallback.available());
            transitions += nextMode != mode ? 1U : 0U;
            mode = nextMode;
        }
    }
    const auto& correctedPositions = slam.correctedFramePositions();
    if (correctedPositions.size() != results.size()) {
        throw std::logic_error(
            "SLAM corrected trajectory size does not match processed frames");
    }
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (!results[i].inertialFallback) {
            results[i].position = correctedPositions[i];
        }
    }

    const EnergyMeasurement energy = energyMeter.finish();
    const double wallSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart)
            .count();
    const double cpuSeconds =
        static_cast<double>(std::clock() - cpuStart) / CLOCKS_PER_SEC;
    const std::int64_t rssAfter = processRssBytes();
    if (rssAfter >= 0) {
        peakRss = std::max(peakRss, rssAfter);
    }

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
    summary.peakRssBytes = peakRss;
    summary.energyJoules = energy.joules;
    summary.energyMethod = energy.method;
    summary.energyScope = energy.scope;
    summary.energyStatus = energy.status;
    summary.controllerEnterThreshold = enterRecoveryBelow;
    summary.controllerExitThreshold = exitRecoveryAbove;
    summary.controllerEnterVisualInertialThreshold =
        enterVisualInertialBelow;
    summary.controllerExitVisualInertialThreshold = exitVisualInertialAbove;
    summary.modeTransitions = transitions;
    summary.loopClosures = slam.loopClosures();
    summary.maxLoopDriftEstimateUnits =
        slam.maxLoopDriftEstimateUnits();

    std::vector<double> latencies;
    std::size_t recoveryFrames = 0;
    std::size_t visualInertialFrames = 0;
    std::vector<std::optional<cv::Vec3d>> groundTruthByFrame(results.size());
    for (std::size_t i = 0; i < results.size(); ++i) {
        const FrameResult& result = results[i];
        latencies.push_back(result.elapsedMs);
        recoveryFrames += result.mode == TrackingMode::Recovery ? 1U : 0U;
        visualInertialFrames +=
            result.mode == TrackingMode::VisualInertial ? 1U : 0U;
        summary.inertialFallbackFrames += result.inertialFallback ? 1U : 0U;
        if (i > 0 && !result.trackingSucceeded) {
            ++summary.trackingFailures;
        }
        summary.framesWithPose += result.trackingSucceeded ? 1U : 0U;
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
    summary.visualInertialFraction =
        static_cast<double>(visualInertialFrames) /
        static_cast<double>(std::max<std::size_t>(1, results.size()));
    if (sequence.groundTruth.has_value()) {
        std::vector<cv::Vec3d> matchedEstimates;
        std::vector<cv::Vec3d> matchedGroundTruth;
        matchedEstimates.reserve(results.size());
        matchedGroundTruth.reserve(results.size());
        for (std::size_t i = 0; i < results.size(); ++i) {
            const auto truth =
                interpolateGroundTruth(results[i].timestampNs,
                                       *sequence.groundTruth);
            if (truth.has_value()) {
                ++summary.groundTruthMatchedFrames;
                groundTruthByFrame[i] = truth;
                if (results[i].trackingSucceeded) {
                    matchedEstimates.push_back(results[i].position);
                    matchedGroundTruth.push_back(*truth);
                }
            }
        }
        if (summary.groundTruthMatchedFrames > 0) {
            summary.poseCoverage =
                static_cast<double>(matchedEstimates.size()) /
                static_cast<double>(summary.groundTruthMatchedFrames);
        }
        if (summary.groundTruthMatchedFrames == 0) {
            summary.trajectoryStatus = "no_matched_ground_truth";
        } else if (summary.poseCoverage < 0.5) {
            summary.trajectoryStatus = "insufficient_pose_coverage";
        } else {
            const auto alignment =
                estimateSimilarityAlignment(matchedEstimates, matchedGroundTruth);
            if (!alignment.has_value()) {
                summary.trajectoryStatus = "alignment_failed";
            } else {
                summary.ateRmse =
                    alignedAte(matchedEstimates, matchedGroundTruth, *alignment);
                summary.rpeTranslationRmse = alignedRpeTranslationRmse(
                    results, groundTruthByFrame, *alignment);
                summary.trajectoryStatus = "evaluated";
            }
        }
    }

    std::filesystem::create_directories(outputPath);
    const std::string stem = configuration + "_" + degradation;
    writeFrameRows(outputPath / (stem + "_frames.csv"), results);
    writeTrajectory(outputPath / (stem + "_trajectory.csv"), results);
    writeMap(outputPath / (stem + "_map.csv"), slam);
    if (guiActive) {
        mapWindow->spin();
        cv::destroyWindow(windowName);
    }
    return summary;
}

std::vector<RunSummary> runBenchmark(
    const std::filesystem::path& sequencePath,
    const std::filesystem::path& outputPath,
    const std::vector<std::string>& degradations,
    double enterRecoveryBelow,
    double exitRecoveryAbove,
    double enterVisualInertialBelow,
    double exitVisualInertialAbove,
    const std::vector<std::string>& configurations,
    std::size_t maxFrames,
    bool visualize) {
    if (degradations.empty()) {
        throw std::invalid_argument("at least one degradation is required");
    }
    for (const auto& degradation : degradations) {
        if (std::find(kDegradations.begin(), kDegradations.end(), degradation) ==
            kDegradations.end()) {
            throw std::invalid_argument("unsupported degradation: " + degradation);
        }
    }
    if (configurations.empty()) {
        throw std::invalid_argument("at least one configuration is required");
    }
    for (std::size_t i = 0; i < configurations.size(); ++i) {
        const auto previousEnd =
            configurations.begin() + static_cast<std::ptrdiff_t>(i);
        if (std::find(kModes.begin(), kModes.end(), configurations[i]) ==
                kModes.end() ||
            std::find(configurations.begin(), previousEnd, configurations[i]) !=
                previousEnd) {
            throw std::invalid_argument(
                "configurations must be a unique subset of fixed_low, "
                "fixed_high, adaptive");
        }
    }
    std::vector<RunSummary> summaries;
    summaries.reserve(degradations.size() * configurations.size());
    for (const auto& degradation : degradations) {
        for (const auto& configuration : configurations) {
            summaries.push_back(runConfiguration(sequencePath, outputPath,
                                                  configuration, degradation, 17,
                                                  enterRecoveryBelow,
                                                  exitRecoveryAbove,
                                                  enterVisualInertialBelow,
                                                  exitVisualInertialAbove,
                                                  maxFrames, visualize));
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
              "frames_with_map_pose,frames_with_pose,"
              "ground_truth_matched_frames,pose_coverage,trajectory_status,"
              "ate_rmse_similarity_aligned,"
              "rpe_translation_rmse_similarity_aligned,"
              "tracking_failure_frames,tracking_failure_rate,mean_latency_ms,"
              "inertial_fallback_frames,"
              "p95_latency_ms,wall_time_s,cpu_time_s,rss_delta_bytes,"
              "rss_after_bytes,peak_rss_bytes,energy_joules,energy_method,"
              "energy_scope,energy_status,recovery_mode_fraction,"
              "visual_inertial_mode_fraction,"
              "mode_transitions,controller_enter_threshold,"
              "controller_exit_threshold,"
              "controller_enter_visual_inertial_threshold,"
              "controller_exit_visual_inertial_threshold,loop_closures,"
              "max_loop_drift_before_optimization_estimate_units,"
              "monocular_scale\n"
           << std::setprecision(10);
    for (const RunSummary& summary : summaries) {
        output << summary.sequence << ',' << summary.configuration << ','
               << summary.degradation << ',' << summary.frames << ','
               << summary.landmarks << ',' << summary.keyframes << ','
               << summary.framesWithMapPose << ',' << summary.framesWithPose
               << ',' << summary.groundTruthMatchedFrames << ','
               << summary.poseCoverage << ',' << summary.trajectoryStatus << ','
               << summary.ateRmse << ','
               << summary.rpeTranslationRmse << ','
               << summary.trackingFailures << ',' << summary.failureRate << ','
               << summary.meanLatencyMs << ','
               << summary.inertialFallbackFrames << ','
               << summary.p95LatencyMs << ','
               << summary.wallTimeSeconds << ',' << summary.cpuTimeSeconds << ','
               << summary.rssDeltaBytes << ',' << summary.rssAfterBytes << ','
               << summary.peakRssBytes << ',';
        if (std::isfinite(summary.energyJoules)) {
            output << summary.energyJoules;
        } else {
            output << "not_measured";
        }
        output << ',' << summary.energyMethod << ',' << summary.energyScope
               << ',' << summary.energyStatus << ','
               << summary.recoveryFraction << ','
               << summary.visualInertialFraction << ','
               << summary.modeTransitions
        << ',' << summary.controllerEnterThreshold << ','
        << summary.controllerExitThreshold << ','
        << summary.controllerEnterVisualInertialThreshold << ','
        << summary.controllerExitVisualInertialThreshold << ','
        << summary.loopClosures
        << ',' << summary.maxLoopDriftEstimateUnits
               << ",unknown_similarity_aligned\n";
    }
}

}  // namespace adaptive_slam
