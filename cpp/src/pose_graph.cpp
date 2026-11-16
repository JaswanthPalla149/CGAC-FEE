#include "adaptive_slam/pose_graph.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <iostream>

namespace adaptive_slam {
namespace {

constexpr int kPoseDimensions = 6;
constexpr double kHuberThreshold = 3.0;
constexpr double kDamping = 1e-6;
using Residual = cv::Vec<double, kPoseDimensions>;
using Jacobian = cv::Matx<double, kPoseDimensions, kPoseDimensions>;

struct LinearizedConstraint {
    const PoseGraphConstraint* constraint{};
    Residual residual{};
    Jacobian fromJacobian{};
    Jacobian toJacobian{};
    double robustWeight{1.0};
};

cv::Matx33d rodrigues(const cv::Vec3d& vector) {
    cv::Mat input(3, 1, CV_64F);
    input.at<double>(0) = vector[0];
    input.at<double>(1) = vector[1];
    input.at<double>(2) = vector[2];
    cv::Mat output;
    cv::Rodrigues(input, output);
    cv::Matx33d rotation;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            rotation(row, column) = output.at<double>(row, column);
        }
    }
    return rotation;
}

cv::Vec3d rotationVector(const cv::Matx33d& rotation) {
    cv::Mat input(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            input.at<double>(row, column) = rotation(row, column);
        }
    }
    cv::Mat output;
    cv::Rodrigues(input, output);
    return {output.at<double>(0), output.at<double>(1), output.at<double>(2)};
}

PoseGraphPose perturb(const PoseGraphPose& pose, int dimension,
                      double amount) {
    PoseGraphPose result = pose;
    if (dimension < 3) {
        result.positionWorld[dimension] += amount;
    } else {
        cv::Vec3d delta(0.0, 0.0, 0.0);
        delta[dimension - 3] = amount;
        result.rotationWorldFromBody = rodrigues(delta) * pose.rotationWorldFromBody;
    }
    return result;
}

Residual calculateResidual(const PoseGraphPose& from,
                           const PoseGraphPose& to,
                           const PoseGraphConstraint& constraint) {
    const cv::Matx33d relativeRotation =
        from.rotationWorldFromBody.t() * to.rotationWorldFromBody;
    const cv::Vec3d relativeTranslation =
        from.rotationWorldFromBody.t() * (to.positionWorld - from.positionWorld);
    const cv::Vec3d translationError =
        relativeTranslation - constraint.translationFromTo;
    const cv::Vec3d rotationError = rotationVector(
        constraint.rotationFromTo.t() * relativeRotation);
    return {translationError[0], translationError[1], translationError[2],
            rotationError[0], rotationError[1], rotationError[2]};
}

Jacobian calculateJacobian(const PoseGraphPose& from, const PoseGraphPose& to,
                           const PoseGraphConstraint& constraint,
                           bool perturbFrom) {
    constexpr double epsilon = 1e-6;
    Jacobian jacobian = Jacobian::zeros();
    const Residual baseline = calculateResidual(from, to, constraint);
    for (int column = 0; column < kPoseDimensions; ++column) {
        const PoseGraphPose changedFrom =
            perturbFrom ? perturb(from, column, epsilon) : from;
        const PoseGraphPose changedTo =
            perturbFrom ? to : perturb(to, column, epsilon);
        const Residual changed =
            calculateResidual(changedFrom, changedTo, constraint);
        for (int row = 0; row < kPoseDimensions; ++row) {
            jacobian(row, column) = (changed[row] - baseline[row]) / epsilon;
        }
    }
    return jacobian;
}

double informationFor(const PoseGraphConstraint& constraint, int row) {
    return row < 3 ? constraint.translationInformation
                   : constraint.rotationInformation;
}

std::vector<LinearizedConstraint> linearize(
    const std::vector<PoseGraphPose>& poses,
    const std::vector<PoseGraphConstraint>& constraints) {
    std::vector<LinearizedConstraint> result;
    result.reserve(constraints.size());
    for (const auto& constraint : constraints) {
        if (constraint.from >= poses.size() || constraint.to >= poses.size() ||
            constraint.from == constraint.to) {
            continue;
        }
        LinearizedConstraint factor;
        factor.constraint = &constraint;
        factor.residual =
            calculateResidual(poses[constraint.from], poses[constraint.to],
                              constraint);
        factor.fromJacobian = calculateJacobian(
            poses[constraint.from], poses[constraint.to], constraint, true);
        factor.toJacobian = calculateJacobian(
            poses[constraint.from], poses[constraint.to], constraint, false);

        double weightedSquaredNorm = 0.0;
        for (int row = 0; row < kPoseDimensions; ++row) {
            weightedSquaredNorm += informationFor(constraint, row) *
                                   factor.residual[row] * factor.residual[row];
        }
        const double weightedNorm = std::sqrt(weightedSquaredNorm);
        if (!constraint.loopClosure && weightedNorm > kHuberThreshold) {
            factor.robustWeight = kHuberThreshold / weightedNorm;
        }
        result.push_back(factor);
    }
    return result;
}

int offsetFor(std::size_t node, int dimension) {
    return static_cast<int>((node - 1) * kPoseDimensions) + dimension;
}

void accumulateNormalEquations(const std::vector<LinearizedConstraint>& factors,
                               std::vector<double>& rhs,
                               std::vector<double>& diagonal) {
    const int dimensions = static_cast<int>(rhs.size());
    for (const auto& factor : factors) {
        const auto& constraint = *factor.constraint;
        for (int variable = 0; variable < 2; ++variable) {
            const std::size_t node = variable == 0 ? constraint.from : constraint.to;
            if (node == 0) {
                continue;
            }
            const Jacobian& jacobian =
                variable == 0 ? factor.fromJacobian : factor.toJacobian;
            for (int column = 0; column < kPoseDimensions; ++column) {
                const int index = offsetFor(node, column);
                double gradient = 0.0;
                double diagonalValue = 0.0;
                for (int row = 0; row < kPoseDimensions; ++row) {
                    const double weight =
                        informationFor(constraint, row) * factor.robustWeight;
                    const double derivative = jacobian(row, column);
                    gradient += derivative * weight * factor.residual[row];
                    diagonalValue += derivative * derivative * weight;
                }
                rhs[index] -= gradient;
                diagonal[index] += diagonalValue;
            }
        }
    }
    for (int index = 0; index < dimensions; ++index) {
        diagonal[index] += kDamping;
    }
}

bool solveDense(const std::vector<LinearizedConstraint>& factors,
                const std::vector<double>& rhs,
                std::vector<double>& solution) {
    const int dimensions = static_cast<int>(rhs.size());
    cv::Mat normal = cv::Mat::zeros(dimensions, dimensions, CV_64F);
    for (const auto& factor : factors) {
        const auto& constraint = *factor.constraint;
        const std::array<std::size_t, 2> nodes{constraint.from, constraint.to};
        const std::array<const Jacobian*, 2> jacobians{
            &factor.fromJacobian, &factor.toJacobian};
        for (int first = 0; first < 2; ++first) {
            if (nodes[first] == 0) {
                continue;
            }
            for (int second = 0; second < 2; ++second) {
                if (nodes[second] == 0) {
                    continue;
                }
                const int firstOffset = offsetFor(nodes[first], 0);
                const int secondOffset = offsetFor(nodes[second], 0);
                for (int firstColumn = 0; firstColumn < kPoseDimensions;
                     ++firstColumn) {
                    for (int secondColumn = 0;
                         secondColumn < kPoseDimensions; ++secondColumn) {
                        double value = 0.0;
                        for (int row = 0; row < kPoseDimensions; ++row) {
                            value += (*jacobians[first])(row, firstColumn) *
                                     informationFor(constraint, row) *
                                     factor.robustWeight *
                                     (*jacobians[second])(row, secondColumn);
                        }
                        normal.at<double>(firstOffset + firstColumn,
                                          secondOffset + secondColumn) += value;
                    }
                }
            }
        }
    }
    for (int index = 0; index < dimensions; ++index) {
        normal.at<double>(index, index) += kDamping;
    }
    cv::Mat rightHandSide(dimensions, 1, CV_64F);
    for (int index = 0; index < dimensions; ++index) {
        rightHandSide.at<double>(index) = rhs[static_cast<std::size_t>(index)];
    }
    cv::Mat solutionMatrix;
    bool solved =
        cv::solve(normal, rightHandSide, solutionMatrix, cv::DECOMP_CHOLESKY);
    if (!solved) {
        solved =
            cv::solve(normal, rightHandSide, solutionMatrix, cv::DECOMP_SVD);
    }
    if (!solved) {
        std::cerr << "pose graph dense solve failed\n";
        return false;
    }
    solution.resize(rhs.size());
    for (int index = 0; index < dimensions; ++index) {
        solution[static_cast<std::size_t>(index)] =
            solutionMatrix.at<double>(index);
        if (!std::isfinite(solution[static_cast<std::size_t>(index)])) {
            std::cerr << "pose graph dense solve returned a non-finite update\n";
            return false;
        }
    }
    return true;
}

std::vector<double> applyNormalMatrix(
    const std::vector<LinearizedConstraint>& factors,
    const std::vector<double>& input, std::size_t poseCount) {
    std::vector<double> output(input.size(), 0.0);
    for (const auto& factor : factors) {
        const auto& constraint = *factor.constraint;
        std::array<double, kPoseDimensions> projected{};
        for (int row = 0; row < kPoseDimensions; ++row) {
            if (constraint.from != 0) {
                const int offset = offsetFor(constraint.from, 0);
                for (int column = 0; column < kPoseDimensions; ++column) {
                    projected[row] += factor.fromJacobian(row, column) *
                                      input[offset + column];
                }
            }
            if (constraint.to != 0) {
                const int offset = offsetFor(constraint.to, 0);
                for (int column = 0; column < kPoseDimensions; ++column) {
                    projected[row] += factor.toJacobian(row, column) *
                                      input[offset + column];
                }
            }
            projected[row] *= informationFor(constraint, row) *
                              factor.robustWeight;
        }
        for (int variable = 0; variable < 2; ++variable) {
            const std::size_t node = variable == 0 ? constraint.from : constraint.to;
            if (node == 0) {
                continue;
            }
            const Jacobian& jacobian =
                variable == 0 ? factor.fromJacobian : factor.toJacobian;
            const int offset = offsetFor(node, 0);
            for (int column = 0; column < kPoseDimensions; ++column) {
                for (int row = 0; row < kPoseDimensions; ++row) {
                    output[offset + column] +=
                        jacobian(row, column) * projected[row];
                }
            }
        }
    }
    for (std::size_t index = 0; index < input.size(); ++index) {
        output[index] += kDamping * input[index];
    }
    (void)poseCount;
    return output;
}

double dot(const std::vector<double>& first,
           const std::vector<double>& second) {
    return std::inner_product(first.begin(), first.end(), second.begin(), 0.0);
}

bool solveWithConjugateGradient(const std::vector<LinearizedConstraint>& factors,
                                const std::vector<double>& rhs,
                                const std::vector<double>& diagonal,
                                std::size_t poseCount,
                                std::vector<double>& solution) {
    solution.assign(rhs.size(), 0.0);
    const double rhsNorm = std::sqrt(dot(rhs, rhs));
    if (!std::isfinite(rhsNorm)) {
        return false;
    }
    if (rhsNorm < 1e-12) {
        return true;
    }

    std::vector<double> residual = rhs;
    std::vector<double> preconditioned(rhs.size());
    for (std::size_t index = 0; index < rhs.size(); ++index) {
        preconditioned[index] = residual[index] / diagonal[index];
    }
    std::vector<double> direction = preconditioned;
    double residualPreconditioned = dot(residual, preconditioned);
    const int maxCgIterations =
        std::min(10000, std::max(500, static_cast<int>(rhs.size()) * 8));
    for (int iteration = 0; iteration < maxCgIterations; ++iteration) {
        const std::vector<double> product =
            applyNormalMatrix(factors, direction, poseCount);
        const double denominator = dot(direction, product);
        if (!std::isfinite(denominator) || denominator <= 1e-30) {
            return false;
        }
        const double alpha = residualPreconditioned / denominator;
        for (std::size_t index = 0; index < rhs.size(); ++index) {
            solution[index] += alpha * direction[index];
            residual[index] -= alpha * product[index];
        }
        if (std::sqrt(dot(residual, residual)) <= rhsNorm * 1e-6) {
            return true;
        }
        for (std::size_t index = 0; index < rhs.size(); ++index) {
            preconditioned[index] = residual[index] / diagonal[index];
        }
        const double nextResidualPreconditioned = dot(residual, preconditioned);
        if (!std::isfinite(nextResidualPreconditioned) ||
            std::abs(residualPreconditioned) < 1e-30) {
            return false;
        }
        const double beta = nextResidualPreconditioned / residualPreconditioned;
        for (std::size_t index = 0; index < rhs.size(); ++index) {
            direction[index] =
                preconditioned[index] + beta * direction[index];
        }
        residualPreconditioned = nextResidualPreconditioned;
    }
    return std::sqrt(dot(residual, residual)) <= rhsNorm * 1e-4;
}

}  // namespace

PoseGraphConstraint makePoseGraphConstraint(
    std::size_t from, std::size_t to, const PoseGraphPose& poseFrom,
    const PoseGraphPose& poseTo, double translationInformation,
    double rotationInformation, bool loopClosure) {
    PoseGraphConstraint constraint;
    constraint.from = from;
    constraint.to = to;
    constraint.rotationFromTo =
        poseFrom.rotationWorldFromBody.t() * poseTo.rotationWorldFromBody;
    constraint.translationFromTo =
        poseFrom.rotationWorldFromBody.t() *
        (poseTo.positionWorld - poseFrom.positionWorld);
    constraint.translationInformation = translationInformation;
    constraint.rotationInformation = rotationInformation;
    constraint.loopClosure = loopClosure;
    return constraint;
}

bool optimizePoseGraph(std::vector<PoseGraphPose>& poses,
                       const std::vector<PoseGraphConstraint>& constraints,
                       int maxIterations) {
    if (poses.size() < 2 || constraints.empty() || maxIterations <= 0) {
        return false;
    }
    for (const auto& constraint : constraints) {
        if (constraint.from >= poses.size() || constraint.to >= poses.size() ||
            constraint.from == constraint.to ||
            !std::isfinite(constraint.translationInformation) ||
            !std::isfinite(constraint.rotationInformation) ||
            constraint.translationInformation <= 0.0 ||
            constraint.rotationInformation <= 0.0) {
            return false;
        }
    }

    std::vector<PoseGraphPose> optimized = poses;
    const std::size_t variablePoseCount = poses.size() - 1;
    const std::size_t dimensions = variablePoseCount * kPoseDimensions;
    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        const std::vector<LinearizedConstraint> factors =
            linearize(optimized, constraints);
        std::vector<double> rhs(dimensions, 0.0);
        std::vector<double> diagonal(dimensions, 0.0);
        accumulateNormalEquations(factors, rhs, diagonal);
        std::vector<double> delta;
        const bool solved =
            dimensions <= 600
                ? solveDense(factors, rhs, delta)
                : solveWithConjugateGradient(factors, rhs, diagonal,
                                             poses.size(), delta);
        if (!solved) {
            std::cerr << "pose graph linear solve did not converge (nodes="
                      << poses.size() << ", constraints="
                      << constraints.size() << ", iteration=" << iteration
                      << ")\n";
            return false;
        }

        double largestUpdate = 0.0;
        for (std::size_t node = 1; node < optimized.size(); ++node) {
            const int offset = offsetFor(node, 0);
            cv::Vec3d translationDelta(delta[offset], delta[offset + 1],
                                       delta[offset + 2]);
            cv::Vec3d rotationDelta(delta[offset + 3], delta[offset + 4],
                                    delta[offset + 5]);
            const double translationNorm = cv::norm(translationDelta);
            const double rotationNorm = cv::norm(rotationDelta);
            if (translationNorm > 10.0) {
                translationDelta *= 10.0 / translationNorm;
            }
            if (rotationNorm > 0.5) {
                rotationDelta *= 0.5 / rotationNorm;
            }
            optimized[node].positionWorld += translationDelta;
            optimized[node].rotationWorldFromBody =
                rodrigues(rotationDelta) *
                optimized[node].rotationWorldFromBody;
            largestUpdate = std::max(
                largestUpdate, std::max(cv::norm(translationDelta),
                                        cv::norm(rotationDelta)));
        }
        if (largestUpdate < 1e-7) {
            break;
        }
    }
    poses = std::move(optimized);
    return true;
}

}  // namespace adaptive_slam
