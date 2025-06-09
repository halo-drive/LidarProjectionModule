#include "camera_calibrator.hpp"
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <cmath>

// ROS logging compatibility - fallback to std::cout if ROS not available
#ifdef ROSCONSOLE_CONSOLE_H_
#include <ros/console.h>
#define LOG_ERROR(msg) ROS_ERROR("[CameraCalibrator] %s", (msg).c_str())
#define LOG_WARN(msg)  ROS_WARN("[CameraCalibrator] %s", (msg).c_str())
#define LOG_INFO(msg)  ROS_INFO("[CameraCalibrator] %s", (msg).c_str())
#else
#include <iostream>
#define LOG_ERROR(msg) std::cerr << "[CameraCalibrator ERROR] " << (msg) << std::endl
#define LOG_WARN(msg)  std::cout << "[CameraCalibrator WARN] " << (msg) << std::endl
#define LOG_INFO(msg)  std::cout << "[CameraCalibrator INFO] " << (msg) << std::endl
#endif

namespace calibration {

CameraCalibrator::CameraCalibrator()
    : maps_initialized_(false), enable_map_caching_(true),
      last_interpolation_method_(cv::INTER_LINEAR) {
    reset();
}

CameraCalibrator::~CameraCalibrator() {
    // Explicit cleanup for embedded systems
    map1_.release();
    map2_.release();
    parameters_.camera_matrix.release();
    parameters_.distortion_coeffs.release();
}

bool CameraCalibrator::loadCalibration(const std::string& config_path) {
    if (config_path.empty()) {
        logError("Empty calibration file path provided");
        return false;
    }

    if (!calib_utils::validateCalibrationFile(config_path)) {
        logError("Calibration file validation failed: " + config_path);
        return false;
    }

    return parseCalibrationFile(config_path);
}

bool CameraCalibrator::loadCalibration(const cv::Mat& camera_matrix,
                                      const cv::Mat& dist_coeffs,
                                      const cv::Size& image_size) {
    // Validate input parameters
    if (!validateCameraMatrix(camera_matrix)) {
        logError("Invalid camera matrix provided");
        return false;
    }

    if (!validateDistortionCoeffs(dist_coeffs)) {
        logError("Invalid distortion coefficients provided");
        return false;
    }

    if (image_size.width <= 0 || image_size.height <= 0) {
        logError("Invalid image size provided");
        return false;
    }

    // Store calibration parameters
    camera_matrix.copyTo(parameters_.camera_matrix);
    dist_coeffs.copyTo(parameters_.distortion_coeffs);
    parameters_.image_size = image_size;
    parameters_.reprojection_error = 0.0; // Unknown for manually loaded parameters
    parameters_.is_valid = true;

    // Reset undistortion maps to force recomputation
    maps_initialized_ = false;
    map1_.release();
    map2_.release();

    logInfo("Camera calibration loaded successfully from parameters");
    return true;
}

bool CameraCalibrator::undistortImage(const cv::Mat& input, cv::Mat& output) {
    return undistortImage(input, output, cv::INTER_LINEAR);
}

bool CameraCalibrator::undistortImage(const cv::Mat& input, cv::Mat& output, int interpolation) {
    if (!parameters_.is_valid) {
        logError("Cannot undistort image: calibration not loaded");
        return false;
    }

    if (input.empty()) {
        logError("Cannot undistort empty input image");
        return false;
    }

    // Validate image size consistency
    if (input.size() != parameters_.image_size) {
        logWarning("Input image size differs from calibration size - proceeding with caution");
    }

    try {
        // Initialize or update undistortion maps if necessary
        if (!maps_initialized_ || last_interpolation_method_ != interpolation) {
            if (!initializeUndistortMaps(interpolation)) {
                logError("Failed to initialize undistortion maps");
                return false;
            }
        }

        // Apply undistortion using precomputed maps for optimal performance
        if (enable_map_caching_ && maps_initialized_) {
            cv::remap(input, output, map1_, map2_, interpolation, cv::BORDER_CONSTANT);
        } else {
            // Direct undistortion - slower but more memory efficient
            cv::undistort(input, output, parameters_.camera_matrix,
                         parameters_.distortion_coeffs, parameters_.camera_matrix);
        }

        return true;

    } catch (const cv::Exception& e) {
        logError("OpenCV exception during undistortion: " + std::string(e.what()));
        return false;
    } catch (const std::exception& e) {
        logError("Exception during undistortion: " + std::string(e.what()));
        return false;
    }
}

bool CameraCalibrator::projectPoint(const cv::Point3f& object_point, cv::Point2f& image_point) {
    std::vector<cv::Point3f> object_points = {object_point};
    std::vector<cv::Point2f> image_points;

    if (!projectPoints(object_points, image_points)) {
        return false;
    }

    if (image_points.empty()) {
        logError("Point projection failed - no output points");
        return false;
    }

    image_point = image_points[0];
    return true;
}

bool CameraCalibrator::projectPoints(const std::vector<cv::Point3f>& object_points,
                                    std::vector<cv::Point2f>& image_points) {
    if (!parameters_.is_valid) {
        logError("Cannot project points: calibration not loaded");
        return false;
    }

    if (object_points.empty()) {
        logError("Cannot project empty point set");
        return false;
    }

    try {
        // Use identity rotation and translation (points assumed in camera coordinate system)
        cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);

        cv::projectPoints(object_points, rvec, tvec, parameters_.camera_matrix,
                         parameters_.distortion_coeffs, image_points);

        return true;

    } catch (const cv::Exception& e) {
        logError("OpenCV exception during point projection: " + std::string(e.what()));
        return false;
    } catch (const std::exception& e) {
        logError("Exception during point projection: " + std::string(e.what()));
        return false;
    }
}

bool CameraCalibrator::saveCalibration(const std::string& output_path) const {
    if (!parameters_.is_valid) {
        logError("Cannot save invalid calibration");
        return false;
    }

    try {
        cv::FileStorage fs(output_path, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            logError("Failed to open calibration file for writing: " + output_path);
            return false;
        }

        // Write calibration parameters in standard OpenCV format
        fs << "camera_matrix" << parameters_.camera_matrix;
        fs << "distortion_coefficients" << parameters_.distortion_coeffs;
        fs << "image_width" << parameters_.image_size.width;
        fs << "image_height" << parameters_.image_size.height;
        fs << "reprojection_error" << parameters_.reprojection_error;

        // Additional metadata
        fs << "calibration_time" << cv::format("%s", cv::getCVObjInfo());

        fs.release();
        logInfo("Calibration saved successfully to: " + output_path);
        return true;

    } catch (const cv::Exception& e) {
        logError("OpenCV exception saving calibration: " + std::string(e.what()));
        return false;
    } catch (const std::exception& e) {
        logError("Exception saving calibration: " + std::string(e.what()));
        return false;
    }
}

bool CameraCalibrator::validateCalibration() const {
    if (!parameters_.is_valid) {
        return false;
    }

    // Validate camera matrix properties
    if (!validateCameraMatrix(parameters_.camera_matrix)) {
        return false;
    }

    // Validate distortion coefficients
    if (!validateDistortionCoeffs(parameters_.distortion_coeffs)) {
        return false;
    }

    // Validate image size
    if (parameters_.image_size.width <= 0 || parameters_.image_size.height <= 0) {
        return false;
    }

    // Validate reprojection error (should be reasonable for real calibrations)
    if (parameters_.reprojection_error < 0.0 || parameters_.reprojection_error > 10.0) {
        logWarning("Reprojection error outside typical range: " +
                  std::to_string(parameters_.reprojection_error));
    }

    return true;
}

void CameraCalibrator::reset() {
    parameters_ = CameraParameters();
    maps_initialized_ = false;
    map1_.release();
    map2_.release();
    last_interpolation_method_ = cv::INTER_LINEAR;
}

bool CameraCalibrator::initializeUndistortMaps(int interpolation) {
    if (!parameters_.is_valid) {
        logError("Cannot initialize undistortion maps: invalid calibration");
        return false;
    }

    try {
        // Generate undistortion and rectification maps
        cv::initUndistortRectifyMap(
            parameters_.camera_matrix,
            parameters_.distortion_coeffs,
            cv::Mat(), // No rectification
            parameters_.camera_matrix, // Keep same camera matrix
            parameters_.image_size,
            CV_16SC2, // Optimized map type for embedded systems
            map1_,
            map2_
        );

        maps_initialized_ = true;
        last_interpolation_method_ = interpolation;

        logInfo("Undistortion maps initialized successfully");
        return true;

    } catch (const cv::Exception& e) {
        logError("OpenCV exception initializing undistortion maps: " + std::string(e.what()));
        return false;
    } catch (const std::exception& e) {
        logError("Exception initializing undistortion maps: " + std::string(e.what()));
        return false;
    }
}

bool CameraCalibrator::parseCalibrationFile(const std::string& config_path) {
    try {
        cv::FileStorage fs(config_path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            logError("Failed to open calibration file: " + config_path);
            return false;
        }

        cv::Mat camera_matrix, dist_coeffs;
        int image_width = 0, image_height = 0;
        double reprojection_error = 0.0;

        // Read calibration parameters with fallback names
        cv::FileNode camera_node = fs["camera_matrix"];
        if (camera_node.empty()) {
            camera_node = fs["camera_intrinsics"];
        }
        if (camera_node.empty()) {
            logError("Camera matrix not found in calibration file");
            return false;
        }
        camera_node >> camera_matrix;

        cv::FileNode dist_node = fs["distortion_coefficients"];
        if (dist_node.empty()) {
            dist_node = fs["distortion_coeffs"];
        }
        if (dist_node.empty()) {
            logError("Distortion coefficients not found in calibration file");
            return false;
        }
        dist_node >> dist_coeffs;

        // Read image dimensions
        cv::FileNode width_node = fs["image_width"];
        cv::FileNode height_node = fs["image_height"];
        if (!width_node.empty() && !height_node.empty()) {
            width_node >> image_width;
            height_node >> image_height;
        } else {
            // Fallback: try to infer from camera matrix
            if (camera_matrix.rows == 3 && camera_matrix.cols == 3) {
                double cx = camera_matrix.at<double>(0, 2);
                double cy = camera_matrix.at<double>(1, 2);
                image_width = static_cast<int>(cx * 2);
                image_height = static_cast<int>(cy * 2);
                logWarning("Image dimensions not found, estimated from principal point");
            } else {
                logError("Cannot determine image dimensions from calibration file");
                return false;
            }
        }

        // Read reprojection error (optional)
        cv::FileNode error_node = fs["reprojection_error"];
        if (!error_node.empty()) {
            error_node >> reprojection_error;
        }

        fs.release();

        // Validate and store parameters
        if (!loadCalibration(camera_matrix, dist_coeffs, cv::Size(image_width, image_height))) {
            logError("Failed to validate loaded calibration parameters");
            return false;
        }

        parameters_.reprojection_error = reprojection_error;
        logInfo("Calibration loaded successfully from file: " + config_path);
        return true;

    } catch (const cv::Exception& e) {
        logError("OpenCV exception parsing calibration file: " + std::string(e.what()));
        return false;
    } catch (const std::exception& e) {
        logError("Exception parsing calibration file: " + std::string(e.what()));
        return false;
    }
}

bool CameraCalibrator::validateCameraMatrix(const cv::Mat& camera_matrix) const {
    if (camera_matrix.empty() || camera_matrix.rows != 3 || camera_matrix.cols != 3) {
        logError("Camera matrix must be 3x3");
        return false;
    }

    if (camera_matrix.type() != CV_64F && camera_matrix.type() != CV_32F) {
        logError("Camera matrix must be floating point type");
        return false;
    }

    // Check for reasonable focal length values
    double fx = camera_matrix.at<double>(0, 0);
    double fy = camera_matrix.at<double>(1, 1);

    if (fx <= 0.0 || fy <= 0.0) {
        logError("Focal lengths must be positive");
        return false;
    }

    if (fx < 50.0 || fx > 10000.0 || fy < 50.0 || fy > 10000.0) {
        logWarning("Focal lengths outside typical range: fx=" + std::to_string(fx) +
                  ", fy=" + std::to_string(fy));
    }

    // Check principal point
    double cx = camera_matrix.at<double>(0, 2);
    double cy = camera_matrix.at<double>(1, 2);

    if (cx < 0.0 || cy < 0.0) {
        logWarning("Principal point has negative coordinates");
    }

    return true;
}

bool CameraCalibrator::validateDistortionCoeffs(const cv::Mat& dist_coeffs) const {
    if (dist_coeffs.empty()) {
        logWarning("Empty distortion coefficients - assuming no distortion");
        return true;
    }

    if (dist_coeffs.type() != CV_64F && dist_coeffs.type() != CV_32F) {
        logError("Distortion coefficients must be floating point type");
        return false;
    }

    int num_coeffs = dist_coeffs.rows * dist_coeffs.cols;
    if (num_coeffs < 4 || num_coeffs > 14) {
        logError("Invalid number of distortion coefficients: " + std::to_string(num_coeffs));
        return false;
    }

    // Check for reasonable coefficient magnitudes
    for (int i = 0; i < num_coeffs; ++i) {
        double coeff = dist_coeffs.at<double>(i);
        if (std::abs(coeff) > 10.0) {
            logWarning("Large distortion coefficient detected: " + std::to_string(coeff));
        }
    }

    return true;
}

void CameraCalibrator::logError(const std::string& message) const {
    LOG_ERROR(message);
}

void CameraCalibrator::logWarning(const std::string& message) const {
    LOG_WARN(message);
}

void CameraCalibrator::logInfo(const std::string& message) const {
    LOG_INFO(message);
}

// Factory function implementation
std::unique_ptr<CameraCalibrator> createCameraCalibrator(const std::string& config_path) {
    auto calibrator = std::make_unique<CameraCalibrator>();

    if (!config_path.empty()) {
        if (!calibrator->loadCalibration(config_path)) {
            logError("Failed to create calibrator from config: " + config_path);
            return nullptr;
        }
    }

    return calibrator;
}

// Utility functions implementation
namespace calib_utils {

bool validateCalibrationFile(const std::string& file_path) {
    if (file_path.empty()) {
        return false;
    }

    std::ifstream file(file_path);
    if (!file.is_open()) {
        return false;
    }

    file.close();

    // Check file extension
    std::string extension = file_path.substr(file_path.find_last_of(".") + 1);
    if (extension != "yml" && extension != "yaml" && extension != "xml") {
        LOG_WARN("Unexpected calibration file extension: " + extension);
    }

    return true;
}

cv::Point2f extractFocalLength(const cv::Mat& camera_matrix) {
    if (camera_matrix.rows != 3 || camera_matrix.cols != 3) {
        return cv::Point2f(0, 0);
    }

    double fx = camera_matrix.at<double>(0, 0);
    double fy = camera_matrix.at<double>(1, 1);

    return cv::Point2f(static_cast<float>(fx), static_cast<float>(fy));
}

cv::Point2f extractPrincipalPoint(const cv::Mat& camera_matrix) {
    if (camera_matrix.rows != 3 || camera_matrix.cols != 3) {
        return cv::Point2f(0, 0);
    }

    double cx = camera_matrix.at<double>(0, 2);
    double cy = camera_matrix.at<double>(1, 2);

    return cv::Point2f(static_cast<float>(cx), static_cast<float>(cy));
}

cv::Point2f calculateFieldOfView(const cv::Mat& camera_matrix, const cv::Size& image_size) {
    cv::Point2f focal_length = extractFocalLength(camera_matrix);

    if (focal_length.x <= 0 || focal_length.y <= 0 ||
        image_size.width <= 0 || image_size.height <= 0) {
        return cv::Point2f(0, 0);
    }

    double fov_x = 2.0 * std::atan(image_size.width / (2.0 * focal_length.x)) * 180.0 / CV_PI;
    double fov_y = 2.0 * std::atan(image_size.height / (2.0 * focal_length.y)) * 180.0 / CV_PI;

    return cv::Point2f(static_cast<float>(fov_x), static_cast<float>(fov_y));
}

} // namespace calib_utils

} // namespace calibration