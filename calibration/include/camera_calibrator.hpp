#ifndef CALIBRATION_CAMERA_CALIBRATOR_HPP
#define CALIBRATION_CAMERA_CALIBRATOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <string>
#include <memory>

namespace calibration {

/**
 * @brief Camera calibration parameters structure
 */
struct CameraParameters {
    cv::Mat camera_matrix;          // 3x3 intrinsic camera matrix
    cv::Mat distortion_coeffs;      // Distortion coefficients (k1,k2,p1,p2,k3,...)
    cv::Size image_size;            // Calibrated image dimensions
    double reprojection_error;      // RMS reprojection error
    bool is_valid;                  // Calibration validity flag

    CameraParameters() : reprojection_error(0.0), is_valid(false) {}
};

/**
 * @brief High-performance camera calibrator for embedded vision systems
 *
 * Optimized for NVIDIA Drive platforms with minimal memory footprint
 * and deterministic performance characteristics. Supports standard
 * OpenCV calibration workflows with enhanced error handling.
 */
class CameraCalibrator {
public:
    CameraCalibrator();
    ~CameraCalibrator();

    // Non-copyable for embedded safety
    CameraCalibrator(const CameraCalibrator&) = delete;
    CameraCalibrator& operator=(const CameraCalibrator&) = delete;

    /**
     * @brief Load camera calibration from YAML configuration file
     * @param config_path Path to calibration configuration file
     * @return True on successful calibration load
     */
    bool loadCalibration(const std::string& config_path);

    /**
     * @brief Load calibration from OpenCV calibration parameters
     * @param camera_matrix 3x3 intrinsic camera matrix
     * @param dist_coeffs Distortion coefficients vector
     * @param image_size Image dimensions for calibration
     * @return True on successful parameter validation
     */
    bool loadCalibration(const cv::Mat& camera_matrix,
                        const cv::Mat& dist_coeffs,
                        const cv::Size& image_size);

    /**
     * @brief Apply distortion correction to input image
     * @param input Source image with lens distortion
     * @param output Undistorted output image
     * @return True on successful undistortion
     */
    bool undistortImage(const cv::Mat& input, cv::Mat& output);

    /**
     * @brief Apply distortion correction with custom interpolation
     * @param input Source image with lens distortion
     * @param output Undistorted output image
     * @param interpolation OpenCV interpolation method
     * @return True on successful undistortion
     */
    bool undistortImage(const cv::Mat& input, cv::Mat& output, int interpolation);

    /**
     * @brief Project 3D point to 2D image coordinates
     * @param object_point 3D point in camera coordinate system
     * @param image_point Resulting 2D image coordinates
     * @return True on successful projection
     */
    bool projectPoint(const cv::Point3f& object_point, cv::Point2f& image_point);

    /**
     * @brief Project multiple 3D points to 2D image coordinates
     * @param object_points Vector of 3D points
     * @param image_points Resulting 2D image coordinates
     * @return True on successful projection
     */
    bool projectPoints(const std::vector<cv::Point3f>& object_points,
                      std::vector<cv::Point2f>& image_points);

    // Validation and status
    bool isCalibrationValid() const { return parameters_.is_valid; }
    const CameraParameters& getParameters() const { return parameters_; }
    double getReprojectionError() const { return parameters_.reprojection_error; }
    cv::Size getImageSize() const { return parameters_.image_size; }

    /**
     * @brief Save current calibration to YAML file
     * @param output_path Path for calibration output file
     * @return True on successful save
     */
    bool saveCalibration(const std::string& output_path) const;

    /**
     * @brief Validate calibration parameters for geometric consistency
     * @return True if calibration parameters are geometrically valid
     */
    bool validateCalibration() const;

    /**
     * @brief Reset calibration to invalid state
     */
    void reset();

private:
    CameraParameters parameters_;

    // Precomputed undistortion maps for performance optimization
    cv::Mat map1_, map2_;
    bool maps_initialized_;

    // Performance optimization flags
    bool enable_map_caching_;
    int last_interpolation_method_;

    /**
     * @brief Initialize undistortion maps for optimal performance
     * @param interpolation OpenCV interpolation method
     * @return True on successful map initialization
     */
    bool initializeUndistortMaps(int interpolation = cv::INTER_LINEAR);

    /**
     * @brief Parse calibration from OpenCV YAML format
     * @param config_path Path to calibration file
     * @return True on successful parsing
     */
    bool parseCalibrationFile(const std::string& config_path);

    /**
     * @brief Validate camera matrix properties
     * @param camera_matrix Input camera matrix to validate
     * @return True if matrix is geometrically valid
     */
    bool validateCameraMatrix(const cv::Mat& camera_matrix) const;

    /**
     * @brief Validate distortion coefficients
     * @param dist_coeffs Input distortion coefficients
     * @return True if coefficients are within reasonable bounds
     */
    bool validateDistortionCoeffs(const cv::Mat& dist_coeffs) const;

    /**
     * @brief Log error message with class context
     */
    void logError(const std::string& message) const;

    /**
     * @brief Log warning message with class context
     */
    void logWarning(const std::string& message) const;

    /**
     * @brief Log info message with class context
     */
    void logInfo(const std::string& message) const;
};

/**
 * @brief Factory function to create calibrator from configuration
 * @param config_path Path to calibration configuration file
 * @return Unique pointer to configured calibrator, nullptr on failure
 */
std::unique_ptr<CameraCalibrator> createCameraCalibrator(const std::string& config_path);

/**
 * @brief Utility functions for camera calibration
 */
namespace calib_utils {
    /**
     * @brief Check if calibration file exists and is readable
     */
    bool validateCalibrationFile(const std::string& file_path);

    /**
     * @brief Extract focal length from camera matrix
     */
    cv::Point2f extractFocalLength(const cv::Mat& camera_matrix);

    /**
     * @brief Extract principal point from camera matrix
     */
    cv::Point2f extractPrincipalPoint(const cv::Mat& camera_matrix);

    /**
     * @brief Calculate field of view from camera parameters
     */
    cv::Point2f calculateFieldOfView(const cv::Mat& camera_matrix, const cv::Size& image_size);
}

} // namespace calibration

#endif // CALIBRATION_CAMERA_CALIBRATOR_HPP