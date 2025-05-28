#ifndef LANE_DETECTION_LANE_SEGMENTATION_HPP
#define LANE_DETECTION_LANE_SEGMENTATION_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>
#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Float64MultiArray.h>

namespace lane_detection {

/**
 * @brief Lane segment representation
 */
struct LaneSegment {
    cv::Rect bounding_box;          // Bounding box of the segment
    cv::Point2f start_point;        // Start point of the segment
    cv::Point2f end_point;          // End point of the segment
    float width;                    // Average width of the lane segment
    float confidence;               // Confidence score [0, 1]
    std::vector<cv::Point> contour; // Detailed contour points
};

/**
 * @brief Lane line representation with geometric parameters
 */
struct LaneLine {
    int lane_id;                           // Unique identifier for the lane
    std::vector<cv::Point2f> points;       // Points defining the lane line
    cv::Vec4f line_params;                 // Line parameters (ax + by + c = 0, unused)
    float confidence;                      // Confidence score [0, 1]
    cv::Scalar color;                      // Color for visualization
    std::string lane_type;                 // Lane type: "solid", "dashed", "double"

    LaneLine() : lane_id(-1), confidence(0.0f),
                color(cv::Scalar(255, 255, 255)), lane_type("solid") {}
};

/**
 * @brief Complete lane model for a single frame
 */
struct LaneModel {
    ros::Time timestamp;                   // Timestamp of the frame
    std::string frame_id;                  // ROS frame identifier

    std::vector<LaneLine> left_lanes;      // Left lane lines
    std::vector<LaneLine> right_lanes;     // Right lane lines
    std::vector<LaneLine> center_lanes;    // Center/divider lanes

    cv::Point2f vanishing_point;           // Estimated vanishing point
    float lane_confidence;                 // Overall lane detection confidence

    LaneModel() : vanishing_point(-1, -1), lane_confidence(0.0f) {}
};

/**
 * @brief Configuration for lane segmentation processing
 */
struct LaneSegmentationConfig {
    // Preprocessing parameters
    bool use_roi = true;                          // Apply region of interest filtering
    cv::Rect region_of_interest;                  // ROI rectangle (empty = auto)

    // Morphological operations
    cv::Size morph_kernel = cv::Size(3, 3);       // Morphological kernel size
    int morph_iterations = 2;                     // Number of morphological iterations
    int close_iterations = 1;                     // Number of closing iterations

    // Contour filtering
    double min_area_threshold = 100.0;            // Minimum contour area
    double max_area_threshold = 50000.0;          // Maximum contour area
    double contour_epsilon = 2.0;                 // Contour approximation epsilon

    // Hough line detection
    double hough_rho = 1.0;                       // Distance resolution (pixels)
    double hough_theta = CV_PI / 180.0;           // Angular resolution (radians)
    int hough_threshold = 50;                     // Accumulator threshold
    double min_line_length = 30.0;                // Minimum line length
    double max_line_gap = 10.0;                   // Maximum gap between line segments

    // Lane geometry filtering
    double min_slope_threshold = 0.1;             // Minimum absolute slope
    double max_slope_threshold = 10.0;            // Maximum absolute slope

    // Lane merging parameters
    double merge_distance = 50.0;                 // Distance threshold for merging lanes
    double parallel_threshold = 0.1;              // Slope difference for parallel lines

    // Visualization colors
    cv::Scalar left_lane_color = cv::Scalar(0, 255, 0);   // Green for left lanes
    cv::Scalar right_lane_color = cv::Scalar(0, 0, 255);  // Red for right lanes
    cv::Scalar center_lane_color = cv::Scalar(255, 0, 0); // Blue for center lanes

    // Debug options
    bool enable_debug_output = false;             // Enable debug visualizations
    bool enable_performance_monitoring = true;    // Enable performance tracking
};

/**
 * @brief Processing statistics for performance monitoring
 */
struct ProcessingStatistics {
    int processed_frames = 0;
    double total_processing_time = 0.0;
    int detected_lanes_count = 0;
    double average_confidence = 0.0;
};

/**
 * @brief Main lane segmentation processor class
 */
class LaneSegmentationProcessor {
public:
    explicit LaneSegmentationProcessor(const LaneSegmentationConfig& config);

    /**
     * @brief Process lane mask to extract lane model
     * @param lane_mask Binary lane mask from YOLO detector
     * @param original_image Original camera image (for context)
     * @param timestamp ROS timestamp for the frame
     * @return Processed lane model
     */
    LaneModel processLaneMask(const cv::Mat& lane_mask,
                             const cv::Mat& original_image,
                             const ros::Time& timestamp);

    /**
     * @brief Create debug visualization of lane model
     * @param image Original image
     * @param model Lane model to visualize
     * @return Image with overlaid lane visualization
     */
    cv::Mat drawLaneModel(const cv::Mat& image, const LaneModel& model);

    /**
     * @brief Convert lane model to RViz marker for visualization
     * @param model Lane model to convert
     * @param frame_id ROS frame identifier
     * @return Visualization marker for RViz
     */
    visualization_msgs::Marker laneModelToMarker(const LaneModel& model,
                                                 const std::string& frame_id);

    // Configuration management
    void updateConfig(const LaneSegmentationConfig& new_config);
    const LaneSegmentationConfig& getConfig() const { return config_; }

    // Performance monitoring
    double getLastProcessingTime() const;
    const ProcessingStatistics& getStatistics() const { return stats_; }
    void printStatistics() const;

private:
    LaneSegmentationConfig config_;
    ProcessingStatistics stats_;
    double last_processing_time_;

    // Main processing pipeline
    cv::Mat preprocessMask(const cv::Mat& input_mask);
    cv::Mat applyROI(const cv::Mat& mask);
    cv::Mat morphologicalProcessing(const cv::Mat& mask);

    std::vector<std::vector<cv::Point>> extractLaneContours(const cv::Mat& mask);
    std::vector<LaneSegment> contoursToSegments(const std::vector<std::vector<cv::Point>>& contours);

    std::vector<LaneLine> detectLaneLines(const cv::Mat& mask);
    std::vector<LaneLine> classifyLanes(const std::vector<LaneLine>& raw_lines,
                                       const cv::Size& image_size);

    // Lane processing utilities
    void assignLaneIDs(std::vector<LaneLine>& lanes);
    std::vector<LaneLine> filterLanesByGeometry(const std::vector<LaneLine>& lanes);
    std::vector<LaneLine> mergeSimilarLanes(const std::vector<LaneLine>& lanes);

    bool validateLaneGeometry(const LaneLine& lane, const cv::Size& image_size);
    cv::Point2f estimateVanishingPoint(const std::vector<LaneLine>& lanes);

    // Helper functions
    double calculateLineSlope(const cv::Vec4f& line);
    double calculatePointDistance(const cv::Point2f& p1, const cv::Point2f& p2);
    cv::Point2f getLineIntersection(const cv::Vec4f& line1, const cv::Vec4f& line2);

    bool shouldMergeLines(const LaneLine& line1, const LaneLine& line2);
    bool areLinesParallel(const LaneLine& line1, const LaneLine& line2);
    double calculateLineDistance(const LaneLine& line1, const LaneLine& line2);
    LaneLine mergeLines(const LaneLine& line1, const LaneLine& line2);

    std::string classifyLaneType(const LaneLine& lane, const cv::Mat& mask);
    cv::Vec4f pointsToLine(const std::vector<cv::Point2f>& points);

    // Logging utilities
    void logError(const std::string& message);
    void logWarning(const std::string& message);
    void logInfo(const std::string& message);
};

/**
 * @brief Factory function to create lane segmentation processor
 * @param config_path Path to configuration file (optional)
 * @return Unique pointer to processor instance
 */
std::unique_ptr<LaneSegmentationProcessor> createLaneSegmentationProcessor(
    const std::string& config_path = "");

/**
 * @brief Utility functions for lane processing
 */
namespace lane_utils {
    /**
     * @brief Convert OpenCV points to ROS geometry points
     */
    std::vector<geometry_msgs::Point> cvPointsToRosPoints(const std::vector<cv::Point2f>& cv_points);

    /**
     * @brief Calculate lane curvature from points
     */
    double calculateLaneCurvature(const std::vector<cv::Point2f>& points);

    /**
     * @brief Smooth lane points using moving average
     */
    std::vector<cv::Point2f> smoothLanePoints(const std::vector<cv::Point2f>& points, int window_size = 5);

    /**
     * @brief Validate lane geometry constraints
     */
    bool isValidLaneGeometry(const LaneLine& lane, const cv::Size& image_size);
}

} // namespace lane_detection

#endif // LANE_DETECTION_LANE_SEGMENTATION_HPP