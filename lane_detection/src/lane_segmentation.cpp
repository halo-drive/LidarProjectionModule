#include "lane_segmentation.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <ros/ros.h>

namespace lane_detection {

LaneSegmentationProcessor::LaneSegmentationProcessor(const LaneSegmentationConfig& config)
    : config_(config), last_processing_time_(0.0) {
    stats_.processed_frames = 0;
    stats_.total_processing_time = 0.0;
    stats_.detected_lanes_count = 0;
    stats_.average_confidence = 0.0;
}

LaneModel LaneSegmentationProcessor::processLaneMask(const cv::Mat& lane_mask,
                                                   const cv::Mat& original_image,
                                                   const ros::Time& timestamp) {
    auto start_time = std::chrono::high_resolution_clock::now();

    LaneModel model;
    model.timestamp = timestamp;
    model.frame_id = "camera0_link";

    if (lane_mask.empty()) {
        ROS_WARN("Empty lane mask provided");
        return model;
    }

    try {
        // Step 1: Preprocess the mask
        cv::Mat processed_mask = preprocessMask(lane_mask);

        // Step 2: Extract lane contours
        std::vector<std::vector<cv::Point>> contours = extractLaneContours(processed_mask);

        // Step 3: Convert contours to lane segments
        std::vector<LaneSegment> segments = contoursToSegments(contours);

        // Step 4: Detect lane lines
        std::vector<LaneLine> raw_lines = detectLaneLines(processed_mask);

        // Step 5: Classify and filter lanes
        std::vector<LaneLine> classified_lanes = classifyLanes(raw_lines, original_image.size());

        // Step 6: Assign to model structure
        for (const auto& lane : classified_lanes) {
            cv::Point2f center = cv::Point2f(original_image.cols / 2.0f, original_image.rows / 2.0f);

            // Determine if lane is left, right, or center based on position
            if (lane.points.size() > 0) {
                float avg_x = 0;
                for (const auto& pt : lane.points) {
                    avg_x += pt.x;
                }
                avg_x /= lane.points.size();

                if (avg_x < center.x - 50) {
                    model.left_lanes.push_back(lane);
                } else if (avg_x > center.x + 50) {
                    model.right_lanes.push_back(lane);
                } else {
                    model.center_lanes.push_back(lane);
                }
            }
        }

        // Step 7: Estimate vanishing point
        model.vanishing_point = estimateVanishingPoint(classified_lanes);

        // Step 8: Calculate overall confidence
        float total_confidence = 0.0f;
        int lane_count = 0;
        for (const auto& lane : classified_lanes) {
            total_confidence += lane.confidence;
            lane_count++;
        }
        model.lane_confidence = (lane_count > 0) ? total_confidence / lane_count : 0.0f;

        // Update statistics
        auto end_time = std::chrono::high_resolution_clock::now();
        last_processing_time_ = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();

        stats_.processed_frames++;
        stats_.total_processing_time += last_processing_time_;
        stats_.detected_lanes_count += lane_count;
        stats_.average_confidence = (stats_.average_confidence * (stats_.processed_frames - 1) +
                                   model.lane_confidence) / stats_.processed_frames;

        return model;

    } catch (const std::exception& e) {
        ROS_ERROR("Exception in lane segmentation processing: %s", e.what());
        return model;
    }
}

cv::Mat LaneSegmentationProcessor::preprocessMask(const cv::Mat& input_mask) {
    cv::Mat processed;

    // Convert to grayscale if needed
    if (input_mask.channels() == 3) {
        cv::cvtColor(input_mask, processed, cv::COLOR_BGR2GRAY);
    } else {
        input_mask.copyTo(processed);
    }

    // Apply threshold to ensure binary mask
    cv::threshold(processed, processed, 127, 255, cv::THRESH_BINARY);

    // Apply ROI if enabled
    if (config_.use_roi) {
        processed = applyROI(processed);
    }

    // Apply morphological operations
    processed = morphologicalProcessing(processed);

    return processed;
}

cv::Mat LaneSegmentationProcessor::applyROI(const cv::Mat& mask) {
    cv::Mat roi_mask = mask.clone();

    if (config_.region_of_interest.area() > 0) {
        cv::Mat roi_filter = cv::Mat::zeros(mask.size(), CV_8UC1);
        cv::rectangle(roi_filter, config_.region_of_interest, cv::Scalar(255), -1);
        cv::bitwise_and(roi_mask, roi_filter, roi_mask);
    } else {
        // Default ROI: lower 60% of image
        cv::Mat roi_filter = cv::Mat::zeros(mask.size(), CV_8UC1);
        cv::Rect default_roi(0, mask.rows * 0.4, mask.cols, mask.rows * 0.6);
        cv::rectangle(roi_filter, default_roi, cv::Scalar(255), -1);
        cv::bitwise_and(roi_mask, roi_filter, roi_mask);
    }

    return roi_mask;
}

cv::Mat LaneSegmentationProcessor::morphologicalProcessing(const cv::Mat& mask) {
    cv::Mat processed = mask.clone();

    // Create morphological kernel
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, config_.morph_kernel);

    // Apply closing to fill gaps
    cv::morphologyEx(processed, processed, cv::MORPH_CLOSE, kernel,
                    cv::Point(-1, -1), config_.close_iterations);

    // Apply opening to remove noise
    cv::morphologyEx(processed, processed, cv::MORPH_OPEN, kernel,
                    cv::Point(-1, -1), config_.morph_iterations);

    return processed;
}

std::vector<std::vector<cv::Point>> LaneSegmentationProcessor::extractLaneContours(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Filter contours by area
    std::vector<std::vector<cv::Point>> filtered_contours;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area >= config_.min_area_threshold && area <= config_.max_area_threshold) {
            // Simplify contour
            std::vector<cv::Point> simplified_contour;
            cv::approxPolyDP(contour, simplified_contour, config_.contour_epsilon, true);
            filtered_contours.push_back(simplified_contour);
        }
    }

    return filtered_contours;
}

std::vector<LaneSegment> LaneSegmentationProcessor::contoursToSegments(
    const std::vector<std::vector<cv::Point>>& contours) {

    std::vector<LaneSegment> segments;

    for (const auto& contour : contours) {
        if (contour.size() < 2) continue;

        LaneSegment segment;

        // Find bounding box
        segment.bounding_box = cv::boundingRect(contour);

        // Set start and end points
        segment.start_point = cv::Point2f(contour.front());
        segment.end_point = cv::Point2f(contour.back());

        // Estimate width (average distance between parallel edges)
        segment.width = std::sqrt(segment.bounding_box.area() /
                                std::max(segment.bounding_box.width, segment.bounding_box.height));

        // Calculate confidence based on contour properties
        double perimeter = cv::arcLength(contour, false);
        double area = cv::contourArea(contour);
        segment.confidence = std::min(1.0, area / (perimeter * perimeter)); // Compactness measure

        segments.push_back(segment);
    }

    return segments;
}

std::vector<LaneLine> LaneSegmentationProcessor::detectLaneLines(const cv::Mat& mask) {
    std::vector<LaneLine> lanes;

    // Apply Hough Line Transform
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(mask, lines, config_.hough_rho, config_.hough_theta,
                   config_.hough_threshold, config_.min_line_length, config_.max_line_gap);

    // Convert cv::Vec4i to LaneLine objects
    for (const auto& line : lines) {
        LaneLine lane;

        // Extract line points
        cv::Point2f start(line[0], line[1]);
        cv::Point2f end(line[2], line[3]);

        lane.points.push_back(start);
        lane.points.push_back(end);

        // Calculate line parameters (ax + by + c = 0)
        float dx = end.x - start.x;
        float dy = end.y - start.y;

        if (std::abs(dx) > 0.001) { // Avoid division by zero
            float slope = dy / dx;
            float intercept = start.y - slope * start.x;

            // Convert to standard form: ax + by + c = 0
            lane.line_params = cv::Vec4f(-slope, 1, -intercept, 0);
        } else {
            // Vertical line
            lane.line_params = cv::Vec4f(1, 0, -start.x, 0);
        }

        // Calculate confidence based on line length
        float length = std::sqrt(dx * dx + dy * dy);
        lane.confidence = std::min(1.0, length / 100.0); // Normalize to reasonable range

        // Assign color (will be updated in classification)
        lane.color = cv::Scalar(255, 255, 255);

        lanes.push_back(lane);
    }

    return lanes;
}

std::vector<LaneLine> LaneSegmentationProcessor::classifyLanes(const std::vector<LaneLine>& raw_lines,
                                                              const cv::Size& image_size) {
    std::vector<LaneLine> classified_lanes;

    for (auto lane : raw_lines) {
        // Filter by geometry
        if (!validateLaneGeometry(lane, image_size)) {
            continue;
        }

        // Calculate slope for classification
        double slope = calculateLineSlope(lane.line_params);

        // Filter by slope thresholds
        if (std::abs(slope) < config_.min_slope_threshold ||
            std::abs(slope) > config_.max_slope_threshold) {
            continue;
        }

        // Assign lane type based on analysis
        lane.lane_type = classifyLaneType(lane, cv::Mat()); // Pass empty mask for now

        // Assign color based on position
        float center_x = image_size.width / 2.0f;
        float lane_x = (lane.points[0].x + lane.points[1].x) / 2.0f;

        if (lane_x < center_x) {
            lane.color = config_.left_lane_color;
        } else {
            lane.color = config_.right_lane_color;
        }

        classified_lanes.push_back(lane);
    }

    // Merge similar lanes
    classified_lanes = mergeSimilarLanes(classified_lanes);

    // Assign unique IDs
    assignLaneIDs(classified_lanes);

    return classified_lanes;
}

void LaneSegmentationProcessor::assignLaneIDs(std::vector<LaneLine>& lanes) {
    for (size_t i = 0; i < lanes.size(); ++i) {
        lanes[i].lane_id = static_cast<int>(i);
    }
}

std::vector<LaneLine> LaneSegmentationProcessor::filterLanesByGeometry(
    const std::vector<LaneLine>& lanes) {

    std::vector<LaneLine> filtered;

    for (const auto& lane : lanes) {
        if (lane.points.size() >= 2) {
            // Check if line has reasonable slope
            double slope = calculateLineSlope(lane.line_params);
            if (std::abs(slope) >= config_.min_slope_threshold &&
                std::abs(slope) <= config_.max_slope_threshold) {
                filtered.push_back(lane);
            }
        }
    }

    return filtered;
}

std::vector<LaneLine> LaneSegmentationProcessor::mergeSimilarLanes(
    const std::vector<LaneLine>& lanes) {

    std::vector<LaneLine> merged;
    std::vector<bool> used(lanes.size(), false);

    for (size_t i = 0; i < lanes.size(); ++i) {
        if (used[i]) continue;

        LaneLine current_lane = lanes[i];
        used[i] = true;

        // Look for similar lanes to merge
        for (size_t j = i + 1; j < lanes.size(); ++j) {
            if (used[j]) continue;

            if (shouldMergeLines(current_lane, lanes[j])) {
                current_lane = mergeLines(current_lane, lanes[j]);
                used[j] = true;
            }
        }

        merged.push_back(current_lane);
    }

    return merged;
}

bool LaneSegmentationProcessor::validateLaneGeometry(const LaneLine& lane,
                                                    const cv::Size& image_size) {
    if (lane.points.size() < 2) return false;

    // Check if points are within image bounds
    for (const auto& point : lane.points) {
        if (point.x < 0 || point.x >= image_size.width ||
            point.y < 0 || point.y >= image_size.height) {
            return false;
        }
    }

    // Check minimum line length
    float length = calculatePointDistance(lane.points[0], lane.points[1]);
    if (length < config_.min_line_length) {
        return false;
    }

    return true;
}

cv::Point2f LaneSegmentationProcessor::estimateVanishingPoint(
    const std::vector<LaneLine>& lanes) {

    if (lanes.size() < 2) {
        return cv::Point2f(-1, -1); // Invalid point
    }

    std::vector<cv::Point2f> intersections;

    // Find intersections between lane pairs
    for (size_t i = 0; i < lanes.size(); ++i) {
        for (size_t j = i + 1; j < lanes.size(); ++j) {
            cv::Point2f intersection = getLineIntersection(lanes[i].line_params,
                                                          lanes[j].line_params);
            if (intersection.x >= 0 && intersection.y >= 0) {
                intersections.push_back(intersection);
            }
        }
    }

    if (intersections.empty()) {
        return cv::Point2f(-1, -1);
    }

    // Calculate average intersection point
    cv::Point2f vanishing_point(0, 0);
    for (const auto& pt : intersections) {
        vanishing_point += pt;
    }
    vanishing_point.x /= intersections.size();
    vanishing_point.y /= intersections.size();

    return vanishing_point;
}

cv::Mat LaneSegmentationProcessor::drawLaneModel(const cv::Mat& image, const LaneModel& model) {
    cv::Mat result;
    image.copyTo(result);

    // Draw left lanes
    for (const auto& lane : model.left_lanes) {
        for (size_t i = 0; i < lane.points.size() - 1; ++i) {
            cv::line(result, lane.points[i], lane.points[i + 1], lane.color, 3);
        }
    }

    // Draw right lanes
    for (const auto& lane : model.right_lanes) {
        for (size_t i = 0; i < lane.points.size() - 1; ++i) {
            cv::line(result, lane.points[i], lane.points[i + 1], lane.color, 3);
        }
    }

    // Draw center lanes
    for (const auto& lane : model.center_lanes) {
        for (size_t i = 0; i < lane.points.size() - 1; ++i) {
            cv::line(result, lane.points[i], lane.points[i + 1], lane.color, 3);
        }
    }

    // Draw vanishing point
    if (model.vanishing_point.x >= 0 && model.vanishing_point.y >= 0) {
        cv::circle(result, model.vanishing_point, 5, cv::Scalar(0, 0, 255), -1);
    }

    // Add confidence text
    std::string conf_text = "Confidence: " + std::to_string(model.lane_confidence);
    cv::putText(result, conf_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
               0.7, cv::Scalar(255, 255, 255), 2);

    return result;
}

visualization_msgs::Marker LaneSegmentationProcessor::laneModelToMarker(
    const LaneModel& model, const std::string& frame_id) {

    visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = model.timestamp;
    marker.id = 0;
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;

    marker.scale.x = 0.05; // Line width
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;

    // Add all lane points
    auto addLanePoints = [&](const std::vector<LaneLine>& lanes) {
        for (const auto& lane : lanes) {
            for (size_t i = 0; i < lane.points.size() - 1; ++i) {
                geometry_msgs::Point p1, p2;
                p1.x = lane.points[i].x / 100.0;     // Convert pixels to meters (rough)
                p1.y = -lane.points[i].y / 100.0;    // Flip Y axis
                p1.z = 0.0;

                p2.x = lane.points[i + 1].x / 100.0;
                p2.y = -lane.points[i + 1].y / 100.0;
                p2.z = 0.0;

                marker.points.push_back(p1);
                marker.points.push_back(p2);
            }
        }
    };

    addLanePoints(model.left_lanes);
    addLanePoints(model.right_lanes);
    addLanePoints(model.center_lanes);

    return marker;
}

void LaneSegmentationProcessor::updateConfig(const LaneSegmentationConfig& new_config) {
    config_ = new_config;
}

double LaneSegmentationProcessor::getLastProcessingTime() const {
    return last_processing_time_;
}

void LaneSegmentationProcessor::printStatistics() const {
    ROS_INFO("=== Lane Segmentation Statistics ===");
    ROS_INFO("Processed frames: %d", stats_.processed_frames);
    ROS_INFO("Average processing time: %.2f ms",
            stats_.processed_frames > 0 ? stats_.total_processing_time / stats_.processed_frames : 0.0);
    ROS_INFO("Total detected lanes: %d", stats_.detected_lanes_count);
    ROS_INFO("Average confidence: %.3f", stats_.average_confidence);
    ROS_INFO("===================================");
}

// Private helper methods
double LaneSegmentationProcessor::calculateLineSlope(const cv::Vec4f& line) {
    // Line format: ax + by + c = 0
    if (std::abs(line[1]) > 0.001) {
        return -line[0] / line[1]; // slope = -a/b
    }
    return std::numeric_limits<double>::infinity(); // Vertical line
}

double LaneSegmentationProcessor::calculatePointDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

cv::Point2f LaneSegmentationProcessor::getLineIntersection(const cv::Vec4f& line1, const cv::Vec4f& line2) {
    // Solve system of equations: a1*x + b1*y + c1 = 0, a2*x + b2*y + c2 = 0
    float det = line1[0] * line2[1] - line2[0] * line1[1];

    if (std::abs(det) < 0.001) {
        return cv::Point2f(-1, -1); // Lines are parallel
    }

    float x = (line2[0] * line1[2] - line1[0] * line2[2]) / det;
    float y = (line1[1] * line2[2] - line2[1] * line1[2]) / det;

    return cv::Point2f(x, y);
}

bool LaneSegmentationProcessor::shouldMergeLines(const LaneLine& line1, const LaneLine& line2) {
    // Check if lines are parallel
    if (!areLinesParallel(line1, line2)) {
        return false;
    }

    // Check distance between lines
    double distance = calculateLineDistance(line1, line2);
    return distance < config_.merge_distance;
}

bool LaneSegmentationProcessor::areLinesParallel(const LaneLine& line1, const LaneLine& line2) {
    double slope1 = calculateLineSlope(line1.line_params);
    double slope2 = calculateLineSlope(line2.line_params);

    return std::abs(slope1 - slope2) < config_.parallel_threshold;
}

double LaneSegmentationProcessor::calculateLineDistance(const LaneLine& line1, const LaneLine& line2) {
    // Simplified distance calculation using average point distances
    if (line1.points.empty() || line2.points.empty()) return std::numeric_limits<double>::max();

    cv::Point2f center1 = line1.points[0];
    cv::Point2f center2 = line2.points[0];

    return calculatePointDistance(center1, center2);
}

LaneLine LaneSegmentationProcessor::mergeLines(const LaneLine& line1, const LaneLine& line2) {
    LaneLine merged = line1;

    // Combine points
    merged.points.insert(merged.points.end(), line2.points.begin(), line2.points.end());

    // Average confidence
    merged.confidence = (line1.confidence + line2.confidence) / 2.0f;

    // Recompute line parameters based on all points
    if (merged.points.size() >= 2) {
        merged.line_params = pointsToLine(merged.points);
    }

    return merged;
}

std::string LaneSegmentationProcessor::classifyLaneType(const LaneLine& lane, const cv::Mat& mask) {
    // Simplified lane type classification
    // In a full implementation, this would analyze the mask pattern
    return "solid"; // Default to solid lines
}

cv::Vec4f LaneSegmentationProcessor::pointsToLine(const std::vector<cv::Point2f>& points) {
    if (points.size() < 2) {
        return cv::Vec4f(0, 0, 0, 0);
    }

    // Use least squares fitting for multiple points
    cv::Vec4f line;
    cv::fitLine(points, line, cv::DIST_L2, 0, 0.01, 0.01);

    // Convert from parametric to standard form
    float vx = line[0];
    float vy = line[1];
    float x0 = line[2];
    float y0 = line[3];

    // Standard form: ax + by + c = 0
    float a = -vy;
    float b = vx;
    float c = vy * x0 - vx * y0;

    return cv::Vec4f(a, b, c, 0);
}

void LaneSegmentationProcessor::logError(const std::string& message) {
    ROS_ERROR("[LaneSegmentationProcessor] %s", message.c_str());
}

void LaneSegmentationProcessor::logWarning(const std::string& message) {
    ROS_WARN("[LaneSegmentationProcessor] %s", message.c_str());
}

void LaneSegmentationProcessor::logInfo(const std::string& message) {
    ROS_INFO("[LaneSegmentationProcessor] %s", message.c_str());
}

} // namespace lane_detection