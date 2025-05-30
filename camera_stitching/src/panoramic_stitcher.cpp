#include "panoramic_stitcher.hpp"
#include <ros/ros.h>

namespace camera_stitching {
    PanoramicStitcher::PanoramicStitcher() {
        ROS_INFO("PanoramicStitcher constructor");
    }
    
    bool PanoramicStitcher::stitch(const cv::Mat& left, const cv::Mat& right, cv::Mat& output) {
        ROS_INFO("PanoramicStitcher stitch - placeholder implementation");
        output = left.clone();  // Placeholder: just return left image
        return true;
    }
}
