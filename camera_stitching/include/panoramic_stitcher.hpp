#ifndef PANORAMIC_STITCHER_HPP
#define PANORAMIC_STITCHER_HPP

#include <opencv2/opencv.hpp>

namespace camera_stitching {
    class PanoramicStitcher {
    public:
        PanoramicStitcher();
        bool stitch(const cv::Mat& left, const cv::Mat& right, cv::Mat& output);
    };
}

#endif
