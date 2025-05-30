#include <ros/ros.h>
#include "camera_synchronizer.hpp"
#include "panoramic_stitcher.hpp"

int main(int argc, char** argv) {
    ros::init(argc, argv, "camera_stitching_node");
    ros::NodeHandle nh;
    
    ROS_INFO("Camera Stitching Node starting - Phase 3/4 build test");
    
    camera_stitching::CameraSynchronizer synchronizer;
    camera_stitching::PanoramicStitcher stitcher;
    
    if (!synchronizer.initialize()) {
        ROS_ERROR("Failed to initialize camera synchronizer");
        return -1;
    }
    
    ROS_INFO("Camera Stitching Node initialized successfully");
    ros::spin();
    
    return 0;
}
