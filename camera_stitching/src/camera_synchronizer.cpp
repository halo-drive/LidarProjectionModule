#include "camera_synchronizer.hpp"
#include <ros/ros.h>

namespace camera_stitching {
    CameraSynchronizer::CameraSynchronizer() {
        ROS_INFO("CameraSynchronizer constructor");
    }
    
    bool CameraSynchronizer::initialize() {
        ROS_INFO("CameraSynchronizer initialize");
        return true;
    }
}
