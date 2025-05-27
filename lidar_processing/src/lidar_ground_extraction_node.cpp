#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Header.h>

#include "lidar_processing/point_cloud_proc.hpp"
#include "lidar_processing/ground_extraction.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

namespace lidar_processing {

class LidarGroundExtractionNode {
public:
    LidarGroundExtractionNode() : nh_("~"), tf_listener_(tf_buffer_) {
        
        // Initialize parameters
        initializeParameters();
        
        // Initialize processing components
        processor_.reset(new PointCloudProcessor(filter_params_));
        extractor_.reset(new GroundExtractor(extraction_params_));
        
        // Setup subscribers for both LiDAR sensors
        setupSubscribers();
        
        // Setup publishers
        setupPublishers();
        
        // Setup timers
        setupTimers();
        
        ROS_INFO("LiDAR Ground Extraction Node initialized");
    }

private:
    // ROS components
    ros::NodeHandle nh_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    
    // Processing components
    boost::shared_ptr<PointCloudProcessor> processor_;
    boost::shared_ptr<GroundExtractor> extractor_;
    
    // Parameters
    PointCloudProcessor::FilterParams filter_params_;
    GroundExtractor::GroundExtractionParams extraction_params_;
    
    // Synchronized subscribers for two LiDAR sensors
    boost::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> vlp16_puck_sub_;
    boost::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> vlp16_highres_sub_;
    
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::PointCloud2, 
        sensor_msgs::PointCloud2> SyncPolicy;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    
    // Publishers
    ros::Publisher merged_cloud_pub_;
    ros::Publisher ground_cloud_pub_;
    ros::Publisher non_ground_cloud_pub_;
    ros::Publisher ground_plane_pub_;
    ros::Publisher ground_normal_pub_;
    
    // Timers
    ros::Timer stats_timer_;
    
    // Configuration parameters
    std::string base_frame_;
    std::string vlp16_puck_frame_;
    std::string vlp16_highres_frame_;
    std::string vlp16_puck_topic_;
    std::string vlp16_highres_topic_;
    
    bool publish_merged_cloud_;
    bool publish_ground_visualization_;
    double processing_frequency_;
    
    // Statistics
    struct ProcessingStats {
        size_t total_processed;
        size_t successful_extractions;
        double avg_processing_time;
        double avg_ground_confidence;
        ros::Time last_processing_time;
        
        ProcessingStats() : 
            total_processed(0), successful_extractions(0), avg_processing_time(0.0),
            avg_ground_confidence(0.0) {}
    } stats_;
    
    void initializeParameters() {
        // Get parameters with defaults
        // Get parameters with defaults that match colleague's relay topics
        nh_.param<std::string>("base_frame", base_frame_, "base_link");
        nh_.param<std::string>("vlp16_puck_frame", vlp16_puck_frame_, "velodyne");        // colleague's frame
        nh_.param<std::string>("vlp16_highres_frame", vlp16_highres_frame_, "velodyne2"); // colleague's frame
        nh_.param<std::string>("vlp16_puck_topic", vlp16_puck_topic_, "/lidar0/points");     // colleague's relay topic
        nh_.param<std::string>("vlp16_highres_topic", vlp16_highres_topic_, "/lidar1/points"); // colleague's relay topic
      
        nh_.param<bool>("publish_merged_cloud", publish_merged_cloud_, true);
        nh_.param<bool>("publish_ground_visualization", publish_ground_visualization_, true);
        nh_.param<double>("processing_frequency", processing_frequency_, 10.0);
        
        // Filter parameters
        nh_.param<double>("filter/voxel_size", filter_params_.voxel_size, 0.1);
        nh_.param<int>("filter/statistical_k", filter_params_.statistical_k, 50);
        nh_.param<double>("filter/statistical_std_mul", filter_params_.statistical_std_mul, 1.0);
        nh_.param<double>("filter/min_range", filter_params_.min_range, 0.9);
        nh_.param<double>("filter/max_range", filter_params_.max_range, 100.0);
        nh_.param<double>("filter/min_height", filter_params_.min_height, -3.0);
        nh_.param<double>("filter/max_height", filter_params_.max_height, 5.0);
        
        // Ground extraction parameters
        nh_.param<double>("extraction/distance_threshold", extraction_params_.distance_threshold, 0.05);
        nh_.param<int>("extraction/max_iterations", extraction_params_.max_iterations, 1000);
        nh_.param<double>("extraction/probability", extraction_params_.probability, 0.99);
        nh_.param<double>("extraction/max_slope", extraction_params_.max_slope, 0.3);
        nh_.param<bool>("extraction/use_cuda", extraction_params_.use_cuda, true);
        
        ROS_INFO("Parameters initialized");
        ROS_INFO("Base frame: %s", base_frame_.c_str());
        ROS_INFO("VLP-16 Puck topic: %s", vlp16_puck_topic_.c_str());
        ROS_INFO("VLP-16 High-res topic: %s", vlp16_highres_topic_.c_str());
    }
    
    void setupSubscribers() {
        // Create synchronized subscribers for both LiDAR sensors
        vlp16_puck_sub_.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(
            nh_, vlp16_puck_topic_, 10));
        vlp16_highres_sub_.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(
            nh_, vlp16_highres_topic_, 10));
        
        // Create synchronizer
        sync_.reset(new message_filters::Synchronizer<SyncPolicy>(
            SyncPolicy(10), *vlp16_puck_sub_, *vlp16_highres_sub_));
        
        sync_->registerCallback(
            boost::bind(&LidarGroundExtractionNode::syncedCallback, this, _1, _2));
        
        ROS_INFO("Subscribers configured for synchronized processing");
    }
    
    void setupPublishers() {
        // Point cloud publishers
        merged_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("merged_pointcloud", 10);
        ground_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("ground_points", 10);
        non_ground_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("non_ground_points", 10);
        
        // Visualization publishers
        ground_plane_pub_ = nh_.advertise<visualization_msgs::Marker>("ground_plane_marker", 10);
        ground_normal_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("ground_normal", 10);
        
        ROS_INFO("Publishers configured");
    }
    
    void setupTimers() {
        // Statistics reporting timer
        stats_timer_ = nh_.createTimer(ros::Duration(5.0),
            &LidarGroundExtractionNode::publishStats, this);
    }
    
    void syncedCallback(const sensor_msgs::PointCloud2::ConstPtr& puck_msg,
                       const sensor_msgs::PointCloud2::ConstPtr& highres_msg) {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        stats_.total_processed++;
        
        try {
            // Convert ROS messages to PCL
            auto puck_cloud = processor_->convertFromROS(puck_msg, SENSOR_VLP16_PUCK);
            auto highres_cloud = processor_->convertFromROS(highres_msg, SENSOR_VLP16_HIGH_RES);
            
            // Get transformation between sensors
            Eigen::Matrix4f transform = getTransformation(vlp16_highres_frame_, vlp16_puck_frame_, puck_msg->header.stamp);
            
            // Merge point clouds
            auto merged_cloud = processor_->mergePointClouds(
                convertToXYZI(puck_cloud), convertToXYZI(highres_cloud), transform);
            
            // Apply filtering
            auto filtered_cloud = processor_->applyFilters(merged_cloud);
            
            // Extract ground plane
            PointCloudGround::Ptr ground_cloud;
            PointCloudXYZIR::Ptr non_ground_cloud;
            
            Eigen::Vector4f plane_coeffs = extractor_->extractGroundPlane(
                filtered_cloud, ground_cloud, non_ground_cloud);
            
            // Publish results
            publishResults(puck_msg->header, filtered_cloud, ground_cloud, non_ground_cloud, plane_coeffs);
            
            // Update statistics
            stats_.successful_extractions++;
            stats_.avg_ground_confidence = 
                (stats_.avg_ground_confidence * (stats_.successful_extractions - 1) + 
                 extractor_->getLastStats().validation_confidence) / stats_.successful_extractions;
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double processing_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            stats_.avg_processing_time = 
                (stats_.avg_processing_time * (stats_.successful_extractions - 1) + processing_time) / 
                stats_.successful_extractions;
            
            stats_.last_processing_time = ros::Time::now();
            
            ROS_DEBUG("Processed frame: %zu ground, %zu non-ground points in %.2f ms",
                ground_cloud->size(), non_ground_cloud->size(), processing_time);
                
        } catch (const std::exception& e) {
            ROS_ERROR("Processing failed: %s", e.what());
        }
    }
    
    Eigen::Matrix4f getTransformation(const std::string& source_frame, 
                                     const std::string& target_frame,
                                     const ros::Time& time) {
        try {
            geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform(
                target_frame, source_frame, time, ros::Duration(0.1));
            
            Eigen::Isometry3d eigen_transform;
            eigen_transform = tf2::transformToEigen(transform);
            
            return eigen_transform.matrix().cast<float>();
            
        } catch (const tf2::TransformException& ex) {
            ROS_WARN("Transform lookup failed: %s", ex.what());
            return Eigen::Matrix4f::Identity();
        }
    }
    
    PointCloudXYZI::Ptr convertToXYZI(const PointCloudXYZIR::ConstPtr& input) {
        auto output = boost::make_shared<PointCloudXYZI>();
        output->reserve(input->size());
        
        for (const auto& point : input->points) {
            pcl::PointXYZI xyz_point;
            xyz_point.x = point.x;
            xyz_point.y = point.y;
            xyz_point.z = point.z;
            xyz_point.intensity = point.intensity;
            output->push_back(xyz_point);
        }
        
        output->header = input->header;
        output->is_dense = input->is_dense;
        return output;
    }
    
    void publishResults(const std_msgs::Header& header,
                       const PointCloudXYZIR::ConstPtr& merged_cloud,
                       const PointCloudGround::ConstPtr& ground_cloud,
                       const PointCloudXYZIR::ConstPtr& non_ground_cloud,
                       const Eigen::Vector4f& plane_coeffs) {
        
        // Publish merged cloud
        if (publish_merged_cloud_ && merged_cloud_pub_.getNumSubscribers() > 0) {
            sensor_msgs::PointCloud2 merged_msg;
            pcl::toROSMsg(*merged_cloud, merged_msg);
            merged_msg.header = header;
            merged_msg.header.frame_id = base_frame_;
            merged_cloud_pub_.publish(merged_msg);
        }
        
        // Publish ground cloud
        if (ground_cloud_pub_.getNumSubscribers() > 0) {
            sensor_msgs::PointCloud2 ground_msg;
            pcl::toROSMsg(*ground_cloud, ground_msg);
            ground_msg.header = header;
            ground_msg.header.frame_id = base_frame_;
            ground_cloud_pub_.publish(ground_msg);
        }
        
        // Publish non-ground cloud
        if (non_ground_cloud_pub_.getNumSubscribers() > 0) {
            sensor_msgs::PointCloud2 non_ground_msg;
            pcl::toROSMsg(*non_ground_cloud, non_ground_msg);
            non_ground_msg.header = header;
            non_ground_msg.header.frame_id = base_frame_;
            non_ground_cloud_pub_.publish(non_ground_msg);
        }
        
        // Publish ground plane visualization
        if (publish_ground_visualization_ && ground_plane_pub_.getNumSubscribers() > 0) {
            publishGroundPlaneMarker(header, plane_coeffs);
        }
    }
    
    void publishGroundPlaneMarker(const std_msgs::Header& header, 
                                 const Eigen::Vector4f& plane_coeffs) {
        visualization_msgs::Marker marker;
        marker.header = header;
        marker.header.frame_id = base_frame_;
        marker.ns = "ground_plane";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        
        // Calculate ground plane position and orientation
        Eigen::Vector3f normal(plane_coeffs[0], plane_coeffs[1], plane_coeffs[2]);
        normal.normalize();
        
        // Position at origin projected onto plane
        float ground_height = -plane_coeffs[3] / plane_coeffs[2];
        marker.pose.position.x = 0.0;
        marker.pose.position.y = 0.0;
        marker.pose.position.z = ground_height;
        
        // Orientation from normal vector
        Eigen::Vector3f z_axis(0.0f, 0.0f, 1.0f);
        Eigen::Vector3f rotation_axis = z_axis.cross(normal);
        float rotation_angle = std::acos(z_axis.dot(normal));
        
        if (rotation_axis.norm() > 1e-6) {
            rotation_axis.normalize();
            
            Eigen::AngleAxisf rotation(rotation_angle, rotation_axis);
            Eigen::Quaternionf quat(rotation);
            
            marker.pose.orientation.x = quat.x();
            marker.pose.orientation.y = quat.y();
            marker.pose.orientation.z = quat.z();
            marker.pose.orientation.w = quat.w();
        } else {
            marker.pose.orientation.w = 1.0;
        }
        
        // Scale (large flat plane)
        marker.scale.x = 20.0;
        marker.scale.y = 20.0;
        marker.scale.z = 0.01;
        
        // Color (semi-transparent green)
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 0.3;
        
        marker.lifetime = ros::Duration(1.0);
        
        ground_plane_pub_.publish(marker);
    }
    
    void publishStats(const ros::TimerEvent&) {
        if (stats_.total_processed == 0) return;
        
        double success_rate = 100.0 * stats_.successful_extractions / stats_.total_processed;
        
        ROS_INFO("Processing Statistics:\n"
            "  Total frames: %zu\n"
            "  Successful extractions: %zu (%.1f%%)\n"
            "  Average processing time: %.2f ms\n"
            "  Average ground confidence: %.3f",
            stats_.total_processed,
            stats_.successful_extractions,
            success_rate,
            stats_.avg_processing_time,
            stats_.avg_ground_confidence);
    }
};

} // namespace lidar_processing

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_ground_extraction_node");
    
    try {
        lidar_processing::LidarGroundExtractionNode node;
        ROS_INFO("Starting LiDAR Ground Extraction Node...");
        ros::spin();
    } catch (const std::exception& e) {
        ROS_ERROR("Node crashed: %s", e.what());
        return 1;
    }
    
    return 0;
}