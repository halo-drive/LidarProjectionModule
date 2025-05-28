#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Header.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

class LidarGroundExtractionNode {
public:
    LidarGroundExtractionNode() : nh_("~") {
        
        // Initialize parameters
        initializeParameters();
        
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
    
    // Filter parameters (simplified)
    float voxel_size_;
    int statistical_k_;
    double statistical_std_mul_;
    float min_range_;
    float max_range_;
    float min_height_;
    float max_height_;
    
    // Ground extraction parameters
    double distance_threshold_;
    int max_iterations_;
    double probability_;
    
    // Statistics
    struct ProcessingStats {
        size_t total_processed;
        size_t successful_extractions;
        double avg_processing_time;
        
        ProcessingStats() : 
            total_processed(0), successful_extractions(0), avg_processing_time(0.0) {}
    } stats_;
   

    void initializeParameters() {
        // Frame and topic parameters
        nh_.param<std::string>("base_frame", base_frame_, "base_link");
        nh_.param<std::string>("vlp16_puck_frame", vlp16_puck_frame_, "lidar0_link");
        nh_.param<std::string>("vlp16_highres_frame", vlp16_highres_frame_, "lidar1_link");
        nh_.param<std::string>("vlp16_puck_topic", vlp16_puck_topic_, "/lidar0/velodyne_points");
        nh_.param<std::string>("vlp16_highres_topic", vlp16_highres_topic_, "/lidar1/velodyne_points");
      
        // Publishing options
        nh_.param<bool>("publish_merged_cloud", publish_merged_cloud_, true);
        nh_.param<bool>("publish_ground_visualization", publish_ground_visualization_, true);
        nh_.param<double>("processing_frequency", processing_frequency_, 10.0);
        
        // Basic filter parameters (only the ones we actually use)
        nh_.param<float>("filter/voxel_size", voxel_size_, 0.1f);
        nh_.param<int>("filter/statistical_k", statistical_k_, 50);
        nh_.param<double>("filter/statistical_std_mul", statistical_std_mul_, 1.0);
        nh_.param<float>("filter/min_range", min_range_, 0.9f);
        nh_.param<float>("filter/max_range", max_range_, 100.0f);
        nh_.param<float>("filter/min_height", min_height_, -2.5f);   // STRICT: Only these heights
        nh_.param<float>("filter/max_height", max_height_, 0.5f);    // STRICT: Much lower max
        
        // Ground extraction parameters (IMPROVED - stricter values)
        nh_.param<double>("extraction/distance_threshold", distance_threshold_, 0.02);  // STRICT: 2cm instead of 5cm
        nh_.param<int>("extraction/max_iterations", max_iterations_, 500);              // REDUCED: Faster processing  
        nh_.param<double>("extraction/probability", probability_, 0.99);                // Keep high confidence
        
        // REMOVED: grid_size, height_tolerance, normal_z_threshold variables
        // These are hardcoded in the GroundExtractor class, no need to pass them here
        
        ROS_INFO("=== CLEANED GROUND EXTRACTION PARAMETERS ===");
        ROS_INFO("Topics: %s + %s", vlp16_puck_topic_.c_str(), vlp16_highres_topic_.c_str());
        ROS_INFO("Distance threshold: %.3f m (MUCH STRICTER)", distance_threshold_);
        ROS_INFO("Height range: %.1f to %.1f m (STRICT)", min_height_, max_height_);
        ROS_INFO("Max iterations: %d (REDUCED for speed)", max_iterations_);
        ROS_INFO("============================================");
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
        
        // ROS_INFO("Publishers configured");
        ROS_INFO("Publishers configured: merged_pointcloud, ground_points, non_ground_points");

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
// STEP 1: Convert ROS messages to PCL
pcl::PointCloud<pcl::PointXYZI>::Ptr puck_cloud(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr highres_cloud(new pcl::PointCloud<pcl::PointXYZI>);

pcl::fromROSMsg(*puck_msg, *puck_cloud);
pcl::fromROSMsg(*highres_msg, *highres_cloud);

// STEP 2: Simple concatenation - NO data loss
pcl::PointCloud<pcl::PointXYZI>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZI>);
*merged_cloud = *puck_cloud;
*merged_cloud += *highres_cloud;

ROS_INFO("Merged: %zu + %zu = %zu points", 
      puck_cloud->size(), highres_cloud->size(), merged_cloud->size());

// STEP 3: Apply filtering ONLY for ground extraction processing
auto filtered_cloud = applyFilters(merged_cloud);

// STEP 4: Extract ground plane
pcl::PointCloud<pcl::PointXYZI>::Ptr ground_cloud;
pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground_cloud;

Eigen::Vector4f plane_coeffs = extractGroundPlane(filtered_cloud, ground_cloud, non_ground_cloud);

// STEP 5: Publish all 3 outputs
publishResults(puck_msg->header, merged_cloud, ground_cloud, non_ground_cloud, plane_coeffs);

// Update statistics
stats_.successful_extractions++;

auto end_time = std::chrono::high_resolution_clock::now();
double processing_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

stats_.avg_processing_time = 
 (stats_.avg_processing_time * (stats_.successful_extractions - 1) + processing_time) / 
 stats_.successful_extractions;

ROS_INFO("Published: %zu merged, %zu ground, %zu non-ground points in %.2f ms",
 merged_cloud->size(), ground_cloud->size(), non_ground_cloud->size(), processing_time);
 
} catch (const std::exception& e) {
ROS_ERROR("Processing failed: %s", e.what());
}
}
    

    pcl::PointCloud<pcl::PointXYZI>::Ptr applyFilters(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input) {
        // Range filtering
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>);
        
        for (const auto& point : input->points) {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
                continue;
            }
            
            float range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            
            if (range >= min_range_ && range <= max_range_ &&
                point.z >= min_height_ && point.z <= max_height_) {
                filtered->push_back(point);
            }
        }
        
        // Voxel grid filtering
        pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
        voxel_filter.setInputCloud(filtered);
        voxel_filter.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr voxel_filtered(new pcl::PointCloud<pcl::PointXYZI>);
        voxel_filter.filter(*voxel_filtered);
        
        // Statistical outlier removal
        pcl::StatisticalOutlierRemoval<pcl::PointXYZI> statistical_filter;
        statistical_filter.setInputCloud(voxel_filtered);
        statistical_filter.setMeanK(statistical_k_);
        statistical_filter.setStddevMulThresh(statistical_std_mul_);
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr final_filtered(new pcl::PointCloud<pcl::PointXYZI>);
        statistical_filter.filter(*final_filtered);
        
        return final_filtered;
    }


    Eigen::Vector4f extractGroundPlane(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& input,
        pcl::PointCloud<pcl::PointXYZI>::Ptr& ground_cloud,
        pcl::PointCloud<pcl::PointXYZI>::Ptr& non_ground_cloud) {
        
        ground_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
        non_ground_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
        
        if (input->empty()) {
            ROS_WARN("Input cloud is empty");
            return Eigen::Vector4f::Zero();
        }
        
        ROS_INFO("Processing %zu points for ground extraction", input->size());
        
        // STEP 1: Apply strict height pre-filtering
        pcl::PointCloud<pcl::PointXYZI>::Ptr height_filtered(new pcl::PointCloud<pcl::PointXYZI>);
        
        for (const auto& point : input->points) {
            // CRITICAL: For BOTH LiDARs at 1m height, ground points should be 
            // around -1.0m relative to each sensor
            // We expect ground points to be BELOW the sensor (negative Z)
            if (point.z >= min_height_ && point.z <= max_height_) {
                height_filtered->push_back(point);
            }
        }
        
        ROS_INFO("Height filtering: %zu -> %zu points (kept points with Z between %.2f and %.2f)", 
                 input->size(), height_filtered->size(), min_height_, max_height_);
        
        if (height_filtered->size() < 100) {
            ROS_ERROR("Too few points after height filtering. Check your height parameters!");
            ROS_ERROR("Expected: Ground points should have Z values around -0.5m to -0.75m");
            return Eigen::Vector4f::Zero();
        }
        
        // STEP 2: RANSAC plane segmentation on height-filtered points
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        
        pcl::SACSegmentation<pcl::PointXYZI> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(max_iterations_);
        seg.setDistanceThreshold(distance_threshold_);
        seg.setProbability(probability_);
        
        seg.setInputCloud(height_filtered);
        seg.segment(*inliers, *coefficients);
        
        if (inliers->indices.empty()) {
            ROS_ERROR("RANSAC failed to find ground plane in height-filtered points");
            return Eigen::Vector4f::Zero();
        }
        
        Eigen::Vector4f plane_coeffs(coefficients->values[0], coefficients->values[1], 
                                    coefficients->values[2], coefficients->values[3]);
        
        // STEP 3: Validate plane normal (must point upward)
        Eigen::Vector3f normal(plane_coeffs[0], plane_coeffs[1], plane_coeffs[2]);
        normal.normalize();
        
        // CRITICAL: Normal should point upward (positive Z component)
        if (normal.z() < 0.7) {  // At least 70% upward
            ROS_WARN("Detected plane normal not pointing upward enough: Z=%.3f", normal.z());
            ROS_WARN("This might be a wall or ceiling, not ground");
            return Eigen::Vector4f::Zero();
        }
        
        // STEP 4: Calculate expected ground height and validate
        float expected_ground_z = -plane_coeffs[3] / plane_coeffs[2];  // Z when X=Y=0
        ROS_INFO("Detected ground plane at Z=%.3f (sensor coordinates)", expected_ground_z);
        
        // For BOTH LiDARs at 1m height, ground should be around -1.0m
        if (expected_ground_z > -0.7 || expected_ground_z < -1.3) {
            ROS_WARN("Ground plane at unexpected height: %.3f (expected: -0.7 to -1.3)", expected_ground_z);
        }
        
        // STEP 5: Extract final ground and non-ground points from original cloud
        for (size_t i = 0; i < input->size(); ++i) {
            const auto& point = input->points[i];
            
            // Calculate distance to detected plane
            float distance = std::abs(plane_coeffs[0] * point.x + plane_coeffs[1] * point.y + 
                                    plane_coeffs[2] * point.z + plane_coeffs[3]) /
                            std::sqrt(plane_coeffs[0] * plane_coeffs[0] + 
                                    plane_coeffs[1] * plane_coeffs[1] + 
                                    plane_coeffs[2] * plane_coeffs[2]);
            
            // Classification: ground if close to plane AND in reasonable height range
            bool is_ground = (distance < distance_threshold_) && 
                            (point.z >= min_height_) && 
                            (point.z <= max_height_);
            
            if (is_ground) {
                ground_cloud->push_back(point);
            } else {
                non_ground_cloud->push_back(point);
            }
        }
        
        ROS_INFO("Final ground extraction: %zu ground, %zu non-ground points",
            ground_cloud->size(), non_ground_cloud->size());
        
        // STEP 6: Additional validation
        if (ground_cloud->size() < 500) {
            ROS_WARN("Very few ground points detected (%zu). Check parameters!", ground_cloud->size());
        }
        
        double ground_percentage = 100.0 * ground_cloud->size() / input->size();
        ROS_INFO("Ground coverage: %.1f%% of total points", ground_percentage);
        
        if (ground_percentage < 5.0 || ground_percentage > 80.0) {
            ROS_WARN("Unusual ground percentage: %.1f%% (expected: 10-60%%)", ground_percentage);
        }
        
        return plane_coeffs;
    }

    void publishResults(const std_msgs::Header& header,
                       const pcl::PointCloud<pcl::PointXYZI>::Ptr& merged_cloud,
                       const pcl::PointCloud<pcl::PointXYZI>::Ptr& ground_cloud,
                       const pcl::PointCloud<pcl::PointXYZI>::Ptr& non_ground_cloud,
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
        if (plane_coeffs.isZero()) return;
        
        visualization_msgs::Marker marker;
        marker.header = header;
        marker.header.frame_id = base_frame_;
        marker.ns = "ground_plane";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        
        // Calculate ground plane position
        float ground_height = -plane_coeffs[3] / plane_coeffs[2];
        marker.pose.position.x = 0.0;
        marker.pose.position.y = 0.0;
        marker.pose.position.z = ground_height;
        
        // Default orientation
        marker.pose.orientation.w = 1.0;
        
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
            "  Average processing time: %.2f ms",
            stats_.total_processed,
            stats_.successful_extractions,
            success_rate,
            stats_.avg_processing_time);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_ground_extraction_node");
    
    try {
        LidarGroundExtractionNode node;
        ROS_INFO("Starting LiDAR Ground Extraction Node...");
        ros::spin();
    } catch (const std::exception& e) {
        ROS_ERROR("Node crashed: %s", e.what());
        return 1;
    }
    
    return 0;
}

