#!/usr/bin/env python3
"""
Multi-Camera Calibration Data Collection Script - Phase 2
LidarProjectionLane: Camera Intrinsic and Extrinsic Calibration

This script collects synchronized camera-LiDAR data for:
- Camera intrinsic calibration (lens distortion, focal length)
- Camera-LiDAR extrinsic calibration (spatial relationships)
- Multi-camera system calibration

Hardware Setup:
- Camera 0: /dev/video0 → /camera0/image_raw
- Camera 1: /dev/video1 → /camera1/image_raw
- LiDAR 0:  /velodyne_points → /lidar0/points
- LiDAR 1:  /velodyne2/velodyne_points → /lidar1/points

Usage:
    python3 calibration_collection.py [camera_id]

    camera_id: 0 or 1 (default: 0)

Example:
    python3 calibration_collection.py 0  # Calibrate camera 0
    python3 calibration_collection.py 1  # Calibrate camera 1

Author: Lane Fusion Development Team
Phase: 2 - Sensor Drivers & Data Acquisition
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import yaml
import os
import sys
import argparse
from datetime import datetime
import threading
import time

class MultiCameraCalibrationCollector:
    """
    Advanced calibration data collection system for dual-camera setup
    Supports both intrinsic and extrinsic calibration workflows
    """

    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        rospy.init_node(f'calibration_collector_camera{camera_id}', anonymous=True)

        self.bridge = CvBridge()
        self.collected_count = 0
        self.target_count = 25  # Optimized for efficient calibration

        # Create timestamped output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f"/workspace/LidarProjectionLane/calibration_data_camera{camera_id}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        # Checkerboard configuration (standard calibration target)
        self.checkerboard = (9, 6)  # 9x6 internal corners
        self.square_size = 0.025    # 2.5cm squares (adjust based on your target)

        # Prepare 3D object points for checkerboard
        self.objp = np.zeros((self.checkerboard[0] * self.checkerboard[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.checkerboard[0], 0:self.checkerboard[1]].T.reshape(-1, 2)
        self.objp *= self.square_size

        # Calibration data storage
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        self.images = []     # Captured images
        self.pointclouds = [] # Corresponding point clouds (for extrinsic calibration)

        # Threading and synchronization
        self.lock = threading.Lock()
        self.latest_image = None
        self.latest_pointcloud = None

        # UI state
        self.display_image = None
        self.checkerboard_detected = False

        # Setup subscribers
        self.setup_subscribers()

        # Calibration quality tracking
        self.quality_metrics = {
            'reprojection_errors': [],
            'corner_distances': [],
            'coverage_score': 0.0
        }

        rospy.loginfo(f"=== Camera {camera_id} Calibration Collector Initialized ===")
        rospy.loginfo(f"Target samples: {self.target_count}")
        rospy.loginfo(f"Output directory: {self.output_dir}")
        rospy.loginfo(f"Checkerboard: {self.checkerboard[0]}x{self.checkerboard[1]} (internal corners)")
        rospy.loginfo(f"Square size: {self.square_size*1000:.1f}mm")
        rospy.loginfo("")
        rospy.loginfo("Instructions:")
        rospy.loginfo("  1. Hold checkerboard in camera view")
        rospy.loginfo("  2. Wait for green detection confirmation")
        rospy.loginfo("  3. Press SPACE to capture sample")
        rospy.loginfo("  4. Move to different positions/angles")
        rospy.loginfo("  5. Press ESC to finish and compute calibration")

    def setup_subscribers(self):
        """Setup camera and LiDAR subscribers with synchronization"""
        # Camera topic based on camera ID
        camera_topic = f'/camera{self.camera_id}/image_raw'
        rospy.loginfo(f"Subscribing to camera: {camera_topic}")

        # Setup message filters for potential synchronization
        self.camera_sub = message_filters.Subscriber(camera_topic, Image)
        self.lidar_sub = message_filters.Subscriber('/lidar0/points', PointCloud2)

        # Use approximate time synchronization for extrinsic calibration data
        # (exact sync not required for intrinsic calibration)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.camera_sub, self.lidar_sub],
            queue_size=10,
            slop=0.1  # 100ms tolerance
        )
        self.ts.registerCallback(self.synchronized_callback)

        # Also subscribe to camera independently for real-time display
        self.image_sub = rospy.Subscriber(camera_topic, Image, self.image_callback, queue_size=1)

    def image_callback(self, image_msg):
        """Process camera images for real-time display"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            with self.lock:
                self.latest_image = cv_image.copy()

            # Process for checkerboard detection
            self.process_image_for_display(cv_image)

        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")

    def synchronized_callback(self, image_msg, pointcloud_msg):
        """Process synchronized camera and LiDAR data"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            with self.lock:
                self.latest_image = cv_image.copy()
                self.latest_pointcloud = pointcloud_msg

        except CvBridgeError as e:
            rospy.logerr(f"Synchronized callback error: {e}")

    def process_image_for_display(self, cv_image):
        """Process image for real-time checkerboard detection and display"""
        if cv_image is None:
            return

        # Convert to grayscale for checkerboard detection
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard, None)

        # Create display image
        display_image = cv_image.copy()

        if ret:
            # Refine corner positions for better accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw refined corners
            cv2.drawChessboardCorners(display_image, self.checkerboard, corners_refined, ret)

            # Calculate corner quality metrics
            corner_quality = self.assess_corner_quality(corners_refined, gray.shape)

            # Status indicators
            self.checkerboard_detected = True
            status_color = (0, 255, 0)  # Green
            status_text = f"DETECTED - Quality: {corner_quality:.2f}"

        else:
            self.checkerboard_detected = False
            status_color = (0, 0, 255)  # Red
            status_text = "NO CHECKERBOARD"

        # Add UI overlay
        self.add_ui_overlay(display_image, status_text, status_color)

        # Update display image
        with self.lock:
            self.display_image = display_image

    def assess_corner_quality(self, corners, image_shape):
        """Assess quality of detected corners for calibration"""
        if corners is None or len(corners) == 0:
            return 0.0

        # Calculate coverage score (how well corners cover the image)
        h, w = image_shape
        corners_2d = corners.reshape(-1, 2)

        # Normalize corner positions to [0, 1]
        norm_corners = corners_2d / np.array([w, h])

        # Calculate coverage as std deviation of positions
        coverage_x = np.std(norm_corners[:, 0])
        coverage_y = np.std(norm_corners[:, 1])
        coverage_score = (coverage_x + coverage_y) / 2.0

        # Calculate corner sharpness (measure of detection confidence)
        min_distance = float('inf')
        for i in range(len(corners_2d)):
            for j in range(i + 1, len(corners_2d)):
                dist = np.linalg.norm(corners_2d[i] - corners_2d[j])
                min_distance = min(min_distance, dist)

        # Combine metrics (coverage is more important)
        quality_score = coverage_score * 0.8 + min(min_distance / 100.0, 1.0) * 0.2

        return min(quality_score, 1.0)

    def add_ui_overlay(self, image, status_text, status_color):
        """Add user interface overlay to image"""
        h, w = image.shape[:2]

        # Semi-transparent overlay for text background
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        # Title and camera info
        cv2.putText(image, f"Camera {self.camera_id} Calibration",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Sample progress
        cv2.putText(image, f"Samples: {self.collected_count}/{self.target_count}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Detection status
        cv2.putText(image, status_text,
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Instructions
        instructions = "SPACE: Capture | ESC: Finish | Q: Quit"
        cv2.putText(image, instructions,
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Quality indicators
        if self.checkerboard_detected:
            cv2.circle(image, (w - 30, 30), 10, (0, 255, 0), -1)  # Green circle
        else:
            cv2.circle(image, (w - 30, 30), 10, (0, 0, 255), -1)  # Red circle

    def capture_sample(self):
        """Capture a calibration sample"""
        with self.lock:
            if self.latest_image is None:
                rospy.logwarn("No image available for capture")
                return False

            current_image = self.latest_image.copy()
            current_pointcloud = self.latest_pointcloud

        # Detect checkerboard in current image
        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard, None)

        if not ret:
            rospy.logwarn("No checkerboard detected in current image")
            return False

        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Assess quality
        quality = self.assess_corner_quality(corners_refined, gray.shape)
        if quality < 0.3:  # Minimum quality threshold
            rospy.logwarn(f"Corner quality too low: {quality:.2f} < 0.3")
            return False

        # Store calibration data
        self.objpoints.append(self.objp)
        self.imgpoints.append(corners_refined)
        self.images.append(current_image)

        if current_pointcloud is not None:
            self.pointclouds.append(current_pointcloud)

        # Save sample to disk
        sample_filename = f"{self.output_dir}/sample_{self.collected_count:03d}.png"
        cv2.imwrite(sample_filename, current_image)

        # Save corner data
        corner_filename = f"{self.output_dir}/corners_{self.collected_count:03d}.yaml"
        corner_data = {
            'corners': corners_refined.tolist(),
            'quality': float(quality),
            'timestamp': rospy.get_time()
        }
        with open(corner_filename, 'w') as f:
            yaml.dump(corner_data, f)

        self.collected_count += 1
        self.quality_metrics['corner_distances'].append(quality)

        rospy.loginfo(f"✅ Sample {self.collected_count}/{self.target_count} captured (quality: {quality:.2f})")

        return True

    def run_calibration(self):
        """Main calibration collection loop"""
        rospy.loginfo("Starting calibration collection...")
        rospy.loginfo("Waiting for camera data...")

        # Wait for first image
        while not rospy.is_shutdown() and self.latest_image is None:
            time.sleep(0.1)

        rospy.loginfo("Camera data received. Starting collection interface.")

        try:
            while not rospy.is_shutdown() and self.collected_count < self.target_count:
                # Display current image
                with self.lock:
                    if self.display_image is not None:
                        cv2.imshow(f'Camera {self.camera_id} Calibration', self.display_image)

                # Handle keyboard input
                key = cv2.waitKey(30) & 0xFF

                if key == ord(' '):  # Space to capture
                    if self.checkerboard_detected:
                        self.capture_sample()
                    else:
                        rospy.logwarn("No checkerboard detected - cannot capture")

                elif key == 27 or key == ord('q'):  # ESC or Q to exit
                    rospy.loginfo("Manual exit requested")
                    break

                time.sleep(0.03)  # ~30 FPS display rate

        except KeyboardInterrupt:
            rospy.loginfo("Keyboard interrupt received")
        finally:
            cv2.destroyAllWindows()

        # Compute calibration if we have enough samples
        if self.collected_count >= 10:
            self.compute_calibration()
        else:
            rospy.logwarn(f"Insufficient samples collected: {self.collected_count} < 10")

    def compute_calibration(self):
        """Compute camera calibration from collected data"""
        rospy.loginfo("Computing camera calibration...")

        if len(self.images) == 0:
            rospy.logerr("No images available for calibration")
            return

        # Get image dimensions
        h, w = self.images[0].shape[:2]

        # Perform camera calibration
        try:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self.objpoints, self.imgpoints, (w, h), None, None,
                flags=cv2.CALIB_RATIONAL_MODEL  # Use rational model for wide FOV
            )

            if not ret:
                rospy.logerr("Camera calibration failed")
                return

        except Exception as e:
            rospy.logerr(f"Calibration computation error: {e}")
            return

        # Calculate calibration quality metrics
        total_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error

        mean_error = total_error / len(self.objpoints)

        rospy.loginfo(f"✅ Calibration completed successfully!")
        rospy.loginfo(f"   Samples used: {len(self.objpoints)}")
        rospy.loginfo(f"   Mean reprojection error: {mean_error:.3f} pixels")

        # Assess calibration quality
        if mean_error < 0.5:
            quality_assessment = "EXCELLENT"
        elif mean_error < 1.0:
            quality_assessment = "GOOD"
        elif mean_error < 2.0:
            quality_assessment = "ACCEPTABLE"
        else:
            quality_assessment = "POOR - Consider recalibrating"

        rospy.loginfo(f"   Quality assessment: {quality_assessment}")

        # Save calibration results in ROS format
        self.save_calibration_results(camera_matrix, dist_coeffs, (w, h), mean_error)

        # Save detailed calibration data
        self.save_detailed_results(camera_matrix, dist_coeffs, rvecs, tvecs, mean_error)

    def save_calibration_results(self, camera_matrix, dist_coeffs, image_size, reprojection_error):
        """Save calibration results in ROS camera_info format"""
        w, h = image_size

        # Create ROS camera_info compatible format
        calibration_data = {
            'camera_info': {
                'width': w,
                'height': h,
                'distortion_model': 'plumb_bob'
            },
            'camera_matrix': {
                'rows': 3, 'cols': 3,
                'data': [
                    float(camera_matrix[0, 0]), 0.0, float(camera_matrix[0, 2]),
                    0.0, float(camera_matrix[1, 1]), float(camera_matrix[1, 2]),
                    0.0, 0.0, 1.0
                ]
            },
            'distortion_coefficients': {
                'rows': 1, 'cols': 5,
                'data': [float(x) for x in dist_coeffs[0]]
            },
            'rectification_matrix': {
                'rows': 3, 'cols': 3,
                'data': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            },
            'projection_matrix': {
                'rows': 3, 'cols': 4,
                'data': [
                    float(camera_matrix[0, 0]), 0.0, float(camera_matrix[0, 2]), 0.0,
                    0.0, float(camera_matrix[1, 1]), float(camera_matrix[1, 2]), 0.0,
                    0.0, 0.0, 1.0, 0.0
                ]
            },
            'frame_id': f'camera{self.camera_id}_link',
            'calibration_status': 'CALIBRATED',
            'calibration_date': datetime.now().isoformat(),
            'calibration_method': 'opencv_checkerboard',
            'reprojection_error': float(reprojection_error),
            'samples_used': len(self.objpoints)
        }

        # Save to project config directory
        config_filename = f"/workspace/LidarProjectionLane/config/camera_params/camera{self.camera_id}.yaml"
        with open(config_filename, 'w') as f:
            yaml.dump(calibration_data, f, default_flow_style=False)

        # Also save to calibration data directory
        backup_filename = f"{self.output_dir}/camera{self.camera_id}_calibration.yaml"
        with open(backup_filename, 'w') as f:
            yaml.dump(calibration_data, f, default_flow_style=False)

        rospy.loginfo(f"📁 Calibration saved to: {config_filename}")
        rospy.loginfo(f"📁 Backup saved to: {backup_filename}")

    def save_detailed_results(self, camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error):
        """Save detailed calibration data for analysis"""
        detailed_data = {
            'calibration_details': {
                'camera_matrix': camera_matrix.tolist(),
                'distortion_coefficients': dist_coeffs.tolist(),
                'rotation_vectors': [r.tolist() for r in rvecs],
                'translation_vectors': [t.tolist() for t in tvecs],
                'reprojection_error': float(reprojection_error),
                'samples_count': len(self.objpoints),
                'checkerboard_size': self.checkerboard,
                'square_size': self.square_size,
                'timestamp': datetime.now().isoformat()
            }
        }

        detailed_filename = f"{self.output_dir}/detailed_calibration.yaml"
        with open(detailed_filename, 'w') as f:
            yaml.dump(detailed_data, f, default_flow_style=False)

        rospy.loginfo(f"📊 Detailed results saved to: {detailed_filename}")

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='Camera Calibration Data Collection')
    parser.add_argument('camera_id', type=int, nargs='?', default=0,
                        help='Camera ID to calibrate (0 or 1)')
    parser.add_argument('--samples', type=int, default=25,
                        help='Number of calibration samples to collect')
    parser.add_argument('--checkerboard', type=str, default='9x6',
                        help='Checkerboard size as WIDTHxHEIGHT (e.g., 9x6)')
    parser.add_argument('--square-size', type=float, default=0.025,
                        help='Checkerboard square size in meters')

    args = parser.parse_args()

    # Validate camera ID
    if args.camera_id not in [0, 1]:
        rospy.logerr("Camera ID must be 0 or 1")
        sys.exit(1)

    try:
        # Create calibration collector
        collector = MultiCameraCalibrationCollector(args.camera_id)

        # Apply custom parameters if provided
        if args.samples != 25:
            collector.target_count = args.samples

        if args.checkerboard != '9x6':
            try:
                w, h = map(int, args.checkerboard.split('x'))
                collector.checkerboard = (w, h)
                # Recalculate object points
                collector.objp = np.zeros((w * h, 3), np.float32)
                collector.objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
                collector.objp *= args.square_size
            except ValueError:
                rospy.logerr("Invalid checkerboard format. Use WIDTHxHEIGHT (e.g., 9x6)")
                sys.exit(1)

        if args.square_size != 0.025:
            collector.square_size = args.square_size
            collector.objp *= args.square_size

        # Run calibration
        collector.run_calibration()

    except rospy.ROSInterruptException:
        rospy.loginfo("ROS shutdown requested")
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupt received")
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()