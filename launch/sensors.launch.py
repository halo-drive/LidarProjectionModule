#!/usr/bin/env python3
"""
ROS1 Sensor Launch Script - Phase 2
LidarProjectionLane: Sensor Drivers & Data Acquisition

This script launches and manages:
- Dual USB Camera drivers (camera0, camera1)
- Dual VLP-16 LiDAR data routing
- Static coordinate transforms
- Parameter loading and sensor monitoring

Hardware Configuration:
- Camera 0: /dev/video0 → /camera0/image_raw
- Camera 1: /dev/video1 → /camera1/image_raw
- LiDAR 0:  /velodyne_points → /lidar0/points
- LiDAR 1:  /velodyne2/velodyne_points → /lidar1/points

Author: Lane Fusion Development Team
Phase: 2 - Sensor Drivers & Data Acquisition
"""

import rospy
import subprocess
import sys
import os
import yaml
import signal
import time
from threading import Thread, Lock
from collections import defaultdict

class SensorLauncher:
    """
    Comprehensive sensor launch and management system for LidarProjectionLane
    Handles dual cameras, dual LiDARs, coordinate transforms, and health monitoring
    """

    def __init__(self):
        rospy.init_node('sensor_launcher', anonymous=True)

        # Get project root directory
        self.project_root = self._find_project_root()
        self.config_dir = os.path.join(self.project_root, 'config')

        # Process management
        self.processes = []
        self.process_lock = Lock()
        self.shutdown_requested = False

        # Health monitoring
        self.health_status = defaultdict(dict)
        self.monitoring_thread = None

        # Load system configuration
        self.config = self._load_system_config()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        rospy.loginfo("=== LidarProjectionLane Sensor Launcher Initialized ===")
        rospy.loginfo(f"Project root: {self.project_root}")
        rospy.loginfo(f"ROS Master: {os.environ.get('ROS_MASTER_URI', 'Not set')}")

    def _find_project_root(self):
        """Find the project root directory"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up from launch/ to project root
        return os.path.dirname(script_dir)

    def _load_system_config(self):
        """Load system configuration with fallback defaults"""
        config_file = os.path.join(self.config_dir, 'system_config.yaml')
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                rospy.loginfo(f"Loaded system config from {config_file}")
                return config
        except Exception as e:
            rospy.logwarn(f"Could not load system config: {e}")
            rospy.loginfo("Using default configuration")
            return self._get_default_config()

    def _get_default_config(self):
        """Get default configuration if file loading fails"""
        return {
            'system': {
                'debug_mode': True,
                'visualization': True,
                'log_level': 'INFO'
            },
            'sensors': {
                'enable_camera0': True,
                'enable_camera1': True,
                'enable_lidar0': True,
                'enable_lidar1': True
            },
            'performance': {
                'enable_health_monitoring': True,
                'health_check_interval': 10
            }
        }

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        rospy.loginfo(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        self.cleanup()
        sys.exit(0)

    def _start_process(self, cmd, name, env=None):
        """Start a subprocess with error handling and tracking"""
        try:
            if env is None:
                env = os.environ.copy()

            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create process group for cleanup
            )

            with self.process_lock:
                self.processes.append({
                    'process': process,
                    'name': name,
                    'cmd': cmd,
                    'start_time': time.time()
                })

            rospy.loginfo(f"Started {name} (PID: {process.pid})")
            return process

        except Exception as e:
            rospy.logerr(f"Failed to start {name}: {e}")
            return None

    def launch_camera(self, camera_id):
        """Launch USB camera driver with comprehensive error handling"""
        rospy.loginfo(f"Launching camera{camera_id}...")

        # Check if camera device exists
        device = f"/dev/video{camera_id}"
        if not os.path.exists(device):
            rospy.logerr(f"Camera device {device} not found")
            return False

        # Camera parameters
        frame_id = f"camera{camera_id}_link"
        topic_prefix = f"camera{camera_id}"

        # Load camera configuration
        camera_config_file = os.path.join(
            self.config_dir, 'camera_params', f'camera{camera_id}.yaml'
        )

        if not os.path.exists(camera_config_file):
            rospy.logwarn(f"Camera config {camera_config_file} not found")

        # USB camera node command with optimized parameters
        cmd = [
            'rosrun', 'usb_cam', 'usb_cam_node',
            f'_video_device:={device}',
            f'_image_width:=640',
            f'_image_height:=480',
            f'_pixel_format:=yuyv',          # Fixed buffer mismatch issue
            f'_camera_frame_id:={frame_id}',
            f'_io_method:=mmap',
            f'_framerate:=30',
            f'_camera_name:=camera{camera_id}',
            f'_camera_info_url:=file://{camera_config_file}',
            f'__name:=usb_cam_node_{camera_id}'
        ]

        # Set topic remappings via ROS namespace
        env = os.environ.copy()
        env['ROS_NAMESPACE'] = topic_prefix

        process = self._start_process(cmd, f"camera{camera_id}", env)
        return process is not None

    def launch_static_transforms(self):
        """Launch static transform publishers for all sensor frames"""
        rospy.loginfo("Setting up static coordinate transforms...")

        # Define transform tree based on verified hardware setup
        transforms = [
            # Base coordinate system
            {
                'parent': 'base_link',
                'child': 'camera0_link',
                'xyz': [0.0, -0.1, 0.0],      # Slightly left of center
                'rpy': [0.0, 0.0, 0.0],
                'description': 'Camera 0 mount'
            },
            {
                'parent': 'base_link',
                'child': 'camera1_link',
                'xyz': [0.0, 0.1, 0.0],       # Slightly right of center
                'rpy': [0.0, 0.0, 0.0],
                'description': 'Camera 1 mount'
            },
            # LiDAR coordinate systems (verified frame IDs)
            {
                'parent': 'base_link',
                'child': 'velodyne',           # LiDAR 0 frame ID
                'xyz': [0.0, 0.0, 1.8],       # Typical roof mount height
                'rpy': [0.0, 0.0, 0.0],
                'description': 'Primary VLP-16 LiDAR'
            },
            {
                'parent': 'base_link',
                'child': 'velodyne2',          # LiDAR 1 frame ID
                'xyz': [0.0, 0.0, 1.8],       # Same height, different position
                'rpy': [0.0, 0.0, 0.0],
                'description': 'Secondary VLP-16 LiDAR'
            }
        ]

        success_count = 0
        for i, tf in enumerate(transforms):
            cmd = [
                'rosrun', 'tf2_ros', 'static_transform_publisher',
                str(tf['xyz'][0]), str(tf['xyz'][1]), str(tf['xyz'][2]),
                str(tf['rpy'][0]), str(tf['rpy'][1]), str(tf['rpy'][2]),
                tf['parent'], tf['child'],
                f'__name:=static_tf_publisher_{i}'
            ]

            process = self._start_process(cmd, f"transform_{tf['child']}")
            if process:
                success_count += 1
                rospy.loginfo(f"📐 Transform: {tf['parent']} → {tf['child']} ({tf['description']})")

        rospy.loginfo(f"✅ Published {success_count}/{len(transforms)} static transforms")
        return success_count == len(transforms)

    def setup_lidar_relays(self):
        """Setup LiDAR topic relays for consistent naming"""
        rospy.loginfo("Setting up LiDAR topic routing...")

        # Define LiDAR topic routing based on verified hardware
        lidar_relays = [
            {
                'source': '/velodyne_points',           # Primary VLP-16
                'target': '/lidar0/points',
                'name': 'lidar0_relay',
                'description': 'Primary VLP-16 relay'
            },
            {
                'source': '/velodyne2/velodyne_points',  # Secondary VLP-16
                'target': '/lidar1/points',
                'name': 'lidar1_relay',
                'description': 'Secondary VLP-16 relay'
            }
        ]

        success_count = 0
        for relay in lidar_relays:
            cmd = [
                'rosrun', 'topic_tools', 'relay',
                relay['source'], relay['target'],
                f'__name:={relay["name"]}'
            ]

            process = self._start_process(cmd, relay['name'])
            if process:
                success_count += 1
                rospy.loginfo(f"Relay: {relay['source']} → {relay['target']}")

        rospy.loginfo(f"Established {success_count}/{len(lidar_relays)} LiDAR relays")
        return success_count == len(lidar_relays)

    def start_health_monitoring(self):
        """Start health monitoring thread"""
        if not self.config.get('performance', {}).get('enable_health_monitoring', True):
            return

        self.monitoring_thread = Thread(target=self._health_monitor_loop, daemon=True)
        self.monitoring_thread.start()
        rospy.loginfo("Health monitoring started")

    def _health_monitor_loop(self):
        """Health monitoring loop"""
        interval = self.config.get('performance', {}).get('health_check_interval', 10)

        while not self.shutdown_requested:
            try:
                self._check_process_health()
                self._check_topic_health()
                time.sleep(interval)
            except Exception as e:
                rospy.logwarn(f"Health monitoring error: {e}")
                time.sleep(interval)

    def _check_process_health(self):
        """Check health of launched processes"""
        with self.process_lock:
            for proc_info in self.processes:
                process = proc_info['process']
                name = proc_info['name']

                if process.poll() is not None:
                    rospy.logwarn(f"Process {name} has terminated (exit code: {process.returncode})")
                    # Could implement restart logic here

    def _check_topic_health(self):
        """Check health of expected topics"""
        try:
            # Use rostopic list to check active topics
            result = subprocess.run(['rostopic', 'list'],
                                    capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                active_topics = set(result.stdout.strip().split('\n'))

                # Expected topics based on configuration
                expected_topics = []
                if self.config['sensors']['enable_camera0']:
                    expected_topics.append('/camera0/image_raw')
                if self.config['sensors']['enable_camera1']:
                    expected_topics.append('/camera1/image_raw')
                if self.config['sensors']['enable_lidar0']:
                    expected_topics.append('/lidar0/points')
                if self.config['sensors']['enable_lidar1']:
                    expected_topics.append('/lidar1/points')

                # Check for missing topics
                missing_topics = set(expected_topics) - active_topics
                if missing_topics:
                    rospy.logwarn(f"Missing topics: {list(missing_topics)}")

        except subprocess.TimeoutExpired:
            rospy.logwarn("Topic health check timed out")
        except Exception as e:
            rospy.logwarn(f"Topic health check failed: {e}")

    def launch_all_sensors(self):
        """Launch all configured sensors and supporting systems"""
        rospy.loginfo("=== Starting Sensor Launch Sequence ===")

        success_count = 0
        total_components = 0

        # Launch cameras
        for camera_id in [0, 1]:
            if self.config['sensors'].get(f'enable_camera{camera_id}', True):
                total_components += 1
                if self.launch_camera(camera_id):
                    success_count += 1
                # Small delay between camera launches
                time.sleep(1)

        # Setup coordinate transforms
        total_components += 1
        if self.launch_static_transforms():
            success_count += 1

        # Setup LiDAR routing
        total_components += 1
        if self.setup_lidar_relays():
            success_count += 1

        # Start health monitoring
        self.start_health_monitoring()

        # Launch summary
        rospy.loginfo("=== Sensor Launch Complete ===")
        rospy.loginfo(f"Successfully launched: {success_count}/{total_components} components")

        if success_count == total_components:
            rospy.loginfo("All systems operational!")
        else:
            rospy.logwarn(f"{total_components - success_count} components failed to start")

        # Display active topics
        self._display_topic_summary()

        return success_count == total_components

    def _display_topic_summary(self):
        """Display summary of expected active topics"""
        rospy.loginfo("")
        rospy.loginfo("Expected active topics:")
        rospy.loginfo("Camera Topics:")
        rospy.loginfo("  - /camera0/image_raw       (USB Camera 0)")
        rospy.loginfo("  - /camera1/image_raw       (USB Camera 1)")
        rospy.loginfo("LiDAR Relay Topics:")
        rospy.loginfo("  - /lidar0/points           (from /velodyne_points)")
        rospy.loginfo("  - /lidar1/points           (from /velodyne2/velodyne_points)")
        rospy.loginfo("Transform Topics:")
        rospy.loginfo("  - /tf_static               (coordinate transforms)")
        rospy.loginfo("")
        rospy.loginfo("To verify topics:")
        rospy.loginfo("  rostopic list | grep -E '(camera|lidar|tf)'")
        rospy.loginfo("  rostopic hz /camera0/image_raw")
        rospy.loginfo("  rostopic hz /lidar0/points")

    def cleanup(self):
        """Cleanup all launched processes"""
        rospy.loginfo("Shutting down sensor nodes...")

        self.shutdown_requested = True

        with self.process_lock:
            for proc_info in self.processes:
                process = proc_info['process']
                name = proc_info['name']

                try:
                    # Send SIGTERM to process group
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)

                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=5)
                        rospy.loginfo(f"{name} shutdown gracefully")
                    except subprocess.TimeoutExpired:
                        # Force kill if necessary
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        rospy.logwarn(f"{name} force killed")

                except Exception as e:
                    rospy.logwarn(f"Error stopping {name}: {e}")

        rospy.loginfo("🏁 Sensor shutdown complete")

def main():
    """Main entry point"""
    launcher = None
    try:
        launcher = SensorLauncher()

        # Launch all sensors
        success = launcher.launch_all_sensors()

        if not success:
            rospy.logwarn("Some components failed to launch, but continuing...")

        # Keep the launcher running
        rospy.loginfo("Sensor launcher running. Press Ctrl+C to exit.")
        rospy.loginfo("")
        rospy.loginfo("Next steps:")
        rospy.loginfo("  1. Verify camera topics: rostopic hz /camera0/image_raw")
        rospy.loginfo("  2. Check LiDAR relays: rostopic hz /lidar0/points")
        rospy.loginfo("  3. Test calibration: python3 scripts/calibration_collection.py 0")
        rospy.loginfo("  4. Proceed to Phase 3: YOLOv8 model integration")

        # Keep running until shutdown
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("ROS shutdown requested")
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupt received")
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
    finally:
        if launcher:
            launcher.cleanup()

if __name__ == '__main__':
    main()