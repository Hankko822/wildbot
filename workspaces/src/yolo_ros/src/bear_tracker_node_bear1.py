#!/usr/bin/env python3
import rclpy
import time
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Point
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from ultralytics import YOLO
import cv2
import numpy as np
import message_filters
import math
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

import sys

# 1. 這裡必須填寫「包含 models 資料夾」的那個目錄 (因為你在 Docker 裡，所以用 /workspaces)
sys.path.append('/workspaces/src/yolo_ros/weights') 

try:
    # 2. 這裡必須是 from "models.scconv" (迎合 PyTorch 的記憶)
    from models.scconv import SCConv  
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    # 雙重魔法註冊
    setattr(modules, 'SCConv', SCConv)
    setattr(tasks, 'SCConv', SCConv)
    print("✅ 成功註冊 SCConv 模組！")
except Exception as e:
    print(f"❌ 找不到 SCConv 模組！錯誤訊息: {e}")

class Bear1TrackerNode(Node):
    def __init__(self):
        super().__init__('bear1_tracker_node')
        self.model = YOLO("/workspaces/src/yolo_ros/weights/best.pt")
        self.prev_frame_time = 0
        self.bridge = CvBridge()
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw', qos_profile=qos_profile_sensor_data)
        self.dep_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw', qos_profile=qos_profile_sensor_data)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.dep_sub], queue_size=5, slop=0.05)
        self.ts.registerCallback(self.image_callback)
        
        # 🌟 輸出 3 個數值：使用 Point (x=dist, y=angle, z=height)
        self.pub = self.create_publisher(Point, '/yolo/targets_info', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.fx, self.fy, self.cx, self.cy = 640.0, 640.0, 640.0, 360.0
        self.get_logger().info("🐻 Bear1 節點啟動：專注尋找最近的熊")

    def image_callback(self, rgb_msg, dep_msg):
        try:
            current_time = time.time()
            fps = 0.0
            if self.prev_frame_time != 0:
                fps = 1.0 / (current_time - self.prev_frame_time)
            self.prev_frame_time = current_time
            trans = self.tf_buffer.lookup_transform('base_link', rgb_msg.header.frame_id, rclpy.time.Time())
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            dep_img = self.bridge.imgmsg_to_cv2(dep_msg, "16UC1")
            img_height, img_width = rgb_img.shape[:2]
            self.cx, self.cy = img_width / 2.0, img_height / 2.0

            # 🌟 假設熊的 ID 是 0
            results = self.model(rgb_img, conf=0.5, classes=[0], verbose=False, device=0)
            annotated_frame = results[0].plot()

            closest_bear = None
            min_dist = float('inf')

            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    u, v = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)
                    
                    depth_roi = dep_img[max(0, v-2):min(img_height, v+3), max(0, u-2):min(img_width, u+3)]
                    valid_depths = depth_roi[depth_roi > 0]

                    if valid_depths.size > 0:
                        depth_m = float(np.median(valid_depths) / 1000.0)
                        pt_cam = PointStamped()
                        pt_cam.header.frame_id = rgb_msg.header.frame_id
                        pt_cam.point.x = (u - self.cx) * depth_m / self.fx
                        pt_cam.point.y = (v - self.cy) * depth_m / self.fy
                        pt_cam.point.z = depth_m

                        pt_car = tf2_geometry_msgs.do_transform_point(pt_cam, trans)
                        real_dist = math.hypot(pt_car.point.x, pt_car.point.y)
                        
                        # 記錄最近的熊
                        if real_dist < min_dist:
                            min_dist = real_dist
                            real_angle_deg = math.degrees(math.atan2(pt_car.point.y, pt_car.point.x))
                            closest_bear = Point(x=round(real_dist, 3), y=round(real_angle_deg, 1), z=round(pt_car.point.z, 3))

            # 發布最近的熊
            if closest_bear:
                self.pub.publish(closest_bear)
                cv2.putText(annotated_frame, f"Dist:{closest_bear.x}m Ang:{closest_bear.y} H:{closest_bear.z}m", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Bear1 Tracker", annotated_frame)
            cv2.waitKey(1)
        except Exception as e:
            pass

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(Bear1TrackerNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()