#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Quaternion
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
    
class Bear2TrackerNode(Node):
    def __init__(self):
        super().__init__('bear2_tracker_node')
        self.model = YOLO("/workspaces/src/yolo_ros/weights/best.pt")
        self.bridge = CvBridge()
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw', qos_profile=qos_profile_sensor_data)
        self.dep_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw', qos_profile=qos_profile_sensor_data)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.dep_sub], queue_size=5, slop=0.05)
        self.ts.registerCallback(self.image_callback)
        
        # 🌟 輸出 4 個數值：借用 Quaternion (x=熊距, y=熊角, z=熊高, w=橋偏移)
        self.pub = self.create_publisher(Quaternion, '/yolo/targets_info', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.fx, self.fy, self.cx, self.cy = 640.0, 640.0, 640.0, 360.0
        self.get_logger().info("🐻🌉 Bear2 節點啟動：過橋與找熊混合模式")

    def image_callback(self, rgb_msg, dep_msg):
        try:
            trans = self.tf_buffer.lookup_transform('base_link', rgb_msg.header.frame_id, rclpy.time.Time())
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            dep_img = self.bridge.imgmsg_to_cv2(dep_msg, "16UC1")
            img_height, img_width = rgb_img.shape[:2]
            self.cx, self.cy = img_width / 2.0, img_height / 2.0

            # 🌟 假設熊=1，橋=2
            results = self.model(rgb_img, conf=0.5, classes=[1, 2], verbose=False, device=0)
            annotated_frame = results[0].plot()

            # 預設值 (0.0 代表沒看到)
            bear_dist, bear_angle, bear_height, bridge_offset = 0.0, 0.0, 0.0, 0.0
            bridge_masks = []
            min_bear_dist = float('inf')

            if len(results[0].boxes) > 0:
                for i, box in enumerate(results[0].boxes):
                    cls_id = int(box.cls[0].item())
                    
                    if cls_id == 2: # 是橋
                        if results[0].masks is not None:
                            bridge_masks.append(results[0].masks.data[i].cpu().numpy())
                            
                    elif cls_id == 1: # 是熊
                        xyxy = box.xyxy[0].cpu().numpy()
                        u, v = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)
                        depth_roi = dep_img[max(0, v-2):min(img_height, v+3), max(0, u-2):min(img_width, u+3)]
                        valid_depths = depth_roi[depth_roi > 0]
                        if valid_depths.size > 0:
                            depth_m = float(np.median(valid_depths) / 1000.0)
                            pt_cam = PointStamped()
                            pt_cam.header.frame_id = rgb_msg.header.frame_id
                            pt_cam.point.x, pt_cam.point.y, pt_cam.point.z = (u - self.cx) * depth_m / self.fx, (v - self.cy) * depth_m / self.fy, depth_m
                            pt_car = tf2_geometry_msgs.do_transform_point(pt_cam, trans)
                            r_dist = math.hypot(pt_car.point.x, pt_car.point.y)
                            if r_dist < min_bear_dist:
                                min_bear_dist = r_dist
                                bear_dist, bear_angle, bear_height = round(r_dist, 3), round(math.degrees(math.atan2(pt_car.point.y, pt_car.point.x)), 1), round(pt_car.point.z, 3)

            # 處理橋的偏移
            if len(bridge_masks) > 0:
                master_mask = np.logical_or.reduce(bridge_masks).astype(np.uint8)
                y_coords, x_coords = np.where(master_mask > 0.5)
                if len(y_coords) > 0:
                    bottom_x_pixels = x_coords[np.where(y_coords >= max(0, np.max(y_coords) - 20))[0]]
                    if len(bottom_x_pixels) > 0:
                        bridge_offset = float(int(np.mean(bottom_x_pixels)) - int(self.cx))

            # 發布 Quaternion (包裝 4 個數值)
            msg = Quaternion(x=bear_dist, y=bear_angle, z=bear_height, w=bridge_offset)
            self.pub.publish(msg)
            
            cv2.putText(annotated_frame, f"B_Dist:{msg.x}m Ang:{msg.y} Offset:{msg.w}px", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Bear2 Hybrid Tracker", annotated_frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"⚠️ 視覺迴圈發生錯誤: {e}")

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(Bear2TrackerNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()