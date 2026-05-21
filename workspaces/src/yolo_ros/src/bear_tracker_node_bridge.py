#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from ultralytics import YOLO
import cv2
import numpy as np
import message_filters

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

class BridgeTrackerNode(Node):
    def __init__(self):
        super().__init__('bridge_tracker_node')
        self.model = YOLO("/workspaces/src/yolo_ros/weights/best.pt")
        self.bridge = CvBridge()
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw', qos_profile=qos_profile_sensor_data)
        
        # 不需要深度圖了，因為只算像素偏移
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub], queue_size=5, slop=0.05)
        self.ts.registerCallback(self.image_callback)
        
        # 🌟 輸出 1 個數值：使用 Float32
        self.pub = self.create_publisher(Float32, '/yolo/targets_info', 10)
        self.get_logger().info("🌉 Bridge 節點啟動：純粹算出偏移像素")

    def image_callback(self, rgb_msg):
        try:
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            img_width = rgb_img.shape[1]
            cx = img_width / 2.0

            # 🌟 假設橋的 ID 是 2
            results = self.model(rgb_img, conf=0.5, classes=[2], verbose=False, device=0)
            annotated_frame = results[0].plot()

            if len(results[0].boxes) > 0 and results[0].masks is not None:
                bridge_masks = [results[0].masks.data[i].cpu().numpy() for i in range(len(results[0].boxes))]
                master_mask = np.logical_or.reduce(bridge_masks).astype(np.uint8)
                y_coords, x_coords = np.where(master_mask > 0.5)
                
                if len(y_coords) > 0:
                    bottom_y = np.max(y_coords)
                    safe_y_threshold = max(0, bottom_y - 20)
                    bottom_area_indices = np.where(y_coords >= safe_y_threshold)[0]
                    bottom_x_pixels = x_coords[bottom_area_indices]
                    
                    if len(bottom_x_pixels) > 0:
                        target_x = int(np.mean(bottom_x_pixels))
                        pixel_error_x = target_x - int(cx)
                        
                        # 發布 Float32
                        msg = Float32(data=float(pixel_error_x))
                        self.pub.publish(msg)
                        
                        cv2.circle(annotated_frame, (target_x, int(np.mean(y_coords[bottom_area_indices]))), 8, (0, 0, 255), -1)
                        cv2.putText(annotated_frame, f"Pixel Offset: {pixel_error_x}px", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Bridge Tracker", annotated_frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"⚠️ 視覺迴圈發生錯誤: {e}")

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(BridgeTrackerNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()