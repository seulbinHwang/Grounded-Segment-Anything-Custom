import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import os

# Bag 파일과 토픽 설정
bag_file_path = '~/d064_dataset_3laps_manual/d064_dynamic_0.db3'
topic_name = 'rgb3'

# 이미지를 저장할 디렉토리 생성

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            CompressedImage,
            'rgb3',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.img_count = 0
        self.save_path = 'workspace/data/images'
        os.makedirs(self.save_path, exist_ok=True)

    def listener_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image_np is not None:
            img_filename = f"{self.save_path}/image_{self.img_count}.jpg"
            cv2.imwrite(img_filename, image_np)
            self.get_logger().info(f'Saved {img_filename}')
            self.img_count += 1
        else:
            self.get_logger().warn('Failed to decode image')

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
