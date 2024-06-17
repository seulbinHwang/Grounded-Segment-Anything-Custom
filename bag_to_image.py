import cv2
import os
import numpy as np
from rosbag2_py import Reader, StorageOptions, ConverterOptions

# Bag 파일과 토픽 설정
bag_file_path = '~/d064_dataset_3laps_manual/d064_dynamic_0.db3'
topic_name = 'rgb3'

# 이미지를 저장할 디렉토리 생성
output_directory = 'workspace/data/images'
os.makedirs(output_directory, exist_ok=True)

# Bag 파일 열기
storage_options = StorageOptions(uri=bag_file_path, storage_id='sqlite3')
converter_options = ConverterOptions()
reader = Reader()
reader.open(storage_options, converter_options)

# 토픽과 일치하는 메시지를 찾아 처리
if reader.has_next():
    for (topic, data, timestamp) in reader.read_messages(topics=[topic_name]):
        np_arr = np.frombuffer(data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # CompressedImage는 JPEG 포맷으로 디코드

        # 이미지 파일로 저장
        image_filename = f'{output_directory}/image_{timestamp}.jpg'
        cv2.imwrite(image_filename, image_np)
        print(f'Saved {image_filename}')

# Reader 닫기
reader.close()
