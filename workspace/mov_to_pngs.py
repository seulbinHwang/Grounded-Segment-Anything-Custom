import cv2
from typing import Union, Optional
import os
from datetime import datetime

def extract_frames_from_mov(video_path: str,
                            output_dir: str,
                            start_time: str,
                            end_time: str,
                            interval_sec: float = 0.1,
                            img_format: str = 'png',
                            max_frames_number: Optional[int] = None) -> Optional[int]:
    """
    주어진 MOV 영상 파일에서 지정된 시작시간과 종료시간 사이의 구간에서,
    지정된 간격(interval_sec)으로 프레임을 추출하여 이미지 파일로 저장한다.

    Args:
        video_path (str): 분석할 MOV 파일의 경로.
        output_dir (str): 추출된 이미지들을 저장할 디렉토리 경로.
        start_time (str): 추출을 시작할 시간 ('hh:mm:ss.x' 형식).
        end_time (str): 추출을 종료할 시간 ('hh:mm:ss.x' 형식).
        interval_sec (float): 프레임을 추출할 시간 간격 (초).
        img_format (str): 저장할 이미지의 포맷 (기본값: 'png').

    Returns:
        Optional[int]: 추출된 이미지의 개수. 오류 발생 시 None을 반환.
    """
    def time_to_seconds(t: str) -> float:
        # 시간 문자열을 초로 변환
        time_obj = datetime.strptime(t, '%H:%M:%S.%f')
        total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
        return total_seconds

    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Video file could not be opened.")
        return None

    # 영상의 FPS 값과 각종 필요한 프레임 인덱스 계산
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_seconds = time_to_seconds(start_time)
    end_seconds = time_to_seconds(end_time)
    start_frame = int(fps * start_seconds)
    end_frame = int(fps * end_seconds)
    frame_interval = int(fps * interval_sec)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    frame_index = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while True:
        ret, frame = cap.read()
        frame_index += 1
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if not ret or frame_id > end_frame:
            break
        if frame_index % frame_interval == 0:
            output_filename = f"{output_dir}/{count:04d}.{img_format}"
            cv2.imwrite(output_filename, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            count += 1
            if max_frames_number and count >= max_frames_number:
                break

    # 모든 작업이 끝난 후 자원 해제
    cap.release()
    return count


video_name = "2_right.mov"
video_name_, _ = os.path.splitext(video_name) # '2_right'
video_path = os.path.join('data', video_name) # 'data/2_right.MOV'
output_dir = os.path.join("data", video_name_) # 'data/2_right'

frames_extracted = extract_frames_from_mov(video_path, output_dir, start_time = '00:04:52.0', end_time = '00:04:57.0')
print(f"Extracted {frames_extracted} frames.")
