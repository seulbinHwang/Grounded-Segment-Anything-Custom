import cv2
from typing import Union, Optional
import os


def extract_frames_from_mov(video_path: str,
                            output_dir: str,
                            img_format: str = 'jpg') -> Optional[int]:
    """
    주어진 MOV 영상 파일에서 각 프레임을 이미지 파일로 추출하고 저장한다.

    Args:
        video_path (str): 분석할 MOV 파일의 경로.
        output_dir (str): 추출된 이미지들을 저장할 디렉토리 경로.
        img_format (str): 저장할 이미지의 포맷 (기본값: 'jpg').

    Returns:
        Optional[int]: 추출된 이미지의 개수. 오류 발생 시 None을 반환.
    """
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Video file could not be opened.")
        return None

    # 영상의 FPS 값을 얻음
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 이미지 파일 이름 설정 (프레임 번호 포함)
        output_filename = f"{output_dir}/{count:04d}.{img_format}"
        # 이미지 저장
        cv2.imwrite(output_filename, frame)
        count += 1

    # 모든 작업이 끝난 후 자원 해제
    cap.release()
    return count


video_name = "test.MOV"
video_name_, _ = os.path.splitext(video_name)
video_path = os.path.join('data', video_name)
output_dir = os.path.join("data", video_name_)

frames_extracted = extract_frames_from_mov(video_path, output_dir)
print(f"Extracted {frames_extracted} frames.")
