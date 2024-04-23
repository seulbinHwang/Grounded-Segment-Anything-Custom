from typing import List
import os
from PIL import Image


def create_gif_from_images(directory: str, output_path: str) -> None:
    """
    특정 폴더(directory) 내의 모든 이미지를 이름 순으로 정렬하여 GIF 파일로 만들고 저장하는 함수.

    Args:
        directory (str): 이미지 파일이 있는 폴더의 경로.
        output_path (str): 생성된 GIF 파일을 저장할 경로.
        duration (int, optional): 각 이미지의 지속 시간(밀리초). 기본값은 500ms.

    Raises:
        ValueError: 지정된 디렉토리에 이미지 파일이 없는 경우 에러 발생.
    """
    # 주어진 디렉토리에서 모든 파일 목록을 가져오고, 이미지 파일만 필터링
    fps = 30
    period = 1 / fps
    duration = int(period * 1000)
    images = [
        os.path.join(directory, file)
        for file in sorted(os.listdir(directory))
        if file.endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not images:
        raise ValueError("No images found in the directory.")

    # 이미지 파일을 열어서 리스트에 담기
    image_frames: List[Image.Image] = [Image.open(img) for img in images]

    # 리스트의 첫 번째 이미지를 기준으로 GIF 생성
    image_frames[0].save(output_path,
                         format='GIF',
                         append_images=image_frames[1:],
                         save_all=True,
                         duration=duration,
                         loop=0)

    # 열려있는 모든 이미지 파일 닫기
    for img in image_frames:
        img.close()


# # 사용 예시
# directory = 'data/long_test'
# create_gif_from_images('path/to/images', 'path/to/output.gif')
