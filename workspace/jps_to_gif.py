import os
import re
from typing import List
from PIL import Image

def make_gif_from_images(image_dir: str, gif_path: str, duration: int = 150) -> None:
    """
    주어진 디렉토리에서 이미지 파일을 숫자 순서대로 정렬하고, 하나의 GIF 파일로 합칩니다.

    Args:
        image_dir (str): 이미지 파일들이 있는 디렉토리 경로.
        gif_path (str): 생성할 GIF 파일의 저장 경로 및 파일 이름.
        duration (int): 각 이미지가 GIF에서 보여질 시간 (밀리초 단위).

    Returns:
        None: 함수는 GIF 이미지를 생성하고 저장합니다.
    """

    # 이미지 파일 목록을 가져와서 숫자 순으로 정렬
    files = sorted(os.listdir(image_dir), key=lambda x: int(re.search(r'\d+', x).group()))
    images = [Image.open(os.path.join(image_dir, file)) for file in files if file.endswith((".png", ".jpg", ".jpeg"))]
    # get 1000 images.
    images = images[:1000]

    # 첫 번째 이미지를 기준으로 GIF 생성
    if images:
        images[0].save(gif_path, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)

# 사용 예:
if __name__ == '__main__':
    image_directory = 'results/images/0958_box_0.4/all'
    output_gif_path = 'results/images/0958_box_0.4/animation_1000_.gif'
    make_gif_from_images(image_directory, output_gif_path)
