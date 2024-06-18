import os
import argparse
from PIL import Image
from typing import Tuple


def resize_images(source_dir: str, new_width: int, new_height: int) -> None:
    target_size: Tuple[int, int] = (new_width, new_height)
    target_dir: str = f"{source_dir}_{new_width}_{new_height}"

    # 대상 디렉토리가 존재하지 않으면 생성
    os.makedirs(target_dir, exist_ok=True)

    # 소스 디렉토리에서 모든 파일 목록을 가져옴
    for filename in os.listdir(source_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # 이미지 파일만 처리
            source_path: str = os.path.join(source_dir, filename)
            target_path: str = os.path.join(target_dir, filename)

            # 이미지 열기 및 크기 조정
            with Image.open(source_path) as img:
                resized_img = img.resize(target_size,
                                         Image.Resampling.LANCZOS)  # ANTIALIAS 대신 LANCZOS 사용

                # 변경된 크기의 이미지 저장
                resized_img.save(target_path)
                print(f"Resized and saved {filename} to {target_path}")


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='Resize images to specified dimensions and save them to a new directory.')
    parser.add_argument('--source_directory', type=str,
                        help='The directory where the source images are stored.',
                        default='workspace/data/images')
    parser.add_argument('--width', type=int,
                        help='The width of the resized images.')
    parser.add_argument('--height', type=int,
                        help='The height of the resized images.')

    args: argparse.Namespace = parser.parse_args()

    resize_images(args.source_directory, args.width, args.height)


if __name__ == '__main__':
    main()
