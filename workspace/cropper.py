import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from matplotlib.widgets import Button
import argparse
from typing import List, Tuple, NoReturn
from tqdm import tqdm
# 사용자 입력을 통해 좌표를 저장하는 데 사용될 글로벌 변수
coords: List[Tuple[float, float]] = []


def onclick(event) -> NoReturn:
    # 마우스 클릭 이벤트 처리
    global coords
    if len(coords) < 4:  # 최대 4개의 좌표만 저장
        ix, iy = event.xdata, event.ydata
        coords.append((ix, iy))
        plt.plot(ix, iy, 'ro')  # 클릭한 위치에 빨간색 점 표시

        if len(coords) == 4:  # 4개의 점이 모두 찍혔을 때
            # 첫 번째 점과 마지막 점을 연결
            coords.append(coords[0])
            # 빨간 선분으로 사각형 표시
            plt.plot(*zip(*coords), color='red')

        plt.draw()


def onkeypress(event) -> NoReturn:
    global coords
    if event.key == 'enter' and len(coords) == 5:  # 엔터를 누르면 크롭 시작
        plt.close()
    elif event.key == 'r':  # 'r'을 누르면 리셋
        coords = []
        plt.cla()
        fig.canvas.mpl_disconnect(cid)
        fig.canvas.mpl_connect('button_press_event', onclick)
        img = plt.imread(first_image_path)
        ax.imshow(img)
        plt.draw()


def crop_images(args, source_folder: str, dest_folder: str,
                crop_coords: List[Tuple[float, float]]) -> NoReturn:
    # 소스 폴더에서 이미지 파일 이름을 가져옴
    images = [
        f for f in os.listdir(source_folder)
        if os.path.isfile(os.path.join(source_folder, f)) and
        f.lower().endswith(('.png'))
    ]
    # sort images.
    crop_size = None
    first_image = Image.open(os.path.join(source_folder, images[0]))
    images.sort()
    for image_name in tqdm(images):
        # 이미지 열기
        with Image.open(os.path.join(source_folder, image_name)) as img:
            # 크롭 영역 계산 (좌측 상단, 우측 하단)
            left = int(min(crop_coords, key=lambda x: x[0])[0])
            upper = int(min(crop_coords, key=lambda x: x[1])[1])
            right = int(max(crop_coords, key=lambda x: x[0])[0])
            lower = int(max(crop_coords, key=lambda x: x[1])[1])
            if crop_size is None and args.dense_crop:
                crop_size = (right - left, lower - upper)
                first_crop_image = first_image.crop((left, upper, right, lower))
                mask_img = Image.new('L', first_crop_image.size, 0)
                cropped_coords = [(x - left, y - upper) for x, y in crop_coords]
                ImageDraw.Draw(mask_img).polygon(cropped_coords,
                                                 outline=1,
                                                 fill=1)
                mask = np.array(mask_img)  # (H, W)

            # 이미지 크롭 및 저장
            if args.dense_crop:
                final_img = np.zeros_like(np.array(first_crop_image))
                final_img[mask == 1] = np.array(cropped_img)[mask ==
                                                             1]  # (H, W, 3)
                final_img[mask == 0] = np.array(first_crop_image)[
                    mask == 0]  # (H, W, 3)
                # save final_img
                final_img = Image.fromarray(final_img)
                final_img.save(os.path.join(dest_folder, image_name), "PNG")
            else:
                cropped_img = img.crop((left, upper, right, lower))
                cropped_img.save(os.path.join(dest_folder, image_name), "PNG")


def main() -> NoReturn:
    parser = argparse.ArgumentParser(
        description="Crop images in a folder based on user-selected area.")
    parser.add_argument("--source_folder",
                        type=str,
                        help="Source folder containing images to crop.",
                        default='data/2_right')
    parser.add_argument("--dense_crop", type=bool, default=False)
    args = parser.parse_args()

    global coords, cid, first_image_path, fig, ax
    source_folder = args.source_folder
    dest_folder = source_folder + "_cropped"

    # 대상 폴더가 없으면 생성
    if os.path.exists(dest_folder):
        # delete
        import shutil
        shutil.rmtree(dest_folder)
    os.makedirs(dest_folder)

    # 첫 번째 이미지를 사용하여 크롭 영역 선택
    first_image_path = os.path.join(source_folder, os.listdir(source_folder)[0])
    img = plt.imread(first_image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkeypress)
    plt.show()

    # 선택한 영역으로 모든 이미지 크롭 (마지막 점 제외)
    crop_images(args, source_folder, dest_folder, coords[:-1])


if __name__ == '__main__':
    main()
