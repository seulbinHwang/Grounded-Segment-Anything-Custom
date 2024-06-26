import argparse
import os
import sys
from typing import Tuple, List, Callable, Dict, Any, Optional, Union

from PIL import Image
from huggingface_hub import hf_hub_download
from supervision.draw.color import Color
from torchvision.ops import box_convert
from tqdm import tqdm
import cv2
import numpy as np
import supervision as sv
import torch
from torch.utils import cpp_extension
from torch.utils.cpp_extension import load
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import load_image, predict, batch_predict, transform_image
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from segment_anything import SamPredictor, build_sam
import object_prompt
import jps_to_gif


class ImageProcessingApp:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.det_model = self.load_model_hf(
            repo_id="ShilongLiu/GroundingDINO",
            filename="groundingdino_swinb_cogcoor.pth",
            ckpt_config_filename="GroundingDINO_SwinB.cfg.py")
        self.sam_predictor = SamPredictor(
            build_sam(checkpoint=args.checkpoint).to(self.device))
        self.objects_prompt = self._set_object_prompt()
        self.object_names_list: List[str] = self.objects_prompt.get_name_list()

    def color_map(self, det_phase: str) -> Color:
        """
        탐지된 객체의 문구에 따라 색상을 반환합니다.

        Args:
            det_phase (str): 탐지된 객체의 문구.
            objects_prompt (object_prompt.ObjectsPrompt): 객체 탐지를 위한 텍스트 프롬프트.

        Returns:
            Color: 반환할 색상.
        """
        for object_prompt_ in self.objects_prompt.objects:
            if det_phase in object_prompt_.prompt:
                return object_prompt_.color
        raise ValueError(f"Invalid detection phrase: {det_phase}")

    def load_model_hf(self, repo_id: str, filename: str,
                      ckpt_config_filename: str) -> torch.nn.Module:
        cache_config_file = hf_hub_download(repo_id=repo_id,
                                            filename=ckpt_config_filename)
        args = SLConfig.fromfile(cache_config_file)
        args.device = self.device
        model = build_model(args)
        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location=self.device)
        model.load_state_dict(clean_state_dict(checkpoint["model"]),
                              strict=False)
        model.eval()
        return model

    def _set_object_prompt(self):
        prompts = [
            object_prompt.ObjectPrompt("person", "blue"),
            object_prompt.ObjectPrompt("interior wall", "green"),
            object_prompt.ObjectPrompt("door with a handle", "red"),
            object_prompt.ObjectPrompt("interior floor", "white"),
            object_prompt.ObjectPrompt("glass wall", "black")
        ]
        return object_prompt.ObjectsPrompt(prompts)

    def make_a_result_dir(self, box_threshold: float,
                          result_dir: str) -> Tuple[str, List[str]]:
        # experiment_folder: box_0.4
        experiment_folder = self.args.experiment_folder + f"box_{box_threshold}"
        # a_result_dir: /results/test/box_0.4
        a_result_dir = os.path.join(result_dir, experiment_folder)
        if not os.path.exists(a_result_dir):
            os.makedirs(a_result_dir)
        # object_names_list: ["person", "wall", ]
        result_dir_per_class_list: List[str] = []
        for object_prompt_ in self.object_names_list:
            # a_result_dir_per_class: /results/test/box_0.4/person
            a_result_dir_per_class = os.path.join(a_result_dir, object_prompt_)
            result_dir_per_class_list.append(a_result_dir_per_class)
            if not os.path.exists(a_result_dir_per_class):
                os.makedirs(a_result_dir_per_class)
        return a_result_dir, result_dir_per_class_list

    @staticmethod
    def save_result_image(result_dir_per_class_list: List[str],
                          img_segdet_per_class: List[np.ndarray],
                          result_frame_name: str) -> None:
        for class_idx, a_result_dir_per_class in tqdm(
                enumerate(result_dir_per_class_list)):
            image_segdet = img_segdet_per_class[class_idx]
            Image.fromarray(image_segdet).save(
                os.path.join(a_result_dir_per_class, result_frame_name))

    @staticmethod
    def save_result_gif(result_dir_per_class_list: List[str],
                        a_result_dir: str) -> None:
        for a_result_dir_per_class in tqdm(result_dir_per_class_list):
            # results_per_class_dir: /results/test/person
            # results_segdet_per_class_dir: /results-segdet/test/person
            # save as gif.
            class_name = os.path.basename(a_result_dir_per_class)
            result_gif_path = os.path.join(a_result_dir, class_name + ".gif")
            jps_to_gif.create_gif_from_images(a_result_dir_per_class,
                                              result_gif_path)

    def segment(self, image: np.ndarray, boxes: torch.Tensor) -> torch.Tensor:
        """ 주어진 이미지와 바운딩 박스를 사용하여 세그먼트를 수행합니다.

        Args:
            image: 원본 이미지 배열 (shape: HxWxC). RGB
            boxes: 바운딩 박스 텐서 (shape: Nx4).

        Returns:
            mask: 세그먼트 마스크 텐서 (shape: Nx1xHxW).
        """
        # TODO: 어짜피 n개의 boxes를 받을 수 있으니까, 그냥 class별 boxes를 for문 돌리는게 아니라,
        #  한번에 처리할 수 있도록 수정?
        # set_image 이미지 전처리
        self.sam_predictor.set_image(image)
        H, W, _ = image.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor(
            [W, H, W, H]).to(boxes.device)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy, image.shape[:2]).to(self.device)
        if transformed_boxes.shape[0] == 0:
            return torch.empty(0, 1, H, W)
        masks, _, _ = self.sam_predictor.predict_torch(point_coords=None,
                                                       point_labels=None,
                                                       boxes=transformed_boxes,
                                                       multimask_output=False)
        # mask: (N, 1, H, W)
        return masks.cpu()

    def inference(
        self, np_image: np.ndarray, box_threshold: float, text_threshold: float
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor, List[str]]],
               List[torch.Tensor]]:
        """
        탐지 및 세그먼트 모델을 사용하여 이미지에서 객체를 탐지하고 세그먼트 마스크를 생성합니다.

        Args:
            np_image (np.ndarray): 원본 이미지 배열 (shape: HxWxC).
            box_threshold (float): 바운딩 박스 임계값.
            text_threshold (float): 텍스트 임계값.

        Returns:
            det_results_per_class
                - 클래스 별로, 객체 탐지 결과를 담은 리스트.
                - List[ Tuple[torch.Tensor, torch.Tensor, List[str]] ]
                - (det_boxes, det_logits, det_phrases)
                    - det_boxes: (n, 4)
                    - det_logits: (n,)
                    - det_phrases: List[str]
            seg_masks_per_class
                - 클래스 별로, 세그먼트 마스크를 담은 리스트.
                - List[torch.Tensor]
                    - mask: (N, 1, H, W)

        """
        transformed_image = transform_image(np_image)

        # det_boxes_per_class: List[torch.Tensor] # (n, 4)
        # det_logits_per_class: List[torch.Tensor] # (n,)
        # det_phrases_per_class: List[List[str]] # List[str]
        (det_boxes_per_class, det_logits_per_class,
         det_phrases_per_class) = batch_predict(model=self.det_model,
                                                image=transformed_image,
                                                captions=self.object_names_list,
                                                box_threshold=box_threshold,
                                                text_threshold=text_threshold,
                                                device=self.device)
        # det_results_per_class:
        # List[ Tuple[torch.Tensor, torch.Tensor, List[str]] ]
        det_results_per_class = list(
            zip(det_boxes_per_class, det_logits_per_class,
                det_phrases_per_class))
        seg_masks_per_class = []
        for det_boxes in det_boxes_per_class:
            # det_boxes: (n, 4)
            seg_masks = self.segment(np_image, boxes=det_boxes)
            seg_masks_per_class.append(seg_masks)

        return det_results_per_class, seg_masks_per_class

    def draw(self,
             img: np.ndarray,
             det_results: Tuple[torch.Tensor, torch.Tensor, List[str]],
             seg_masks: torch.Tensor,
             fill: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """ 이미지에 객체 탐지 및 세그먼트 결과를 그립니다.

        Args:
            img (np.ndarray): 원본 이미지.
            det_results (Tuple[torch.Tensor, torch.Tensor, List[str]])
                - List[Tuple[torch.Tensor, torch.Tensor, List[str]]]
                - (det_boxes, det_logits, det_phrases)
                - det_boxes: (n, 4)
                - det_logits: (n,)
                - det_phrases: List[str]
            seg_masks: torch.Tensor
                - mask: (N, 1, H, W)
        """
        # det_phrases_per_class: ["person", "person", ... ]
        (det_boxes_per_class, det_logits_per_class,
         det_phrases_per_class) = det_results
        colors = [self.color_map(e) for e in det_phrases_per_class]

        if fill:
            img = np.zeros_like(img)
        if seg_masks.shape[0] > 0:
            img_seg = self.draw_seg(img, seg_masks, colors)
        else:
            img_seg = img.copy()
        img_segdet = self.draw_det(img_seg.copy(), det_boxes_per_class,
                                   det_logits_per_class, det_phrases_per_class,
                                   colors)

        return img_seg, img_segdet

    @staticmethod
    def draw_seg(image: np.ndarray, seg_masks: torch.Tensor,
                 colors: List[Color]) -> np.ndarray:
        """
        세그먼트 마스크를 이미지 위에 그립니다.

        Args:
            image (np.ndarray): 원본 이미지 배열 (shape: HxWxC).
            seg_masks (torch.Tensor): 세그먼트 마스크 텐서 (shape: N, 1, H, W).
            colors (List[Color]): 각 세그먼트 마스크에 사용할 색상.

        Returns:
            np.ndarray: 세그먼트가 적용된 이미지.
        """
        for seg_mask, color in zip(seg_masks, colors):
            # TODO: batch로 처리할 수 있어야 함
            seg_mask = seg_mask[0]  # (N, 1, H, W) -> (1, H, W)
            h, w = seg_mask.shape[-2:]
            mask_image = seg_mask.reshape(h, w, 1) * np.array(
                [*[e * 255 for e in color.as_rgb()], 1.0]).reshape(1, 1, -1)

            annotated_frame_pil = Image.fromarray(image).convert("RGBA")
            mask_image_pil = Image.fromarray(
                (mask_image.cpu().numpy() * 255).astype(
                    np.uint8)).convert("RGBA")

            image = np.array(
                Image.alpha_composite(annotated_frame_pil,
                                      mask_image_pil).convert("RGBA"))
        return image

    @staticmethod
    def draw_det(image_source: np.ndarray,
                 boxes: torch.Tensor,
                 logits: torch.Tensor,
                 phrases: List[str],
                 colors: List[Color],
                 font: int = cv2.FONT_HERSHEY_SIMPLEX,
                 thickness: int = 2,
                 text_color: Color = Color.white(),
                 text_scale: float = 0.5,
                 text_thickness: int = 1,
                 text_padding: int = 10) -> np.ndarray:
        """ 탐지된 객체에 대한 정보를 이미지에 그립니다.
        Args:
            image_source (np.ndarray): 원본 이미지 배열 (shape: HxWxC).
            boxes (torch.Tensor): 바운딩 박스 텐서 (shape: Nx4).
            logits (torch.Tensor): 각 박스의 신뢰도 점수 (shape: N).
            phrases (List[str]): 각 박스에 대응하는 설명 텍스트.
            colors (List[Color]): 각 박스의 테두리 색상.
            font (int): 텍스트 폰트.
            thickness (int): 테두리 두께.
            text_color (Color): 텍스트 색상.
            text_scale (float): 텍스트 크기.
            text_thickness (int): 텍스트 두께.
            text_padding (int): 텍스트 패딩.

        Returns:
            np.ndarray: 정보가 추가된 이미지.
        """
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        detections = sv.Detections(xyxy=xyxy, confidence=logits.numpy())

        labels = [
            f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)
        ]

        scene = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
        for i, (xyxy, confidence) in enumerate(
                zip(detections.xyxy, detections.confidence)):
            color = colors[i]

            x1, y1, x2, y2 = xyxy.astype(int)
            cv2.rectangle(
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=thickness,
            )

            text = (f"{confidence:0.2f}" if
                    (labels is None or
                     len(detections) != len(labels)) else labels[i])

            text_width, text_height = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=text_scale,
                thickness=text_thickness,
            )[0]

            text_x = x1 + text_padding
            text_y = y1 - text_padding

            text_background_x1 = x1
            text_background_y1 = y1 - 2 * text_padding - text_height

            text_background_x2 = x1 + 2 * text_padding + text_width
            text_background_y2 = y1

            cv2.rectangle(
                img=scene,
                pt1=(text_background_x1, text_background_y1),
                pt2=(text_background_x2, text_background_y2),
                color=color.as_bgr(),
                thickness=cv2.FILLED,
            )
            cv2.putText(
                img=scene,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=text_scale,
                color=text_color.as_bgr(),
                thickness=text_thickness,
                lineType=cv2.LINE_AA,
            )
        scene = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)

        return scene

    def run(
        self,
        np_image: np.ndarray,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[np.ndarray]]:
        """ 이미지 경로와 텍스트 프롬프트를 사용하여, 객체 탐지 및 세그먼트 과정을 실행하고, 결과 이미지를 생성

        Args:
            np_image (np.ndarray): 원본 이미지.
            objects_prompt (str): 탐지를 위한 텍스트 프롬프트.
            det_model (torch.nn.Module): 객체 탐지 모델.
            seg_model (torch.nn.Module): 세그먼트 모델.
            color_map (Callable[[str, object_prompt.ObjectsPrompt], Color]):
                탐지된 객체의 문구에 따라 색상을 반환하는 함수.
            device (torch.device): 사용할 디바이스.
            box_threshold (float): 바운딩 박스 임계값.
            text_threshold (float): 텍스트 임계값.

        Returns:
            bounding_boxes_per_class: List[torch.Tensor] # boxes: (n, 4)
            seg_masks_per_class: List[torch.Tensor] # mask: (N, 1, H, W)
            img_segdet_per_class: List[np.ndarray]
                - 객체 탐지 및 세그먼트 결과 이미지.
        """
        # TODO: image_fpath가 directory이면, batch 이미지를 불러오는지 확인
        # object_names_list: ["person", "wall", ... ]
        # det_results_per_class: List[ Tuple[torch.Tensor, torch.Tensor, List[str]] ]
        # seg_masks_per_class: List[torch.Tensor]
        det_results_per_class, seg_masks_per_class = self.inference(
            np_image, box_threshold, text_threshold)
        img_segdet_per_class = []
        bounding_boxes_per_class: List[np.ndarray] = [
            det_results[0] for det_results in det_results_per_class
        ]
        for det_results, seg_masks in zip(det_results_per_class,
                                          seg_masks_per_class):
            # det_results: Tuple[torch.Tensor, torch.Tensor, List[str]]
            # det_results: (det_boxes, det_logits, det_phrases)
            # boxes: (n, 4)
            # logits: (n,)
            # phrases: List[str]
            # seg_masks: torch.Tensor # # mask: (N, 1, H, W)
            # TODO: cp -> seg_masks를 (H, W) 형태로 출력
            _, img_segdet = self.draw(np_image.copy(), det_results, seg_masks)
            img_segdet_per_class.append(img_segdet)
        return bounding_boxes_per_class, seg_masks_per_class, img_segdet_per_class

    def run_on_files(self):
        input_dir = os.path.join(args.input_parent_dir, args.target_folder)
        # 사진 한장을 target한 경우, 에러 뜨게끔 했음
        assert os.path.isdir(input_dir), f"{input_dir} is not a directory."
        # input_frames: /data/test 에 있는 파일들 중, .png 또는 .jpg로 끝나는 파일들
        input_frames = filter(lambda e: e.endswith((".png", ".jpg")),
                              os.listdir(input_dir))
        input_frames = sorted(input_frames,
                              key=lambda e: int(os.path.splitext(e)[0]))
        box_thresholds = [0.4]
        # result_dir: /results/test
        result_dir = os.path.join(args.result_parent_dir, args.target_folder)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for box_threshold in tqdm(box_thresholds):
            a_result_dir, result_dir_per_class_list = self.make_a_result_dir(
                box_threshold, result_dir)
            for a_input_frame in tqdm(input_frames):
                # a_input_frame: 0660.jpg
                # input_dir: /data/test
                # input_image_path: /data/test/0660.jpg
                input_image_path = os.path.join(input_dir, a_input_frame)
                # result_frame_name: 0660.jpg
                result_frame_name = f"{os.path.splitext(a_input_frame)[0]}.jpg"
                # np_image: rgb order
                np_image, _ = load_image(input_image_path)

                (bounding_boxes_per_class, seg_masks_per_class,
                 img_segdet_per_class) = self.run(
                     np_image=np_image,  # /data/test/0660.jpg
                     box_threshold=box_threshold,
                 )
                if args.save_result:
                    self.save_result_image(result_dir_per_class_list,
                                           img_segdet_per_class,
                                           result_frame_name)

            if args.save_result_gif:
                self.save_result_gif(result_dir_per_class_list, a_result_dir)

    def run_on_realtime(
        self,
        np_image: np.ndarray,
        box_threshold: float = 0.4
    ) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
        """ 실시간 이미지를 처리합니다.

        Args:
            np_image (np.ndarray): 원본 이미지.
            box_threshold (float): 바운딩 박스 임계값.

        Returns:
            bounding_boxes_per_class: List[np.ndarray]
                - 클래스 별로, 바운딩 박스를 담은 리스트. (n, 4)
            seg_masks_per_class: List[torch.Tensor]
                - 클래스 별로, 세그먼트 마스크를 담은 리스트. (N, 1, H, W)

        """
        (bounding_boxes_per_class, seg_masks_per_class,
         img_segdet_per_class) = self.run(
             np_image=np_image,  # /data/test/0660.jpg
             box_threshold=box_threshold,
         )
        return bounding_boxes_per_class, seg_masks_per_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # if target_folder is empty, it means it gets a real-time image.
    parser.add_argument("--target_folder", default="")
    parser.add_argument("--save_result", type=bool, default=True)
    parser.add_argument("--save_result_gif", type=bool, default=False)
    parser.add_argument("--experiment_folder", default="")
    parser.add_argument("--input_parent_dir", default="data")
    parser.add_argument("--result_parent_dir", default="results")
    parser.add_argument("--output_results_segdet_dir", default="results-segdet")
    parser.add_argument(
        'checkpoint',
        choices=['b', 'l', 'h'],
        help="Big model(636M), Large model(308M), Huge model(91M).")
    args = parser.parse_args()
    if args.checkpoint == "b":
        args.checkpoint = "sam_vit_b_01ec64.pth"
    elif args.checkpoint == "l":
        args.checkpoint = "sam_vit_l_0b3195.pth"
    elif args.checkpoint == "h":
        args.checkpoint = "sam_vit_h_4b8939.pth"
    return args


if __name__ == "__main__":
    args = parse_args()
    app = ImageProcessingApp(args)
    if args.target_folder:
        app.run_on_files()
    else:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        app.run_on_realtime(image)
