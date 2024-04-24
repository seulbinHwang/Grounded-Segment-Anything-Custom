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
from groundingdino.util.inference import load_image, predict, batch_predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from segment_anything import SamPredictor, build_sam
import object_prompt
import jps_to_gif


def load_model_hf(repo_id: str,
                  filename: str,
                  ckpt_config_filename: str,
                  device: torch.device
                  ) -> torch.nn.Module:
    """ Hugging Face Hub에서 모델과 설정 파일을 다운로드하고, 모델을 빌드 및 로드

    Args:
        repo_id: 저장소 ID.
        filename: 모델 체크포인트 파일 이름.
        ckpt_config_filename: 설정 파일 이름.
        device: 모델을 로드할 디바이스

    Returns:
        로드된 모델.
    """
    cache_config_file = hf_hub_download(repo_id=repo_id,
                                        filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    args.device = device
    model = build_model(args)

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']),
                                strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


def detect(
    image_tsr: torch.Tensor, text_prompt: List[str], model: torch.nn.Module,
    box_threshold: float, text_threshold: float
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
    """
    이미지와 텍스트 프롬프트를 사용하여 객체 탐지를 수행합니다.

    Args:
        image_tsr: 입력 이미지 (torch Tensor).
        text_prompt: 탐지를 위한 텍스트 프롬프트.
        model: 탐지 모델.
        box_threshold: 바운딩 박스 임계값.
        text_threshold: 텍스트 임계값.

    Returns:
        탐지된 바운딩 박스, 로짓스, 관련 문구.
    """
    boxes_per_caption, all_logits, phrases_per_caption = batch_predict(
        model=model,
        image=image_tsr,
        captions=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold)
    return boxes_per_caption, all_logits, phrases_per_caption


def segment(image: np.ndarray, sam_model: SamPredictor, device: torch.device,
            boxes: torch.Tensor) -> torch.Tensor:
    """
    주어진 이미지와 바운딩 박스를 사용하여 세그먼트를 수행합니다.

    Args:
        image: 원본 이미지 (numpy 배열).
        sam_model: 세그먼트 모델.
        boxes: 바운딩 박스 (torch Tensor).

    Returns:
        세그먼트된 마스크 (torch Tensor).

        self.features, self.interm_features = self.model.image_encoder(input_image)
        self.is_image_set = True
    """
    sam_model.set_image(image)
    H, W, _ = image.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor(
        [W, H, W, H]).to(boxes.device)
    transformed_boxes = sam_model.transform.apply_boxes_torch(
        boxes_xyxy, image.shape[:2]).to(device)
    # transformed_boxes to device
    if transformed_boxes.shape[0] == 0:
        return torch.empty(0, 1, H, W)
    masks, _, _ = sam_model.predict_torch(point_coords=None,
                                          point_labels=None,
                                          boxes=transformed_boxes,
                                          multimask_output=False)
    # mask: (N, 1, H, W)
    return masks.cpu()


def inference(
    img_npy: np.ndarray, img_tsr: torch.Tensor, text_prompt: List[str],
    det_model: torch.nn.Module, seg_model: SamPredictor, device: torch.device,
    box_threshold: float, text_threshold: float
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor, List[str]]],
           List[torch.Tensor]]:
    """
    탐지 및 세그먼트 모델을 사용하여 이미지에서 객체를 탐지하고 세그먼트 마스크를 생성합니다.

    Args:
        img_npy (np.ndarray): 원본 이미지 배열 (shape: HxWxC).
        img_tsr (torch.Tensor): 이미지를 나타내는 텐서 (shape: CxHxW).
        text_prompt (str): 탐지를 위한 텍스트 프롬프트.
        det_model (torch.nn.Module): 객체 탐지 모델.
        seg_model (SamPredictor): 세그먼트 모델.
        box_threshold (float): 바운딩 박스 임계값.
        text_threshold (float): 텍스트 임계값.

    Returns:
        Tuple[Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]], List[torch.Tensor]]:
            객체 탐지 결과와 세그먼트 마스크.
    """
    # Tuple[
    # List[torch.Tensor],
    # List[torch.Tensor],
    # List[List[str]]
    # ]
    det_boxes_per_caption, det_logits_per_caption, det_phrases_per_caption = detect(
        img_tsr,
        text_prompt=text_prompt,
        model=det_model,
        box_threshold=box_threshold,
        text_threshold=text_threshold)
    # det_results_per_caption: List[ Tuple[torch.Tensor, torch.Tensor, List[str]] ]
    det_results_per_caption = list(
        zip(det_boxes_per_caption, det_logits_per_caption,
            det_phrases_per_caption))
    seg_masks_per_caption = []
    for det_boxes in det_boxes_per_caption:
        seg_masks = segment(img_npy, seg_model, device, boxes=det_boxes)
        seg_masks_per_caption.append(seg_masks)

    return det_results_per_caption, seg_masks_per_caption


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
        seg_mask = seg_mask[0]
        h, w = seg_mask.shape[-2:]
        mask_image = seg_mask.reshape(h, w, 1) * np.array(
            [*[e * 255 for e in color.as_rgb()], 1.0]).reshape(1, 1, -1)

        annotated_frame_pil = Image.fromarray(image).convert("RGBA")
        mask_image_pil = Image.fromarray(
            (mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

        image = np.array(
            Image.alpha_composite(annotated_frame_pil,
                                  mask_image_pil).convert("RGBA"))
    return image


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
    """
    탐지된 객체에 대한 정보를 이미지에 그립니다.

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

    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]

    scene = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    for i, (xyxy,
            confidence) in enumerate(zip(detections.xyxy,
                                         detections.confidence)):
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


def draw(img: np.ndarray,
         det_results: Tuple[torch.Tensor, torch.Tensor, List[str]],
         seg_masks: torch.Tensor,
         color_map: Callable[[str, object_prompt.ObjectsPrompt], Color],
         text_prompt: object_prompt.ObjectsPrompt,
         fill: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    객체 탐지 결과와 세그먼트 마스크를 이미지에 그립니다.

    Args:
        img (np.ndarray): 원본 이미지 (shape: HxWxC).
        det_results (Tuple[torch.Tensor, torch.Tensor, List[str]]): 탐지 결과로, 바운딩 박스, 로짓, 문구를 포함.
        seg_masks (torch.Tensor): 세그먼트 마스크 (shape: N, 1, H, W).
        color_map (Callable[[str, object_prompt.ObjectsPrompt], Color]): 탐지된 객체의 문구에 따라 색상을 반환하는 함수.
        text_prompt (object_prompt.ObjectsPrompt): 객체 탐지를 위한 텍스트 프롬프트.
        fill (bool): 이미지를 새로 그릴지 여부 (True면 원본 이미지를 검은색으로 채움).

    Returns:
        Tuple[np.ndarray, np.ndarray]: 세그먼트가 적용된 이미지와 세부 탐지 정보가 추가된 이미지.
    """
    det_boxes_per_caption, det_logits_per_caption, det_phrases_per_caption = det_results
    colors = [color_map(e, text_prompt) for e in det_phrases_per_caption]

    if fill:
        img = np.zeros_like(img)
    if seg_masks.shape[0] > 0:
        img_seg = draw_seg(img, seg_masks, colors)
    else:
        img_seg = img.copy()
    img_segdet = draw_det(img_seg.copy(), det_boxes_per_caption,
                          det_logits_per_caption, det_phrases_per_caption,
                          colors)

    return img_seg, img_segdet


def run(image_fpath: str,
        text_prompt: object_prompt.ObjectsPrompt,
        det_model: torch.nn.Module,
        seg_model: torch.nn.Module,
        color_map: Callable[[str, object_prompt.ObjectsPrompt], Color],
        device: torch.device,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25
       ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    이미지 경로와 텍스트 프롬프트를 사용하여 객체 탐지 및 세그먼트 과정을 실행하고, 결과 이미지를 생성

    Args:
        image_fpath (str): 이미지 파일 경로.
        text_prompt (str): 탐지를 위한 텍스트 프롬프트.
        det_model (torch.nn.Module): 객체 탐지 모델.
        seg_model (torch.nn.Module): 세그먼트 모델.
        color_map (Callable[[str, object_prompt.ObjectsPrompt], Color]): 탐지된 객체의 문구에 따라 색상을 반환하는 함수.
        box_threshold (float): 바운딩 박스 임계값.
        text_threshold (float): 텍스트 임계값.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 원본과 처리된 이미지.
    """
    # TODO: image_fpath가 directory이면, batch 이미지를 불러오는지 확인

    img_npy, img_tsr = load_image(image_fpath)
    text_prompt_: List[str] = text_prompt.get_name_list()
    # TODO: det_results_per_caption, seg_masks_per_caption
    # det_results_per_caption: List[ Tuple[torch.Tensor, torch.Tensor, List[str]] ]
    # seg_masks_per_caption: List[torch.Tensor]
    det_results_per_caption, seg_masks_per_caption = inference(
        img_npy, img_tsr, text_prompt_, det_model, seg_model, device,
        box_threshold, text_threshold)
    img_seg_per_caption = []
    img_segdet_per_caption = []
    for det_results, seg_masks in zip(det_results_per_caption,
                                      seg_masks_per_caption):
        img_seg, img_segdet = draw(img_npy.copy(), det_results, seg_masks,
                                   color_map, text_prompt)
        img_seg_per_caption.append(img_seg)
        img_segdet_per_caption.append(img_segdet)
    return img_seg_per_caption, img_segdet_per_caption


def color_map(det_phase: str,
              text_prompt: object_prompt.ObjectsPrompt) -> Color:
    """
    탐지된 객체의 문구에 따라 색상을 반환합니다.

    Args:
        det_phase (str): 탐지된 객체의 문구.
        text_prompt (object_prompt.ObjectsPrompt): 객체 탐지를 위한 텍스트 프롬프트.

    Returns:
        Color: 반환할 색상.
    """
    for object_prompt_ in text_prompt.objects:
        if det_phase in object_prompt_.prompt:
            return object_prompt_.color
    raise ValueError(f"Invalid detection phrase: {det_phase}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", required=True)
    # post_fix
    parser.add_argument("-p", "--post_fix", default="")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    groundingdino_model = load_model_hf(
        repo_id="ShilongLiu/GroundingDINO",
        filename="groundingdino_swinb_cogcoor.pth",
        ckpt_config_filename="GroundingDINO_SwinB.cfg.py",
        device=device)

    sam_predictor = SamPredictor(
        build_sam(
            # checkpoint="sam_vit_h_4b8939.pth",  # 636M
            # checkpoint="sam_vit_l_0b3195.pth", # 308M
            checkpoint="sam_vit_b_01ec64.pth",  # 91M
        ).to(device))

    in_rdpath = "data"
    out_rdpath = "results"
    out_rdpath_segdet = "results-segdet"
    person0_prompt = object_prompt.ObjectPrompt(prompt="person",
                                                color_str="blue")
    wall0_prompt = object_prompt.ObjectPrompt(prompt="interior wall",
                                              color_str="green")
    door0_prompt = object_prompt.ObjectPrompt(prompt="door with a handle",
                                              color_str="red")
    door2_prompt = object_prompt.ObjectPrompt(prompt="sliding door",
                                              color_str="red")
    floor0_prompt = object_prompt.ObjectPrompt(prompt="interior floor",
                                               color_str="white")
    glass0_prompt = object_prompt.ObjectPrompt(prompt="glass wall",
                                               color_str="black")
    objects_prompt = object_prompt.ObjectsPrompt(objects=[
        person0_prompt,
        wall0_prompt,
        door0_prompt,
        door2_prompt,
        floor0_prompt,
        glass0_prompt,
    ])
    # person0_prompt = object_prompt.ObjectPrompt(
    #     prompt="person", color_str="blue")
    # wall0_prompt = object_prompt.ObjectPrompt(prompt="wall", color_str="green")
    # door0_prompt = object_prompt.ObjectPrompt(prompt="door", color_str="red")
    # floor0_prompt = object_prompt.ObjectPrompt(prompt="floor", color_str="white")
    # objects_prompt = object_prompt.ObjectsPrompt(
    #     objects=[
    #         person0_prompt,
    #         wall0_prompt,
    #         door0_prompt,
    #         floor0_prompt,
    #     ]
    # )
    # e.g. args.target: /examples/0660.jpg
    # in_path: /data/examples/0660.jpg
    in_path = os.path.join(in_rdpath, args.target)
    # 사진 한장인 경우
    if os.path.isfile(in_path):
        # in_fpath: /data/examples/0660.jpg
        in_fpath = in_path
        # out_dname: data/examples
        out_dname = os.path.dirname(args.target)
        # out_dpath: /results/data/examples
        out_dpath = os.path.join(out_rdpath, out_dname)
        if not os.path.exists(out_dpath):
            os.makedirs(out_dpath)
        # out_dpath_segdet: /results-segdet/data/examples
        out_dpath_segdet = os.path.join(out_rdpath_segdet, out_dname)
        if not os.path.exists(out_dpath_segdet):
            os.makedirs(out_dpath_segdet)
        # os.path.basename: 마지막 요소를 반환
        # os.path.basename(args.target): 0660.jpg
        # os.path.splitext[0]: 0660
        # out_fname: 0660.png
        out_fname = f"{os.path.splitext(os.path.basename(args.target))[0]}.png"
        # out_fpath: /results/data/examples/0660.png
        out_fpath = os.path.join(out_dpath, out_fname)
        # out_fpath_segdet: /results/data/examples/0660.png
        out_fpath_segdet = os.path.join(out_dpath_segdet, out_fname)

        img, img_segdet = run(
            image_fpath=in_fpath,  # /data/examples/0660.jpg
            text_prompt=objects_prompt,
            det_model=groundingdino_model,
            seg_model=sam_predictor,
            color_map=color_map,
            device=device,
        )
        # out_fpath: /results/data/examples/0660.png
        Image.fromarray(img).save(out_fpath)
        # out_fpath_segdet: /results/data/examples/0660.png
        Image.fromarray(img_segdet).save(out_fpath_segdet)
    # e.g. args.target: examples
    # in_path: /data/examples
    # 폴더인 경우
    elif os.path.isdir(in_path):
        # in_dpath: /data/examples
        in_dpath = in_path
        # in_fnames: /data/examples에 있는 파일들 중, .png 또는 .jpg로 끝나는 파일들
        in_fnames = filter(lambda e: e.endswith((".png", ".jpg")),
                           os.listdir(in_dpath))
        sorted_in_fnames = sorted(in_fnames,
                                  key=lambda e: int(os.path.splitext(e)[0]))
        # out_dname: examples
        out_dname = args.target

        box_thresholds = [0.4, 0.5]
        for box_threshold in tqdm(box_thresholds):
            post_fix = args.post_fix + f"_box_{box_threshold}"
            out_dpath = os.path.join(out_rdpath, out_dname + post_fix)
            if not os.path.exists(out_dpath):
                os.makedirs(out_dpath)
            # out_dpath_segdet: /results-segdet/examples
            out_dpath_segdet = os.path.join(out_rdpath_segdet,
                                            out_dname + post_fix)
            if not os.path.exists(out_dpath_segdet):
                os.makedirs(out_dpath_segdet)
            # out_dpath: /results/examples

            # object_prompt: ObjectPrompt
            object_names_list: List[str] = objects_prompt.get_name_list()
            out_depth_per_object: List[str] = []
            out_depth_segdet_per_object: List[str] = []
            for object_prompt_ in object_names_list:
                out_dpath2 = os.path.join(out_dpath, object_prompt_)
                out_depth_per_object.append(out_dpath2)
                if not os.path.exists(out_dpath2):
                    os.makedirs(out_dpath2)
                    print(f"Creating {out_dpath2}...")
                out_dpath_segdet2 = os.path.join(out_dpath_segdet,
                                                 object_prompt_)
                out_depth_segdet_per_object.append(out_dpath_segdet2)
                if not os.path.exists(out_dpath_segdet2):
                    os.makedirs(out_dpath_segdet2)
                    print(f"Creating {out_dpath_segdet2}...")
            for in_fname in tqdm(sorted_in_fnames):
                # in_fname: 0660.jpg
                # in_dpath: /data/examples
                # in_fpath: /data/examples/0660.jpg
                in_fpath = os.path.join(in_dpath, in_fname)
                # out_fname: 0660.png
                out_fname = f"{os.path.splitext(in_fname)[0]}.png"

                # img_seg_per_caption, img_segdet_per_caption
                # box_threshold: [0.3, 0.4, 0.5]
                img_seg_per_caption, img_segdet_per_caption = run(
                    image_fpath=in_fpath,  # /data/examples/0660.jpg
                    text_prompt=objects_prompt,
                    det_model=groundingdino_model,
                    seg_model=sam_predictor,
                    color_map=color_map,
                    device=device,
                    box_threshold=box_threshold,
                )
                for idx, (out_dpath2, out_dpath_segdet2) in enumerate(
                        zip(out_depth_per_object, out_depth_segdet_per_object)):
                    img = img_seg_per_caption[idx]
                    img_segdet = img_segdet_per_caption[idx]
                    # out_fpath: /results/examples/0660.png
                    out_fpath = os.path.join(out_dpath2, out_fname)
                    # out_fpath_segdet: /results-segdet/examples/0660.png
                    out_fpath_segdet = os.path.join(out_dpath_segdet2,
                                                    out_fname)
                    # out_fpath: /results/examples/0660.png
                    # Image.fromarray(img).save(out_fpath)
                    # out_fpath_segdet: /results/examples/0660.png
                    Image.fromarray(img_segdet).save(out_fpath_segdet)
            for idx, (out_dpath2, out_dpath_segdet2) in enumerate(
                    zip(out_depth_per_object, out_depth_segdet_per_object)):
                # save as gif.
                class_name = os.path.basename(out_dpath2)
                output_path = os.path.join(out_dpath_segdet,
                                           class_name + ".gif")
                jps_to_gif.create_gif_from_images(out_dpath_segdet2,
                                                  output_path)

    else:
        raise ValueError(f"There's nothing at {args.target}...!")
