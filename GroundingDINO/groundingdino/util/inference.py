from typing import Tuple, List

import re
import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from torchvision.ops import box_convert

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap


# ----------------------------------------------------------------------------------------------------------------------
# OLD API
# ----------------------------------------------------------------------------------------------------------------------


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def transform_image(image: np.array) -> torch.Tensor:
    transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    image_source = Image.fromarray(image)
    image_transformed, _ = transform(image_source, None)
    return image_transformed


def predict(
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    caption_ = preprocess_caption(caption=caption)
    model = model.to(device)
    # TODO: batch image 연산 :
    #      images = torch.stack(images_list)
    #
    image = image.to(device)

    with torch.no_grad():
        # caption: " person, wall, door, floor. "
        # captions = [caption] = [" person, wall, door, floor. "]
        """
        batch 연산을 하고싶으면
        image_: (batch_size, 3, H, W)
        captions: List[str] = [caption_] -> len(captions) = batch_size
            - 여기서 caption_ : " person. "

        하나의 사진에 여러개의 caption(n개)을 한번에 연산하고 싶으면

        image_ -> image 를 p 번 복사
        captions: List[str] = [ "person.", "wall.", "door.", "floor." ]

        outputs["pred_logits"]: (p, 900, 256)
        outputs["pred_boxes"]: (p, 900, 4)
        for _ in range(p):
            prediction_logits: (900, 256)
            prediction_boxes: (900, 4)
            mask: (900,)
            logits: (n, 256) (n=2)
            boxes: (n, 4) (n=2)
            det_logits: (n,)
            phrases: List[str] = [caption for _ in range(logits_number)] -> len(phrases) = n

        boxes_per_caption: (p, n, 4)
        det_logits_per_caption: (p, n)
        phrases_per_caption: List[List[str]] = [phrases for _ in range(p)] -> len(phrases_per_caption) = p
        """
        image_ = image[None]
        outputs = model(image_, captions=[caption_])
        """ non-batch 연산의 경우
        outputs["pred_logits"].cpu().shape: (1, 900, 256) # 1 = 사진 장수
        outputs["pred_boxes"].cpu().shape: (1, 900, 4)
        prediction_logits: (900, 256)
        prediction_boxes: (900, 4)
        mask: (900,)
        logits: (n, 256) (n=2) # 900개 중, threshold를 넘는 갯수가 n개
        boxes: (n, 4) (n=2)
        """
    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)
    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    # caption: " person, wall, door, floor. "
    """
여기서 사용된 토크나이저는 대부분의 트랜스포머 기반 모델 (BERT, GPT 등)에서 널리 사용되는 구조
    input_ids:
        input_ids는 입력 텍스트를 토크나이저가 이해할 수 있는 숫자 시퀀스로 변환한 결과
        각 숫자는 토크나이저의 어휘 사전에 정의된 특정 단어나 토큰
        예를 들어, input_ids가 [101, 2711, 102]라는 것은
            첫 번째와 마지막 토큰이 특수 토큰이며 (101은 보통 BERT에서 [CLS] 토큰을, 102는 [SEP] 토큰을 나타냄), 
            2711은 "person"이라는 단어에 해당하는 토큰
            참고 1010: , (쉼표)

    token_type_ids:
        token_type_ids는 입력 토큰이 속하는 세그먼트를 구분하는 데 사용됩니다. 
        이는 주로 두 가지 또는 그 이상의 서로 다른 문장을 구분하는데 사용되며, 
            예를 들어 BERT의 경우 두 개의 문장을 입력으로 넣을 때 첫 번째 문장은 0, 두 번째 문장은 1로 구분됩니다.
        여기서는 모든 값이 0으로, 하나의 문장만 처리되었음을 나타냅니다.

    attention_mask:
        attention_mask는 토큰이 실제 데이터인지, 아니면 패딩을 위해 추가된 토큰인지를 구분하는 데 사용
        트랜스포머 모델에서는 입력 데이터의 길이를 동일하게 맞추기 위해 짧은 입력에 패딩을 추가할 수 있습니다.
        이 마스크에서 1은 해당 토큰이 실제 데이터임을, 0은 패딩된 토큰임을 나타냅니다. 
        여기서 모든 값이 1이므로, 모든 입력 토큰이 유효한 데이터임을 의미합니다.
    """
    det_logits = logits.max(dim=1)[0]  # det_logits.shape = (n,) 신뢰도 점수를 의미

    tokenized = tokenizer(caption_)
    # TODO: check
    # phrases = [
    #     get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
    #     for logit
    #     in logits
    # ]
    logits_number = logits.shape[0]
    phrases = [caption for _ in range(logits_number)]
    return boxes, det_logits, phrases


def batch_predict(
        model,
        image: torch.Tensor,
        captions: List[str],
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
    captions_ = [preprocess_caption(caption=caption) for caption in captions]
    model = model.to(device)
    # TODO: batch image 연산 :
    #      images = torch.stack(images_list)
    #
    image = image.to(device)

    with torch.no_grad():
        # caption: " person, wall, door, floor. "
        # captions = [caption] = [" person, wall, door, floor. "]
        """
        batch 연산을 하고싶으면
        image_: (batch_size, 3, H, W)
        captions: List[str] = [caption_] -> len(captions) = batch_size
            - 여기서 caption_ : " person. "

        하나의 사진에 여러개의 caption(n개)을 한번에 연산하고 싶으면

        image_ -> image 를 p 번 복사
        captions: List[str] = [ "person.", "wall.", "door.", "floor." ]

        outputs["pred_logits"]: (p, 900, 256)
        outputs["pred_boxes"]: (p, 900, 4)
        for _ in range(p):
            prediction_logits: (900, 256)
            prediction_boxes: (900, 4)
            mask: (900,)
            logits: (n, 256) (n=2)
            boxes: (n, 4) (n=2)
            det_logits: (n,)
            phrases: List[str] = [caption for _ in range(logits_number)] -> len(phrases) = n

        boxes_per_caption: List[(n, 4)] -> len(boxes_per_caption) = p
        det_logits_per_caption: List[(n,)] -> len(det_logits_per_caption) = p
        phrases_per_caption: List[List[str]] = [phrases for _ in range(p)] -> len(phrases_per_caption) = p
        """
        boxes_per_caption = []
        det_logits_per_caption = []
        phrases_per_caption = []
        captions_number = len(captions_)  # p
        # image: (3, H, W) -> image_: (1, 3, H, W) -> image_: (p, 3, H, W)
        image_ = image[None].repeat(captions_number, 1, 1, 1)
        outputs = model(image_, captions=captions_)
        pred_logits = outputs["pred_logits"].cpu().sigmoid()  # pred_logits.shape = (p, 900, 256)
        pred_boxes = outputs["pred_boxes"].cpu()
        for caption_idx, caption in enumerate(captions):
            prediction_logits = pred_logits[caption_idx]  # prediction_logits.shape = (900, 256)
            prediction_boxes = pred_boxes[caption_idx]  # prediction_boxes.shape = (900, 4)
            mask = prediction_logits.max(dim=1)[0] > box_threshold  # mask.shape = (900,)
            logits = prediction_logits[mask]  # logits.shape = (n, 256)
            boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)
            det_logits = logits.max(dim=1)[0]  # det_logits.shape = (n,) 신뢰도 점수를 의미
            logits_number = logits.shape[0]
            phrases = [caption for _ in range(logits_number)]
            boxes_per_caption.append(boxes)
            det_logits_per_caption.append(det_logits)
            phrases_per_caption.append(phrases)

    return boxes_per_caption, det_logits_per_caption, phrases_per_caption


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


# ----------------------------------------------------------------------------------------------------------------------
# NEW API
# ----------------------------------------------------------------------------------------------------------------------


class Model:

    def __init__(
            self,
            model_config_path: str,
            model_checkpoint_path: str,
            device: str = "cuda"
    ):
        self.model = load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=device
        ).to(device)
        self.device = device

    def predict_with_caption(
            self,
            image: np.ndarray,
            caption: str,
            box_threshold: float = 0.35,
            text_threshold: float = 0.25
    ) -> Tuple[sv.Detections, List[str]]:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections, labels = model.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        """
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        return detections, phrases

    def predict_with_classes(
            self,
            image: np.ndarray,
            classes: List[str],
            box_threshold: float,
            text_threshold: float
    ) -> sv.Detections:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections = model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )


        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        """
        caption = ". ".join(classes)
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        class_id = Model.phrases2classes(phrases=phrases, classes=classes)
        detections.class_id = class_id
        return detections

    @staticmethod
    def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pillow, None)
        return image_transformed

    @staticmethod
    def post_process_result(
            source_h: int,
            source_w: int,
            boxes: torch.Tensor,
            logits: torch.Tensor
    ) -> sv.Detections:
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.numpy()
        return sv.Detections(xyxy=xyxy, confidence=confidence)

    @staticmethod
    def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
        class_ids = []
        for phrase in phrases:
            try:
                # class_ids.append(classes.index(phrase))
                class_ids.append(Model.find_index(phrase, classes))
            except ValueError:
                class_ids.append(None)
        return np.array(class_ids)

    @staticmethod
    def find_index(string, lst):
        # if meet string like "lake river" will only keep "lake"
        # this is an hack implementation for visualization which will be updated in the future
        string = string.lower().split()[0]
        for i, s in enumerate(lst):
            if string in s.lower():
                return i
        print(
            "There's a wrong phrase happen, this is because of our post-process merged wrong tokens, which will be modified in the future. We will assign it with a random label at this time.")
        return 0