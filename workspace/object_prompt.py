import dataclasses
from supervision.draw.color import Color
from typing import List, Optional


@dataclasses.dataclass
class ObjectPrompt:
    prompt: str
    color_str: str
    box_threshold: float = 0.3
    rank: Optional[int] = None
    color: Optional[Color] = None

    def str_to_color(self, color_str: str) -> Color:
        color_str = color_str.lower()
        if color_str == "red":
            return Color.red()
        elif color_str == "green":
            return Color.green()
        elif color_str == "blue":
            return Color.blue()
        elif color_str == "white":
            return Color.white()
        elif color_str == "black":
            return Color.black()
        elif color_str == "yellow":
            return Color(r=255, g=255, b=0)
        elif color_str == "purple":
            return Color(r=128, g=0, b=128)
        else:
            raise ValueError(f"Invalid color string: {color_str}")

    def __post_init__(self):
        self.color = self.str_to_color(self.color_str)


@dataclasses.dataclass
class ObjectsPrompt:
    objects: List[ObjectPrompt]

    def get_name_list(self) -> List[str]:
        return [obj.prompt for obj in self.objects]

    def get_box_thresholds(self) -> List[float]:
        return [obj.box_threshold for obj in self.objects]

    def get_rank_list(self) -> List[Optional[int]]:
        return [obj.rank for obj in self.objects]
