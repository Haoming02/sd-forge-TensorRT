from lib_trt.utils import DATABASE
from dataclasses import dataclass
import json
import os


@dataclass
class UNetData:
    name: str
    min_width: int
    max_width: int
    min_height: int
    max_height: int

    def in_range(self, w: int, h: int) -> bool:
        return (
            self.min_width <= w <= self.max_width
            and self.min_height <= h <= self.max_height
        )

    def save(self) -> dict:
        return {
            "name": self.name,
            "min_width": self.min_width,
            "max_width": self.max_width,
            "min_height": self.min_height,
            "max_height": self.max_height,
        }


class TensorRTDatabase:
    database: list[UNetData] = []

    @classmethod
    def serialize(cls):
        data = []
        for unet in cls.database:
            data.append(unet.save())

        with open(DATABASE, "w+", encoding="utf-8") as db:
            json.dump(data, db)

    @classmethod
    def deserialize(cls):
        if not os.path.isfile(DATABASE):
            return

        with open(DATABASE, "r", encoding="utf-8") as db:
            data = json.load(db)

        for obj in data:
            cls.database.append(UNetData(**obj))
