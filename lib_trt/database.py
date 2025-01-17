from dataclasses import dataclass
import json
import os

from lib_trt.utils import DATABASE, OUTPUT_DIR
from lib_trt.logging import logger


@dataclass
class UNetData:
    filename: str
    min_width: int
    max_width: int
    min_height: int
    max_height: int

    def memory_requirement(self) -> int:
        return 6 * 1024 * self.min_width * self.min_height

    def in_range(self, w: int, h: int) -> bool:
        return (
            self.min_width <= w <= self.max_width
            and self.min_height <= h <= self.max_height
        )

    def serialize(self, family: str) -> dict[str, str | int]:
        return {
            "family": family,
            "filename": self.filename,
            "min_width": self.min_width,
            "max_width": self.max_width,
            "min_height": self.min_height,
            "max_height": self.max_height,
        }


class TensorRTDatabase:
    database: dict[str, list[UNetData]] = {}

    @classmethod
    def load(cls) -> dict[str, list[UNetData]]:
        """Load the JSON database from disk"""

        if not os.path.isfile(DATABASE):
            return

        try:
            with open(DATABASE, "r", encoding="utf-8") as db:
                data: list[dict[str, str | int]] = json.load(db)
        except json.JSONDecodeError:
            logger.error("Failed to read database...")
            return

        registered_paths = set()

        for obj in data:
            family: str = obj.pop("family")
            unet = UNetData(**obj)

            if not os.path.isfile(os.path.join(OUTPUT_DIR, f"{unet.filename}.trt")):
                logger.warning(f'Engine "{unet.filename}" does not exist...')
                continue

            if unet.filename in registered_paths:
                logger.error(f'Entry "{unet.filename}" was duplicated...')
                continue

            registered_paths.add(unet.filename)
            if family not in cls.database:
                cls.database[family] = []
            cls.database[family].append(unet)

        logger.debug(cls.database)

    @classmethod
    def delete(cls, family: str, filename: str):
        data = []

        for fam, unets in cls.database.items():
            for unet in unets:
                if fam != family or unet.filename != filename:
                    data.append(unet.serialize(fam))

        with open(DATABASE, "w+", encoding="utf-8") as db:
            json.dump(data, db)

    @classmethod
    def save(cls, family: str, kwargs: dict):
        unet = UNetData(**kwargs)

        if family not in cls.database:
            cls.database[family] = []
        cls.database[family].append(unet)

        data = []

        for fam, unets in cls.database.items():
            for unet in unets:
                data.append(unet.serialize(fam))

        with open(DATABASE, "w+", encoding="utf-8") as db:
            json.dump(data, db)

    @classmethod
    def get_family(cls, name: str) -> str:
        for fam, unets in cls.database.items():
            for unet in unets:
                if unet.filename == name:
                    return fam

        return None

    @classmethod
    def get_suitable(cls, family: str, w: int, h: int) -> str:
        if not family in cls.database:
            return None

        for unet in cls.database[family]:
            if unet.in_range(w, h):
                return f"[TRT] {unet.filename}"

        return None

    @classmethod
    def get_memory(cls, family: str, filename: str) -> int:
        for unet in cls.database[family]:
            if unet.filename == filename:
                return unet.memory_requirement()
