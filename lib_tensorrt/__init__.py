import logging

import torch

from backend.logging import setup_logger

logger = logging.getLogger("ForgeTensorRT")
setup_logger(logger)


def get_dtype(dtype: str) -> torch.dtype:
    match dtype:
        case "fp16":
            return torch.float16
        case "bf16":
            return torch.bfloat16
        case _:
            return torch.float32
