import os.path
import warnings

import torch
from lib_tensorrt import logger
from lib_tensorrt.paths import GAN_ONNX, GAN_TRT
from lib_tensorrt.utilities import Engine

from backend import memory_management
from modules.modelloader import load_spandrel_model
from modules.shared import opts


@torch.no_grad()
def export_upscaler(path: os.PathLike):
    dim = int(getattr(opts, "ESRGAN_tile", 512) or 512)
    logger.info(f"Baking {dim}x{dim} Engine")

    _base = os.path.splitext(os.path.basename(path))[0] + f"-{dim}"

    onnx_path: os.PathLike = os.path.join(GAN_ONNX, _base + ".onnx")
    trt_path: os.PathLike = os.path.join(GAN_TRT, _base + ".trt")

    memory_management.soft_empty_cache()

    descriptor = load_spandrel_model(path, memory_management.cpu, prefer_half=False)
    model: torch.nn.Module = descriptor.model

    inputs: torch.Tensor = torch.ones(
        (1, 3, dim, dim),
        dtype=torch.float32,
        device=memory_management.cpu,
    ).mul_(0.5)

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            model.cuda(),
            inputs.cuda(),
            onnx_path,
            verbose=True,
            input_names=["input"],
            output_names=["output"],
            opset_version=20,
            external_data=False,
            export_params=True,
        )

    memory_management.soft_empty_cache()

    engine = Engine(trt_path)

    engine.build(
        onnx_path,
        input_profile=[
            {"input": [(1, 3, dim, dim), (1, 3, dim, dim), (1, 3, dim, dim)]},
        ],
    )

    memory_management.soft_empty_cache()
    logger.info("Success!")
