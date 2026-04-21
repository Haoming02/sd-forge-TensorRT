import os.path
import warnings

import tensorrt as trt
import torch
from lib_tensorrt import logger
from lib_tensorrt.paths import GAN_ONNX, GAN_TRT

from backend import memory_management
from modules.modelloader import load_spandrel_model
from modules.shared import opts


@torch.no_grad()
def export_upscaler(path: os.PathLike):
    dim = int(getattr(opts, "ESRGAN_tile", 512) or 512)
    logger.info(f'Baking {dim}x{dim} Engine for "{os.path.basename(path)}"')

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

    logger.info("Exporting .onnx")

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            model.cuda(),
            inputs.cuda(),
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=20,
            external_data=False,
            export_params=True,
            verbose=False,
        )

    memory_management.soft_empty_cache()

    trt_logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(0)
    config = builder.create_builder_config()

    parser = trt.OnnxParser(network, trt_logger)
    success = parser.parse_from_file(onnx_path)
    if not success:
        logger.error(f"[TRT] Failed to parse the Onnx Model...")
        return

    logger.info("Exporting .trt")

    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, dim, dim), (1, 3, dim, dim), (1, 3, dim, dim))
    config.add_optimization_profile(profile)
    serialized_engine = builder.build_serialized_network(network, config)

    with open(trt_path, "wb") as f:
        f.write(serialized_engine)

    memory_management.soft_empty_cache()
    logger.info("Success!")
