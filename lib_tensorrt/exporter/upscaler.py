import os.path
import warnings

import tensorrt as trt
import torch
from lib_tensorrt import logger
from lib_tensorrt.paths import GAN_ONNX, GAN_TRT

from backend import memory_management
from modules.modelloader import load_spandrel_model
from modules.shared import opts


@torch.inference_mode()
def export_upscaler(path: os.PathLike, opt: int, its: int):
    dim = int(getattr(opts, "ESRGAN_tile", 512) or 512)
    dtype = "fp32"  # TODO
    logger.info(f'Baking {dim}x{dim} Engine ({dtype}) for "{os.path.basename(path)}"')

    _base = os.path.splitext(os.path.basename(path))[0] + f"-{dim}-{dtype}"

    onnx_path: os.PathLike = os.path.join(GAN_ONNX, _base + ".onnx")
    trt_path: os.PathLike = os.path.join(GAN_TRT, _base + ".trt")

    if not os.path.isfile(onnx_path):
        _build_onnx(path, dim, onnx_path)

    if not os.path.isfile(trt_path):
        _build_trt(dim, onnx_path, trt_path, opt, its)
    else:
        logger.warning(f'Engine "{os.path.basename(trt_path)}" already exists...')


def _build_onnx(path: str, dim: int, onnx_path: str):
    memory_management.soft_empty_cache()

    descriptor = load_spandrel_model(path, memory_management.cpu, prefer_half=False)
    model: torch.nn.Module = descriptor.model

    inputs: torch.Tensor = torch.zeros(
        (1, 3, dim, dim),
        dtype=torch.float32,
        device=memory_management.cpu,
    )

    logger.info("> Exporting .onnx Model...")

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

    logger.info("> Success!")


def _build_trt(dim: int, onnx_path: str, trt_path: str, opt: int, its: int):
    memory_management.soft_empty_cache()

    trt_logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    config.builder_optimization_level = opt
    config.avg_timing_iterations = its

    parser = trt.OnnxParser(network, trt_logger)
    success = parser.parse_from_file(onnx_path)
    if not success:
        logger.error(f"[TRT] Failed to parse the Onnx Model...")
        return

    logger.info("> Exporting .trt Model...")

    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, dim, dim), (1, 3, dim, dim), (1, 3, dim, dim))
    config.add_optimization_profile(profile)
    serialized_engine = builder.build_serialized_network(network, config)

    with open(trt_path, "wb") as f:
        f.write(serialized_engine)

    logger.info("> Success!")
