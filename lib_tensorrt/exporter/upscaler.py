import os.path
import warnings

import tensorrt as trt
import torch
from lib_tensorrt import get_dtype, logger
from lib_tensorrt.paths import GAN_ONNX, GAN_TRT

from backend import memory_management
from modules.modelloader import load_spandrel_model
from modules.shared import opts
from modules.timer import Timer

_EXPORT_TIMER = Timer(False)


@torch.inference_mode()
def export_upscaler(path: os.PathLike, opt: int, its: int, half: bool):
    _EXPORT_TIMER.reset()

    dim = int(getattr(opts, "ESRGAN_tile", 512) or 512)
    model, dtype = _load_upscaler(path, half)
    logger.info(f'Baking {dim}x{dim} Engine ({dtype}) for "{os.path.basename(path)}"')

    _base = os.path.splitext(os.path.basename(path))[0] + f"-{dim}-{dtype}"

    onnx_path: os.PathLike = os.path.join(GAN_ONNX, _base + ".onnx")
    trt_path: os.PathLike = os.path.join(GAN_TRT, _base + ".trt")

    if not os.path.isfile(onnx_path):
        try:
            _build_onnx(model, dim, onnx_path, dtype)
        except Exception:
            logger.error(f"[TRT] Failed to export Onnx Model...")
            return

    if not os.path.isfile(trt_path):
        try:
            _build_trt(dim, onnx_path, trt_path, opt, its, dtype)
        except Exception:
            logger.error(f"[TRT] Failed to export TRT Model...")
            return

    else:
        logger.warning(f'Engine "{os.path.basename(trt_path)}" already exists...')
        return

    print("Took: " + _EXPORT_TIMER.summary())


def _load_upscaler(path: str, half: bool) -> tuple[torch.nn.Module, str]:
    memory_management.soft_empty_cache()

    descriptor = load_spandrel_model(path, memory_management.cpu, prefer_half=False)
    dtype = "fp32"

    if half:
        if descriptor.supports_half:
            descriptor.half()
            dtype = "fp16"
        if descriptor.supports_bfloat16:
            descriptor.bfloat16()
            dtype = "bf16"

    _EXPORT_TIMER.record("Load Upscaler")
    return descriptor.model, dtype


def _build_onnx(model: torch.nn.Module, dim: int, onnx_path: str, dtype: str):
    logger.info("> Exporting .onnx Model...")

    inputs: torch.Tensor = torch.zeros(
        (1, 3, dim, dim),
        dtype=get_dtype(dtype),
        device=memory_management.cpu,
    )

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
    _EXPORT_TIMER.record("Export Onnx")


def _build_trt(dim: int, onnx_path: str, trt_path: str, opt: int, its: int, dtype: str):
    memory_management.soft_empty_cache()

    trt_logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    config.builder_optimization_level = opt
    config.avg_timing_iterations = its

    if dtype == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    if dtype == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)

    parser = trt.OnnxParser(network, trt_logger)
    success = parser.parse_from_file(onnx_path)
    assert success

    _EXPORT_TIMER.record("Parse Onnx")
    logger.info("> Exporting .trt Model...")

    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, dim, dim), (1, 3, dim, dim), (1, 3, dim, dim))
    config.add_optimization_profile(profile)
    serialized_engine = builder.build_serialized_network(network, config)

    with open(trt_path, "wb") as f:
        f.write(serialized_engine)

    logger.info("> Success!")
    _EXPORT_TIMER.record("Export TRT")
