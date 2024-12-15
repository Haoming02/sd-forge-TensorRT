from ldm_patched.modules import model_management
from modules_forge.unet_patcher import UnetPatcher
from modules.script_callbacks import on_ui_tabs
from modules import shared

import tensorrt as trt
import gradio as gr
import torch
import os

from lib_trt.utils import TIMING_CACHE, TEMP_DIR, OUTPUT_DIR
from lib_trt.tqdm import TQDMProgressMonitor
from lib_trt.logging import logger


class TensorRTConverter:

    @staticmethod
    def setup_timing_cache(config: trt.IBuilderConfig):
        """Sets up the builder to use the timing cache file, and creates it if it does not already exist"""

        if os.path.exists(TIMING_CACHE):
            with open(TIMING_CACHE, mode="rb") as timing_cache_file:
                buffer = timing_cache_file.read()
                logger.debug(f"Read {len(buffer)} bytes from timing cache")
        else:
            buffer = b""
            logger.debug("No timing cache found; Initializing a new one.")

        timing_cache: trt.ITimingCache = config.create_timing_cache(buffer)
        config.set_timing_cache(timing_cache, ignore_mismatch=True)

    @staticmethod
    def save_timing_cache(config: trt.IBuilderConfig):
        """Saves the config's timing cache to file"""
        timing_cache: trt.ITimingCache = config.get_timing_cache()
        with open(TIMING_CACHE, "wb") as timing_cache_file:
            timing_cache_file.write(memoryview(timing_cache.serialize()))

    @staticmethod
    def convert(
        model: UnetPatcher,
        filename: str,
        batch_size_min: int,
        batch_size_opt: int,
        batch_size_max: int,
        width_min: int,
        width_opt: int,
        width_max: int,
        height_min: int,
        height_opt: int,
        height_max: int,
        context_min: int,
        context_opt: int,
        context_max: int,
    ) -> bool:

        model_management.unload_all_models()
        model_management.load_models_gpu([model])
        unet = model.model.diffusion_model

        context_dim = model.model.model_config.unet_config.get("context_dim", None)
        if context_dim is None:
            logger.error("Model is not supported...")
            return False

        context_len = 77
        y_dim = model.model.adm_channels
        extra_input = {}
        dtype = torch.float16

        input_names = ["x", "timesteps", "context"]
        output_names = ["h"]

        dynamic_axes = {
            "x": {0: "batch", 2: "height", 3: "width"},
            "timesteps": {0: "batch"},
            "context": {0: "batch", 1: "num_embeds"},
        }

        transformer_options = model.model_options["transformer_options"].copy()

        class UNET(torch.nn.Module):
            def forward(self, x, timesteps, context, *args):
                extras = input_names[3:]
                extra_args = {}
                for i in range(len(extras)):
                    extra_args[extras[i]] = args[i]
                return self.unet(
                    x,
                    timesteps,
                    context,
                    transformer_options=self.transformer_options,
                    **extra_args,
                )

        _unet = UNET()
        _unet.unet = unet
        _unet.transformer_options = transformer_options
        unet = _unet

        input_channels = model.model.model_config.unet_config.get("in_channels", 4)

        inputs_shapes_min = (
            (batch_size_min, input_channels, height_min // 8, width_min // 8),
            (batch_size_min,),
            (batch_size_min, context_len * context_min, context_dim),
        )
        inputs_shapes_opt = (
            (batch_size_opt, input_channels, height_opt // 8, width_opt // 8),
            (batch_size_opt,),
            (batch_size_opt, context_len * context_opt, context_dim),
        )
        inputs_shapes_max = (
            (batch_size_max, input_channels, height_max // 8, width_max // 8),
            (batch_size_max,),
            (batch_size_max, context_len * context_max, context_dim),
        )

        if y_dim > 0:
            input_names.append("y")
            dynamic_axes["y"] = {0: "batch"}
            inputs_shapes_min += ((batch_size_min, y_dim),)
            inputs_shapes_opt += ((batch_size_opt, y_dim),)
            inputs_shapes_max += ((batch_size_max, y_dim),)

        for k in extra_input:
            input_names.append(k)
            dynamic_axes[k] = {0: "batch"}
            inputs_shapes_min += ((batch_size_min,) + extra_input[k],)
            inputs_shapes_opt += ((batch_size_opt,) + extra_input[k],)
            inputs_shapes_max += ((batch_size_max,) + extra_input[k],)

        device = model_management.get_torch_device()
        inputs = tuple(
            [
                torch.randn(shape, dtype=dtype, device=device)
                for shape in inputs_shapes_opt
            ]
        )

        output_onnx = os.path.join(os.path.join(TEMP_DIR, filename), "model.onnx")
        os.makedirs(os.path.dirname(output_onnx), exist_ok=True)

        if not os.path.isfile(output_onnx):
            torch.onnx.export(
                unet,
                inputs,
                output_onnx,
                verbose=False,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=17,
            )

        model_management.unload_all_models()
        model_management.soft_empty_cache()

        # TRT conversion starts here
        trt_logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(trt_logger)

        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, trt_logger)
        success = parser.parse_from_file(output_onnx)

        if not success:
            logger.error("Failed to load the Onnx Model:")
            for idx in range(parser.num_errors):
                print(parser.get_error(idx))
            return False

        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        TensorRTConverter.setup_timing_cache(config)
        config.progress_monitor = TQDMProgressMonitor()

        prefix_encode = ""
        for k in range(len(input_names)):
            min_shape = inputs_shapes_min[k]
            opt_shape = inputs_shapes_opt[k]
            max_shape = inputs_shapes_max[k]
            profile.set_shape(input_names[k], min_shape, opt_shape, max_shape)

            # Encode shapes to filename
            encode = lambda a: ".".join(map(lambda x: str(x), a))
            prefix_encode += "{}#{}#{}#{};".format(
                input_names[k], encode(min_shape), encode(opt_shape), encode(max_shape)
            )

        assert dtype == torch.float16
        config.set_flag(trt.BuilderFlag.FP16)
        config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)

        output_trt = os.path.join(OUTPUT_DIR, f"{filename}.trt")
        os.makedirs(os.path.dirname(output_trt), exist_ok=True)
        with open(output_trt, "wb") as f:
            f.write(serialized_engine)

        TensorRTConverter.save_timing_cache(config)


def _validate(*args: list[int]) -> bool:
    if (
        shared.sd_model.forge_objects.unet.model.diffusion_model.dtype
        is not torch.float16
    ):
        logger.error("Only fp16 precision UNet is supported...")
        return False

    if all([args[i * 3 + 0] <= args[i * 3 + 1] <= args[i * 3 + 2] for i in range(4)]):
        return True

    logger.error("Invalid Value Range(s)...")
    return False


def on_convert(*args: list[int]):
    if not _validate(*args):
        return

    ckpt = shared.sd_model.sd_model_checkpoint
    TensorRTConverter.convert(
        shared.sd_model.forge_objects.unet,
        os.path.splitext(os.path.basename(ckpt))[0],
        *args,
    )


class Sliders:

    @staticmethod
    def batch(var: str) -> gr.Slider:
        return gr.Slider(
            label=var,
            minimum=1,
            maximum=8,
            value=1,
            step=1,
        )

    @staticmethod
    def dim(var: str, val: int) -> gr.Slider:
        return gr.Slider(
            label=var,
            minimum=256,
            maximum=2048,
            value=val,
            step=64,
        )

    @staticmethod
    def context(var: str) -> gr.Slider:
        return gr.Slider(
            label=var,
            minimum=1,
            maximum=8,
            value=2,
            step=1,
        )


def trt_ui():
    with gr.Blocks() as TRT:
        with gr.Row():
            with gr.Group(elem_id="trt_sliders"):
                args = []
                gr.HTML('<p align="center">Batch Size</p>')
                with gr.Row():
                    args.append(Sliders.batch("Min"))
                    args.append(Sliders.batch("Opt"))
                    args.append(Sliders.batch("Max"))
                with gr.Row():
                    with gr.Column():
                        gr.HTML('<p align="center">Width</p>')
                        args.append(Sliders.dim("Min", 896))
                        args.append(Sliders.dim("Opt", 1024))
                        args.append(Sliders.dim("Max", 1152))
                    with gr.Column():
                        gr.HTML('<p align="center">Height</p>')
                        args.append(Sliders.dim("Min", 896))
                        args.append(Sliders.dim("Opt", 1024))
                        args.append(Sliders.dim("Max", 1152))
                gr.HTML('<p align="center">Context Length</p>')
                with gr.Row():
                    args.append(Sliders.context("Min"))
                    args.append(Sliders.context("Opt"))
                    args.append(Sliders.context("Max"))

                btn = gr.Button("Convert Engine", variant="primary")
                btn.click(fn=on_convert, inputs=args)

                for comp in args:
                    comp.do_not_save_to_config = True

            with gr.Group(elem_id="trt_docs"):
                gr.HTML("Tutorial W.I.P")

    return [(TRT, "TensorRT", "sd-forge-trt")]


on_ui_tabs(trt_ui)
