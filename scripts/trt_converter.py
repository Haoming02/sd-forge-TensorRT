from ldm_patched.modules import model_management
from modules_forge.unet_patcher import UnetPatcher
from modules.script_callbacks import on_ui_tabs
from modules import shared

import tensorrt as trt
import gradio as gr
import warnings
import torch
import os

from lib_trt.utils import TUTORIAL, TIMING_CACHE, TEMP_DIR, OUTPUT_DIR
from lib_trt.tqdm import TQDMProgressMonitor
from lib_trt.logging import logger


with open(TUTORIAL, "r", encoding="utf-8") as md:
    docs = md.read()


class TensorRTConverter:

    @staticmethod
    def load_timing_cache(config: trt.IBuilderConfig):
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
    def convert(*args: list[int]) -> str:
        logger.info("initializing...")

        err: bool | str = TensorRTConverter.validate(*args)
        if err:
            logger.error(err)
            return err

        (
            batch_size_min,
            batch_size_opt,
            batch_size_max,
            width_min,
            width_opt,
            width_max,
            height_min,
            height_opt,
            height_max,
            context_min,
            context_opt,
            context_max,
            opt_level,
        ) = args

        model: UnetPatcher = shared.sd_model.forge_objects.unet

        context_dim = model.model.model_config.unet_config.get("context_dim", None)
        if context_dim is None:
            logger.error("Model is not supported...")
            return "Model is not supported..."

        y_dim = model.model.adm_channels
        context_len = 77

        input_names = ["x", "timesteps", "context"]
        output_names = ["h"]

        dynamic_axes = {
            "x": {0: "batch", 2: "height", 3: "width"},
            "timesteps": {0: "batch"},
            "context": {0: "batch", 1: "num_embeds"},
        }

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

        ckpt = shared.sd_model.sd_model_checkpoint
        filename: str = os.path.splitext(os.path.basename(ckpt))[0]

        output_onnx = os.path.join(os.path.join(TEMP_DIR, filename), "model.onnx")
        if not os.path.isfile(output_onnx):

            model_management.unload_all_models()
            model_management.load_model_gpu(model)
            unet = model.model.diffusion_model

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

            dt, de = torch.float16, model_management.get_torch_device()
            inputs = tuple(
                [torch.randn(shape, dtype=dt, device=de) for shape in inputs_shapes_opt]
            )

            logger.info("exporting Onnx...")
            os.makedirs(os.path.dirname(output_onnx))
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=torch.jit.TracerWarning)
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

        trt_logger = trt.Logger(trt.Logger.WARNING)
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
            return "Failed to load the Onnx Model..."

        logger.info("converting TensorRT...")

        config = builder.create_builder_config()
        TensorRTConverter.load_timing_cache(config)
        config.progress_monitor = TQDMProgressMonitor()
        config.flags = 1 << int(trt.BuilderFlag.FP16)
        config.builder_optimization_level = opt_level

        profile = builder.create_optimization_profile()

        prefix_encode = ""
        for k in range(len(input_names)):
            min_shape = inputs_shapes_min[k]
            opt_shape = inputs_shapes_opt[k]
            max_shape = inputs_shapes_max[k]
            profile.set_shape(input_names[k], min_shape, opt_shape, max_shape)

            encode = lambda a: ".".join(map(lambda x: str(x), a))
            prefix_encode += "{}#{}#{}#{};".format(
                input_names[k], encode(min_shape), encode(opt_shape), encode(max_shape)
            )

        config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            logger.error("Failed to convert the Engine...")
            return "Failed to convert the Engine..."

        output_trt = os.path.join(OUTPUT_DIR, f"{filename}.trt")
        with open(output_trt, "wb") as f:
            f.write(serialized_engine)

        # TensorRTConverter.save_timing_cache(config)
        logger.info("Success!")
        return "Success!"

    @staticmethod
    def validate(*args: list[int]) -> str:
        if (
            shared.sd_model.forge_objects.unet.model.diffusion_model.dtype
            is not torch.float16
        ):
            return "Only fp16 precision UNet is supported..."

        if model_management.XFORMERS_IS_AVAILABLE is True:
            return "Only PyTorch attention is supported..."

        for i in range(4):
            if not args[i * 3 + 0] <= args[i * 3 + 1] <= args[i * 3 + 2]:
                return "Invalid Value Range(s)..."

        return False


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
            value=1,
            step=1,
        )


def trt_ui():
    with gr.Blocks() as TRT:
        with gr.Row():
            with gr.Column(elem_classes="trt_block"):
                args = []
                gr.HTML('<h2 align="center">Batch Size</h2>')
                with gr.Row():
                    args.append(Sliders.batch("Min"))
                    args.append(Sliders.batch("Opt"))
                    args.append(Sliders.batch("Max"))
                with gr.Row():
                    with gr.Column():
                        gr.HTML('<h3 align="center">Width</h3>')
                        args.append(Sliders.dim("Min", 896))
                        args.append(Sliders.dim("Opt", 1024))
                        args.append(Sliders.dim("Max", 1152))
                    with gr.Column():
                        gr.HTML('<h3 align="center">Height</h3>')
                        args.append(Sliders.dim("Min", 896))
                        args.append(Sliders.dim("Opt", 1024))
                        args.append(Sliders.dim("Max", 1152))
                gr.HTML('<h2 align="center">Context Length</h2>')
                with gr.Row():
                    args.append(Sliders.context("Min"))
                    args.append(Sliders.context("Opt"))
                    args.append(Sliders.context("Max"))

            with gr.Column(elem_classes="trt_block"):
                gr.Markdown(docs)

        with gr.Row():
            with gr.Column():
                args.append(
                    gr.Slider(
                        label="Optimization Level",
                        minimum=0,
                        maximum=5,
                        step=1,
                        value=3,
                    )
                )
                args.append(gr.Button("Convert Engine", variant="primary"))
            args.append(gr.Textbox(label="Status", value=None, interactive=False))

        for comp in args:
            comp.do_not_save_to_config = True

        *params, btn, status = args
        btn.click(fn=lambda: gr.update(interactive=False), outputs=[btn]).then(
            fn=TensorRTConverter.convert, inputs=[*params], outputs=[status]
        ).then(fn=lambda: gr.update(interactive=True), outputs=[btn])

    return [(TRT, "TensorRT", "sd-forge-trt")]


on_ui_tabs(trt_ui)
