import gradio as gr
from lib_tensorrt.ui.diffusion import unet_ui
from lib_tensorrt.ui.upscaler import gan_ui

from modules.script_callbacks import on_ui_tabs


def exporter_ui():
    with gr.Blocks() as TRT_EXPORTER:
        gr.HTML('<h1 align="center">TensorRT Forge</h1>')

        with gr.Accordion(label="Bake Upscaler Engine", open=True):
            gan_ui()
        with gr.Accordion(label="Bake Diffusion Engine", open=False):
            unet_ui()

    return [(TRT_EXPORTER, "TensorRT", "sd-forge-trt")]


on_ui_tabs(exporter_ui)
