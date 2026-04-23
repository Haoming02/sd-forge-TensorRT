import gradio as gr
from lib_tensorrt.exporter.upscaler import export_upscaler


def gan_ui():
    from modules import shared
    from modules.modelloader import load_models

    models: list[str] = load_models(
        model_path=shared.cmd_opts.esrgan_models_path,
        ext_filter=[".pt", ".pth", ".safetensors"],
    )

    with gr.Row():
        target = gr.Dropdown(
            label="Model to Export",
            value=next(iter(models), None),
            choices=models,
            type="value",
            scale=4,
        )
        with gr.Column(scale=1):
            button = gr.Button(
                value="Export",
                variant="primary",
                interactive=bool(models),
            )
            half = gr.Checkbox(False, label="Prefer Half Precision")

    with gr.Row():
        opt = gr.Slider(
            label="Optimization Level",
            info="Setting a higher optimization level allows TensorRT to spend longer engine building time searching for more optimization options.",
            minimum=0,
            maximum=5,
            step=1,
            value=3,
            scale=6,
        )
        its = gr.Slider(
            label="Timing Iterations",
            info="When timing layers, the builder minimizes over a set of average times for layer execution. This parameter controls the number of iterations used in averaging.",
            minimum=1,
            maximum=10,
            step=1,
            value=1,
            scale=7,
        )

    for comp in (target, button, half, opt, its):
        comp.do_not_save_to_config = True

    def _pre():
        gr.Info("Export Started...", duration=5.0)
        return gr.update(interactive=False)

    def _post():
        gr.Info("Export Finished...", duration=10.0)
        return gr.update(interactive=True)

    button.click(fn=_pre, outputs=[button]).then(
        fn=export_upscaler, inputs=[target, opt, its, half]
    ).then(fn=_post, outputs=[button])
