import os.path

from modules.paths import models_path

_TRT = os.path.join(models_path, "TensorRT")

GAN_TRT = os.path.join(_TRT, "Upscaler")
DIT_TRT = os.path.join(_TRT, "Diffusion")

GAN_ONNX = os.path.join(GAN_TRT, "_onnx")
DIT_ONNX = os.path.join(DIT_TRT, "_onnx")

os.makedirs(GAN_ONNX, exist_ok=True)
os.makedirs(DIT_ONNX, exist_ok=True)
