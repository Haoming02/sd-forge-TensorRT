from modules.paths import models_path

import tensorrt as trt
import torch
import os


ext = os.path.dirname(os.path.realpath(__file__))
TIMING_CACHE = os.path.join(os.path.dirname(ext), "timing_cache.dat")
DATABASE = os.path.join(os.path.dirname(ext), "database.json")

OUTPUT_DIR = os.path.normpath(os.path.join(models_path, "unet-trt"))
TEMP_DIR = os.path.normpath(os.path.join(models_path, "unet-onnx"))


def trt_datatype_to_torch(datatype):
    if datatype == trt.float16:
        return torch.float16
    elif datatype == trt.float32:
        return torch.float32
    elif datatype == trt.int32:
        return torch.int32
    elif datatype == trt.bfloat16:
        return torch.bfloat16
