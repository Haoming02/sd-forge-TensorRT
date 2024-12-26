from ldm_patched.modules.model_management import soft_empty_cache, current_loaded_models, get_torch_device
from modules.script_callbacks import on_list_unets, on_script_unloaded
from modules_forge.unet_patcher import UnetPatcher
from modules_forge.forge_loader import ForgeSD
from modules import sd_unet, shared

from typing import Callable
import tensorrt as trt
import torch
import gc
import os

from lib_trt.utils import trt_datatype_to_torch, OUTPUT_DIR
from lib_trt.utils import logger as trt_logger
from lib_trt.database import TensorRTDatabase
from lib_trt.patcher import LoadedEngine
from lib_trt.logging import logger


CUDA_STREAM = torch.cuda.current_stream().cuda_stream
RUNTIME = trt.Runtime(trt_logger)
TensorRTDatabase.load()


class TrtUnet:
    dtype = torch.float16

    @staticmethod
    def load_unet(unet_name: str) -> "TrtUnet":
        unet_path = os.path.join(OUTPUT_DIR, f"{unet_name}.trt")
        if not os.path.isfile(unet_path):
            raise FileNotFoundError(f'Engine "{unet_path}" does not exist...')

        return TrtUnet(unet_path)

    def __init__(self, engine_path: str):
        with open(engine_path, "rb") as f:
            self.engine = RUNTIME.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def __del__(self):
        del self.context
        del self.engine

    def set_bindings_shape(self, inputs: dict, split_batch: int):
        for k in inputs:
            shape = inputs[k].shape
            shape = [shape[0] // split_batch] + list(shape[1:])
            self.context.set_input_shape(k, shape)

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor = None,
        *args,
        **kwargs,
    ):
        model_inputs = {"x": x, "timesteps": timesteps, "context": context}

        if y is not None:
            model_inputs["y"] = y

        for i in range(len(model_inputs), self.engine.num_io_tensors - 1):
            name = self.engine.get_tensor_name(i)
            model_inputs[name] = kwargs[name]

        batch_size = x.shape[0]
        dims = self.engine.get_tensor_profile_shape(self.engine.get_tensor_name(0), 0)
        min_batch = dims[0][0]
        max_batch = dims[2][0]

        for i in range(max_batch, min_batch - 1, -1):
            if batch_size % i == 0:
                curr_split_batch = batch_size // i
                break

        self.set_bindings_shape(model_inputs, curr_split_batch)

        model_inputs_converted = {}
        for k in model_inputs:
            data_type = self.engine.get_tensor_dtype(k)
            model_inputs_converted[k] = model_inputs[k].to(
                dtype=trt_datatype_to_torch(data_type)
            )

        output_binding_name = self.engine.get_tensor_name(len(model_inputs))
        out_shape = self.engine.get_tensor_shape(output_binding_name)
        out_shape = list(out_shape)

        for idx in range(len(out_shape)):
            if out_shape[idx] == -1:
                out_shape[idx] = x.shape[idx]
            else:
                if idx == 0:
                    out_shape[idx] *= curr_split_batch

        out = torch.empty(
            out_shape,
            device=x.device,
            dtype=trt_datatype_to_torch(
                self.engine.get_tensor_dtype(output_binding_name)
            ),
        )

        model_inputs_converted[output_binding_name] = out

        for i in range(curr_split_batch):
            for k in model_inputs_converted:
                x = model_inputs_converted[k]
                self.context.set_tensor_address(
                    k, x[(x.shape[0] // curr_split_batch) * i :].data_ptr()
                )
            self.context.execute_async_v3(stream_handle=CUDA_STREAM)

        return out


class UnetOption(sd_unet.SdUnetOption):
    def __init__(self, name: str, family: str):
        self.label = f"[TRT] {name}"
        self.model_name = family
        self.unet = name
        self.mem = TensorRTDatabase.get_memory(family, name)

    def create_unet(self):
        return SDUnet(self.unet, self.mem)


class SDUnet(sd_unet.SdUnet):

    def __init__(self, model_name: str, memory: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_requirement = memory
        self.model_name = model_name

        self.original: Callable = None
        self.unet: TrtUnet = None
        self.FSD: ForgeSD = None

    def __del__(self):
        if self.unet is not None:
            del self.unet

    @torch.no_grad()
    def activate(self):
        if self.FSD is not None:
            return

        self.unet = TrtUnet.load_unet(self.model_name)
        engine = LoadedEngine(self.unet, self.memory_requirement)
        logger.info(f'Loaded Engine: "{self.model_name}"')

        m = shared.sd_model
        objs = m.forge_objects

        unet = objs.unet
        assert isinstance(unet, UnetPatcher)

        for loaded in current_loaded_models:
            if loaded.model is unet:
                logger.debug(f"Unloading: {loaded.model.__class__.__name__}")
                current_loaded_models.remove(loaded)
                loaded.model_unload()
                del loaded

                soft_empty_cache(force=True)
                gc.collect()
                break

        current_loaded_models.insert(0, engine)
        self.FSD = m.forge_objects_original.shallow_copy()

        unet.load_device = unet.offload_device = torch.device("cpu")
        assert isinstance(unet.model.diffusion_model.forward, Callable)
        self.original = unet.model.diffusion_model.forward

        unet.model.diffusion_model.forward = self.unet.__call__
        m.forge_objects_original = objs.shallow_copy()
        m.forge_objects_after_applying_lora = objs.shallow_copy()

    @torch.no_grad()
    def deactivate(self):
        if self.FSD is None:
            return

        for loaded in current_loaded_models:
            if isinstance(loaded, LoadedEngine):
                logger.debug(f"Unloading: {loaded.model.__class__.__name__}")
                current_loaded_models.remove(loaded)
                del loaded
                break

        m = shared.sd_model

        m.forge_objects = self.FSD.shallow_copy()
        m.forge_objects.unet.load_device = get_torch_device()
        m.forge_objects.unet.model.diffusion_model.forward = self.original

        m.forge_objects_original = m.forge_objects.shallow_copy()
        m.forge_objects_after_applying_lora = m.forge_objects.shallow_copy()

        del self.FSD
        self.FSD = None
        del self.original
        self.original = None
        del self.unet
        self.unet = None

        soft_empty_cache(force=True)
        gc.collect()


def list_unets(unet_list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for obj in os.listdir(OUTPUT_DIR):
        filename, ext = os.path.splitext(obj)

        if ext == ".trt":
            family = TensorRTDatabase.get_family(filename)
            unet_list.append(UnetOption(filename, family))

    if getattr(shared.opts, "trt_auto_select", False):

        class NullUnet(sd_unet.SdUnetOption):
            def __init__(self):
                self.label = "null"
                self.model_name = None

            def create_unet(self):
                null = lambda: None
                null.activate = lambda: None
                null.deactivate = lambda: None
                null.option = None
                return null

        unet_list.append(NullUnet())


def unload():
    global CUDA_STREAM, RUNTIME
    del CUDA_STREAM
    del RUNTIME
    gc.collect()
    soft_empty_cache(force=True)


on_list_unets(list_unets)
on_script_unloaded(unload)
