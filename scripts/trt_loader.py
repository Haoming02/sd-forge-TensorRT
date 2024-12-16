from modules.script_callbacks import on_list_unets
from modules import sd_unet, shared

from typing import Callable
import tensorrt as trt
import torch
import os

from lib_trt.utils import trt_datatype_to_torch, OUTPUT_DIR
from lib_trt.logging import logger


trt.init_libnvinfer_plugins(None, "")


class TrtUnet:

    @staticmethod
    def load_unet(unet_name):
        unet_path = os.path.join(OUTPUT_DIR, f"{unet_name}.trt")
        if not os.path.isfile(unet_path):
            raise FileNotFoundError(f'Engine "{unet_path}" does not exist...')

        return TrtUnet(unet_path)

    def __init__(self, engine_path: str):
        self.trt_logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.trt_logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.cudaStream = torch.cuda.current_stream().cuda_stream

    def set_bindings_shape(self, inputs, split_batch):
        for k in inputs:
            shape = inputs[k].shape
            shape = [shape[0] // split_batch] + list(shape[1:])
            self.context.set_input_shape(k, shape)

    def __call__(
        self,
        x,
        timesteps,
        context,
        y=None,
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
            self.context.execute_async_v3(stream_handle=self.cudaStream)

        return out


class UnetOption(sd_unet.SdUnetOption):
    def __init__(self, name: str):
        self.label = f"[TRT] {name}"
        self.model_name = name

    def create_unet(self):
        return SDUnet(self.model_name)


class SDUnet(sd_unet.SdUnet):

    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.configs = {"name": model_name, "backend": "trt"}
        self.original_forward: Callable = None
        self.engine = None

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        return self.engine(x, timesteps, context, *args, **kwargs)

    def activate(self):
        if getattr(self, "engine", None) is None:
            setattr(self, "engine", TrtUnet.load_unet(self.model_name))
            logger.info(f'Loaded Engine: "{self.model_name}"')

        if self.original_forward is None:
            unet = shared.sd_model.forge_objects.unet
            self.original_forward = unet.model.diffusion_model.forward
            unet.model.diffusion_model.forward = self.forward

    def deactivate(self):
        unet = shared.sd_model.forge_objects.unet
        unet.model.diffusion_model.forward = self.original_forward
        self.original_forward = None
        del self.engine


def list_unets(unet_list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for obj in os.listdir(OUTPUT_DIR):
        if obj.endswith(".trt"):
            unet_list.append(UnetOption(obj.split(".trt", 1)[0]))


on_list_unets(list_unets)
