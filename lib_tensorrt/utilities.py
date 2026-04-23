# https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT/blob/torch_tensorrt/utilities.py

# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path
from collections import OrderedDict
from typing import Final

import tensorrt as trt
import torch
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes
from polygraphy.logger import G_LOGGER
from torch.cuda import nvtx
from tqdm import tqdm

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
G_LOGGER.module_severity = G_LOGGER.ERROR


trt_to_torch_dtype_dict: Final[dict[trt.DataType, torch.dtype]] = {
    trt.DataType.BF16: torch.bfloat16,
    trt.DataType.HALF: torch.float16,
    trt.DataType.FLOAT: torch.float32,
}


class TQDMProgressMonitor(trt.IProgressMonitor):
    def __init__(self):
        trt.IProgressMonitor.__init__(self)
        self._active_phases = {}
        self._step_result = True
        self.max_indent = 5

    def phase_start(self, phase_name, parent_phase, num_steps):
        leave = False
        try:
            if parent_phase is not None:
                nbIndents = (
                    self._active_phases.get(parent_phase, {}).get(
                        "nbIndents", self.max_indent
                    )
                    + 1
                )
                if nbIndents >= self.max_indent:
                    return
            else:
                nbIndents = 0
                leave = True
            self._active_phases[phase_name] = {
                "tq": tqdm(
                    total=num_steps, desc=phase_name, leave=leave, position=nbIndents
                ),
                "nbIndents": nbIndents,
                "parent_phase": parent_phase,
            }
        except KeyboardInterrupt:
            self._step_result = False

    def phase_finish(self, phase_name):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    self._active_phases[phase_name]["tq"].total
                    - self._active_phases[phase_name]["tq"].n
                )

                parent_phase = self._active_phases[phase_name].get("parent_phase", None)
                while parent_phase is not None:
                    self._active_phases[parent_phase]["tq"].refresh()
                    parent_phase = self._active_phases[parent_phase].get(
                        "parent_phase", None
                    )
                if (
                    self._active_phases[phase_name]["parent_phase"]
                    in self._active_phases.keys()
                ):
                    self._active_phases[
                        self._active_phases[phase_name]["parent_phase"]
                    ]["tq"].refresh()
                del self._active_phases[phase_name]
        except KeyboardInterrupt:
            self._step_result = False

    def step_complete(self, phase_name, step):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    step - self._active_phases[phase_name]["tq"].n
                )
            return self._step_result
        except KeyboardInterrupt:
            return False


class Engine:
    def __init__(self, engine_path: os.PathLike):
        self.engine_path: os.PathLike = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def reset(self, *args, **kwargs):
        del self.context
        del self.buffers
        del self.tensors

        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.inputs = {}
        self.outputs = {}

    def build(self, *args, **kwargs):
        raise NotImplementedError

    def load(self):
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=False):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda", additional_shapes=None):
        nvtx.range_push("allocate_buffers")
        for binding in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(binding)

            if shape_dict and name in shape_dict:
                shape = shape_dict[name].shape
            elif additional_shapes and name in additional_shapes:
                shape = additional_shapes[name]
            else:
                shape = self.context.get_tensor_shape(name)

            dtype = trt_to_torch_dtype_dict[self.engine.get_tensor_dtype(name)]
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            tensor = torch.zeros(tuple(shape), dtype=dtype).to(device=device)
            self.tensors[name] = tensor
        nvtx.range_pop()

    def infer(self, feed_dict, stream, use_cuda_graph=False):
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        noerror = self.context.execute_async_v3(stream)
        if not noerror:
            raise SystemError("inference failed...")

        return self.tensors

    def __str__(self):
        return f"[TRT] Engine ({os.path.basename(self.engine_path)})"
