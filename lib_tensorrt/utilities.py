# https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT

# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import os.path
from collections import OrderedDict
from typing import Final

import numpy as np
import tensorrt_rtx as trt
import torch
from polygraphy.backend.trt import Profile
from polygraphy.logger import G_LOGGER
from torch.cuda import nvtx
from tqdm import tqdm

from lib_tensorrt import logger

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
G_LOGGER.module_severity = G_LOGGER.ERROR

numpy_to_torch_dtype_dict: Final[dict[np.dtype, torch.dtype]] = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
    np.bool_: torch.bool,
}

torch_to_numpy_dtype_dict: Final[dict[torch.dtype, np.dtype]] = {
    value: key for (key, value) in numpy_to_torch_dtype_dict.items()
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

    def build(
        self,
        onnx_path: os.PathLike,
        input_profile: list[tuple[int, int, int]],
        enable_refit: bool = False,
        enable_all_tactics: bool = False,
    ) -> int:

        p = [Profile() for _ in range(len(input_profile))]
        for _p, i_profile in zip(p, input_profile):
            for name, dims in i_profile.items():
                assert len(dims) == 3
                _p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}
        if not enable_all_tactics:
            config_kwargs["tactic_sources"] = []

        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, TRT_LOGGER)
        success = parser.parse_from_file(onnx_path)
        if not success:
            raise RuntimeError(f"Failed to parse the ONNX file: {onnx_path}")

        config = builder.create_builder_config()
        config.progress_monitor = TQDMProgressMonitor()
        if enable_refit:
            config.set_flag(trt.BuilderFlag.REFIT)

        profiles = copy.deepcopy(p)
        for profile in profiles:
            calib_profile = profile.fill_defaults(
                network[1] if not True else network
            ).to_trt(builder, network[1] if not True else network)
            config.add_optimization_profile(calib_profile)

        try:
            engine = builder.build_serialized_network(network, config)
        except Exception as e:
            logger.error(f"Failed to build engine: {e}")
            return 1

        try:
            with open(self.engine_path, "wb") as f:
                f.write(engine)
        except Exception as e:
            logger.error(f"Failed to save engine: {e}")
            return 1

        return 0

    def load(self):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(self.engine_path, "rb") as f:
            engine_bytes = f.read()
        buffer = memoryview(engine_bytes)
        self.engine = runtime.deserialize_cuda_engine(buffer)

    def activate(self, reuse_device_memory=False):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        nvtx.range_push("allocate_buffers")
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]["shape"]
            else:
                shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
            ).to(device=device)
            self.tensors[binding] = tensor
        nvtx.range_pop()

    def infer(self, feed_dict, stream, use_cuda_graph=False):
        nvtx.range_push("set_tensors")
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        nvtx.range_pop()
        nvtx.range_push("execute")
        noerror = self.context.execute_async_v3(stream)
        if not noerror:
            raise SystemError("inference failed...")
        nvtx.range_pop()
        return self.tensors

    def __str__(self):
        return f"[TRT] Engine ({os.path.basename(self.engine_path)})"
