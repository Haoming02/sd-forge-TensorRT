from backend import memory_management

assert memory_management.is_nvidia()

import os.path
from functools import wraps

import numpy as np
import torch
import tqdm
from lib_tensorrt import get_dtype, logger
from lib_tensorrt.paths import GAN_TRT
from lib_tensorrt.utilities import Engine
from PIL import Image

from modules import images, modelloader, shared
from modules.script_callbacks import on_script_unloaded
from modules.upscaler import Upscaler, UpscalerData


class UpscalerTRT(Upscaler):
    name = "TensorRT"
    cudaStream = None

    def __init__(self):
        super().__init__(False)
        self.scalers = []

        for file in os.listdir(GAN_TRT):
            if not file.endswith(".trt"):
                continue

            self.scalers.append(
                UpscalerData(
                    name=f'[TRT] {file.replace(".trt", "")}',
                    path=os.path.join(GAN_TRT, file),
                    upscaler=self,
                    scale=1,
                ),
            )

    @classmethod
    def do_upscale(cls, model: Engine, tile: Image.Image, dtype: torch.dtype):
        data = torch.from_numpy(np.array(tile)).to(dtype=dtype)
        img = data.permute(2, 0, 1).div_(255.0).clip_(0.0, 1.0).unsqueeze(0)

        result = model.infer({"input": img}, cls.cudaStream)
        output: torch.Tensor = result["output"].squeeze(0).float()

        output = output.mul_(255.0).round_().clip_(0.0, 255.0).permute(1, 2, 0)
        data = output.cpu().numpy().astype(np.uint8)
        return Image.fromarray(data)

    @torch.inference_mode()
    def upscale(self, img: Image.Image, scale: float, selected_model: os.PathLike):
        memory_management.soft_empty_cache()

        model: str = os.path.splitext(os.path.basename(selected_model))[0]

        try:
            name, tile, dtype = model.rsplit("-", 2)
        except Exception:
            logger.error("Failed to parse Model params (do not rename the model)...")
            return img
        else:
            tile = int(tile)
            dtype = get_dtype(dtype)

        logger.info(f'Model: "{name}" (TileSize: {tile})')

        engine = Engine(selected_model)
        engine.load()

        shape_dict = {"input": torch.empty((1, 3, tile, tile))}

        engine.activate()
        engine.allocate_buffers(shape_dict=shape_dict)

        image: Image.Image = img.convert("RGB")

        grid = images.split_grid(image, tile, tile, shared.opts.ESRGAN_tile_overlap)
        new_tiles = []

        if UpscalerTRT.cudaStream is None:
            UpscalerTRT.cudaStream = torch.cuda.current_stream().cuda_stream

        memory_management.soft_empty_cache()

        with tqdm.tqdm(
            total=grid.tile_count,
            desc="Tiled Upscale (TRT)",
            disable=not shared.opts.enable_upscale_progressbar,
        ) as p:
            for y, h, row in grid.tiles:
                newrow = []
                for x, w, tile in row:
                    if shared.state.interrupted:
                        break
                    output = self.do_upscale(engine, tile, dtype)
                    scale: int = output.width // tile.width
                    newrow.append([x * scale, w * scale, output])
                    p.update(1)
                new_tiles.append([y * scale, h * scale, newrow])

        if shared.state.interrupted:
            engine.reset()
            del engine
            return img

        newgrid = images.Grid(
            new_tiles,
            tile_w=grid.tile_w * scale,
            tile_h=grid.tile_h * scale,
            image_w=grid.image_w * scale,
            image_h=grid.image_h * scale,
            overlap=grid.overlap * scale,
        )

        engine.reset()
        del engine

        memory_management.soft_empty_cache()
        return images.combine_grid(newgrid)

    def load_model(self, *args, **kwargs):
        raise NotImplementedError

    def find_models(self, *args, **kwargs):
        raise NotImplementedError


orig = modelloader.load_upscalers


@wraps(orig)
def extra_upscalers():
    orig()
    shared.sd_upscalers.extend(UpscalerTRT().scalers)


modelloader.load_upscalers = extra_upscalers


def revert():
    modelloader.load_upscalers = orig


on_script_unloaded(revert)
