from ldm_patched.modules.model_management import LoadedModel, get_torch_device
from ldm_patched.modules.model_patcher import ModelPatcher

from lib_trt.logging import logger


class EnginePatcher(ModelPatcher):

    def __init__(self, engine, memory_required):
        self.model = engine
        self.size = memory_required

        self.load_device = get_torch_device()
        self.offload_device = get_torch_device()
        self.current_device = self.load_device
        self.weight_inplace_update = False

    def model_size(self):
        return self.size

    def clone(self):
        raise NotImplementedError

    def is_clone(self, *args, **kwargs):
        return False

    def memory_required(self, *args, **kwargs):
        return self.size


class LoadedEngine(LoadedModel):

    def __init__(self, engine, memory_required):
        self.model = EnginePatcher(engine, memory_required)
        self.memory_required = memory_required
        self.model_accelerated = False
        self.device = get_torch_device()

    def model_memory(self):
        return self.memory_required

    def model_memory_required(self):
        return self.memory_required

    def model_load(self, *args, **kwargs):
        logger.warning("Engine is always loaded...")
        return False

    def model_unload(self, *args, **kwargs):
        logger.warning("Unloading Engine is not supported...")
        return False

    def __eq__(self, *args, **kwargs):
        return False
