from modules.script_callbacks import on_ui_settings
from modules import sd_unet, scripts, shared
import os

from lib_trt.database import TensorRTDatabase
from lib_trt.logging import logger


class TensorRTSelector(scripts.Script):

    def title(self):
        return "TensorRT"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    @staticmethod
    def verify_unet(option: str, family: str):
        unet: sd_unet.SdUnetOption = sd_unet.get_unet_option(option)
        if "[TRT]" not in unet.label:
            return False

        if unet.model_name != family:
            logger.warning(f'Unet "{unet.label}" does not match "{family}"!')

        return True

    def setup(self, p, *args, **kwargs):
        if not getattr(shared.opts, "trt_auto_select", False):
            return

        option: None | str = getattr(shared.opts, "sd_unet", None)
        if option is None or option == "None":
            return

        ckpt = shared.sd_model.sd_model_checkpoint
        family: str = os.path.splitext(os.path.basename(ckpt))[0]

        if option not in ("Automatic", "null"):
            if not TensorRTSelector.verify_unet(option, family):
                return

        unet = TensorRTDatabase.get_suitable(family, p.width, p.height)
        if unet is None:
            if option != "null":
                logger.info("deselected engine")
                setattr(shared.opts, "sd_unet", "null")
        elif unet != option:
            logger.info(f'selected "{unet}"')
            setattr(shared.opts, "sd_unet", unet)


def trt_settings():
    section = ("trt", "TensorRT")
    shared.opts.add_option(
        "trt_auto_select",
        shared.OptionInfo(
            False,
            "Automatically select the most suitable Unet",
            section=section,
            category_id="sd",
        ).needs_reload_ui(),
    )


on_ui_settings(trt_settings)
