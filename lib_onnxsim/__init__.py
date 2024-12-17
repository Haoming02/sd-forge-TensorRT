"""
Credits: luchangli03
License: MIT
https://github.com/luchangli03/onnxsim_large_model
"""

from onnxsim import simplify
import onnx

from .initializer import compress, decompress


def simplify_onnx(input_path: str, output_path: str, large: bool) -> bool:
    assert input_path.endswith(".onnx") and output_path.endswith(".onnx")
    model = onnx.load(input_path)

    if large:
        model, removed_inits = compress(model)

    model, success = simplify(model)
    if not success:
        return False

    if large:
        model = decompress(model, removed_inits)

    if large:
        onnx.save(model, output_path, save_as_external_data=True, location="weights")
    else:
        onnx.save(model, output_path, save_as_external_data=False)

    return True
