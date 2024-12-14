import launch

try:
    import tensorrt
except ImportError:
    launch.run_pip("install tensorrt>=10.0.1", "TensorRT")

try:
    import onnx
except ImportError:
    launch.run_pip("install onnx~=1.17.0", "TensorRT")
