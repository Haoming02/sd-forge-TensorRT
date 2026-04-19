import launch

if not launch.is_installed("onnx"):
    launch.run_pip("install onnx==1.20.1", "Onnx")

if not launch.is_installed("tensorrt-rtx"):
    launch.run_pip("install tensorrt-rtx==1.4.0.76", "TensorRT")
