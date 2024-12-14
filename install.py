import launch

if not launch.is_installed("tensorrt"):
    launch.run_pip("install tensorrt~=10.7.0", "TensorRT")
    launch.run_pip("install tensorrt_cu12~=10.7.0", "TensorRT")
    launch.run_pip("install tensorrt_cu12_bindings~=10.7.0", "TensorRT")
    launch.run_pip("install tensorrt_cu12_libs~=10.7.0", "TensorRT")

if not launch.is_installed("onnx"):
    launch.run_pip("install onnx~=1.17.0", "Onnx")
