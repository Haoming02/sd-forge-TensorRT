import launch

if not launch.is_installed("onnx"):
    launch.run_pip("install onnx==1.20.1", "onnx")

if not launch.is_installed("onnxscript"):
    launch.run_pip("install onnxscript==0.6.2", "onnxscript")

if not launch.is_installed("tensorrt"):
    launch.run_pip("install tensorrt==10.16.1.11", "tensorrt")

if not launch.is_installed("polygraphy"):
    launch.run_pip("install polygraphy==0.49.26", "polygraphy")
