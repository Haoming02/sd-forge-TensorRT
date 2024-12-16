## TensorRT Conversion

Tutorials W.I.P

## Troubleshooting

<details>
<summary>Only fp16 precision UNet is supported</summary>

The conversion only works on `fp16` checkpoint. Disable the `Store UNet Weights in fp8` option under **Optimizations** in the **Settings**.

</details>

<details>
<summary>Only PyTorch attention is supported</summary>

The conversion does not work with `xformers` attention. Add `--attention-pytorch` flag to the `webui-user.bat` commandline args to force PyTorch attention instead. *(you can remove the flag after the conversion is finished)*

</details>

<details>
<summary>Invalid Value Range(s)</summary>

- `Min` ≤ `Opt` ≤ `Max`

</details>

<details>
<summary>Model is not supported</summary>

Only `SD1` and `SDXL` checkpoints are supported as of now

</details>

<details>
<summary>Failed to load the Onnx Model</summary>

The exported Onnx model is somehow corrupted, try deleting the folder and convert again.

</details>

<details>
<summary>Failed to convert the Engine</summary>

The conversion to TensorRT failed. Check the console for the reason. *(**eg.** Out of Memory)*

</details>
