import onnx

DTYPE_BYTES = {
    onnx.TensorProto.FLOAT: 4,
    onnx.TensorProto.FLOAT16: 2,
    onnx.TensorProto.BFLOAT16: 2,
}


def del_onnx_initializers(graph, del_init_names):
    indices = []
    for idx, tensor_proto in enumerate(graph.initializer):
        if tensor_proto.name in del_init_names:
            indices.append(idx)

    indices = sorted(indices, reverse=True)
    for idx in indices:
        del graph.initializer[idx]


def get_onnx_tensor_proto_shape(onnx_tensor_proto):
    shape = [elem for elem in onnx_tensor_proto.dims]
    return shape


def get_onnx_tensor_proto_dtype(onnx_tensor_proto):
    return onnx_tensor_proto.data_type


def shape_elem_num(shape):
    elem_num = 1
    for elem in shape:
        elem_num *= elem
    return elem_num
