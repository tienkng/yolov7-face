import os
import onnx
import numpy as np
import onnxruntime as rt
import onnx_graphsurgeon as gs

from onnx_inference.inference import read_img

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Export the cutout model")
    parser.add_argument(
        "--original_model_path",
        type=str,
        default="yolo-person.onnx",
        help="Path to the original model",
    )
    parser.add_argument(
        "--input_image_path",
        type=str,
        default="../data/images/22_Picnic_Picnic_22_10.jpg",
        help="Path to the input image",
    )
    parser.add_argument(
        "--node_name",
        type=str,
        default="act/Mul",
        help="Name of the node to cut out",
    )

    return parser.parse_args()


def find_nodes(model, output_names, node_name="act/LeakyRelu"):
    layers, sorted_output_names = [], []
    # for node in model.graph.node:
    #     if all("end2end" in _node for _node in node.output) and \
    #         any("model" in _node for _node in node.input) and \
    #             any("Concat" in _node for _node in node.input) and \
    #             node.input != []:

    #         layer = [s for s in node.input if "model" in s][0]
    #         if layer in layers:
    #             continue
    #         layers.append(layer)

    for node in model.graph.node:
        if node_name in node.name:
            layers.append(node)
        if node.output[0] in output_names:
            sorted_output_names.append(node.output[0])

    layers = sorted(
        layers, key=lambda layer: int(layer.name.split("/")[2].split(".")[-1])
    )[-3:]
    layers = [layer.output[0] for layer in layers]
    outputs = {}
    for tensor in model.graph.value_info:
        if tensor.name in layers:
            outputs[tensor.name] = [
                dim.dim_value if dim.dim_value else None
                for dim in tensor.type.tensor_type.shape.dim
            ]

    return outputs, sorted_output_names


def cut_out_model(
    model,
    input_names,
    output_names,
    input_shape=None,
    output_path="subgraph.onnx",
    output_shape=[[1, 25200, 21]],
):
    graph = gs.import_onnx(model)
    tensors = graph.tensors()

    for idx, input_name in enumerate(
        input_names
        if isinstance(input_names, list) or isinstance(input_names, type({}.keys()))
        else [input_names]
    ):
        if tensors[input_name].shape is None:
            tensors[input_name].shape = (
                input_shape[idx] if len(input_shape) > 1 else input_shape[0]
            )
        if tensors[input_name].dtype is None:
            tensors[input_name].dtype = np.float32

    for idx, output_name in enumerate(
        output_names
        if isinstance(output_names, list) or isinstance(output_names, type({}.keys()))
        else [output_names]
    ):
        if tensors[output_name].shape is None:
            tensors[output_name].shape = (
                output_shape[idx] if len(output_shape) > 1 else output_shape[0]
            )
        if tensors[output_name].dtype is None:
            tensors[output_name].dtype = np.float32

    graph.inputs = [tensors[input_name] for input_name in input_names]
    graph.outputs = [tensors[output_name] for output_name in output_names]
    graph.cleanup()

    onnx.save(gs.export_onnx(graph), output_path)


def test_model(original_model_path, backbone_model_path, head_names, test_image_path):
    original_model = rt.InferenceSession(original_model_path)
    backbone_model = rt.InferenceSession(backbone_model_path)

    input_data, _, _ = read_img(test_image_path)
    original_output = original_model.run(
        None, {original_model.get_inputs()[0].name: input_data}
    )
    backbone_output = backbone_model.run(
        None, {backbone_model.get_inputs()[0].name: input_data}
    )

    for i, head_name in enumerate(head_names):
        head_model = rt.InferenceSession(f"{original_model_name}-{head_name}.onnx")
        head_output = head_model.run(
            None,
            {k.name: v for k, v in zip(backbone_model.get_outputs(), backbone_output)},
        )
        original_head_output = original_output[i]

        print("Test head model:", head_name)
        np.testing.assert_allclose(
            original_head_output, head_output[0], rtol=1e-3, atol=1e-3
        )
        print("Passed")

    # face_output = face_model.run(
    #     None, {k.name: v for k, v in zip(face_model.get_inputs(), backbone_output)}
    # )

    # print("Original output shape:", original_output[0].shape)
    # print("Backbone output shape:", backbone_output[0].shape)
    # print("Face output shape:", face_output[0].shape)

    # np.testing.assert_allclose(original_output[0], face_output[0], rtol=1e-3, atol=1e-3)
    # print(original_output[0] - face_output[0])
    # print(original_output, face_output)


def get_name_shape(model):
    inputs, outputs = {}, {}
    for input in model.graph.input:
        in_shape = []
        for shape in input.type.tensor_type.shape.dim:
            if shape.HasField("dim_value"):
                in_shape.append(shape.dim_value)
            elif shape.HasField("dim_param"):
                in_shape.append(None)
            else:
                raise ValueError("Unsupported shape")
        inputs[input.name] = in_shape

    for output in model.graph.output:
        out_shape = []
        for shape in output.type.tensor_type.shape.dim:
            if shape.HasField("dim_value"):
                out_shape.append(shape.dim_value)
            elif shape.HasField("dim_param"):
                out_shape.append(None)
            else:
                raise ValueError("Unsupported shape")
        outputs[output.name] = out_shape

    return inputs, outputs


if __name__ == "__main__":
    # Load the model
    args = parse_args()

    original_model_name = os.path.basename(args.original_model_path).split(".")[0]
    model = onnx.load(args.original_model_path)
    inputs, outputs = get_name_shape(model)

    # Take the backbone model
    middles, sorted_output_names = find_nodes(
        model, outputs.keys(), node_name=args.node_name
    )

    cut_out_model(
        model,
        inputs.keys(),
        middles.keys(),
        output_path=f"{original_model_name}-backbone.onnx",
        output_shape=middles.values(),
    )

    _model = onnx.load(f"{original_model_name}-backbone.onnx")
    onnx.checker.check_model(_model)
    _model.ir_version = 8
    onnx.save(_model, f"{original_model_name}-backbone.onnx")

    # Take the head model
    for output_name in sorted_output_names:
        cut_out_model(
            model,
            middles.keys(),
            [output_name],
            output_path=f"{original_model_name}-{output_name}.onnx",
            input_shape=middles.values(),
            output_shape=[outputs[output_name]],
        )
        _model = onnx.load(f"{original_model_name}-{output_name}.onnx")
        onnx.checker.check_model(_model)
        _model.ir_version = 8
        onnx.save(
            _model,
            f"{original_model_name}-{output_name}.onnx",
        )

    # Test the model
    test_model(
        original_model_path=args.original_model_path,
        backbone_model_path=f"{original_model_name}-backbone.onnx",
        head_names=outputs.keys(),
        test_image_path=args.input_image_path,
    )
