# Split Model to backbone and head

This is a simple example of how to split a model into two parts: backbone and head. The backbone is the first part of the model, and the head is the second part of the model. The backbone is used to extract features from the input image, and the head is used to make predictions based on the features extracted by the backbone.

## Usage

```sh
python export_cutout.py \
    --original_model_path 'weights/yolov7-tiny-v0.onnx' \
    --input_image_path 'data/images/22_Picnic_Picnic_22_36.jpg' \
    --node_name 'act/LeakyRelu'
```

## Arguments

- `original_model_path`: Path to the original model file (ONNX format).
- `input_image_path`: Path to the input image file.
- `node_name`: Name of the node that separates the backbone and head.

## Output

The script will output two ONNX file paths: `backbone.onnx` and `head.onnx` (depends on how many heads the model have). The `backbone.onnx` file contains the backbone part of the model, and the `head.onnx` file contains the head part of the model.

## How to select the node name

You can use `netron` to visualize the model and select the node name. The node name is the name of the node that separates the backbone and head. For example, in the YOLOv7-tiny model, the node name is `act/LeakyRelu` that separates the 3 convolutional layers from the backbone and the heads.

