import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import onnx
from onnxconverter_common import float16

input_model = "weights/yolo-person-backbone.preprocess.onnx"
test_input_model = "weights/yolo-person-backbone.onnx"
output_model = "weights/yolo-person-backbone.f16.onnx"

model = onnx.load(input_model)
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, output_model)

# Cleanup
import onnx_graphsurgeon as gs

model = onnx.load(output_model)

graph = gs.import_onnx(model)
graph.cleanup().toposort().fold_constants().cleanup()

optimized_model = gs.export_onnx(graph)
onnx.save(optimized_model, output_model)

# Test time
import onnxruntime as ort
from int8.test import time_test_normal
import numpy as np

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

print("Testing input model...")
time_test_normal(
    test_input_model,
    num_runs=10,
    provider=["CPUExecutionProvider", "CUDAExecutionProvider"],
    session_options=session_options,
)
print("Testing output model...")
# session_options.log_severity_level = 1
time_test_normal(
    output_model,
    num_runs=10,
    provider=["CPUExecutionProvider", "CUDAExecutionProvider"],
    session_options=session_options,
    fp16=True,
)

# Visualize
from int8.test import visualize

model = ort.InferenceSession(
    test_input_model,
    sess_options=session_options,
    providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
)
visualize(model)

model = ort.InferenceSession(
    output_model,
    sess_options=session_options,
    providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
)

visualize(model, fp16=True)
