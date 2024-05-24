import argparse
from glob import glob
import numpy as np
import onnxruntime
import time
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

from dataloader import DataReader, extract_data


def benchmark(model_path):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 10
    input_data = np.zeros((1, 3, 640, 640), np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for _ in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quant_format",
        default=QuantFormat.QDQ,
        type=QuantFormat.from_string,
        choices=list(QuantFormat),
    )
    parser.add_argument("--per_channel", default=False, type=bool)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_model_path = "weights/yolo-person-backbone.preprocess.onnx"
    output_model_path = "weights/yolo-person-backbone.augmented.onnx"

    dataset_names = ["crowdhuman-body", "widerface-fhb", "scuthead"]
    datasets = []
    for name in dataset_names:
        dataset = glob(f"datahub/{name}/images/**/*.jpg")
        datasets.append(extract_data(dataset))
        length = len(dataset)
        print(f"{name}: {length}")

    dr = DataReader(np.concatenate(datasets), input_model_path)

    print("Collecting calibration data...")
    # Calibrate and quantize model
    # Turn off model optimization during quantization
    quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=args.quant_format,
        per_channel=args.per_channel,
        weight_type=QuantType.QInt8,
    )
    print("Calibrated and quantized model saved.")

    print("benchmarking fp32 model...")
    benchmark(input_model_path)

    print("benchmarking int8 model...")
    benchmark(output_model_path)


if __name__ == "__main__":
    main()
