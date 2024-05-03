# Yolov7-tiny human detection

## Installed
```sh
pip install -r requirements.txt
```

### Download checkpoint model

 ```python
mkdir -p weights
cd weights
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
cd ..
 ```
## Dataset
- For pretrain stage, we need setup a dataset with full of classes (the same number of head layers). For example, we use WIDERFACE as main of pretrain, so that I need to label for `body` and `head` which use another repo labeling. It can be define as `face = 0`, `head = 1` and `body = 2`.
 - For finetune stage, we use original dataset and change the setup code in `train.py` and `utils/datasets.py` depend on 
 ```
# train.py

# Only calculate face layer 
loss_fn = {
        #'IKeypoint' : ComputeLoss(model, kpt_label=kpt_label),
        #'IDetectHead' : ComputeLoss(model),
        'IDetectBody' : ComputeLoss(model)
}
 ```
 ```
# utils/dataset.py

# Comment head, face, body label
label = torch.cat(label,0)

#face_label = label[label[:,1] == 0]
#head_label = label[label[:,1] == 1]
#body_label = label[label[:,1] == 2]
#head_label[:, 1], body_label[: 1] = 0, 0

return torch.stack(img, 0), {'IDetectBody':label}, path, shapes
 ```

## Training
### 1. Pretrain stage
For pretrain stage you run command line with 
```
bash scripts/pretrain.sh
```
### 2. Finetune stage (body/head/face)
For finetune stage we need to change `lr0=0.01` in `data/hyp.scratch.tiny` config to `lr0=0.001`
```
bash scripts/finetune_body.sh
```

## Testing
Change `detect-layer` to determine which your target ('body', 'head')
```
python hf_detect.py \
    --weights weights/yolov7-tiny-v0.pt \
    --device 0 \
    --source data/images \
    --detect-layer 'face' \
    --kpt-label 5 \
    --save-txt
```
![](data/images/result.jpg)

## Evaluation
Download original WIDER_val dataset put in the widerface_evaluate and setup evaluate
```
cd widerface_evaluate
!gdown 1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q
!unzip WIDER_val.zip
python setup.py build_ext --inplace # setup evaluate
cd ..
```
Then, predict image 

```
python test_widerface.py \
  --weights weights/yolov7-tiny-v0.pt \
  --dataset_folder 'widerface_evaluate/WIDER_val/images' \
  --save_folder predict_txt/tiny-preface
```
and running evaluate
```
python widerface_evaluate/evaluation.py \
  -p predict_txt/tiny-preface \
  -g widerface_evaluate/ground_truth
```
| Method           |  Model Size | Easy  | Medium | Hard  | Google |
| -----------------| ---------- | ----- | ------ | ----- | -------------- |
| github/derronqi    | 12MB        | 94.7  | 92.6   | 82.1  | [yolov7-face](https://github.com/derronqi/yolov7-face)
| github/hiennguyen9874  | 16MB        | 94.9  | 93.12   | 82.8  | [yolov7-face-detection](https://github.com/hiennguyen9874/yolov7-face-detection/tree/main) |
| Our   | 16MB        | 93.4  | 91.4   | 79.6  | [our]() |
## Convert pytorch to ONNX
```sh
python export.py \
  --weights weights/yolov7-tiny-v0.pt \
  --img-size 640 --batch-size 1 \
  --dynamic-batch --grid --end2end --max-wh 640 --topk-all 100 \
  --iou-thres 0.5 --conf-thres 0.2 --device 'cpu' --simplify --cleanup
```

## ONNX inference
```sh
python onnx_inference/inference.py \
  --model-path 'weights/yolov7-tiny-v0.onnx' \
  --img-path 'data/images' \
  --dst-path '/predicts/output' \
  --get-layer 'face' \
  --face-thres 0.78
```

## Convert ONNX to TensorRT
- Convert ONNX to TensorRT
  ```sh
  python export.py \
    --weights weights/yolov7-tiny-v0.pt \
    --img-size 640 --batch-size 1 \
    --grid --end2end --topk-all 100 \
    --iou-thres 0.5 --conf-thres 0.2 --device 'cpu' \ 
    --simplify --cleanup
  ```
- Convert ONNX to TensorRT
  ```sh
  python export_trt.py \
    --onnx-path 'weights/yolov7-tiny-v0.onnx' \
    --engine 'weights/yolov7-tiny-v0.trt'
  ```

## TensorRT inference
```sh
python trt_inference/inference.py \
  --input './data/images/22_Picnic_Picnic_22_36.jpg' \
  --output './output' \
  --weight 'weights/yolov7-tiny-v0.trt' \
  --nms_thresh 0.5 \
  --body_conf 0.4 \
  --head_conf 0.2 \
  --face_conf 0.4 \
  --warmup
```

#### Dataset

[WiderFace](http://shuoyang1213.me/WIDERFACE/)

[yolov7-face-label](https://drive.google.com/file/d/1FsZ0ACah386yUufi0E_PVsRW_0VtZ1bd/view?usp=sharing)


#### Demo

* [ncnn_Android_face](https://github.com/FeiGeChuanShu/ncnn_Android_face)

* [yolov7-detect-face-onnxrun-cpp-py](https://github.com/hpc203/yolov7-detect-face-onnxrun-cpp-py)

#### References

* [https://github.com/deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)

* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)

* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

* [https://github.com/ppogg/YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite)
