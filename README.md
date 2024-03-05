# yolov7-head-face detection

## Installed
```sh
pip install -r requirements.txt
```

#### Download checkpoint model
 Get checkpoint model [here](https://drive.google.com/drive/folders/1wbgZeBCgiqazGlmMldIxTXaHwD2wNZSQ?usp=sharing)

## Test 
Change `detect-layer` to determine which your target ('face', 'head')
```
python hf_detect.py \
    --weights yolov7-headface.best.pt \
    --device 0 \
    --source data/images \
    --detect-layer 'face' \
    --kpt-label 5 \
    --save-txt
```
![](data/images/result.jpg)

## Convert pytorch to ONNX
```
python export.py \
  --weights yolov7-headface-v1.pt \
  --img-size 640 --batch-size 1 \
  --dynamic-batch --grid --end2end --max-wh 640 --topk-all 100 \
  --iou-thres 0.5 --conf-thres 0.2 --device 'cpu' --simplify --cleanup
```

## ONNX inference
```python
python onnx_inference/inference.py \
  --model-path 'yolov7-headface-v1.onnx' \
  --img-path 'data/images' \
  --dst-path '/predicts/output' \
  --get-layer 'face'
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
