# yolov7-head-face detection

## Installed
```sh
pip install -r requirements.txt
```

#### Download checkpoint model
 Get checkpoint model [here](https://drive.google.com/drive/folders/1yUP8G5dyp9FK0ayM144SfztoIGyN8N_W?usp=sharing)

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
