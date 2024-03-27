CUDA_VISIBLE_DEVICE=0,1 python train.py \
  --epochs 100 \
  --workers 4 \
  --device 0,1 \
  --batch-size 72 \
  --data data/widerface.yaml \
  --img 640 640 \
  --cfg cfg/yolov7-headface.yaml \
  --name headface \
  --hyp data/hyp.scratch.tiny.yaml \
  --weight weights/yolov7-face.pt \
  --freeze 101