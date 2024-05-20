# change lr0:0.001 in data/hyp.scratch.tiny.yaml

python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py \
  --epochs 30 \
  --workers 2 \
  --device 0,1 \
  --batch-size 128 \
  --data data/widerface.yaml \
  --img 640 640 \
  --cfg cfg/yolov7-tiny.yaml \
  --name yolov7-tiny-bodyheadface \
  --hyp data/hyp.scratch.tiny.yaml \
  --weight pretrained/yolov7-tiny-bodyhead/weights/best.pt \
  --multilosses True \
  --kpt-label 5 \
  --detect-layer 'IKeypoint' \
  --sync-bn \
  --freeze '0-79'