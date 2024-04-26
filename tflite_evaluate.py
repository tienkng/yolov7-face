import argparse
import time

import tensorflow as tf
import numpy as np
import os
import cv2
import torch
import glob

from tqdm import tqdm
from utils.general import check_requirements, non_max_suppression, \
    scale_coords, set_logging
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, time_synchronized



def dataset_letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return img, ratio, (dw, dh)

def detect(opt):
    weights, imgsz, kpt_label = opt.weights, opt.img_size, opt.kpt_label
    backend = 'tflite'
    
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load TFLite model and allocate tensors
    print("\nLoading checkpoint from: ", weights[0])
    interpreter = tf.lite.Interpreter(model_path=weights[0])
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    t0 = time.time()

    # testing dataset
    testset_folder = opt.dataset_folder
    test_dataset = glob.glob(f'{testset_folder}/*')

    for folder in test_dataset:
        parent_path = folder.split('/')[-1]
        children_path = glob.glob(f'{folder}/*')
        for image_path in tqdm(children_path):
            img_name = image_path.split('/')[-1]
            img0 = cv2.imread(image_path)  # BGR
            img = dataset_letterbox(img0, imgsz, auto=backend=='pytorch', stride=None)[0]
            
            img = img[:, :, ::-1].transpose(2, 0, 1) #astype('float32')  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            
            # convert
            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            input_data = img.permute(0, 2, 3, 1).cpu().numpy()
            interpreter.set_tensor(input_details[0]['index'], input_data)   #1,320,320,3
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]['index'])

            pred = torch.Tensor(pred.transpose(0, 2, 1))
            names = 'HiFace'
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, kpt_label=kpt_label)
            t2 = time_synchronized()

            save_name = os.path.join(opt.save_folder, parent_path, f'{img_name[:-4]}.txt')
            dirname = os.path.dirname(save_name)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            with open(save_name, "w") as fd:
                file_name = os.path.basename(save_name)[:-4] + "\n"
                bboxs_num = str(len(pred[0])) + "\n"
                fd.write(file_name)
                fd.write(bboxs_num)
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        scale_coords(img.shape[2:], det[:, :4], img0.shape, kpt_label=False).round()
                        if kpt_label:
                            scale_coords(img.shape[2:], det[:, 6:], img0.shape, kpt_label=kpt_label, step=3)

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                        
                        # Write results
                        for det_index, (*xyxy, conf, cls) in enumerate(det[:,:6]):
                            c = int(cls)  # integer class
                            label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                            kpts = det[det_index, 6:]
                            x1 = int(xyxy[0] + 0.5)
                            y1 = int(xyxy[1] + 0.5)
                            x2 = int(xyxy[2] + 0.5)
                            y2 = int(xyxy[3] + 0.5)
                            
                            fd.write('%d %d %d %d %.03f' % (x1, y1, x2-x1, y2-y1, conf if conf <= 1 else 1) + '\n')
                #             plot_one_box(xyxy, img0, label=label, color=colors(c, True), line_thickness=opt.line_thickness, kpt_label=kpt_label, kpts=kpts, steps=3, orig_shape=img0.shape[:2])
                # cv2.imwrite('hi-face-result.jpg', img0)
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', nargs= '+', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--kpt-label', type=int, default=0, help='number of keypoints')
    parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
    parser.add_argument('--dataset_folder', default='data/widerface/widerface/val/images/', type=str, help='dataset path')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    with torch.no_grad():
       detect(opt=opt)