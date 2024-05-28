import cv2
import numpy as np


def letterbox(
    im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, r, (dw, dh)


def read_img(img_file):
    image = cv2.imread(img_file)[:, :, ::-1]
    image, ratio, dwdh = letterbox(image, auto=False)

    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    image = image.astype(np.float32)
    image /= 255

    return image, ratio, dwdh


def plot_skeleton_kpts(im, kpts, pdg=0, ratio=0, steps=3, radius=2):
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = (255, 0, 255)
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]

        x_coord -= np.array(pdg[0])
        y_coord -= np.array(pdg[1])

        conf = kpts[steps * kid + 2]

        if conf > 0.5:  # Confidence of a keypoint has to be greater than 0.5
            cv2.circle(
                im,
                (int(x_coord / ratio), int(y_coord / ratio)),
                radius,
                (int(r), int(g), int(b)),
                -1,
            )


def postprocess(img_file, output, dwdh=0, score_threshold=0.3, ratio=0, get_layer=None):
    """
    Draw bounding boxes on the input image. Dump boxes in a txt file.
    """
    img = cv2.imread(img_file)

    padding = dwdh * 2
    det_bboxes, det_scores, det_labels = output[:, 1:5], output[:, 6], output[:, 5]

    if get_layer == "face":
        kpts = output[:, 7:]

    for idx in range(len(det_bboxes)):
        color_map = (0, 255, 0)
        det_bbox = det_bboxes[idx]
        det_bbox -= np.array(padding)
        det_bbox /= ratio
        det_bbox = det_bbox.round().astype(np.int32).tolist()

        # get class_id & score
        # cls_id = int(det_labels[idx])
        score = round(float(det_scores[idx]), 3)

        if score > score_threshold:
            cv2.rectangle(img, det_bbox[:2], det_bbox[2:], color_map[::-1], 2)
            cv2.putText(
                img,
                str(score),
                (det_bbox[0], det_bbox[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color_map[::-1],
                thickness=2,
            )

        # draw keypoints
        if get_layer == "face":
            plot_skeleton_kpts(
                img, kpts[idx], pdg=padding, ratio=ratio, steps=3, radius=2
            )

    return img
