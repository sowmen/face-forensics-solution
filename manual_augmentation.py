import random
import math
import cv2
import numpy as np

from scipy.ndimage import binary_erosion, binary_dilation
import skimage
from skimage import measure


def remove_landmark(image, landmarks, cutout_fill):
    if random.random() > 0.5:
        image = remove_eyes(image, landmarks, cutout_fill)
    elif random.random() > 0.5:
        image = remove_mouth(image, landmarks, cutout_fill)
    elif random.random() > 0.5:
        image = remove_nose(image, landmarks, cutout_fill)
    return image

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def remove_eyes(image, landmarks, cutout_fill):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[:2]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    if cutout_fill == 0:
        image[line, :] = 0
    else:
       image[line, :] = np.random.rand(0,255,image[line,:].shape)
    return image

def remove_mouth(image, landmarks, cutout_fill):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[-2:]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 3)
    line = binary_dilation(line, iterations=dilation)
    if cutout_fill == 0:
        image[line, :] = 0
    else:
       image[line, :] = np.random.rand(0,255,image[line,:].shape)
    return image

def remove_nose(image, landmarks, cutout_fill):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[:2]
    x3, y3 = landmarks[2]
    mask = np.zeros_like(image[..., 0])
    x4 = int((x1 + x2) / 2)
    y4 = int((y1 + y2) / 2)
    line = cv2.line(mask, (x3, y3), (x4, y4), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    if cutout_fill == 0:
        image[line, :] = 0
    else:
       image[line, :] = np.random.rand(0,255,image[line,:].shape)
    return image

def blackout_convex_hull(img, detector, predictor, cutout_fill):
    rects = detector(img)
    if(len(rects) == 0):
        return
    sp = predictor(img, rects[0])
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])
    outline = landmarks[[*range(17), *range(26, 16, -1)]]
    Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])
    cropped_img = np.zeros(img.shape[:2], dtype=np.uint8)
    cropped_img[Y, X] = 1
    # if random.random() > 0.5:
    #     img[cropped_img == 0] = 0
    #     #leave only face
    #     return img
    y, x = measure.centroid(cropped_img)
    y = int(y)
    x = int(x)
    first = random.random() > 0.5
    if random.random() > 0.5:
        if first:
            cropped_img[:y, :] = 0
        else:
            cropped_img[y:, :] = 0
    else:
        if first:
            cropped_img[:, :x] = 0
        else:
            cropped_img[:, x:] = 0
    if(cutout_fill == 0):
        img[cropped_img > 0] = 0
    else:
        img[cropped_img > 0] = np.random.rand(0,255,img[cropped_img > 0].shape)
    

def prepare_bit_masks(mask):
    h, w = mask.shape
    mid_w = w // 2
    mid_h = w // 2
    masks = []
    ones = np.ones_like(mask)
    ones[:mid_h] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[mid_h:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:, :mid_w] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:, mid_w:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:mid_h, :mid_w] = 0
    ones[mid_h:, mid_w:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:mid_h, mid_w:] = 0
    ones[mid_h:, :mid_w] = 0
    masks.append(ones)
    return masks

def change_padding(image, part=5):
    h, w = image.shape[:2]
    # original padding was done with 1/3 from each side, too much
    pad_h = int(((3 / 5) * h) / part)
    pad_w = int(((3 / 5) * w) / part)
    image = image[h // 5 - pad_h:-h // 5 + pad_h, w // 5 - pad_w:-w // 5 + pad_w]
    return image

def blackout_random(image, mask, label):
    binary_mask = mask > 0.4 * 255
    h, w = binary_mask.shape[:2]

    tries = 50
    current_try = 1
    while current_try < tries:
        first = random.random() < 0.5
        if random.random() < 0.5:
            pivot = random.randint(h // 2 - h // 5, h // 2 + h // 5)
            bitmap_msk = np.ones_like(binary_mask)
            if first:
                bitmap_msk[:pivot, :] = 0
            else:
                bitmap_msk[pivot:, :] = 0
        else:
            pivot = random.randint(w // 2 - w // 5, w // 2 + w // 5)
            bitmap_msk = np.ones_like(binary_mask)
            if first:
                bitmap_msk[:, :pivot] = 0
            else:
                bitmap_msk[:, pivot:] = 0

        if label < 0.5 and np.count_nonzero(image * np.expand_dims(bitmap_msk, axis=-1)) / 3 > (h * w) / 5 \
                or np.count_nonzero(binary_mask * bitmap_msk) > 40:
            mask *= bitmap_msk
            image *= np.expand_dims(bitmap_msk, axis=-1)
            break
        current_try += 1
    return image