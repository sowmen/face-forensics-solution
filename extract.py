import json
import os
from os import cpu_count
from pathlib import Path
import pandas as pd 

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from functools import partial
from glob import glob
from multiprocessing.pool import Pool
from pathlib import Path

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from tqdm import tqdm


def extract_video(param, root_dir, crops_dir):
    video_name = param['name']
    video = param['vid_path']
    bboxes_path = param['bbox_path']
    
    with open(bboxes_path, "r") as f:
        bboxes_dict = json.load(f)
        
    capture = cv2.VideoCapture(video)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    done = False
    for i in range(frames_num):
        capture.grab()
        
        if i % 10 != 0: # Takes every 10th frame. Add heuristic here
            continue
        
        success, frame = capture.retrieve()
        if not success or str(i) not in bboxes_dict or bboxes_dict[str(i)] is None:
            continue
        id = video_name
        crops = []
        bboxes = bboxes_dict[str(i)]
        if bboxes is None:
            continue
        
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
            w = xmax - xmin
            h = ymax - ymin
            p_h = h // 3
            p_w = w // 3
            crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
            h, w = crop.shape[:2]
            crops.append(crop)
            
        img_dir = os.path.join(root_dir, crops_dir, id)
        os.makedirs(img_dir, exist_ok=True)

        for j, crop in enumerate(crops):
            cv2.imwrite(os.path.join(img_dir, "{}_{}.png".format(i, j)), crop)
        done = True

    if(done):
        f = open(os.path.join(img_dir,'done.txt'), 'w')
        f.close()
            

ROOT_DIR = 'face_data'
CROPS_DIR = 'face_crops'

def main():
    os.makedirs(os.path.join(ROOT_DIR, CROPS_DIR), exist_ok=True)
    
    with open("meta.json", "r") as read_file:
        params = json.load(read_file)
    print("Total videos : {}".format(len(params)))
    
    _temp = []
    for item in params:
        video_id = item['name']
        if(os.path.exists(os.path.join(ROOT_DIR, 'face_crops', video_id, 'done.txt'))):
            continue
        else:
            _temp.append((item))
    params = _temp
    print("Remaining : {}".format(len(params)))
    
    with Pool(processes=cpu_count()) as p:
        with tqdm(total=len(params)) as pbar:
            for v in p.imap_unordered(partial(extract_video, root_dir=ROOT_DIR, crops_dir=CROPS_DIR), params):
                pbar.update()
        
if __name__ == "__main__":
    main()
    