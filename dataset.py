import os
import random
import traceback
import sys

import pandas as pd
import numpy as np
import cv2

from torch.utils.data import Dataset

from albumentations.augmentations.functional import  rot90
from albumentations.pytorch.functional import img_to_tensor
from manual_augmentation import remove_landmark, prepare_bit_masks, change_padding,\
                                blackout_convex_hull, blackout_random


import dlib 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('libs/shape_predictor_68_face_landmarks.dat')

class FFPP_Dataset(Dataset):
    
    def __init__(self,
                 data_root,
                 mode = 'train',
                 val_fold = 1,
                 test_fold = 0,
                 cutout_fill = 0,
                 crops_dir = 'face_crops',
                 folds_csv = 'folds.csv',
                 oversample_real = True, # Equalizes number of real and fake frames
                 reduce_val = False, # Reduces number of frames selected for faster validation
                 hardcore = False, # Does hardcore augmentations
                 label_smoothing = 0.01, # Adds label smoothing
                 padding_part = 3, # Removes excess regions around face 
                 transforms = None, # pytorch transforms
                 rotation = False, # Apply 90 degree rotation augmentations
                 normalize={"mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]},
                 ):
        
        super().__init__()
        self.data_root = data_root
        self.mode = mode
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.cutout_fill = cutout_fill
        self.crops_dir = crops_dir
        self.folds_csv = folds_csv
        self.oversample_real = oversample_real
        self.reduce_val = reduce_val
        self.hardcore = hardcore
        self.label_smoothing = label_smoothing
        self.padding_part = padding_part
        self.transforms = transforms
        self.rotation = rotation
        self.normalize = normalize
        
        self.df = pd.read_csv(self.folds_csv)
        
    def __getitem__(self, index : int):
        # while(True):
        video, img_file, label, ori_video, frame, fold = self.data[index]
        
        # try:
        if self.mode == "train":
            label = np.clip(label, self.label_smoothing, 1 - self.label_smoothing)
        
        # Load image and mask  
        img_path = os.path.join(self.data_root, self.crops_dir, video, img_file)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
        # Applying hardcore augmentations without rotation
        if self.mode == "train" and self.hardcore and not self.rotation:
            landmark_path = os.path.join(self.data_root, "landmarks", ori_video, img_file[:-4] + ".npy")
            
            # Remove facial features using landmark informations
            if os.path.exists(landmark_path) and random.random() < 0.7:
                landmarks = np.load(landmark_path)
                image = remove_landmark(image, landmarks, self.cutout_fill)
            
            # Remove facial parts using convex hull
            elif random.random() < 0.4:
                err = 0
                cp = np.copy(image)
                try:
                    blackout_convex_hull(cp, detector, predictor, self.cutout_fill)
                except Exception:
                    err = 1                
                if err == 0:
                    image = cp
                
                
            # Remove parts of image randomly from 6 bitmasks
            # elif random.random() < 0.1:
            #     binary_mask = mask > 0.4 * 255
            #     masks = prepare_bit_masks((binary_mask * 1).astype(np.uint8))
            #     tries = 6
            #     current_try = 1
            #     while current_try < tries:
            #         bitmap_msk = random.choice(masks)
            #         if label < 0.5 or np.count_nonzero(mask * bitmap_msk) > 20:
            #             mask *= bitmap_msk
            #             image *= np.expand_dims(bitmap_msk, axis=-1)
            #             break
            #         current_try += 1
        
        # Resize image and remove extra padding outside face
        if self.mode == "train" and self.padding_part > 3:
            image = change_padding(image, self.padding_part)
        
        # Change label depending on ratio of fake parts in mask
        # TODO : change mask to binary_mask
        # valid_label = np.count_nonzero(mask[mask > 20]) > 32 or label < 0.5
        # valid_label = 1 if valid_label else 0
        
        # Use builtin transforms passed in
        if self.transforms is not None:
            data = self.transforms(image=image)
            image = data["image"]
            # mask = data["mask"]
        
        # If hardcore augmentation and rotation are both true
        # then applies only random blackout
        # if self.mode == "train" and self.hardcore and self.rotation:
        #     dropout = 0.8 if label > 0.5 else 0.6
        #     if self.rotation:
        #         dropout *= 0.7
        #     elif random.random() < dropout:
        #         blackout_random(image, mask, label)
        
        rotation = 0
        # Applies 90 degree rotation
        if self.mode == "train" and self.rotation:
            rotation = random.randint(0, 3)
            image = rot90(image, rotation)
        
        # Saves 20% of the train images
        # if(random.random() < 0.1 and conv == True):
        #     os.makedirs("train_images", exist_ok=True)
        #     cv2.imwrite(os.path.join("train_images", video+ "_" + str(1 if label > 0.5 else 0) + "_"+img_file), image[...,::-1])
        
        image = img_to_tensor(image, self.normalize)
        return {
            "image": image, 
            "label": np.array((label,)), 
            "image_name": os.path.join(video, img_file), 
            "rotations": rotation
        }
            
        # except Exception as e:
        #     print("Broken image", os.path.join(self.data_root, self.crops_dir, video, img_file))
        #     index = random.randint(0, len(self.data) - 1)

    def __len__(self) -> int:
        return len(self.data)
    
    
    def reset(self, seed):
        self.data = self._prepare_data(seed)
       
        
    '''
        Returns selected rows in dataframe as a list of lists
        example : [['kjwnylvwuo', '140_0.png', 1, 'hivnldfvyl', 140, 1]]
    '''
    def _prepare_data(self, seed) -> list:
        df = self.df
        if self.mode == "train":
            rows = df[~df["fold"].isin([self.val_fold, self.test_fold]) ]
        elif self.mode == "val":
            rows = df[df["fold"] == self.val_fold]
        else:
            rows = df[df["fold"] == self.test_fold]
            
        seed += 1

        if self.oversample_real:
            rows = self._oversample(rows, seed)
        
        if self.mode == 'val' and self.reduce_val:
            rows = rows[rows["frame"] % 20 == 0]
        
        print("real {} fakes {} mode {}".format(len(rows[rows["label"] == 0]), len(rows[rows["label"] == 1]), self.mode))
        data = rows.values

        np.random.seed(seed)
        np.random.shuffle(data)
        return data
      
            
    '''
        Equalizes count of fake and real samples
    '''
    def _oversample(self, rows: pd.DataFrame, seed) -> pd.DataFrame:
        real = rows[rows["label"] == 0]
        fakes = rows[rows["label"] == 1]
        num_real = real["video"].count()
        if self.mode == "train":
            fakes = fakes.sample(n=num_real, replace=False, random_state=seed)
        return pd.concat([real, fakes])