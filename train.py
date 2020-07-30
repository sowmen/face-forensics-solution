import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import gc
try:
   import cPickle as pkl
except:
   import pickle as pkl

import wandb

from torch.backends import cudnn
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
cudnn.benchmark = True

from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur
from transforms import IsotropicResize

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_optimizer
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from dataset import FFPP_Dataset
from sklearn import metrics

from utils import EarlyStopping, AverageMeter

import neptune
from neptunecontrib.monitoring.metrics import log_binary_classification_metrics

import scikitplot as skplt
import matplotlib.pyplot as plt

DATA_ROOT = 'ff_data'
OUTPUT_DIR = 'weights'
device = 'cuda'
config_defaults = {
    'epochs' : 20,
    'train_batch_size' : 40,
    'valid_batch_size' : 32,
    'optimizer' : 'radam',
    'learning_rate' : 1e-3,
    'weight_decay' : 0.0005,
    'schedule_patience' : 5,
    'schedule_factor' : 0.25,
    'rand_seed' : 777,
    'cutout_fill' : 1
}
VAL_FOLD = 9
TEST_FOLD = 0

def train(name, run, folds_csv):
    
    wandb.init(project='dfdc', 
               config=config_defaults,
               name=f'{name},val_fold:{VAL_FOLD},run{run}')
    config = wandb.config
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=1)
    model.to(device)
    # model = DataParallel(model).to(device)
    wandb.watch(model)
    
    if config.optimizer == 'radam' :
        optimizer = torch_optimizer.RAdam(model.parameters(), 
                                          lr=config.learning_rate,
                                          weight_decay = config.weight_decay)
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
                              lr=config.learning_rate,
                              weight_decay=config.weight_decay)
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config.schedule_patience,
        threshold=0.001,
        mode="max",
        factor = config.schedule_factor
    )
    criterion = nn.BCEWithLogitsLoss()
    es = EarlyStopping(patience = 10, mode='max')
    
    data_train = FFPP_Dataset(data_root=DATA_ROOT,
                              mode='train',
                              folds_csv=folds_csv,
                              val_fold=VAL_FOLD,
                              test_fold=TEST_FOLD,
                              cutout_fill=config.cutout_fill,
                              hardcore=True,
                              oversample_real=True,
                              transforms=create_train_transforms(size=224))
    data_train.reset(config.rand_seed)
    train_data_loader = DataLoader( data_train, 
                                    batch_size=config.train_batch_size, 
                                    num_workers=8, 
                                    shuffle=True, 
                                    drop_last=True)

    data_val = FFPP_Dataset(data_root=DATA_ROOT,
                            mode='val',
                            folds_csv=folds_csv,
                            val_fold=VAL_FOLD,
                            test_fold=TEST_FOLD,
                            hardcore=False,
                            oversample_real=False,
                            transforms=create_val_transforms(size=224))
    data_val.reset(config.rand_seed)

    val_data_loader = DataLoader(data_val, 
                                 batch_size=config.valid_batch_size, 
                                 num_workers=8, 
                                 shuffle=False, 
                                 drop_last=True)

    data_test = FFPP_Dataset(data_root=DATA_ROOT,
                            mode='test',
                            folds_csv=folds_csv,
                            val_fold=VAL_FOLD,
                            test_fold=TEST_FOLD,
                            hardcore=False,
                            oversample_real=False,
                            transforms=create_val_transforms(size=224))
    data_test.reset(config.rand_seed)

    test_data_loader = DataLoader(data_test, 
                                 batch_size=config.valid_batch_size, 
                                 num_workers=8, 
                                 shuffle=False, 
                                 drop_last=True)
    

    train_history = []
    val_history = []
    test_history = []
    
    for epoch in range(config.epochs):
        print(f"Epoch = {epoch}/{config.epochs-1}")
        print("------------------")
        
        train_metrics = train_epoch(model, train_data_loader, optimizer, criterion, epoch)
        valid_metrics = valid_epoch(model, val_data_loader, criterion, epoch)
        scheduler.step(valid_metrics['valid_auc'])

        print(f"TRAIN_AUC = {train_metrics['train_auc']}, TRAIN_LOSS = {train_metrics['train_loss']}")
        print(f"VALID_AUC = {valid_metrics['valid_auc']}, VALID_LOSS = {valid_metrics['valid_loss']}")
        
        train_history.append(train_metrics)
        val_history.append(valid_metrics)

        es(valid_metrics['valid_auc'], model, model_path=os.path.join(OUTPUT_DIR,f"{name}_fold_{VAL_FOLD}_run_{run}.h5"))
        if es.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(f'weights/{name}_fold_{VAL_FOLD}_run_{run}.h5'))

    neptune.init('sowmen/dfdc')
    neptune.create_experiment(name=f'{name},val_fold:{VAL_FOLD},run{run}')

    test_history = test(model, test_data_loader, criterion)

    try:
        pkl.dump( train_history, open( f"train_history{name}{run}.pkl", "wb" ) )
        pkl.dump( val_history, open( f"val_history{name}{run}.pkl", "wb" ) )
        pkl.dump( test_history, open( f"test_history{name}{run}.pkl", "wb" ) )
    except:
        print("Error pickling")

    wandb.save(f'weights/{name}_fold_{VAL_FOLD}_run_{run}.h5')
      
    
    
    
def train_epoch(model, train_data_loader, optimizer, criterion, epoch):
    model.train()
    
    train_loss = AverageMeter()
    correct_predictions = []
    targets = []
    
    idx = 1
    for batch in tqdm(train_data_loader):
        
        batch_images = batch['image'].to(device)
        batch_labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        out = model(batch_images)
        
        loss = criterion(out, batch_labels.view(-1, 1).type_as(out))
        
        loss.backward()
        optimizer.step()
        
        train_loss.update(loss.item(), train_data_loader.batch_size)
        targets.append((batch_labels.view(-1,1).cpu() >= 0.5) *1)
        correct_predictions.append(torch.sigmoid(out).cpu().detach().numpy())
        
        if(idx % 100 == 0):
            with torch.no_grad():
                temp_t = np.vstack((targets)).ravel()
                temp_correct_preds = np.vstack((correct_predictions)).ravel()

                train_auc = metrics.roc_auc_score(temp_t, temp_correct_preds)
                train_f1_05 = metrics.f1_score(temp_t,(temp_correct_preds >= 0.5)*1)
                train_acc_05 = metrics.accuracy_score(temp_t,(temp_correct_preds >= 0.5)*1)
                train_balanced_acc_05 = metrics.balanced_accuracy_score(temp_t,(temp_correct_preds >= 0.5)*1)
                train_ap = metrics.average_precision_score(temp_t, temp_correct_preds)
                train_log_loss = metrics.log_loss(temp_t, expand_prediction(temp_correct_preds))
                
                train_metrics = {
                    'b_train_loss' : train_loss.avg,
                    'b_train_auc' : train_auc,
                    'b_train_f1_05' : train_f1_05,
                    'b_train_acc_05' : train_acc_05,
                    'b_train_balanced_acc_05' : train_balanced_acc_05,
                    'b_train_batch' : idx,
                    'b_train_ap' : train_ap,
                    'b_train_log_loss' : train_log_loss
                }
                wandb.log(train_metrics)
        idx += 1
        
    with torch.no_grad():
        targets = np.vstack((targets)).ravel()
        correct_predictions = np.vstack((correct_predictions)).ravel()

        train_auc = metrics.roc_auc_score(targets, correct_predictions)
        train_f1_05 = metrics.f1_score(targets,(correct_predictions >= 0.5)*1)
        train_acc_05 = metrics.accuracy_score(targets,(correct_predictions >= 0.5)*1)
        train_balanced_acc_05 = metrics.balanced_accuracy_score(targets,(correct_predictions >= 0.5)*1)
        train_ap = metrics.average_precision_score(targets, correct_predictions)
        train_log_loss = metrics.log_loss(targets, expand_prediction(correct_predictions))

    train_metrics = {
        'train_loss' : train_loss.avg,
        'train_auc' : train_auc,
        'train_f1_05' : train_f1_05,
        'train_acc_05' : train_acc_05,
        'train_balanced_acc_05' : train_balanced_acc_05,
        'train_ap' : train_ap,
        'train_log_loss' : train_log_loss,
        'epoch' : epoch
    }
    wandb.log(train_metrics)
    
    return train_metrics
    
    
def valid_epoch(model, val_data_loader, criterion, epoch):
    model.eval()
    
    valid_loss = AverageMeter()
    correct_predictions = []
    targets = []
    example_images = []
    
    
    with torch.no_grad():   
        idx = 1     
        for batch in tqdm(val_data_loader):
            # batch_image_names = batch['image_name']
            batch_images = batch['image'].to(device).float()
            batch_labels = batch['label'].to(device).float()
            
            out = model(batch_images)
            loss = criterion(out, batch_labels.view(-1, 1).type_as(out))
            
            valid_loss.update(loss.item(), val_data_loader.batch_size)
            batch_targets = (batch_labels.view(-1,1).cpu() >= 0.5) *1
            batch_preds = torch.sigmoid(out).cpu()
            
            targets.append(batch_targets)
            correct_predictions.append(batch_preds)
                
            best_batch_pred_idx = np.argmin(abs(batch_targets - batch_preds))  
            worst_batch_pred_idx = np.argmax(abs(batch_targets - batch_preds))
            example_images.append(wandb.Image(batch_images[best_batch_pred_idx],
                                      caption=f"Pred : {batch_preds[best_batch_pred_idx].item()} Label: {batch_targets[best_batch_pred_idx].item()}"))
            
            example_images.append(wandb.Image(batch_images[worst_batch_pred_idx],
                                      caption=f"Pred : {batch_preds[worst_batch_pred_idx].item()} Label: {batch_targets[worst_batch_pred_idx].item()}"))

            if(idx % 100 == 0):
                temp_t = np.vstack((targets)).ravel()
                temp_correct_preds = np.vstack((correct_predictions)).ravel()

                valid_auc = metrics.roc_auc_score(temp_t, temp_correct_preds)
                valid_f1_05 = metrics.f1_score(temp_t,(temp_correct_preds >= 0.5)*1)
                valid_acc_05 = metrics.accuracy_score(temp_t,(temp_correct_preds >= 0.5)*1)
                valid_balanced_acc_05 = metrics.balanced_accuracy_score(temp_t,(temp_correct_preds >= 0.5)*1)
                valid_ap = metrics.average_precision_score(temp_t, temp_correct_preds)
                valid_log_loss = metrics.log_loss(temp_t, expand_prediction(temp_correct_preds))

                valid_metrics = {
                    'b_valid_loss' : valid_loss.avg,
                    'b_valid_auc' : valid_auc,
                    'b_valid_f1_05' : valid_f1_05,
                    'b_valid_acc_05' : valid_acc_05,
                    'b_valid_balanced_acc_05' : valid_balanced_acc_05,
                    'b_valid_ap' :  valid_ap,
                    'b_valid_log_loss' : valid_log_loss,
                    'b_valid_batch' : idx
                }
                wandb.log(valid_metrics)
            idx += 1
    
    # Logging
    targets = np.vstack((targets)).ravel()
    correct_predictions = np.vstack((correct_predictions)).ravel()

    valid_auc = metrics.roc_auc_score(targets, correct_predictions)
    valid_f1_05 = metrics.f1_score(targets,(correct_predictions >= 0.5)*1)
    valid_acc_05 = metrics.accuracy_score(targets,(correct_predictions >= 0.5)*1)
    valid_balanced_acc_05 = metrics.balanced_accuracy_score(targets,(correct_predictions >= 0.5)*1)
    valid_ap = metrics.average_precision_score(targets, correct_predictions)
    valid_log_loss = metrics.log_loss(targets, expand_prediction(correct_predictions))

    valid_metrics = {
        'valid_loss' : valid_loss.avg,
        'valid_auc' : valid_auc,
        'valid_f1_05' : valid_f1_05,
        'valid_acc_05' : valid_acc_05,
        'valid_balanced_acc_05' : valid_balanced_acc_05,
        'valid_ap' : valid_ap,
        'valid_log_loss' : valid_log_loss,
        'valid_examples' : example_images[-10:],
        'epoch' : epoch
    }
    wandb.log(valid_metrics)

    return valid_metrics

def test(model, test_data_loader, criterion):
    model.eval()
    
    test_loss = AverageMeter()
    correct_predictions = []
    targets = []

    with torch.no_grad():        
        for batch in tqdm(test_data_loader):
            # batch_image_names = batch['image_name']
            batch_images = batch['image'].to(device).float()
            batch_labels = batch['label'].to(device).float()
            
            out = model(batch_images)
            loss = criterion(out, batch_labels.view(-1, 1).type_as(out))
            
            test_loss.update(loss.item(), test_data_loader.batch_size)
            batch_targets = (batch_labels.view(-1,1).cpu() >= 0.5) *1
            batch_preds = torch.sigmoid(out).cpu()
            
            targets.append(batch_targets)
            correct_predictions.append(batch_preds)

    # Logging
    targets = np.vstack((targets)).ravel()
    correct_predictions = np.vstack((correct_predictions)).ravel()

    test_auc = metrics.roc_auc_score(targets, correct_predictions)
    test_f1_05 = metrics.f1_score(targets,(correct_predictions >= 0.5)*1)
    test_acc_05 = metrics.accuracy_score(targets,(correct_predictions >= 0.5)*1)
    test_balanced_acc_05 = metrics.balanced_accuracy_score(targets,(correct_predictions >= 0.5)*1)
    test_ap = metrics.average_precision_score(targets, correct_predictions)
    test_log_loss = metrics.log_loss(targets, expand_prediction(correct_predictions))

    test_metrics = {
        'test_loss' : test_loss.avg,
        'test_auc' : test_auc,
        'test_f1_05' : test_f1_05,
        'test_acc_05' : test_acc_05,
        'test_balanced_acc_05' : test_balanced_acc_05,
        'test_ap' : test_ap,
        'test_log_loss' : test_log_loss
    }
    wandb.log(test_metrics)
    wandb.log({
        'test_roc_auc_curve' : skplt.metrics.plot_roc(targets, expand_prediction(correct_predictions)),
        'test_precision_recall_curve' : skplt.metrics.plot_precision_recall(targets, expand_prediction(correct_predictions))
    })
    log_binary_classification_metrics(targets, expand_prediction(correct_predictions), threshold=0.5)


    return test_metrics

def expand_prediction(arr):
    arr_reshaped = arr.reshape(-1, 1)
    return np.clip(np.concatenate((1.0 - arr_reshaped, arr_reshaped), axis=1), 0.0, 1.0)

def create_train_transforms(size=224):
    return Compose([
        ImageCompression(quality_lower=70, quality_upper=100, p=0.5),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(),
        OneOf([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ]
    ) 
    
def create_val_transforms(size=224):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])

    
if __name__ == "__main__":
    run = 5
    model_name = 'tf_efficientnet_b4_ns'
    train(name='01_FF++_hardcore1_fill,'+model_name, run=run, folds_csv='folds.csv')


