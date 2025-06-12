#-*- coding: utf-8 -*-
import os
import sys
import time
import argparse
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
import math
import random

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from package_utils.transform import (
    final_transform,
    get_center_scale,
    get_affine_transform,
    GeometryTransform,
    ColorJitterTransform,
)
from configs.get_config import load_config
from models import *
from package_utils.utils import vis_heatmap
from package_utils.image_utils import load_image, crop_by_margin
from losses.losses import _sigmoid
from lib.metrics import get_acc_mesure_func, bin_calculate_auc_ap_ar
from datasets import DATASETS, build_dataset
from lib.core_function import AverageMeter
from logs.logger import Logger, LOG_DIR

from restormer.model import RestormerDerainer
import datetime
from torchvision import transforms as T

# composite: first geometry, then color, then final tensor+normalize
#def inference_transform(img_np):
#    img_np = geome(img_np)
#    img_np = colori(img_np)
#    return normie(img_np)


def parse_args(args=None):
    arg_parser = argparse.ArgumentParser('Processing testing...')
    arg_parser.add_argument('--cfg', '-c', help='Config file', required=True)
    arg_parser.add_argument('--image', '-i', type=str, help='Image for the single testing mode!')
    args = arg_parser.parse_args(args)
    
    return args

# Create output folder if it doesn't exist
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

noisy_path = os.path.join(output_dir, f"noisy_{timestamp}.png")
denoised_path = os.path.join(output_dir, f"denoised_{timestamp}.png")

def denormalize(tensor, mean, std):
    """
    tensor: 3×H×W torch.Tensor that was normalized via (x - mean)/std
    mean, std: 3‐element lists or tuples
    Returns a new Tensor in [0..1] range.
    """
    m = torch.tensor(mean, device=tensor.device).view(-1,1,1)
    s = torch.tensor(std,  device=tensor.device).view(-1,1,1)
    return (tensor * s + m).clamp(0, 1)


def noise_and_denoise_batch(inputs, transforms, derainer, device, mean, std, output_dir="debug_images"):
    """
    inputs:    torch.Tensor [B, C, H, W], RGB floats in [0–1]
    transforms:  your noise-adding torchvision or custom transform (expects numpy HWC in [0-1] or similar)
    derainer:  RestormerDerainer instance
    device:    torch.device('cuda') or cpu

    returns:   torch.Tensor [B, C, H, W], RGB floats in [0–1], on `device`
    """
    # make sure save folder exists
    os.makedirs(output_dir, exist_ok=True)

    os.makedirs(output_dir, exist_ok=True)
    B, C, H, W = inputs.shape
    out = []
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(B):
        # A) H×W×C [0–1] numpy
        img_np = inputs[i].cpu().permute(1,2,0).numpy()

        # B) apply your ColorJitter + ToTensor + Normalize
        noisy_tensor = transforms(img_np)               # -> [3,H,W], normalized
        noisy_tensor = noisy_tensor.to(device)

        # C) denormalize *before* saving
        noisy_den = denormalize(noisy_tensor, mean, std)  # [3,H,W] in [0..1]
        noisy_uint8 = (noisy_den.cpu().permute(1,2,0).numpy() * 255.0).round().astype(np.uint8)
        noisy_bgr = cv2.cvtColor(noisy_uint8, cv2.COLOR_RGB2BGR)
        noisy_name = os.path.join(output_dir, f"noisy_b{i}_{ts}.png")
        #cv2.imwrite(noisy_name, noisy_bgr)

        # D) denoise
        denoised_bgr = derainer.derain_image(noisy_bgr)
        denoised_name = os.path.join(output_dir, f"denoised_b{i}_{ts}.png")
        #cv2.imwrite(denoised_name, denoised_bgr)

        # E) back to [0–1] tensor RGB
        den_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
        den_tensor = (torch.from_numpy(den_rgb)
                        .float().div(255.0)
                        .permute(2,0,1))
        out.append(den_tensor)

    batch_den = torch.stack(out, 0).to(device=device, dtype=inputs.dtype)
    return batch_den


if __name__=='__main__':
    if sys.argv[1:] is not None:
        args = sys.argv[1:]
    else:
        args = sys.argv[:-1]
    args = parse_args(args)
    
    # Loading config file
    cfg = load_config(args.cfg)
    logger = Logger(task='testing')

    #Seed
    seed = cfg.SEED
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    task = cfg.TEST.subtask
    flip_test = cfg.TEST.flip_test
    logger.info('Flip Test is used --- {}'.format(flip_test))
    
    derainer = RestormerDerainer()
    
    if task == 'test_img':
        assert args.image is not None, "Image can not be None with single image test mode!"
        logger.info('Turning on single image test mode...')
    else:
        logger.info('Turning on evaluation mode...')
    if task == 'eval' and cfg.DATASET.DATA.TEST.FROM_FILE:
        assert cfg.DATASET.DATA.TEST.ANNO_FILE is not None, "Annotation file can not be None with evaluation test mode!"
        assert len(cfg.DATASET.DATA.TEST.ANNO_FILE), "Annotation file can not be empty with evaluation test mode!"
        # assert os.access(cfg.DATASET.DATA.TEST.ANNO_FILE, os.R_OK), "Annotation file must be valid with evaluation test mode!"
    device_count = torch.cuda.device_count()
    
    # build and load/initiate pretrained model
    model = build_model(cfg.MODEL, MODELS).to(torch.float64)
    logger.info('Loading weight ... {}'.format(cfg.TEST.pretrained))
    model = load_pretrained(model, cfg.TEST.pretrained)
    
    
    if device_count >= 1:
        model = nn.DataParallel(model, device_ids=cfg.TEST.gpus).cuda()
    else:
        model = model.cuda()
    
    # Define essential variables
    image = args.image
    test_file = cfg.TEST.test_file
    video_level = cfg.TEST.video_level
    aspect_ratio = cfg.DATASET.IMAGE_SIZE[1]*1.0 / cfg.DATASET.IMAGE_SIZE[0]
    pixel_std = 200
    rot = 0
    
    # 1. Build your normalization transform once, using your cfg values:
    normalize = T.Normalize(
        mean=cfg.DATASET.TRANSFORM.normalize.mean,
        std= cfg.DATASET.TRANSFORM.normalize.std
    )


    transforms = final_transform(cfg.DATASET)
    #geome = GeometryTransform(cfg.DATASET)
    #colori = ColorJitterTransform(cfg.DATASET)
    #normie  = final_transform(cfg.DATASET)
    #transforms = inference_transform

    metrics_base = cfg.METRICS_BASE
    acc_measure = get_acc_mesure_func(metrics_base)
    
    model.eval()
    if image is not None and task == 'test_img':
        img = load_image(image)
        img = cv2.resize(img, (317, 317))
        img = img[60:(317), 30:(287), :]
        c, s = get_center_scale(img.shape[:2], aspect_ratio, pixel_std)
        trans = get_affine_transform(c, s, rot, cfg.DATASET.IMAGE_SIZE)
        input = cv2.warpAffine(img,
                               trans,
                               (int(cfg.DATASET.IMAGE_SIZE[0]), int(cfg.DATASET.IMAGE_SIZE[1])),
                               flags=cv2.INTER_LINEAR,
                              )
        with torch.no_grad():
            st = time.time()
            # 1) apply your existing noise-adding transform
            noisy = transforms(input/255).to(torch.float64)   # shape [C,H,W]
            noisy = noisy.unsqueeze(0)                        # [1,C,H,W]
            if device_count > 0:
                noisy = noisy.cuda(non_blocking=True)

            # 2) pull back to CPU numpy uint8 BGR so Restormer can process
            noisy_cpu = noisy.squeeze(0).cpu().permute(1,2,0).numpy()  # [H,W,C] in RGB [0–1]
            noisy_uint8 = (noisy_cpu * 255).round().astype('uint8')
            noisy_bgr   = cv2.cvtColor(noisy_uint8, cv2.COLOR_RGB2BGR)
            logger.info(f"[NOISY] ndim={noisy_bgr.ndim}, shape={noisy_bgr.shape}, dtype={noisy_bgr.dtype}")
	    
            #output_dir = "output_images"
            #os.makedirs(output_dir, exist_ok=True)
            #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            #noisy_path = os.path.join(output_dir, f"noisy_{timestamp}.png")

            #cv2.imwrite(noisy_path, noisy_bgr)

            # 3) denoise!
            logger.info(f"[DENOISER] Applying Restormer to this noisy image…")
            denoised_bgr = derainer.derain_image(noisy_bgr)
            logger.info(f"[DENOISED] got output with ndim={denoised_bgr.ndim}, shape={denoised_bgr.shape}, dtype={denoised_bgr.dtype}")

            #denoised_path = os.path.join(output_dir, f"denoised_{timestamp}.png")
            #cv2.imwrite(denoised_path, denoised_bgr)

            # 4) convert back to your tensor+normalize workflow
            denoised_rgb    = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
            denoised_tensor = torch.from_numpy(denoised_rgb)\
                                     .float().div(255.)\
                                     .permute(2,0,1)\
                                     .to(torch.float64)
            img_trans = denoised_tensor.unsqueeze(0)
            if device_count > 0:
                img_trans = img_trans.cuda(non_blocking=True)

            # now run through your classifier
            outputs = model(img_trans)
            hm_outputs = outputs[0]['hm']
            cls_outputs = outputs[0]['cls']
            hm_preds = _sigmoid(hm_outputs).cpu().numpy()
            if cfg.TEST.vis_hm:
                print(f'Heatmap max value --- {hm_preds.max()}')
                vis_heatmap(img, hm_preds[0], 'output_pred.jpg')
            label_pred = cls_outputs.cpu().numpy()
            label = 'Fake' if label_pred[0][-1] > cfg.TEST.threshold else 'Real'
            logger.info('Inferencing time --- {}'.format(time.time() - st))
            logger.info('{} --- {}'.format(label, label_pred[0][-1]))
            logger.info('-----------------***--------------------')
    if task == 'eval':
        logger.info(f'Using metric-base {metrics_base} for evaluation!')
        logger.info(f'Video level evaluation mode: {video_level}')
        st = time.time()
        test_dataset = build_dataset(cfg.DATASET, 
                                     DATASETS,
                                     default_args=dict(split='test', config=cfg.DATASET))
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=cfg.TRAIN.batch_size * len(cfg.TRAIN.gpus),
                                     shuffle=True,
                                     num_workers=cfg.DATASET.NUM_WORKERS)
        #logger.info('Dataset loading time --- {}'.format(time.time() - st))

        # Make sure all tensors in same device
        total_preds = torch.tensor([]).cuda().to(dtype=torch.float64)
        total_labels = torch.tensor([]).cuda().to(dtype=torch.float64)
        vid_preds = {}
        vid_labels = {}
        acc = AverageMeter()
        auc = AverageMeter()
        ar = AverageMeter()
        ap = AverageMeter()
        test_dataloader = tqdm(test_dataloader, dynamic_ncols=True)
        with torch.no_grad():
            for b, (inputs, labels, vid_ids) in enumerate(test_dataloader):
                i_st = time.time()
                if device_count > 0:
                    inputs = inputs.to(dtype=torch.float64).cuda()
                    labels = labels.to(dtype=torch.float64).cuda()
                #inputs = noise_and_denoise_batch(inputs, transforms, derainer, inputs.device, output_dir="output_images")
                # 2. Noise & denoise; returns [B, C, H, W] in [0–1] float32 RGB
                #try:
                den_batch = noise_and_denoise_batch(
                    inputs, transforms, derainer, inputs.device,
                    mean=cfg.DATASET.TRANSFORM.normalize.mean,
                    std= cfg.DATASET.TRANSFORM.normalize.std,
                    output_dir="output_images"
                )
                #except Exception:
		    # if deraining fails, fall back to original inputs
                #    den_batch = inputs
                #    print(f"Denoise failed at batch {b}, using original inputs")

		# 3. Re‑normalize exactly as training pipeline expected
		#    normalize works per-sample on [B,C,H,W] in [0–1]
                den_batch_norm = normalize(den_batch)
                outputs = model(den_batch_norm)
                # Applying Flip test
                if flip_test:
                    outputs_1 = model(inputs.flip(dims=(3,)))
                if isinstance(outputs, list):
                    outputs = outputs[0]
                    if flip_test:
                        outputs_1 = outputs_1[0]
                
                #In case outputs contain a dict key
                if isinstance(outputs, dict):
                    if flip_test:
                        hm_outputs = (outputs['hm'] + outputs_1['hm'])/2
                        cls_outputs = (outputs['cls'] + outputs_1['cls'])/2
                    else:
                        hm_outputs = outputs['hm']
                        cls_outputs = outputs['cls']
                logger.info('Inferencing time --- {}'.format(time.time() - st))

                if not video_level:
                    total_preds = torch.cat((total_preds, cls_outputs), 0)
                    total_labels = torch.cat((total_labels, labels), 0)
                else:
                    for idx, vid_id in enumerate(vid_ids):
                        if vid_id in vid_preds.keys(): 
                            vid_preds[vid_id] = torch.cat((vid_preds[vid_id], torch.unsqueeze(cls_outputs[idx], 0)), 0)
                        else:
                            vid_preds[vid_id] = torch.unsqueeze(cls_outputs[idx].clone().detach(), 0).cuda().to(dtype=torch.float64)
                            vid_labels[vid_id] = torch.unsqueeze(labels[idx].clone().detach(), 0).cuda().to(dtype=torch.float64)
            
            if video_level:
                for k in vid_preds.keys():
                    total_preds = torch.cat((total_preds, torch.mean(vid_preds[k], 0, keepdim=True)), 0)
                    total_labels = torch.cat((total_labels, vid_labels[k]), 0)

                # Calculate accuracy and AUC
                # if metrics_base == 'binary':
            acc_ = acc_measure(total_preds, targets=None, labels=total_labels, threshold=cfg.TEST.threshold)
            auc_, ap_, ar_, mf1_ = bin_calculate_auc_ap_ar(total_preds, total_labels, metrics_base=metrics_base)
                # elif metrics_base == 'heatmap':
                #     acc_ = acc_measure(hm_outputs, targets=None, labels=labels, threshold=cfg.TEST.threshold)
                #     auc_, ap_, ar_ = bin_calculate_auc_ap_ar(cls_outputs, labels, metrics_base=metrics_base)
                # else:
                #     acc_ = acc_measure(hm_outputs, cls_outputs, targets=None, labels=labels, cls_lamda=cfg.TRAIN.loss.cls_lmda)
                #     auc_, ap_, ar_ = bin_calculate_auc_ap_ar(cls_outputs, labels, hm_preds=hm_outputs, cls_lamda=cfg.TRAIN.loss.cls_lmda, metrics_base=metrics_base)
                
            acc.update(acc_, n=inputs.size(0))
            if not math.isnan(float(auc_)): 
                auc.update(auc_, n=inputs.size(0))
                ap.update(ap_, n=inputs.size(0))
                ar.update(ar_, n=inputs.size(0))
            logger.info(f'Current ACC, AUC, AP, AR, mF1 for {cfg.DATASET.DATA.TEST.FAKETYPE} --- {cfg.DATASET.DATA.TEST.LABEL_FOLDER} -- \
                {acc_*100} -- {auc_*100} -- {ap_*100} -- {ar_*100} -- {mf1_*100}')
        #     logger.info(f'Overall ACC, AUC, AP, AR for {cfg.DATASET.DATA.TEST.FAKETYPE}--- {cfg.DATASET.DATA.TEST.LABEL_FOLDER} -- {acc.avg*100} -- {auc.avg*100} -- {ap.avg*100} -- {ar.avg*100}')
        # logger.info('-----------------***--------------------')
