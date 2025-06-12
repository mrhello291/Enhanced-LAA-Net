#-*- coding: utf-8 -*-
from copy import deepcopy
import cv2
import numpy as np

from torchvision import transforms
import albumentations as alb


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center, 
                         scale, 
                         rot, 
                         output_size, 
                         shift=np.array([0, 0], dtype=np.float32), 
                         inv=0, 
                         pixel_std=200):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * pixel_std
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180

    src_dir = get_dir([0, (src_w-1) * -0.5], rot_rad)
    dst_dir = np.array([0, (dst_w-1) * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [(dst_w-1) * 0.5, (dst_h-1) * 0.5]
    dst[1, :] = np.array([(dst_w-1) * 0.5, (dst_h-1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    """
    This function apply the affine transform to each point given by an affine matrix
    """
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def draw_landmarks(image, landmarks):
    """This function is to draw facial landmarks into transformed images
    """
    assert landmarks is not None, "Landmarks can not be None!"
    
    img_cp = deepcopy(image)
    
    for i, p in enumerate(landmarks):
        img_cp = cv2.circle(img_cp, (p[0], p[1]), 2, (0, 255, 0), 1)
    
    return img_cp


def get_center_scale(shape, aspect_ratio, pixel_std=200):
    h, w = shape[0], shape[1]
    center = np.zeros((2), dtype=np.float32)
    center[0] = (shape[1] - 1) / 2
    center[1] = (shape[0] - 1) / 2
    
    if w > h * aspect_ratio:
        h = w * 1.0 / aspect_ratio
    else:
        w = h * 1.0 / aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32
    )
    
    return center, scale


def randaffine(img, mask):
    f = alb.Affine(translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
		           scale=[0.95,1/0.95],
		           fit_output=False,
		           p=1)
			
    g = alb.ElasticTransform(alpha=50, 
                             sigma=7,
		                     alpha_affine=0,
		                     p=1)

    transformed=f(image=img,mask=mask)
    img=transformed['image']
    
    mask=transformed['mask']
    transformed=g(image=img,mask=mask)
    mask=transformed['mask']
    return img, mask

class GeometryTransform:
    """
    Read geometry-related config and apply resize, flip, crop, scale, random erasing.
    """
    def __init__(self, cfg):
        geom = cfg.TRANSFORM.geometry
        self.resize = geom.resize        # [h, w, prob]
        self.hflip_p = geom.horizontal_flip
        self.crop_lim, self.crop_p = geom.cropping
        self.scale_lim, self.scale_p = geom.scale
        self.rand_erasing_p, self.rand_erasing_max = geom.rand_erasing

    def __call__(self, image):
        # Resize
        if np.random.rand() < self.resize[2]:
            image = cv2.resize(image, (self.resize[1], self.resize[0]))
        # Horizontal flip
        if np.random.rand() < self.hflip_p:
            image = cv2.flip(image, 1)
        # Random crop
        if np.random.rand() < self.crop_p:
            h, w = image.shape[:2]
            crop_h = int(h * (1 - self.crop_lim * np.random.rand()))
            crop_w = int(w * (1 - self.crop_lim * np.random.rand()))
            y = np.random.randint(0, h - crop_h)
            x = np.random.randint(0, w - crop_w)
            image = image[y:y+crop_h, x:x+crop_w]
            image = cv2.resize(image, (w, h))
        # Random scale
        if np.random.rand() < self.scale_p:
            factor = 1 + (np.random.rand()*2-1)*self.scale_lim
            h, w = image.shape[:2]
            image = cv2.resize(image, (int(w*factor), int(h*factor)))
            image = cv2.resize(image, (w, h))
        # Random erasing
        if np.random.rand() < self.rand_erasing_p:
            h, w = image.shape[:2]
            for _ in range(int(self.rand_erasing_max)):
                x = np.random.randint(0, w)
                y = np.random.randint(0, h)
                erase_w = np.random.randint(w // 10, w // 3)
                erase_h = np.random.randint(h // 10, h // 3)
                image[y:y+erase_h, x:x+erase_w] = np.random.randint(0, 256)
        return image

class ColorJitterTransform:
    """
    Read color-related config and apply CLAHE, jitter, blur, noise, JPEG compression, etc.
    """
    def __init__(self, cfg):
        col = cfg.TRANSFORM.color
        self.clahe_p = col.clahe
        self.jitter_p = col.colorjitter
        self.blur_p = col.gaussianblur
        self.noise_p = col.gaussnoise
        self.jpeg_p, self.jpeg_low, self.jpeg_high = col.jpegcompression
        self.rgbshift_p = col.rgbshift
        self.contrast_p = col.randomcontrast
        self.gamma_p = col.randomgamma
        self.brightness_p = col.randombrightness
        self.huesat_p = col.huesat

    def __call__(self, image):
        # CLAHE
        #if np.random.rand() < self.clahe_p:
        #    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        #    l, a, b = cv2.split(lab)
         #   clahe = cv2.createCLAHE().apply(l)
          #  lab = cv2.merge((clahe, a, b))
          #  image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # Color jitter (HSV space)
        #if np.random.rand() < self.jitter_p:
        #    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        #    hsv[...,1] *= 1 + (np.random.rand()*2-1)*0.3
        #    hsv[...,2] *= 1 + (np.random.rand()*2-1)*0.3
        #    image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        # Gaussian blur
        #if np.random.rand() < self.blur_p:
        #    k = int(3 + 2*np.random.randint(1,3)) | 1
        #    image = cv2.GaussianBlur(image, (k,k), sigmaX=0)
        # Gaussian noise
        if np.random.rand() < self.noise_p:
            noise = np.random.randn(*image.shape) * 100 * self.noise_p
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        # JPEG compression
        #if np.random.rand() < self.jpeg_p:
        #    q = int(self.jpeg_low + np.random.rand()*(self.jpeg_high-self.jpeg_low))
        #    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        #    _, encimg = cv2.imencode('.jpg', image, encode_param)
        #    image = cv2.imdecode(encimg, 1)
        return image

def final_transform(_cfg):
    #def composed_transform(image_np):
    #    image_np = GeometryTransform(_cfg)(image_np)
    #    image_np = ColorJitterTransform(_cfg)(image_np)
    #    tensor_img = transforms.ToTensor()(image_np)
    #    return transforms.Normalize(mean=_cfg.TRANSFORM.normalize.mean,
    #                               std=_cfg.TRANSFORM.normalize.std)(tensor_img)
    #return composed_transform
    
    #GeometryTransform(_cfg),
    return transforms.Compose([   
        ColorJitterTransform(_cfg),
        transforms.ToTensor(),
        transforms.Normalize(
	    mean=_cfg.TRANSFORM.normalize.mean,
            std=_cfg.TRANSFORM.normalize.std),
    ])

    #return transforms.Compose([
    #            transforms.ToTensor(),
    #            transforms.Normalize(
    #                mean=_cfg.TRANSFORM.normalize.mean, 
    #                std=_cfg.TRANSFORM.normalize.std,
    #            ),
    #        ])
