import torch
import cv2
import numpy as np
import random
from torchvision.transforms import functional as F
import math

class Compose(object):
    def __init__(self, tranforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transfroms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, std

def scale_box(xmin, ymin , w, h , scale):
    s_h = h*scale[0]
    s_w = w*scale[1]
    xmin = xmin - (s_w - 2)/2
    ymin = ymin - (s_h - 2)/2
    return xmin, ymin, s_w, s_h

class HalfBody(object):
    def __init__(self, p=0.3, upper_body_ids=None, lower_body_ids=None):
        self.p = p
        self.upper_body_ids = upper_body_ids
        self.lower_body_ids = lower_body_ids

    def __call__(self, image, target):
        if random.random() < self.p:
            kps = target["keypoints"]
            vis = target["visible"]
            upper_kds = []
            lower_kds = []

            for i, v in enumerate(vis):
                if v > 0.5:
                    if i in self.upper_body_ids:
                        upper_kds.append(kps[i])
                    else:
                        lower_kds.append(kps[i])

            if random.random() < 0.5:
                selected_kps = upper_kps
            else:
                selected_kps = lower_kps
            if len(selected_kps) > 2:
                selected_kps = np.array(selected_kps, dtype=np.float32)
                xmin, ymin = np.min(selected_kps, axis=0).tolist()
                xmax, ymax = np.max(selected_kps, axis=0).tolist()
                w = xmax - xmin
                h = ymax - ymin
                xmin, ymin, w, h = scale_box(xmin, ymin, w, h, (1.5, 1.5))
                target["bbox"] = [xmin, ymin, w, h]
        return image, target

#通过拉伸使图片长宽比固定不变
def adjust_box(xmin, ymin, w, h, fixed_size):
    xmax = xmin + w
    ymax = ymin + h
    hw_ra = fixed_size[0]/fixed_size[1]

    if h/w > hw_ra:
        wi = h/hw_ra
        p_w = (wi - w)/2
        xmin = xmin - p_w
        xmax = xmax + p_w
    else:
        hi = w*hw_ra
        p_h = (hi-h)/2
        ymin = ymin - p_h
        ymax = ymax + p_h
    return xmin, ymin, xmax, ymax

class AffineTransform(object):
    def __init__(self, scale, rotation, fixed_size):
        self.scale = scale
        self.rotation = rotation
        self.fixed_size = fixed_size
    def __call__(self, img, target):
        xmin, ymin, xmax, ymax = adjust_box(*target["box"], self.fixed_size)
        center = np.array[(xmin + xmax)/2, (ymax+ymin)/2]
        h = ymax - ymin
        w = xmax - xmin
        topm = center + np.array([0, -h/2])
        rm = center + np.array[w/2 ,0]

        if self.scale is not None:
            scale = random.uniform(*self.scale)
            w = w*scale
            h = h*scale
            topm = center + np.array([0, -h/2])
            rm = center + np.array([w/2, 0])

        if self.rotation is not None:
            angle = random.randint(*self.rotation)
            angle = angle/180*math.pi
            topm = center + np.array([h/2*math.sin(angle), -h/2*math.cos(angle)])
            rm = center + np.array([w/2*math.cos(angle), w/2*math.sin(angle)])
            
        d_center = np.array([(self.fixed_size[0]-1)/2, (self.fixed_size[1]-1)/2])
        d_topm = np.array([0, (self.fixed_size[1]-1)/2])
        d_rm = np.array([(self.fixed_size[0]-1)/2, (self.fixed_size[1]-1)])
        
        src = np.stack([center, topm , rm])
        dst = np.stack([d_center, d_topm, d_rm])
        trans = cv2.getAffineTransform(src, dist)
        dst /= 4
        reverse_trans = trans = cv2.getAffineTransform(dist, src)
        
        resize_img = cv2.warpAffine(img, trans, tuple(self.fixed_size[::-1]),flags= cv2.INTER_LINEAR)
        target["trans"]=trans
        target['reverse_trans']=reverse_trans
        return resize_img, target

        
