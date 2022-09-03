import torch
import cv2
import numpy as np
import random
from torchvision.transforms import functional as F
import math

class Compose(object):
    def __init__(self, transforms):
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
        return image

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
                selected_kps = upper_kds
            else:
                selected_kps = lower_kds
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
        trans = cv2.getAffineTransform(src, dst)
        dst /= 4
        reverse_trans = trans = cv2.getAffineTransform(dst, src)
        
        resize_img = cv2.warpAffine(img, trans, tuple(self.fixed_size[::-1]), flags=cv2.INTER_LINEAR)
        target["trans"] = trans
        target['reverse_trans'] = reverse_trans
        return resize_img, target

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5, matched_parts=None):
        self.p = p
        self.matched_parts = matched_parts

    def __call__(self, image, target):
        if random.random() < self.p:
            image = np.ascontiguousarray(np.flip(image, axis=1))
            keypoints = target["keypoints"]
            visible = target["visible"]
            width = image.shape[1]

            #图像整体x轴作翻转
            keypoints[:, 0] = width - 1 - keypoints[:, 0]

            #关键点翻转
            for pair in self.matched_parts:
                keypoints[pair[0], :], keypoints[pair[1], :] = keypoints[pair[1], :], keypoints[pair[0], :].copy()
                visible[pair[0]], visible[pair[1]] = visible[pair[1]], visible[pair[0]].copy()
            target["keypoints"] = keypoints
            target["visible"] = visible


class keypointoHeatMap(object):
    def __init__(self, heatmap_hw=(256//4, 192//4), gaussian_sigma=2, keypoint_weights=None):
        self.heatmap_hw = heatmap_hw
        self.sigma = gaussian_sigma
        self.kernel_radius = self.sigma * 3
        self.use_kps_weights = False if keypoint_weights is None else True
        self.kps_weights = keypoint_weights

        kernel_size = 2 * self.kernel_radius + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        x_center = y_center = kernel_size // 2
        for x in range(kernel_size):
            for y in range(kernel_size):
                kernel[y,x] = np.exp(-((x - x_center)**2 + (y - y_center)**2)/(2*self.sigma**2))
        self.kernel = kernel

    def __call__(self, image, target):
        kps = target["keypoints"]
        num_kps = kps.shape[0]
        kps_weights = np.ones((num_kps), dtype=np.float32)
        if "visible" in target:
            visible = target["visible"]
            kps_weights = visible

        heatmap = np.zeros((num_kps, self.heatmap_hw[0], self.heatmap_hw[1]), dtype=np.float32)
        heatmap_kps = (kps/4 + 0.5).astype(np.int)
        for kp_id in range(num_kps):
            v = kps_weights[kp_id]
            if v < 0.5:
                continue
            x, y = heatmap_kps[kp_id]
            ul = [x - self.kernel_radius, y - self.kernel_radius]
            br = [x + self.kernel_radius, y + self.kernel_radius]
            if ul[0] > self.heatmap_hw[1] - 1 or ul[1] > self.heatmap_hw[0] - 1 or br[0] < 0 or br[1] < 0:
                kps_weights[kp_id] = 0
                continue
            g_x = (max(0, -ul[0]), min(br[0], self.heatmap_hw[1] - 1) - ul[0])
            g_y = (max(0, -ul[1]), min(br[1], self.heatmap_hw[0] - 1) - ul[1])

            img_x = (max(0, ul[0]), min(br[0], self.heatmap_hw[1] - 1))
            img_y = (max(0, ul[1]), min(br[1], self.heatmap_hw[0] - 1))
            if kps_weights[kp_id] > 0.5:
                heatmap[kp_id][img_y[0]:img_y[1]+1][img_x[0]:img_x[1]+1] = self.kernel[g_y[0]:g_y[1]+1][g_x[0]:g_x[1]+1]
        if self.use_kps_weights:
            kps_weights = np.multiply(kps_weights, self.kps_weights)

        target["heatmap"] = torch.as_tensor(heatmap, dtype=torch.float32)
        target["kps_weights"] = torch.as_tensor(kps_weights, dtype=torch.float32)
        return image, target








        
