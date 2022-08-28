import os
import torch
import numpy as np
import cv2
from pycocotools.coco import COCO
from torch.utils import data
import copy


class Cocokeypoints(data.Dataset):
    def __int__(self, root, dataset="train", transforms=None, fixed_size=[256, 192]):
        super(Cocokeypoints, self).__int__()
        self.anno_file = root + "/annotations/" + f'person_keypoints_{dataset}2017.json'
        self.image_file = os.path.join(root, "images", dataset, "2017")
        self.mode = dataset
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.coco = COCO(self.anno_file)
        img_ids = list(sorted(self.coco.imgs.keys()))

        self.valid_person_list = []
        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            ann_ids = self.coco.getAnnIds(imIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if ann["category_id"] != 1:
                    continue
                if "keypoints" not in ann:
                    continue
                if max(ann["keypoints"]) == 0:
                    continue
            xmin, ymin, w, h = ann['bbox']
            obj_index = 0
            if w > 0 and h > 0:
                info = {
                    "bbox": [amin, ymin, w, h],
                    "image_id": img_id,
                    "image_path": os.path.join(self.img_root, img_info["file_name"]),
                    "image_width": img_info["width"],
                    "image_height": imge_info["height"],
                    "obj_origin_hw": [h, w],
                    "obj_index": obj_idx,
                    "score": ann["score"] if "score" in ann else 1.
                }
                keypoints = np.array(ann["keypoints"]).reshape(-1, 3)
                visible = keypoints[:, 2]
                keypoints = keypoints[:, :2]
                info["keypoints"] = keypoints
                info["visible"] = visible
                self.valid_person_list.append(info)
                obj_index += 1

    def __getitem__(self, item):
        target = copy.deepcopy(self.valid_person_list[item])
        image = cv2.imread(target["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            image, person_info = self.transforms(image, target)
        return image, target

    def collate_fn(batch):
        imgs_tuple, targets_tuple = tuple(zip(*batch))
        imgs_tensor = torch.stack(imgs_tuple)
        return imgs_tensor, targets_tuple