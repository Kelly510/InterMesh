import os
import cv2
import copy
import math
import json
import torch
import pickle
import numpy as np
from PIL import Image
from .base import BASE
from torchvision import transforms
from configs.paths import dataset_root


class MuPoTS(BASE):

    def __init__(self, split='test', downsample=1, **kwargs):
        super(MuPoTS, self).__init__(**kwargs)
        assert split == 'test'
        assert downsample == 1

        self.ds_name = 'mupots'
        self.split = split
        self.dataset_path = os.path.join(dataset_root, 'mupots')
        self.annots_path = os.path.join(self.dataset_path, 'mupots_annots.npz')
        self.annots = np.load(self.annots_path, allow_pickle=True)['annots'].item()
        self.img_names = list(self.annots.keys())

    def __len__(self):
        return len(self.img_names)

    def get_raw_data(self, idx):
        img_id = idx % len(self.img_names)
        img_name = self.img_names[img_id]
        annots = copy.deepcopy(self.annots[img_name])
        img_path = os.path.join(self.dataset_path, 'MultiPersonTestSet', img_name)

        pnum = len(annots)
        j2d_list = []
        j3d_list = []
        vis_list = []
        for i in range(pnum):
            j2d_list.append(torch.from_numpy(annots[i]['annot_2d']))
            j3d_list.append(torch.from_numpy(annots[i]['annot_3d']))
            vis_list.append(torch.from_numpy(annots[i]['vis']))
        j2ds = torch.stack(j2d_list).float()
        j3ds = torch.stack(j3d_list).float()
        jvis = torch.stack(vis_list).bool()

        raw_data = {'img_path': img_path, 'ds': 'mupots', 'TS': img_name.split('/')[0], 'pnum': pnum, 'j2ds': j2ds, 'j3ds': j3ds, 'jvis': jvis}
        return raw_data

    def process_bbox(self, boxes, meta_data):
        w, h = meta_data['img_size'][1], meta_data['img_size'][0]
        if not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)

        boxes = np.stack([boxes[:, (0, 1)], boxes[:, (2, 1)], boxes[:, (0, 2)], boxes[:, (2, 3)]], axis=1)
        boxes_img = boxes * np.array([[[w, h]]])
        x_min = np.min(boxes_img[:, :, 0], axis=-1) / w
        y_min = np.min(boxes_img[:, :, 1], axis=-1) / h
        x_max = np.max(boxes_img[:, :, 0], axis=-1) / w
        y_max = np.max(boxes_img[:, :, 1], axis=-1) / h

        transformed_boxes = np.stack([x_min, y_min, x_max, y_max], axis=-1)
        transformed_boxes = torch.tensor(transformed_boxes, dtype=torch.float32)
        transformed_boxes = torch.clamp(transformed_boxes, min=0, max=1)
        return transformed_boxes

    def __getitem__(self, index):
        raw_data = copy.deepcopy(self.get_raw_data(index))

        # Load original image
        ori_img = cv2.imread(raw_data['img_path'])
        img_size = torch.tensor(ori_img.shape[:2])
        raw_data['j2ds'][:, :, 0] /= img_size[1]
        raw_data['j2ds'][:, :, 1] /= img_size[0]

        if img_size[1] >= img_size[0]:
            resize_rate = self.input_size / img_size[1]
            img = cv2.resize(ori_img, dsize=(self.input_size, int(resize_rate * img_size[0])))
            img_size = torch.tensor([int(resize_rate * img_size[0]), self.input_size])
        else:
            resize_rate = self.input_size / img_size[0]
            img = cv2.resize(ori_img, dsize=(int(resize_rate * img_size[1]), self.input_size))
            img_size = torch.tensor([self.input_size, int(resize_rate * img_size[1])])
        raw_data.update({'img_size': img_size, 'resize_rate': resize_rate})

        array2tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        patch_size = 14
        if self.use_sat:
            patch_size = 56
        # pad image to support pooling
        pad_img = np.zeros((math.ceil(img.shape[0] / patch_size) * patch_size, math.ceil(img.shape[1] / patch_size) * patch_size, 3), dtype=img.dtype)
        pad_img[:img.shape[0], :img.shape[1]] = img
        assert max(pad_img.shape[:2]) == self.input_size
        pad_img = Image.fromarray(pad_img[:, :, ::-1].copy())
        norm_img = array2tensor(pad_img)

        if self.use_hoi:
            img_clip, _ = self.hoi_normalizer(self.hoi_transform(pad_img, None)[0], None)
            return norm_img, img_clip, None, pad_img, raw_data
        else:
            return norm_img, None, None, None, raw_data
