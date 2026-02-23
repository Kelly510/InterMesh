import os
import cv2
import json
import copy
import math
import pickle
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from math import radians, sin, cos

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from utils.transforms import unNormalize, to_zorder
from utils.visualization import tensor_to_BGR, vis_meshes_img, vis_scale_img, get_colors_rgb

from models.human_models import smpl_gendered
from utils.map import gen_scale_map, build_z_map
from utils.constants import smpl_24_flip, smpl_root_idx
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from models.hoi_detect.detr.datasets import transforms_clip as T


class BASE(Dataset):

    def __init__(self, input_size=1288, aug=True, mode='train', human_type='smpl', sat_cfg=None, aug_cfg=None, hoi_cfg=None):
        self.input_size = input_size
        self.aug = aug
        if mode not in ['train', 'eval', 'infer']:
            raise NotImplementedError
        if human_type not in ['smpl', 'no']:
            raise NotImplementedError
        self.mode = mode
        self.human_type = human_type
        assert sat_cfg is not None
        self.use_sat = sat_cfg['use_sat']
        self.sat_cfg = sat_cfg
        self.use_hoi = hoi_cfg['use_hoi']
        self.hoi_cfg = hoi_cfg

        if self.use_hoi:
            self.hoi_normalizer = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.hoi_transform = T.IResize([336, 336])
            self.detr_normalizer = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.detr_transform = T.RandomResize([800], max_size=1333)

        if self.use_sat:
            assert input_size % 56 == 0

        if self.mode == 'train' and aug_cfg is None:
            aug_cfg = {'rot_range': [-15, 15], 'scale_range': [0.8, 1.8], 'flip_ratio': 0.5, 'crop_ratio': 1.0}
        self.aug_cfg = aug_cfg

        # set default values for missing keys
        if self.mode == 'train':
            if 'rot_range' not in self.aug_cfg:
                self.aug_cfg['rot_range'] = [-0, 0]
            if 'scale_range' not in self.aug_cfg:
                self.aug_cfg['scale_range'] = [1.0, 1.0]
            if 'flip_ratio' not in self.aug_cfg:
                self.aug_cfg['flip_ratio'] = 0.0
            if 'crop_ratio' not in self.aug_cfg:
                self.aug_cfg['crop_ratio'] = 0.0

        if human_type == 'smpl':
            self.poses_flip = smpl_24_flip
            self.num_poses = 24
            self.num_betas = 10
            self.num_kpts = 45
            self.human_model = smpl_gendered

        self.vis_thresh = 4  # least num visible kpts for a valid individual

        self.img_keys = ['img_path', 'ds', 'pnum', 'img_size', 'resize_rate', 'cam_intrinsics', '3d_valid', 'detect_all_people', 'scale_map', 'scale_map_pos', 'scale_map_hw']
        self.human_keys = ['boxes', 'labels', 'poses', 'betas', 'transl', 'verts', 'j3ds', 'j2ds', 'j2ds_mask', 'depths', 'focals', 'genders']

        z_depth = math.ceil(math.log2(self.input_size // 28))
        self.z_order_map, self.y_coords, self.x_coords = build_z_map(z_depth)

    def get_raw_data(self, idx):
        raise NotImplementedError

    def get_aug_dict(self):
        if self.aug:
            rot = random.uniform(*self.aug_cfg['rot_range'])
            flip = random.random() <= self.aug_cfg['flip_ratio']
            scale = random.uniform(*self.aug_cfg['scale_range'])
            crop = random.random() <= self.aug_cfg['crop_ratio']
        else:
            rot = 0.
            flip = False
            scale = 1.
            crop = False

        return {'rot': rot, 'flip': flip, 'scale': scale, 'crop': crop}

    def process_img(self, img, meta_data, rot=0., flip=False, scale=1.0):
        # resize
        img_size = torch.tensor(img.shape[:2])
        if img_size[1] >= img_size[0]:
            resize_rate = self.input_size / img_size[1]
            img = cv2.resize(img, dsize=(self.input_size, int(resize_rate * img_size[0])))
            img_size = torch.tensor([int(resize_rate * img_size[0]), self.input_size])
        else:
            resize_rate = self.input_size / img_size[0]
            img = cv2.resize(img, dsize=(int(resize_rate * img_size[1]), self.input_size))
            img_size = torch.tensor([self.input_size, int(resize_rate * img_size[1])])
        meta_data.update({'img_size': img_size, 'resize_rate': resize_rate})

        # flip
        if flip:
            img = np.flip(img, axis=1)
            rot = -rot

        # rot and scale
        img_valid = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
        M = cv2.getRotationMatrix2D((int(img_size[1] / 2), int(img_size[0] / 2)), rot, scale)
        img = cv2.warpAffine(img, M, dsize=(img.shape[1], img.shape[0]))
        img_valid = cv2.warpAffine(img_valid, M, dsize=(img.shape[1], img.shape[0]))
        meta_data.update({'img_valid': img_valid})

        return img, M

    def occlusion_aug(self, meta_data):
        occ_boxes = []
        imght, imgwidth = meta_data['img_size']
        for bbox in box_cxcywh_to_xyxy(meta_data['boxes']):
            bbox = bbox.clone()
            bbox *= self.input_size
            xmin, ymin = bbox[:2]
            xmax, ymax = bbox[2:]

            if random.random() <= 0.6:
                counter = 0
                while True:
                    # force to break if no suitable occlusion
                    if counter > 5:
                        synth_ymin, synth_h, synth_xmin, synth_w = 0, 0, 0, 0
                        break
                    counter += 1

                    area_min = 0.0
                    area_max = 0.3
                    synth_area = (random.random() * (area_max - area_min) + area_min) * (xmax - xmin) * (ymax - ymin)

                    ratio_min = 0.5
                    ratio_max = 1 / 0.5
                    synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

                    synth_h = math.sqrt(synth_area * synth_ratio)
                    synth_w = math.sqrt(synth_area / synth_ratio)
                    synth_xmin = random.random() * ((xmax - xmin) - synth_w - 1) + xmin
                    synth_ymin = random.random() * ((ymax - ymin) - synth_h - 1) + ymin

                    if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < imgwidth and synth_ymin + synth_h < imght:
                        synth_xmin = int(synth_xmin)
                        synth_ymin = int(synth_ymin)
                        synth_w = int(synth_w)
                        synth_h = int(synth_h)
                        break
            else:
                synth_ymin, synth_h, synth_xmin, synth_w = 0, 0, 0, 0
            occ_boxes.append((synth_ymin, synth_h, synth_xmin, synth_w))
        return occ_boxes

    def get_boxes(self, meta_data):
        j2ds = meta_data['j2ds']
        j2ds_mask = meta_data['j2ds_mask']
        pnum = meta_data['pnum']

        bboxes_list = []

        for i in range(pnum):
            kpts = j2ds[i].clone()
            min_xy = kpts.min(dim=0)[0]
            max_xy = kpts.max(dim=0)[0]
            bbox_xyxy = torch.cat([min_xy, max_xy], dim=0)
            bboxes_list.append(bbox_xyxy)

        imght, imgwidth = meta_data['img_size']
        boxes = box_xyxy_to_cxcywh(torch.stack(bboxes_list)) / self.input_size
        boxes[..., 2:] *= 1.2
        boxes = box_cxcywh_to_xyxy(boxes)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(min=0.000001, max=(imgwidth - 1) / self.input_size)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(min=0.000001, max=(imght - 1) / self.input_size)
        boxes = box_xyxy_to_cxcywh(boxes)

        meta_data.update({'boxes': boxes})

    def process_cam(self, meta_data, rot=0., flip=False, scale=1.):
        img_size = meta_data['img_size']
        resize_rate = meta_data['resize_rate']
        rot_aug_mat = meta_data['rot_aug_mat']
        cam_intrinsics = meta_data['cam_intrinsics']

        # resize
        cam_intrinsics[:, 0:2, 2] *= resize_rate * scale
        cam_intrinsics[:, [0, 1], [0, 1]] *= resize_rate * scale
        cam_intrinsics[:, 0, 2] += (1 - scale) * img_size[1] / 2
        cam_intrinsics[:, 1, 2] += (1 - scale) * img_size[0] / 2
        # rotation
        princpt = cam_intrinsics[:, 0:2, 2].clone()
        princpt[..., 0] -= img_size[1] / 2
        princpt[..., 1] -= img_size[0] / 2
        princpt = torch.matmul(princpt, rot_aug_mat[:2, :2].transpose(-1, -2))
        princpt[..., 0] += img_size[1] / 2
        princpt[..., 1] += img_size[0] / 2
        cam_intrinsics[:, 0:2, 2] = princpt
        # flip
        if flip:
            cam_intrinsics[:, 0, 2] = img_size[1] - cam_intrinsics[:, 0, 2]
        meta_data.update({'cam_intrinsics': cam_intrinsics})

        #cam_ext
        new_cam_rot = torch.matmul(rot_aug_mat.unsqueeze(0), meta_data['cam_rot'])
        new_cam_trans = torch.matmul(meta_data['cam_trans'], rot_aug_mat.transpose(-1, -2))
        meta_data.update({'cam_rot': new_cam_rot, 'cam_trans': new_cam_trans})

    def process_smpl(self, meta_data, rot=0., flip=False, scale=1.):
        poses = meta_data['poses']
        bs = poses.shape[0]
        assert poses.ndim == 2
        assert tuple(poses.shape) == (bs, self.num_poses * 3)
        # Merge rotation to smpl global_orient
        global_orient = poses[:, :3].clone()
        cam_rot = meta_data['cam_rot'].numpy()
        for i in range(global_orient.shape[0]):
            root_pose = global_orient[i].view(1, 3).numpy()
            R = cam_rot[i].reshape(3, 3)
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
            root_pose = torch.from_numpy(root_pose).flatten()
            global_orient[i] = root_pose
        poses[:, :3] = global_orient

        # Flip smpl parameters
        if flip:
            poses = poses.reshape(bs, self.num_poses, 3)
            poses = poses[:, self.poses_flip, :]
            poses[..., 1:3] *= -1  # multiply -1 to y and z axis of axis-angle
            poses = poses.reshape(bs, -1)

        # Update all pose params
        meta_data.update({'poses': poses})

        # Get vertices and joints in cam_coords
        with torch.no_grad():
            smpl_kwargs = {'poses': meta_data['poses'], 'betas': meta_data['betas']}
            if 'genders' in meta_data:
                smpl_kwargs.update({'genders': meta_data['genders']})
            verts, j3ds = self.human_model(**smpl_kwargs)

        j3ds = j3ds[:, :self.num_kpts, :]
        root = j3ds[:, smpl_root_idx, :].clone()  # smpl root
        # new translation in cam_coords
        transl = torch.bmm((root+meta_data['transl']).reshape(-1,1,3),meta_data['cam_rot'].transpose(-1,-2)).reshape(-1,3)\
            +meta_data['cam_trans']-root
        if flip:
            transl[..., 0] = -transl[..., 0]

        meta_data.update({'transl': transl})

        verts = verts + transl.reshape(-1, 1, 3)
        j3ds = j3ds + transl.reshape(-1, 1, 3)
        meta_data.update({'verts': verts, 'j3ds': j3ds})

    def project_joints(self, meta_data):
        j3ds = meta_data['j3ds']
        cam_intrinsics = meta_data['cam_intrinsics']
        j2ds_homo = torch.matmul(j3ds, cam_intrinsics.transpose(-1, -2))
        j2ds = j2ds_homo[..., :2] / (j2ds_homo[..., 2, None])

        meta_data.update({'j3ds': j3ds, 'j2ds': j2ds})

    def check_visibility(self, meta_data):
        img_valid = meta_data['img_valid']
        img_size = meta_data['img_size']

        j2ds = meta_data['j2ds']
        j2ds_mask = meta_data['j2ds_mask'] if 'j2ds_mask' in meta_data else torch.ones_like(j2ds, dtype=bool)

        j2ds_vis = torch.from_numpy(img_valid[j2ds[..., 1].int().clip(0, img_size[0] - 1), j2ds[..., 0].int().clip(0, img_size[1] - 1)] > 0)
        j2ds_vis &= (j2ds[..., 1] >= 0) & (j2ds[..., 1] < img_size[0])
        j2ds_vis &= (j2ds[..., 0] >= 0) & (j2ds[..., 0] < img_size[1])

        j2ds_invalid = ~j2ds_vis
        j2ds_mask[j2ds_invalid] = False
        meta_data.update({'j2ds_mask': j2ds_mask})

        vis_cnt = j2ds_mask[..., 0].sum(dim=-1)  # num of visible joints per person
        valid_msk = (vis_cnt >= self.vis_thresh)

        pnum = valid_msk.sum().item()

        if pnum == 0:
            meta_data['pnum'] = pnum
            return

        if pnum < meta_data['pnum']:
            meta_data['pnum'] = pnum
            for key in self.human_keys:
                if key in meta_data:
                    if isinstance(meta_data[key], list):
                        meta_data[key] = np.array(meta_data[key])[valid_msk].tolist()
                    else:
                        meta_data[key] = meta_data[key][valid_msk]
            if 'cam_intrinsics' in meta_data and len(meta_data['cam_intrinsics']) > 1:
                meta_data['cam_intrinsics'] = meta_data['cam_intrinsics'][valid_msk]

    def process_data(self, img, raw_data, rot=0., flip=False, scale=1., crop=False):
        meta_data = copy.deepcopy(raw_data)
        # prepare rotation augmentation mat.
        rot_aug_mat = torch.tensor([[cos(radians(-rot)), -sin(radians(-rot)), 0.], [sin(radians(-rot)), cos(radians(-rot)), 0.], [0., 0., 1.]])
        meta_data.update({'rot_aug_mat': rot_aug_mat})

        img, M = self.process_img(img, meta_data, rot, flip, scale)

        self.process_cam(meta_data, rot, flip, scale)
        self.process_smpl(meta_data, rot, flip, scale)
        self.project_joints(meta_data)

        if crop:
            img_valid = meta_data['img_valid']
            bottom_pelvis_y = torch.max(meta_data['j2ds'][:, 9, 1]).item()  # joint-9 is spine
            orig_img_h, orig_img_w = meta_data['img_size']
            bottom_pelvis_y = max(orig_img_h // 2, bottom_pelvis_y)
            if bottom_pelvis_y < orig_img_h:
                crop_img_h = int(np.random.beta(2, 5) * (orig_img_h - bottom_pelvis_y) + bottom_pelvis_y)
                crop_img_w = orig_img_w
                img = img[:crop_img_h]
                img_valid = img_valid[:crop_img_h]
                if crop_img_h >= crop_img_w:
                    resize_rate = self.input_size / crop_img_h
                    new_img_h = self.input_size
                    new_img_w = int(resize_rate * crop_img_w)
                    img = cv2.resize(img, dsize=(new_img_w, new_img_h))
                    img_valid = cv2.resize(img_valid, dsize=(new_img_w, new_img_h))
                else:
                    resize_rate = self.input_size / crop_img_w
                    new_img_w = self.input_size
                    new_img_h = int(resize_rate * crop_img_h)
                    img = cv2.resize(img, dsize=(new_img_w, new_img_h))
                    img_valid = cv2.resize(img_valid, dsize=(new_img_w, new_img_h))
                meta_data['img_size'] = torch.tensor([new_img_h, new_img_w])
                meta_data['resize_rate'] *= resize_rate
                meta_data['img_valid'] = img_valid
                meta_data['j2ds'] *= resize_rate

                meta_data['cam_intrinsics'][:, [0, 1], [0, 1]] *= resize_rate
                meta_data['cam_intrinsics'][:, :2, 2] = meta_data['cam_intrinsics'][:, :2, 2] * resize_rate + 0.5 * (resize_rate - 1)

                S_scale = np.array([[resize_rate, 0, 0], [0, resize_rate, 0], [0, 0, 1]], dtype=np.float32)
                M_new = S_scale @ np.vstack([M, [0, 0, 1]])
                M = M_new[:2, :]

        self.check_visibility(meta_data)

        matcher_vis = meta_data['j2ds_mask'][:, :22, 0].sum(dim=-1)  # num of visible joints used in Hungarian Matcher
        if meta_data['pnum'] == 0 or not torch.all(matcher_vis):
            if self.mode == 'train':
                meta_data['pnum'] = 0
                return img, meta_data, M

        j3ds = meta_data['j3ds']
        depths = j3ds[:, smpl_root_idx, [2]].clone()
        if len(meta_data['cam_intrinsics']) == 1:
            focals = torch.full_like(depths, meta_data['cam_intrinsics'][0, 0, 0])
        else:
            focals = meta_data['cam_intrinsics'][:, 0, 0][:, None]
        depths = torch.cat([depths, depths / focals], dim=-1)
        meta_data.update({'depths': depths, 'focals': focals})

        self.get_boxes(meta_data)

        meta_data.update({'labels': torch.zeros(meta_data['pnum'], dtype=int)})

        # VI. Occlusion augmentation
        if self.aug:
            occ_boxes = self.occlusion_aug(meta_data)
            for (synth_ymin, synth_h, synth_xmin, synth_w) in occ_boxes:
                img[synth_ymin:synth_ymin + synth_h, synth_xmin:synth_xmin + synth_w, :] = np.random.rand(synth_h, synth_w, 3) * 255

        if self.use_sat:
            # scale map
            boxes = meta_data['boxes']
            scales = boxes[:, 2:].norm(p=2, dim=1).clamp(0., 1.)
            v3ds = meta_data['verts']
            depths_norm = meta_data['depths'][:, 1]
            cam_intrinsics = meta_data['cam_intrinsics']
            sorted_idx = torch.argsort(depths_norm, descending=True)
            map_size = (meta_data['img_size'] + 27) // 28

            scale_map = gen_scale_map(scales[sorted_idx], v3ds[sorted_idx], faces=self.human_model.faces, cam_intrinsics=cam_intrinsics[sorted_idx] if len(cam_intrinsics) > 1 else cam_intrinsics, map_size=map_size, patch_size=28, pad=True)
            scale_map_z, _, pos_y, pos_x = to_zorder(scale_map, z_order_map=self.z_order_map, y_coords=self.y_coords, x_coords=self.x_coords)
            meta_data['scale_map'] = scale_map_z
            meta_data['scale_map_pos'] = {'pos_y': pos_y, 'pos_x': pos_x}
            meta_data['scale_map_hw'] = scale_map.shape[:2]

        return img, meta_data, M

    def process_bbox(self, boxes, meta_data, flip=False, M=None):
        w, h = meta_data['img_size'][1], meta_data['img_size'][0]
        if not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)
        if flip:
            boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]

        boxes = np.stack([boxes[:, (0, 1)], boxes[:, (2, 1)], boxes[:, (0, 3)], boxes[:, (2, 3)]], axis=1)
        boxes_img = boxes * np.array([[[w, h]]])
        boxes_img = cv2.transform(boxes_img, M)  # 100, 4, 2
        x_min = np.min(boxes_img[:, :, 0], axis=-1) / w
        y_min = np.min(boxes_img[:, :, 1], axis=-1) / h
        x_max = np.max(boxes_img[:, :, 0], axis=-1) / w
        y_max = np.max(boxes_img[:, :, 1], axis=-1) / h

        transformed_boxes = np.stack([x_min, y_min, x_max, y_max], axis=-1)
        transformed_boxes = torch.tensor(transformed_boxes, dtype=torch.float32)
        transformed_boxes = torch.clamp(transformed_boxes, min=0, max=1)
        return transformed_boxes

    def __getitem__(self, index):

        raw_data = self.get_raw_data(index)

        # Load original image
        ori_img = cv2.imread(raw_data['img_path'])
        if raw_data['ds'] == 'bedlam' and 'closeup' in raw_data['img_path']:
            ori_img = cv2.rotate(ori_img, cv2.ROTATE_90_CLOCKWISE)
        img_size = torch.tensor(ori_img.shape[:2])
        raw_data.update({'img_size': img_size})

        if self.mode == 'train':
            cnt = 0
            while (True):
                aug_dict = self.get_aug_dict()
                img, meta_data, M = self.process_data(ori_img, raw_data, **aug_dict)
                if meta_data['pnum'] > 0:
                    break
                cnt += 1
                if cnt >= 10:
                    aug_dict.update({'rot': 0., 'scale': 1.})
                    img, meta_data, M = self.process_data(ori_img, raw_data, **aug_dict)
                    if meta_data['pnum'] == 0:
                        print('skipping: ' + meta_data['img_path'])
                    return self.__getitem__(index + 1)

        elif self.mode == 'eval':
            assert not self.aug, f'No need to use augmentation when mode is {self.mode}!'
            aug_dict = self.get_aug_dict()
            img, meta_data, M = self.process_data(ori_img, raw_data, **aug_dict)

        else:
            assert not self.aug, f'No need to use augmentation when mode is {self.mode}!'
            aug_dict = self.get_aug_dict()
            meta_data = raw_data
            img, M = self.process_img(ori_img, meta_data)

        # delete unwanted keys
        if self.mode == 'train':
            for key in list(meta_data.keys()):
                if key not in self.img_keys and key not in self.human_keys:
                    del meta_data[key]

        if self.aug:
            array2tensor = transforms.Compose([transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
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
        pad_img = pad_img.convert('RGB')

        if 'j2ds_mask' in meta_data:
            meta_data['j2ds_mask'][:, :, :] = True

        if self.use_hoi:
            img_clip, _ = self.hoi_normalizer(self.hoi_transform(pad_img, None)[0], None)
            img_detr, _ = self.detr_normalizer(self.detr_transform(pad_img, None)[0], None)
            return norm_img, img_clip, img_detr, meta_data
        else:
            return norm_img, None, None, meta_data

    def visualize(self, results_save_dir=None, vis_num=100, vis_categories=['verts', 'boxes', 'scale_map']):
        if results_save_dir is None:
            results_save_dir = os.path.join('./datasets_visualization', f'{self.ds_name}_{self.split}')
        os.makedirs(results_save_dir, exist_ok=True)

        vis_interval = len(self) // vis_num

        print(f'Visualization results will be saved in {results_save_dir}')

        for idx in tqdm(range(len(self))):
            if idx % vis_interval != 0:
                continue

            norm_img, _, _, _, targets = self.__getitem__(idx)

            ori_img = tensor_to_BGR(unNormalize(norm_img).cpu())
            img_name = targets['img_path'].split('/')[-1].split('.')[-2]
            pnum = targets['pnum']

            if 'verts' in targets and 'verts' in vis_categories:
                colors = get_colors_rgb(len(targets['verts']))
                mesh_img = vis_meshes_img(img=ori_img.copy(), verts=targets['verts'], smpl_faces=self.human_model.faces, cam_intrinsics=targets['cam_intrinsics'].cpu(), colors=colors, padding=False)
                cv2.imwrite(os.path.join(results_save_dir, f'{idx}_{img_name}_mesh.jpg'), mesh_img)

            if 'boxes' in targets and 'boxes' in vis_categories:
                gt_img = ori_img.copy()
                boxes = box_cxcywh_to_xyxy(targets['boxes']) * self.input_size
                for i, bbox in enumerate(boxes):
                    bbox = bbox.int().tolist()
                    cv2.rectangle(gt_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=2)

                cv2.imwrite(os.path.join(results_save_dir, f'{idx}_{img_name}_boxes.jpg'), gt_img)

            if 'scale_map' in targets and 'scale_map' in vis_categories:
                gt_img = ori_img.copy()
                flatten_map = targets['scale_map']
                ys, xs = targets['scale_map_pos']['pos_y'], targets['scale_map_pos']['pos_x']
                h, w = targets['scale_map_hw']
                scale_map = torch.zeros((h, w, 2))
                scale_map[ys, xs] = flatten_map
                img = vis_scale_img(gt_img, scale_map, patch_size=28)

                cv2.imwrite(os.path.join(results_save_dir, f'{idx}_{img_name}_scales.jpg'), img)

            if 'j2ds' in targets:
                gt_img = ori_img.copy()
                j2ds = targets['j2ds']
                j2ds_mask = targets['j2ds_mask']
                for kpts, valids in zip(j2ds, j2ds_mask):
                    for kpt, valid in zip(kpts, valids):
                        if not valid.all():
                            continue
                        kpt_int = kpt.numpy().astype(int)
                        cv2.circle(gt_img, kpt_int, 2, (0, 0, 255), -1)
                cv2.imwrite(os.path.join(results_save_dir, f'{idx}_{img_name}_joints.png'), np.hstack([ori_img, gt_img]))
