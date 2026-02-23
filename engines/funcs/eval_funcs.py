import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict

from utils.transforms import unNormalize
from utils.constants import H36M_EVAL_JOINTS
from utils.visualization import tensor_to_BGR, pad_img
from utils.visualization import vis_meshes_img, get_colors_rgb
from engines.funcs.mupots_utils import mpii_compute_3d_pck, _match_poses, _scale_to_gt
from utils.evaluation import cal_3d_position_error, match_2d_greedy, get_matching_dict, \
    compute_prf1, vectorize_distance


CONNECTIONS = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]


def evaluate_3dpw(model, eval_dataloader, conf_thresh, vis=True, vis_step=40, results_save_path=None, distributed=False, accelerator=None, infer_detr=None):
    assert results_save_path is not None
    assert accelerator is not None
    num_processes = accelerator.num_processes

    os.makedirs(results_save_path, exist_ok=True)
    if vis:
        imgs_save_dir = os.path.join(results_save_path, 'imgs')
        os.makedirs(imgs_save_dir, exist_ok=True)

    step = 0
    total_miss_count = 0
    total_count = 0
    total_fp = 0

    mve, mpjpe, pa_mpjpe, pa_mve = [0.], [0.], [0.], [0.]
    cur_device = next(model.parameters()).device
    smpl_layer = model.module.human_model if hasattr(model, 'module') else model.human_model
    smpl2h36m_regressor = torch.from_numpy(smpl_layer.smpl2h36m_regressor).float().to(cur_device)

    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process, ncols=80)
    progress_bar.set_description('evaluate')
    for itr, (samples, images_clip, images_detr, targets) in enumerate(eval_dataloader):
        if infer_detr is not None:
            target_image_sizes = torch.as_tensor([im.size()[-2:] for im in images_clip])
            with torch.no_grad():
                results_detr = infer_detr(images_detr, target_image_sizes=target_image_sizes)
                images_clip_tensor = torch.stack(images_clip, dim=0)
                x = model.decoder.decoder.hoi_detector.clip_head.image_encoder.encode_image(images_clip_tensor)
                shared_ctx, compound_deeper_prompts = model.decoder.decoder.hoi_detector.clip_head.prompt_learner.get_visual_prompts(x[:, 1:])
                images_clip_ctx = (images_clip_tensor, shared_ctx, compound_deeper_prompts)
        else:
            results_detr = None
        
        with torch.no_grad():
            outputs = model(samples, images_clip_ctx, results_detr, targets)
        
        bs = len(targets)
        for idx in range(bs):
            # gt
            gt_verts = targets[idx]['verts']
            gt_transl = targets[idx]['transl']
            gt_j3ds = torch.einsum('bik,ji->bjk', [gt_verts - gt_transl[:, None, :], smpl2h36m_regressor]) + gt_transl[:, None, :]

            gt_verts = gt_verts.cpu().numpy()
            gt_j3ds = gt_j3ds.cpu().numpy()
            gt_j2ds = targets[idx]['j2ds'].cpu().numpy()[:, :24, :]

            # pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]

            pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach()
            pred_transl = outputs['pred_transl'][idx][select_queries_idx].detach()
            pred_j3ds = torch.einsum('bik,ji->bjk', [pred_verts - pred_transl[:, None, :], smpl2h36m_regressor]) + pred_transl[:, None, :]

            pred_verts = pred_verts.cpu().numpy()
            pred_j3ds = pred_j3ds.cpu().numpy()
            pred_j2ds = outputs['pred_j2ds'][idx][select_queries_idx].detach().cpu().numpy()[:, :24, :]

            matched_verts_idx = []
            assert len(gt_j2ds.shape) == 3 and len(pred_j2ds.shape) == 3
            # matching
            # tuples are (idx_pred_kps, idx_gt_kps)
            greedy_match = match_2d_greedy(pred_j2ds, gt_j2ds)
            matchDict, falsePositive_count = get_matching_dict(greedy_match)

            # align with matching result
            gt_verts_list, pred_verts_list, gt_joints_list, pred_joints_list = [], [], [], []
            gtIdxs = np.arange(len(gt_j3ds))
            miss_flag = []
            for gtIdx in gtIdxs:
                gt_verts_list.append(gt_verts[gtIdx])
                gt_joints_list.append(gt_j3ds[gtIdx])
                if matchDict[str(gtIdx)] == 'miss' or matchDict[str(gtIdx)] == 'invalid':
                    miss_flag.append(1)
                    pred_verts_list.append([])
                    pred_joints_list.append([])
                else:
                    miss_flag.append(0)
                    pred_joints_list.append(pred_j3ds[matchDict[str(gtIdx)]])
                    pred_verts_list.append(pred_verts[matchDict[str(gtIdx)]])
                    matched_verts_idx.append(matchDict[str(gtIdx)])

            # calculating 3d errors
            for i, (gt3d, pred) in enumerate(zip(gt_joints_list, pred_joints_list)):
                total_count += 1

                # Get corresponding ground truth and predicted 3d joints and verts
                if miss_flag[i] == 1:
                    total_miss_count += 1
                    continue

                gt3d = gt3d.reshape(-1, 3)
                pred3d = pred.reshape(-1, 3)
                gt3d_verts = gt_verts_list[i].reshape(-1, 3)
                pred3d_verts = pred_verts_list[i].reshape(-1, 3)

                gt_pelvis = gt3d[[0], :].copy()
                pred_pelvis = pred3d[[0], :].copy()

                gt3d = (gt3d - gt_pelvis)[H36M_EVAL_JOINTS, :].copy()
                gt3d_verts = (gt3d_verts - gt_pelvis).copy()

                pred3d = (pred3d - pred_pelvis)[H36M_EVAL_JOINTS, :].copy()
                pred3d_verts = (pred3d_verts - pred_pelvis).copy()

                # joints
                error_j, pa_error_j = cal_3d_position_error(pred3d, gt3d)
                mpjpe.append(error_j)
                pa_mpjpe.append(pa_error_j)
                # vertices
                error_v, pa_error_v = cal_3d_position_error(pred3d_verts, gt3d_verts)
                mve.append(error_v)
                pa_mve.append(pa_error_v)

            # counting
            step += 1
            total_fp += falsePositive_count

            img_idx = step + accelerator.process_index * \
                len(eval_dataloader) * bs

            if vis and (img_idx % vis_step == 0) and len(matched_verts_idx) > 0:
                img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
                input_size = model.module.input_size if hasattr(model, 'module') else model.input_size
                ori_img = pad_img(ori_img, input_size)

                selected_verts = pred_verts[matched_verts_idx]
                colors = get_colors_rgb(len(selected_verts))
                mesh_img = vis_meshes_img(img=ori_img.copy(), verts=selected_verts, smpl_faces=smpl_layer.faces, colors=colors, cam_intrinsics=outputs['pred_intrinsics'][idx].detach().cpu())

                full_img = np.hstack([ori_img, mesh_img])
                cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)

        progress_bar.update(1)

    if distributed:
        mve = accelerator.gather_for_metrics(mve)
        mpjpe = accelerator.gather_for_metrics(mpjpe)
        pa_mpjpe = accelerator.gather_for_metrics(pa_mpjpe)
        pa_mve = accelerator.gather_for_metrics(pa_mve)

        total_miss_count = sum(accelerator.gather_for_metrics([total_miss_count]))
        total_count = sum(accelerator.gather_for_metrics([total_count]))
        total_fp = sum(accelerator.gather_for_metrics([total_fp]))

    if len(mpjpe) <= num_processes:
        return "Failed to evaluate. Keep training!"

    precision, recall, f1 = compute_prf1(total_count, total_miss_count, total_fp)
    error_dict = {}
    error_dict['recall'] = recall

    error_dict['MPJPE'] = round(float(sum(mpjpe) / (len(mpjpe) - num_processes)), 1)
    error_dict['PA-MPJPE'] = round(float(sum(pa_mpjpe) / (len(pa_mpjpe) - num_processes)), 1)
    error_dict['MVE'] = round(float(sum(mve) / (len(mve) - num_processes)), 1)
    error_dict['PA-MVE'] = round(float(sum(pa_mve) / (len(pa_mve) - num_processes)), 1)

    if accelerator.is_main_process:
        with open(os.path.join(results_save_path, 'results.txt'), 'w') as f:
            for k, v in error_dict.items():
                f.write(f'{k}: {v}\n')

    return error_dict


# Modified from https://github.com/Arthur151/ROMP/tree/master/trace/lib/evaluation/mupots_util
def evaluate_mupots(model, eval_dataloader, conf_thresh, vis=True, vis_step=40, results_save_path=None, distributed=False, accelerator=None, infer_detr=None):
    assert results_save_path is not None
    assert accelerator is not None

    os.makedirs(results_save_path, exist_ok=True)
    if vis:
        imgs_save_dir = os.path.join(results_save_path, 'imgs')
        os.makedirs(imgs_save_dir, exist_ok=True)

    cur_device = next(model.parameters()).device
    smpl_layer = model.module.human_model if hasattr(model, 'module') else model.human_model
    smpl2h36m_regressor = torch.from_numpy(smpl_layer.smpl2h36m_regressor).float().to(cur_device)

    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process, ncols=80)
    progress_bar.set_description('evaluate')

    step = 0
    mismatch_cnt = 0
    pje_all_dict = defaultdict(list)
    pje_all_mask = defaultdict(list)
    pje_match_dict = defaultdict(list)
    pje_match_mask = defaultdict(list)
    for itr, (samples, images_clip, images_detr, targets) in enumerate(eval_dataloader):
        if infer_detr is not None:
            target_image_sizes = torch.as_tensor([im.size()[-2:] for im in images_clip])
            with torch.no_grad():
                results_detr = infer_detr(images_detr, target_image_sizes=target_image_sizes)
                images_clip_tensor = torch.stack(images_clip, dim=0)
                x = model.decoder.decoder.hoi_detector.clip_head.image_encoder.encode_image(images_clip_tensor)
                shared_ctx, compound_deeper_prompts = model.decoder.decoder.hoi_detector.clip_head.prompt_learner.get_visual_prompts(x[:, 1:])
                images_clip_ctx = (images_clip_tensor, shared_ctx, compound_deeper_prompts)
        else:
            results_detr = None
        
        with torch.no_grad():
            outputs = model(samples, images_clip_ctx, results_detr, targets)
        
        bs = len(targets)
        for idx in range(bs):
            step += 1
            ts_name = targets[idx]['TS']

            # gt
            gt_j3ds = targets[idx]['j3ds'].cpu().numpy()
            gt_j2ds = targets[idx]['j2ds'].cpu().numpy()
            gt_jvis = targets[idx]['jvis'].cpu().numpy()
            img_size = targets[idx]['img_size'].cpu().numpy()
            resize_rate = targets[idx]['resize_rate'].cpu().numpy()
            gt_j2ds[:, :, 0] = gt_j2ds[:, :, 0] * img_size[1] / resize_rate
            gt_j2ds[:, :, 1] = gt_j2ds[:, :, 1] * img_size[0] / resize_rate
            gt_visibility = np.ones(gt_j2ds.shape[:2], dtype='bool')

            # pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach()
            pred_transl = outputs['pred_transl'][idx][select_queries_idx].detach()
            pred_j3ds = torch.einsum('bik,ji->bjk', [pred_verts - pred_transl[:, None, :], smpl2h36m_regressor]) + pred_transl[:, None, :]

            pred_intrinsics = outputs['pred_intrinsics'][idx][0]
            pred_j2ds_homo = torch.einsum('bjc,cd->bjd', pred_j3ds, pred_intrinsics.transpose(0, 1))
            pred_j2ds = pred_j2ds_homo[..., :2] / (pred_j2ds_homo[..., 2:] + 1e-6)
            pred_j2ds = pred_j2ds.cpu().numpy() / resize_rate
            pred_j3ds = pred_j3ds.cpu().numpy() * 1000.
            pred_visibility = np.ones(pred_j2ds.shape[:2], dtype='bool')

            H36M_to_MPI = [10, 8, 14, 15, 16, 11, 12, 13, 4, 5, 6, 1, 2, 3, 0, 7, 9]
            gt_j2ds = gt_j2ds[:, H36M_to_MPI]
            gt_j3ds = gt_j3ds[:, H36M_to_MPI]
            pred_j2ds = pred_j2ds[:, H36M_to_MPI]
            pred_j3ds = pred_j3ds[:, H36M_to_MPI]

            # matching
            joints_for_matching = np.arange(1, 14)
            pair_inds = _match_poses(gt_j2ds[:, joints_for_matching], gt_visibility[:, joints_for_matching], pred_j2ds[:, joints_for_matching], pred_visibility[:, joints_for_matching], 40)

            for k in range(len(pair_inds)):
                jvis = gt_jvis[k]
                gtP = gt_j3ds[k, :] - gt_j3ds[k, 14:15]
                if pair_inds[k] != -1:
                    predP = pred_j3ds[pair_inds[k]]
                    predP = predP - predP[14:15]
                    predP = _scale_to_gt(predP, gtP)
                else:
                    mismatch_cnt += 1
                    predP = 100000 * np.ones(gtP.shape)

                errorP = np.sqrt(np.power(predP - gtP, 2).sum(axis=1))
                pje_all_dict[ts_name].append(errorP)
                pje_all_mask[ts_name].append(jvis)
                if pair_inds[k] != -1:
                    pje_match_dict[ts_name].append(errorP)
                    pje_match_mask[ts_name].append(jvis)

            img_idx = step + accelerator.process_index * len(eval_dataloader) * bs
            if vis and img_idx % vis_step == 0:
                img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
                input_size = model.module.input_size if hasattr(model, 'module') else model.input_size
                ori_img = pad_img(ori_img, input_size)

                matches = list(filter(lambda x: x != -1, pair_inds))
                selected_verts = pred_verts[matches].detach().cpu()
                colors = get_colors_rgb(len(selected_verts))
                mesh_img = vis_meshes_img(img=ori_img.copy(), verts=selected_verts, smpl_faces=smpl_layer.faces, colors=colors, cam_intrinsics=outputs['pred_intrinsics'][idx].detach().cpu())

                full_img = np.hstack([ori_img, mesh_img])
                cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)

        progress_bar.update(1)

    if distributed:
        for k in pje_all_dict.keys():
            pje_all_dict[k] = accelerator.gather_for_metrics(pje_all_dict[k])
            pje_all_mask[k] = accelerator.gather_for_metrics(pje_all_mask[k])
            pje_match_dict[k] = accelerator.gather_for_metrics(pje_match_dict[k])
            pje_match_mask[k] = accelerator.gather_for_metrics(pje_match_mask[k])

    pck_all = mpii_compute_3d_pck(pje_all_dict, pje_all_mask)
    pck_match = mpii_compute_3d_pck(pje_match_dict, pje_match_mask)

    error_dict = {}
    error_dict['PCK (All)'] = round(float(pck_all * 100), 1)
    error_dict['PCK (Match)'] = round(float(pck_match * 100), 1)
    error_dict['mismatch'] = mismatch_cnt

    if accelerator.is_main_process:
        with open(os.path.join(results_save_path, 'results.txt'), 'w') as f:
            for k, v in error_dict.items():
                f.write(f'{k}: {v}\n')

    return error_dict


def evaluate_panoptic(model, eval_dataloader, conf_thresh, vis=True, vis_step=40, results_save_path=None, distributed=False, accelerator=None, infer_detr=None):
    assert results_save_path is not None
    assert accelerator is not None
    num_processes = accelerator.num_processes

    seqs = ['haggling', 'mafia', 'ultimatum', 'pizza']
    J24_TO_H36M = [14, 2, 1, 0, 3, 4, 5, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]  # Different SMPL2H36M regressor
    # J24_TO_H36M = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    os.makedirs(results_save_path, exist_ok=True)
    if vis:
        imgs_save_dir = os.path.join(results_save_path, 'imgs')
        os.makedirs(imgs_save_dir, exist_ok=True)

    step = 0

    seq_mpjpe = {'haggling': [0.], 'mafia': [0.], 'ultimatum': [0.], 'pizza': [0.]}
    cur_device = next(model.parameters()).device
    smpl_layer = model.module.human_model if hasattr(model, 'module') else model.human_model
    smpl2h36m_regressor = torch.from_numpy(smpl_layer.smpl2h36m_regressor).float().to(cur_device)
    error_per_sample = {'imagename': [], 'mpjpe': []}

    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process, ncols=80)
    progress_bar.set_description('evaluate')
    for itr, (samples, images_clip, images_detr, targets) in enumerate(eval_dataloader):
        if infer_detr is not None:
            target_image_sizes = torch.as_tensor([im.size()[-2:] for im in images_clip])
            with torch.no_grad():
                results_detr = infer_detr(images_detr, target_image_sizes=target_image_sizes)
                images_clip_tensor = torch.stack(images_clip, dim=0)
                x = model.decoder.decoder.hoi_detector.clip_head.image_encoder.encode_image(images_clip_tensor)
                shared_ctx, compound_deeper_prompts = model.decoder.decoder.hoi_detector.clip_head.prompt_learner.get_visual_prompts(x[:, 1:])
                images_clip_ctx = (images_clip_tensor, shared_ctx, compound_deeper_prompts)
        else:
            results_detr = None
        
        with torch.no_grad():
            outputs = model(samples, images_clip_ctx, results_detr, targets)
        
        bs = len(targets)
        for idx in range(bs):
            #gt
            gt_j3ds = targets[idx]['j3ds'].cpu()
            #pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            while len(select_queries_idx) == 0:
                conf_thresh /= 2
                select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]

            pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach()
            pred_transl = outputs['pred_transl'][idx][select_queries_idx].detach()
            pred_j3ds = torch.einsum('bik,ji->bjk', [pred_verts - pred_transl[:, None, :], smpl2h36m_regressor]) + pred_transl[:, None, :]

            pred_verts = pred_verts.detach().cpu()
            pred_j3ds = pred_j3ds.detach().cpu()

            # following BMP
            gt_j3ds[:, 14, :] -= torch.tensor([0., 0.06, 0., 0.])
            gt_keypoints_3d = gt_j3ds.clone()
            gt_pelvis_smpl = gt_keypoints_3d[:, [14], :-1].clone()
            visible_kpts = gt_keypoints_3d[:, J24_TO_H36M, -1].clone()
            # visible_kpts = targets[idx]['vis'][:, J24_TO_H36M].cpu().clone()
            origin_gt_kpts3d = gt_j3ds.clone()
            origin_gt_kpts3d = origin_gt_kpts3d[:, J24_TO_H36M]
            origin_gt_kpts3d[:, :, :-1] -= gt_pelvis_smpl
            gt_keypoints_3d = gt_keypoints_3d[:, J24_TO_H36M, :-1].clone()
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis_smpl

            # Get 14 predicted joints from the SMPL mesh
            pred_keypoints_3d_smpl = pred_j3ds
            pred_pelvis_smpl = pred_keypoints_3d_smpl[:, [0], :].clone()
            pred_keypoints_3d_smpl = pred_keypoints_3d_smpl - pred_pelvis_smpl

            # To select closest points
            glb_vis = (visible_kpts.sum(0) >= (visible_kpts.shape[0] - 0.1)).float()[None, :, None]  # To avoid in-accuracy in float point number

            dist = vectorize_distance((glb_vis * gt_keypoints_3d).numpy(), (glb_vis * pred_keypoints_3d_smpl).numpy())
            paired_idxs = torch.from_numpy(dist.argmin(1))

            # is_mismatch = len(set(paired_idxs.tolist())) < len(paired_idxs)
            # if is_mismatch:
            #     self.mismatch_cnt += 1

            selected_prediction = pred_keypoints_3d_smpl[paired_idxs]

            # Compute error metrics
            # Absolute error (MPJPE)
            error_smpl = (torch.sqrt(((selected_prediction - gt_keypoints_3d)**2).sum(dim=-1)) * visible_kpts)
            mpjpes = error_smpl.mean(-1) * 1000
            mpjpes = mpjpes.tolist()

            avg_mpjpe = np.mean(mpjpes) if len(mpjpes) else 0
            error_per_sample['imagename'].append(targets[idx]['img_path'])
            error_per_sample['mpjpe'].append(avg_mpjpe)

            for seq in seqs:
                if seq in targets[idx]['img_path']:
                    for mpjpe in mpjpes:
                        seq_mpjpe[seq].append(float(mpjpe))
                    break

            gt_keypoints_3d = gt_keypoints_3d.numpy()
            selected_prediction = selected_prediction.numpy()
            visible_kpts = visible_kpts.bool().numpy()

            #counting
            step += 1
            img_idx = step + accelerator.process_index * len(eval_dataloader) * bs
            if vis and (img_idx % vis_step == 0):
                img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
                input_size = model.module.input_size if hasattr(model, 'module') else model.input_size
                ori_img = pad_img(ori_img, input_size)

                selected_verts = pred_verts[paired_idxs]
                colors = get_colors_rgb(len(selected_verts))
                mesh_img = vis_meshes_img(img=ori_img.copy(), verts=selected_verts, smpl_faces=smpl_layer.faces, colors=colors, cam_intrinsics=outputs['pred_intrinsics'][idx].detach().cpu())

                full_img = np.hstack([ori_img, mesh_img])
                cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)

        progress_bar.update(1)

    if distributed:
        for seq in seqs:
            seq_mpjpe[seq] = accelerator.gather_for_metrics(seq_mpjpe[seq])

    error_dict = {}

    for seq in seqs:
        if len(seq_mpjpe[seq]) <= num_processes:
            return "Failed to evaluate. Keep training!"
        error_dict[seq] = round(sum(seq_mpjpe[seq]) / (len(seq_mpjpe[seq]) - num_processes), 1)

    all_mpjpe = []
    for seq in seqs:
        all_mpjpe += seq_mpjpe[seq]

    error_dict['mean'] = round(sum(all_mpjpe) / (len(all_mpjpe) - num_processes * 4), 1)

    if accelerator.is_main_process:
        with open(os.path.join(results_save_path, 'results.txt'), 'w') as f:
            for k, v in error_dict.items():
                f.write(f'{k}: {v}\n')

    error_per_sample = pd.DataFrame(error_per_sample)
    error_per_sample.to_excel(os.path.join(results_save_path, 'error_per_sample.xlsx'))

    return error_dict


def evaluate_hi4d(model, eval_dataloader, conf_thresh, vis=True, vis_step=40, results_save_path=None, distributed=False, accelerator=None, infer_detr=None):
    assert results_save_path is not None
    assert accelerator is not None
    num_processes = accelerator.num_processes

    os.makedirs(results_save_path, exist_ok=True)
    if vis:
        imgs_save_dir = os.path.join(results_save_path, 'imgs')
        os.makedirs(imgs_save_dir, exist_ok=True)

    step = 0
    total_miss_count = 0
    total_count = 0
    total_fp = 0

    mve, mpjpe, pa_mpjpe, pa_mve = [0.], [0.], [0.], [0.]
    cur_device = next(model.parameters()).device
    smpl_layer = model.module.human_model if hasattr(model, 'module') else model.human_model
    smpl2h36m_regressor = torch.from_numpy(smpl_layer.smpl2h36m_regressor).float().to(cur_device)

    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process, ncols=80)
    progress_bar.set_description('evaluate')
    for itr, (samples, images_clip, images_detr, targets) in enumerate(eval_dataloader):
        if infer_detr is not None:
            target_image_sizes = torch.as_tensor([im.size()[-2:] for im in images_clip])
            with torch.no_grad():
                results_detr = infer_detr(images_detr, target_image_sizes=target_image_sizes)
                images_clip_tensor = torch.stack(images_clip, dim=0)
                x = model.decoder.decoder.hoi_detector.clip_head.image_encoder.encode_image(images_clip_tensor)
                shared_ctx, compound_deeper_prompts = model.decoder.decoder.hoi_detector.clip_head.prompt_learner.get_visual_prompts(x[:, 1:])
                images_clip_ctx = (images_clip_tensor, shared_ctx, compound_deeper_prompts)
        else:
            results_detr = None
        
        with torch.no_grad():
            outputs = model(samples, images_clip_ctx, results_detr, targets)
        
        bs = len(targets)
        for idx in range(bs):
            # gt
            gt_verts = targets[idx]['verts']
            gt_transl = targets[idx]['transl']
            gt_j3ds = torch.einsum('bik,ji->bjk', [gt_verts - gt_transl[:, None, :], smpl2h36m_regressor]) + gt_transl[:, None, :]

            gt_verts = gt_verts.cpu().numpy()
            gt_j3ds = gt_j3ds.cpu().numpy()
            gt_j2ds = targets[idx]['j2ds'].cpu().numpy()[:, :24, :]

            # pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]

            pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach()
            pred_transl = outputs['pred_transl'][idx][select_queries_idx].detach()
            pred_j3ds = torch.einsum('bik,ji->bjk', [pred_verts - pred_transl[:, None, :], smpl2h36m_regressor]) + pred_transl[:, None, :]

            pred_verts = pred_verts.cpu().numpy()
            pred_j3ds = pred_j3ds.cpu().numpy()
            pred_j2ds = outputs['pred_j2ds'][idx][select_queries_idx].detach().cpu().numpy()[:, :24, :]

            matched_verts_idx = []
            assert len(gt_j2ds.shape) == 3 and len(pred_j2ds.shape) == 3
            # matching
            # tuples are (idx_pred_kps, idx_gt_kps)
            greedy_match = match_2d_greedy(pred_j2ds, gt_j2ds)
            matchDict, falsePositive_count = get_matching_dict(greedy_match)

            # align with matching result
            gt_verts_list, pred_verts_list, gt_joints_list, pred_joints_list = [], [], [], []
            gtIdxs = np.arange(len(gt_j3ds))
            miss_flag = []
            for gtIdx in gtIdxs:
                gt_verts_list.append(gt_verts[gtIdx])
                gt_joints_list.append(gt_j3ds[gtIdx])
                if matchDict[str(gtIdx)] == 'miss' or matchDict[str(gtIdx)] == 'invalid':
                    miss_flag.append(1)
                    pred_verts_list.append([])
                    pred_joints_list.append([])
                else:
                    miss_flag.append(0)
                    pred_joints_list.append(pred_j3ds[matchDict[str(gtIdx)]])
                    pred_verts_list.append(pred_verts[matchDict[str(gtIdx)]])
                    matched_verts_idx.append(matchDict[str(gtIdx)])

            # calculating 3d errors
            action = targets[idx]['img_path'].split('/')[-4][:-2]
            for i, (gt3d, pred) in enumerate(zip(gt_joints_list, pred_joints_list)):
                total_count += 1

                # Get corresponding ground truth and predicted 3d joints and verts
                if miss_flag[i] == 1:
                    total_miss_count += 1
                    continue

                gt3d = gt3d.reshape(-1, 3)
                pred3d = pred.reshape(-1, 3)
                gt3d_verts = gt_verts_list[i].reshape(-1, 3)
                pred3d_verts = pred_verts_list[i].reshape(-1, 3)

                gt_pelvis = gt3d[[0], :].copy()
                pred_pelvis = pred3d[[0], :].copy()

                gt3d = (gt3d - gt_pelvis)[H36M_EVAL_JOINTS, :].copy()
                gt3d_verts = (gt3d_verts - gt_pelvis).copy()

                pred3d = (pred3d - pred_pelvis)[H36M_EVAL_JOINTS, :].copy()
                pred3d_verts = (pred3d_verts - pred_pelvis).copy()

                # joints
                error_j, pa_error_j = cal_3d_position_error(pred3d, gt3d)
                mpjpe.append(error_j)
                pa_mpjpe.append(pa_error_j)
                # vertices
                error_v, pa_error_v = cal_3d_position_error(pred3d_verts, gt3d_verts)
                mve.append(error_v)
                pa_mve.append(pa_error_v)

            # counting
            step += 1
            total_fp += falsePositive_count

            img_idx = step + accelerator.process_index * \
                len(eval_dataloader) * bs

            if vis and (img_idx % vis_step == 0) and len(matched_verts_idx) > 0:
                img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
                input_size = model.module.input_size if hasattr(model, 'module') else model.input_size
                ori_img = pad_img(ori_img, input_size)

                selected_verts = pred_verts[matched_verts_idx]
                colors = get_colors_rgb(len(selected_verts))
                mesh_img = vis_meshes_img(img=ori_img.copy(), verts=selected_verts, smpl_faces=smpl_layer.faces, colors=colors, cam_intrinsics=outputs['pred_intrinsics'][idx].detach().cpu())

                full_img = np.hstack([ori_img, mesh_img])
                cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)

        progress_bar.update(1)

    if distributed:
        mve = accelerator.gather_for_metrics(mve)
        mpjpe = accelerator.gather_for_metrics(mpjpe)
        pa_mpjpe = accelerator.gather_for_metrics(pa_mpjpe)
        pa_mve = accelerator.gather_for_metrics(pa_mve)

        total_miss_count = sum(accelerator.gather_for_metrics([total_miss_count]))
        total_count = sum(accelerator.gather_for_metrics([total_count]))
        total_fp = sum(accelerator.gather_for_metrics([total_fp]))

    if len(mpjpe) <= num_processes:
        return "Failed to evaluate. Keep training!"

    precision, recall, f1 = compute_prf1(total_count, total_miss_count, total_fp)
    error_dict = {}
    error_dict['recall'] = recall

    error_dict['MPJPE'] = round(float(sum(mpjpe) / (len(mpjpe) - num_processes)), 1)
    error_dict['PA-MPJPE'] = round(float(sum(pa_mpjpe) / (len(pa_mpjpe) - num_processes)), 1)
    error_dict['MVE'] = round(float(sum(mve) / (len(mve) - num_processes)), 1)
    error_dict['PA-MVE'] = round(float(sum(pa_mve) / (len(pa_mve) - num_processes)), 1)

    if accelerator.is_main_process:
        with open(os.path.join(results_save_path, 'results.txt'), 'w') as f:
            for k, v in error_dict.items():
                f.write(f'{k}: {v}\n')

    return error_dict
