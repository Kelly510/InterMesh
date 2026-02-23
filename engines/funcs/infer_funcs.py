import os
import cv2
import numpy as np
from tqdm.auto import tqdm

import torch
from utils.transforms import unNormalize
from utils.box_ops import box_cxcywh_to_xyxy
from utils.visualization import tensor_to_BGR, pad_img
from utils.visualization import vis_meshes_img, vis_boxes, vis_sat, vis_scale_img, get_colors_rgb


def inference(model, infer_dataloader, conf_thresh, results_save_path=None, distributed=False, accelerator=None, infer_detr=None):
    assert results_save_path is not None
    assert accelerator is not None

    accelerator.print(f'Results will be saved at: {results_save_path}')
    os.makedirs(results_save_path, exist_ok=True)
    smpl_layer = model.human_model

    progress_bar = tqdm(total=len(infer_dataloader), disable=not accelerator.is_local_main_process, ncols=100)
    progress_bar.set_description('inference')

    for itr, (samples, images_clip, images_detr, targets) in enumerate(infer_dataloader):
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
            images_clip_ctx = None
        
        with torch.no_grad():
            outputs = model(samples, images_clip_ctx, results_detr, targets)
        
        bs = len(targets)
        for idx in range(bs):
            img_size = targets[idx]['img_size'].detach().cpu().int().numpy()
            img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]

            # pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach().cpu().numpy()

            ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
            ori_img[img_size[0]:, :, :] = 255
            ori_img[:, img_size[1]:, :] = 255
            ori_img[img_size[0]:, img_size[1]:, :] = 255
            ori_img = pad_img(ori_img, model.input_size, pad_color_offset=255)

            sat_img = vis_sat(ori_img.copy(), input_size=model.input_size, patch_size=14, sat_dict=outputs['sat'], bid=idx)[:img_size[0], :img_size[1]]

            colors = get_colors_rgb(len(pred_verts))
            pred_mesh_img = vis_meshes_img(img=ori_img.copy(), verts=pred_verts, smpl_faces=smpl_layer.faces, cam_intrinsics=outputs['pred_intrinsics'][idx].reshape(3, 3).detach().cpu(), colors=colors)[:img_size[0], :img_size[1]]

            if 'enc_outputs' not in outputs:
                pred_scale_img = np.zeros_like(ori_img)[:img_size[0], :img_size[1]]
            else:
                enc_out = outputs['enc_outputs']
                h, w = enc_out['hw'][idx]
                flatten_map = enc_out['scale_map'].split(enc_out['lens'])[idx].detach().cpu()

                ys = enc_out['pos_y'].split(enc_out['lens'])[idx]
                xs = enc_out['pos_x'].split(enc_out['lens'])[idx]
                scale_map = torch.zeros((h, w, 2))
                scale_map[ys, xs] = flatten_map

                pred_scale_img = vis_scale_img(img=ori_img.copy(), scale_map=scale_map, conf_thresh=model.sat_cfg['conf_thresh'], patch_size=28)[:img_size[0], :img_size[1]]

            pred_boxes = outputs['pred_boxes'][idx][select_queries_idx].detach().cpu()
            pred_boxes = box_cxcywh_to_xyxy(pred_boxes) * model.input_size
            pred_box_img = vis_boxes(ori_img.copy(), pred_boxes, color=(255, 0, 255))[:img_size[0], :img_size[1]]

            cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'), np.vstack([np.hstack([pred_box_img, pred_mesh_img]), np.hstack([pred_scale_img, sat_img])]))

        progress_bar.update(1)
