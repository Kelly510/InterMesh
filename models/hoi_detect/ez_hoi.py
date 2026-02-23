import os
import pdb
import math
import argparse
from PIL import Image
from collections import OrderedDict
from typing import List, Optional, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision
from torchvision.ops.boxes import batched_nms

from . import clip
from .clip_models_adapter import CLIP
from utils.misc import nested_tensor_from_tensor_list
from .detr.models import build_model as build_detr_model
from .hico_text_label import hico_text_label, hico_obj_text_label
from .transformer_module import TransformerDecoderLayer, TransformerDecoderLayer_CA, TransformerSALayer, _get_clones


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Adapter(nn.Module):

    def __init__(self, input_size, dropout=0.1, adapter_scalar="1.0", adapter_num_layers=1, mem_adpt_self=False, SA_only=False, pt_tune=False, prior_size=None, down_size=64):
        super().__init__()
        self.n_embd = input_size
        self.down_size = down_size
        self.scale = float(adapter_scalar)

        self.down_proj_mem = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj_mem = nn.Linear(self.down_size, self.n_embd)
        self.adapter_num_layers = adapter_num_layers
        if prior_size == None:
            self.down_proj_prior = MLP(input_size, 128, self.down_size, 3)
        else:
            self.down_proj_prior = MLP(prior_size, 128, self.down_size, 3)

        self.dropout = dropout
        with torch.no_grad():
            if pt_tune is False:
                nn.init.kaiming_uniform_(self.down_proj_mem.weight, a=math.sqrt(5))
                nn.init.zeros_(self.down_proj_mem.bias)
            nn.init.zeros_(self.up_proj_mem.weight)
            nn.init.zeros_(self.up_proj_mem.bias)

        if SA_only is False and pt_tune is False:
            instance_decoder_layer = TransformerDecoderLayer(self.down_size, 2, self.down_size * 2, self.dropout, 'relu', False)
        elif SA_only is False and pt_tune is True:
            instance_decoder_layer = TransformerDecoderLayer_CA(self.down_size, 2, self.down_size * 2, self.dropout, 'relu', False)
        else:
            instance_decoder_layer = TransformerSALayer(self.down_size, 2, self.down_size * 2, self.dropout, 'relu', False)
        self.mhsa_layers = _get_clones(instance_decoder_layer, adapter_num_layers)
        self.mem_adpt_self = mem_adpt_self

    def forward(self, x, prior=None, verbose=False):
        tempa = self.down_proj_mem(x)
        if prior is None or self.mem_adpt_self is True:
            context = tempa  ## 18(#instance) x batchsize x 64
            mask = None
        else:
            prior, mask = prior
            context = self.down_proj_prior(prior).transpose(0, 1)

        tempa = self.non_linear_func(tempa)  ## 197 x batchsize x 64
        for z, layer in enumerate(self.mhsa_layers):
            tempa = layer(tempa, context, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=mask, pos=None, query_pos=None)

        up = self.up_proj_mem(tempa)
        output = (up * self.scale) + x
        return output


class TextEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text, txtcls_feat=None, txtcls_pt_list=None, origin_ctx=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        if txtcls_feat is not None:
            combined = [x, compound_prompts_deeper_text, 0, txtcls_feat, origin_ctx]
        elif txtcls_pt_list is not None:
            combined = [x, compound_prompts_deeper_text, 0, txtcls_pt_list, origin_ctx]
        else:
            combined = [x, compound_prompts_deeper_text, 0, origin_ctx]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        if isinstance(x, List):
            if len(x) == 5:
                origin_x = x[4]
            else:
                origin_x = None
            x = x[0]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        if origin_x is not None:
            return x, origin_x
        return x


class MultiModalPromptLearner(nn.Module):

    def __init__(self, ctx_dim, n_ctx, compound_prompts_depth, emb_dim):
        super().__init__()
        if ctx_dim == 768:
            vis_dim = 1024
        elif ctx_dim == 512:
            vis_dim = 768

        self.txtcls_ctx_pt = nn.ParameterList([nn.Parameter(torch.randn(n_ctx, ctx_dim)) for _ in range(compound_prompts_depth)])
        self.img_clip_pt_adapter = Adapter(vis_dim, pt_tune=True, prior_size=ctx_dim, down_size=emb_dim)

        self.proj = nn.Linear(ctx_dim, vis_dim)
        single_layer = nn.Linear(ctx_dim, vis_dim)
        self.compound_prompt_proj_vis = _get_clones(single_layer, compound_prompts_depth - 1)

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1)
        return prompts

    def get_visual_prompts(self, img_clip_prior):
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_proj_vis):
            temp_vis_pt = layer(self.txtcls_ctx_pt[index + 1])
            visual_deep_prompts.append(temp_vis_pt)
        first_ly_vis_pt = self.proj(self.txtcls_ctx_pt[0])

        for index, vis_pt_i in enumerate(visual_deep_prompts):
            visual_deep_prompts[index] = self.img_clip_pt_adapter(vis_pt_i.unsqueeze(1).repeat(1, len(img_clip_prior), 1), (img_clip_prior.to(vis_pt_i), None))
        first_ly_vis_pt = self.img_clip_pt_adapter(first_ly_vis_pt.unsqueeze(1).repeat(1, len(img_clip_prior), 1), (img_clip_prior.to(vis_pt_i), None), verbose=True)

        return first_ly_vis_pt, visual_deep_prompts


class CustomCLIP(nn.Module):

    def __init__(self, args, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(ctx_dim=clip_model.ln_final.weight.shape[0], n_ctx=args.N_CTX, compound_prompts_depth=args.compound_prompts_depth, emb_dim=args.emb_dim)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def encode_image(self, image):
        return self.image_encoder(image.type(self.dtype))


class UPT(nn.Module):

    def __init__(self, model: nn.Module, object_embedding: torch.tensor, human_idx: int, num_classes: int, emb_dim: int = 64, box_score_thresh: float = 0.2, min_instances: int = 3, max_instances: int = 15) -> None:
        super().__init__()
        self.clip_head = model
        self.register_buffer("object_embedding", object_embedding)

        self.human_idx = human_idx
        self.num_classes = num_classes
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.box_score_thresh = box_score_thresh
        self.visual_output_dim = model.image_encoder.output_dim
        self.priors_initial_dim = self.visual_output_dim + 5

        self.priors_downproj = MLP(self.priors_initial_dim, 128, 64, 3)  # old 512+5
        self.vis_fuse = nn.Sequential(
            nn.Linear(self.visual_output_dim * 2, self.visual_output_dim),
            nn.ReLU(),
            nn.Linear(self.visual_output_dim, self.visual_output_dim),
            nn.ReLU(),
        )
        self.featmap_dropout = nn.Dropout(0.2)

        self.COCO_CLASSES = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', \
            'fire hydrant','N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',\
            'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', \
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', \
            'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', \
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', \
            'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', \
            'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self.reserve_indices = [idx for (idx, name) in enumerate(self.COCO_CLASSES) if name != 'N/A']
        self.reserve_indices = self.reserve_indices + [91]
        self.reserve_indices = torch.as_tensor(self.reserve_indices)

    def compute_roi_embeddings(self, features: OrderedDict, image_size: Tensor, region_props: List[dict], sub_batch_size: int = 1024):
        device = features.device
        boxes_h_collated = []
        boxes_o_collated = []
        interaction_img_feats = []

        for b_idx, props in enumerate(region_props):
            local_features = features[b_idx]
            boxes = props['boxes']
            labels = props['labels']

            is_human = labels == self.human_idx
            n_h = torch.sum(is_human)
            n = len(boxes)

            # Permute human instances to the top
            if not torch.all(labels[:n_h] == self.human_idx):
                h_idx = torch.nonzero(is_human).squeeze(1)
                o_idx = torch.nonzero(is_human == 0).squeeze(1)
                perm = torch.cat([h_idx, o_idx])
                boxes = boxes[perm]

            # Get the pairwise indices
            x, y = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device))

            # Valid human-object pairs
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid human-object pairs")

            spatial_scale = 1 / (image_size[0, 0] / local_features.shape[1])
            single_features = torchvision.ops.roi_align(local_features.unsqueeze(0), [boxes], output_size=(7, 7), spatial_scale=spatial_scale, aligned=True)
            single_features = self.featmap_dropout(single_features).flatten(2).mean(-1)
            human_features = single_features[x_keep]
            object_features = single_features[y_keep]
            human_features = human_features / human_features.norm(dim=-1, keepdim=True)
            object_features = object_features / object_features.norm(dim=-1, keepdim=True)
            combined_features = torch.cat([human_features, object_features], dim=-1)

            assert human_features.shape[0] == object_features.shape[0]
            total_num_hoi = human_features.shape[0]
            vis_feat_all = []
            for n in range(math.ceil(total_num_hoi / sub_batch_size)):
                start = n * sub_batch_size
                end = min((n + 1) * sub_batch_size, total_num_hoi)
                vis_feat = self.vis_fuse(combined_features[start:end])
                vis_feat_all.append(vis_feat)
            vis_feat_all = torch.concat(vis_feat_all, dim=0)

            # out-of-boundary boxes cause nan features
            # vis_feat_all: (N, 768)
            nan_mask = torch.any(torch.isnan(vis_feat_all), dim=-1)
            if nan_mask.any():
                vis_feat_all[nan_mask] = 0

            interaction_img_feats.append(vis_feat_all)
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)

        return boxes_h_collated, boxes_o_collated, interaction_img_feats

    def get_prior(self, region_props, image_size):  ##  for adapter module training
        max_feat = self.priors_initial_dim
        max_length = max(rep['boxes'].shape[0] for rep in region_props)
        mask = torch.ones((len(region_props), max_length), dtype=torch.bool, device=region_props[0]['boxes'].device)
        priors = torch.zeros((len(region_props), max_length, max_feat), dtype=torch.float32, device=region_props[0]['boxes'].device)
        img_h, img_w = image_size.unbind(-1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        for b_idx, props in enumerate(region_props):
            boxes = props['boxes'] / scale_fct[b_idx][None, :]
            scores = props['scores']
            labels = props['labels']
            is_human = labels == self.human_idx
            n_h = torch.sum(is_human)
            n = len(boxes)
            if n_h == 0 or n <= 1:
                print(n_h, n)

            object_embs = self.object_embedding[labels.to(self.object_embedding.device)]
            mask[b_idx, :n] = False

            priors[b_idx, :n, :5] = torch.cat((scores.unsqueeze(-1), boxes), dim=-1)
            priors[b_idx, :n, 5:self.visual_output_dim + 5] = object_embs

        priors = self.priors_downproj(priors)

        return (priors, mask)

    def prepare_region_proposals(self, results_detr):  ## √ detr extracts the human-object pairs
        region_props = []
        for res in results_detr:
            sc = res['scores']
            lb = res['labels']
            bx = res['boxes']
            keep = batched_nms(bx, sc, lb, 0.5)
            sc = sc[keep].view(-1)
            lb = lb[keep].view(-1)
            bx = bx[keep].view(-1, 4)

            keep = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)

            is_human = lb == self.human_idx
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)
            n_human = is_human[keep].sum()
            n_object = len(keep) - n_human
            if n_human < self.min_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                keep_h = hum[keep_h]
            elif n_human > self.max_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = hum[keep_h]
            else:
                keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                keep_h = keep[keep_h]

            if n_object < self.min_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                keep_o = obj[keep_o]
            elif n_object > self.max_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                keep_o = obj[keep_o]
            else:
                keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                keep_o = keep[keep_o]

            keep = torch.cat([keep_h, keep_o])

            region_props.append(dict(
                boxes=bx[keep],
                scores=sc[keep],
                labels=lb[keep],
            ))

        return region_props

    def extract_hoi_features(self, images_clip_ctx, detections=None, results_detr=None, sub_batch_size=1024):
        batch_size = len(detections)

        if detections is None:
            region_props = self.prepare_region_proposals(results_detr)
        else:
            assert len(detections) == batch_size
            assert len(results_detr) == batch_size
            region_props = []
            for i, res in enumerate(results_detr):
                sc = res['scores']
                lb = res['labels']
                bx = res['boxes']
                keep = batched_nms(bx, sc, lb, 0.5)
                sc = sc[keep].view(-1)
                lb = lb[keep].view(-1)
                bx = bx[keep].view(-1, 4)

                keep = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)
                is_human = lb == self.human_idx
                obj = torch.nonzero(is_human == 0).squeeze(1)
                n_human = is_human[keep].sum()
                n_object = len(keep) - n_human
                if n_object < self.min_instances:
                    keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                    keep_o = obj[keep_o]
                elif n_object > self.max_instances:
                    keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                    keep_o = obj[keep_o]
                else:
                    keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                    keep_o = keep[keep_o]

                if len(detections[i]['boxes']) == 0:
                    region_props.append(dict(
                        boxes=bx[keep_o],
                        scores=sc[keep_o],
                        labels=lb[keep_o],
                    ))
                else:
                    region_props.append(dict(
                        boxes=torch.cat([detections[i]['boxes'], bx[keep_o]], dim=0),
                        scores=torch.cat([detections[i]['scores'], sc[keep_o]], dim=0),
                        labels=torch.cat([detections[i]['labels'], lb[keep_o]], dim=0),
                    ))

        clip_image_size = self.clip_head.image_encoder.input_resolution
        image_sizes = torch.ones(batch_size, 2, device=self.object_embedding.device) * clip_image_size
        priors = self.get_prior(region_props, image_sizes)  # priors: (prior_feat, mask): (batch_size*14*64, batch_size*14)
        feat_global, feat_local = self.clip_head.image_encoder(images_clip_ctx, priors)

        if torch.isnan(feat_local).any():
            # print("different local", feat_local[0], feat_local[1], feat_local[2], feat_local[3])
            pdb.set_trace()

        bh, bo, interaction_img_feats = self.compute_roi_embeddings(feat_local, image_sizes, region_props, sub_batch_size)
        return bh, bo, interaction_img_feats


@torch.no_grad()
def get_origin_text_emb(clip_model, tgt_class_names, obj_class_names):
    text_inputs = torch.cat([clip.tokenize(classname, context_length=77, truncate=True) for classname in tgt_class_names])
    with torch.no_grad():
        origin_text_embedding = clip_model.encode_text(text_inputs)
    origin_text_embedding = origin_text_embedding / origin_text_embedding.norm(dim=-1, keepdim=True)  # text embeddings of hoi 117*512 or 600*512
    obj_text_inputs = torch.cat([clip.tokenize(obj_text) for obj_text in obj_class_names])
    with torch.no_grad():
        obj_text_embedding = clip_model.encode_text(obj_text_inputs)
        object_embedding = obj_text_embedding
    return origin_text_embedding, object_embedding


def build_detector(args):
    upt_args = argparse.Namespace(N_CTX=2, init_txtcls_pt=False, pt_begin_layer=0, human_idx=0, num_classes=117, emb_dim=64, box_score_thresh=0.2, min_instances=3, max_instances=15, compound_prompts_depth=9, obj_embed_path="weights/ez_hoi/obj_embed_hico.pt")
    upt_args.clip_model_name = 'ViT-L/14@336px'

    clip_cfg = {'embed_dim': 768, 'image_resolution': 336, 'vision_layers': 24, 'vision_width': 1024, 'vision_patch_size': 14, 'context_length': 77, 'vocab_size': 49408, 'transformer_width': 768, 'transformer_heads': 12, 'transformer_layers': 12, 'use_adapter': True, 'adapter_layers': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], 'adapter_num_layers': 1, 'multi_cross': False, 'design_details': {'trainer': 'MaPLe', 'vision_depth': 0, 'language_depth': 0, 'vision_ctx': 0, 'language_ctx': 0, 'maple_length': 2, 'init_txtcls_pt': False, 'pt_begin_layer': 0}}
    clip_model = CLIP(**clip_cfg)

    object_embedding = torch.load(upt_args.obj_embed_path, weights_only=True)
    model = CustomCLIP(args=upt_args, clip_model=clip_model)

    upt = UPT(model=model, object_embedding=object_embedding, human_idx=upt_args.human_idx, num_classes=upt_args.num_classes, emb_dim=upt_args.emb_dim, box_score_thresh=upt_args.box_score_thresh, min_instances=upt_args.min_instances, max_instances=upt_args.max_instances)
    checkpoint = torch.load(args.hoi_cfg["upt_pretrained"], map_location="cpu", weights_only=True)
    upt.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return upt


def relocate_to_cuda(x, ignore: bool = False, device: Optional[Union[torch.device, int]] = None, **kwargs):
    if isinstance(x, torch.Tensor):
        return x.cuda(device, **kwargs)
    elif x is None:
        return x
    elif isinstance(x, list):
        return [relocate_to_cuda(item, ignore, device, **kwargs) for item in x]
    elif isinstance(x, tuple):
        return tuple(relocate_to_cuda(item, ignore, device, **kwargs) for item in x)
    elif isinstance(x, dict):
        for key in x:
            x[key] = relocate_to_cuda(x[key], ignore, device, **kwargs)
        return x
    elif isinstance(x, Image.Image):
        return x
    elif not ignore:
        raise TypeError('Unsupported type of data {}'.format(type(x)))


def relocate_to_cpu(x, ignore: bool = False):
    if isinstance(x, Tensor):
        return x.cpu()
    elif x is None:
        return x
    elif isinstance(x, list):
        return [relocate_to_cpu(item, ignore=ignore) for item in x]
    elif isinstance(x, tuple):
        return tuple(relocate_to_cpu(item, ignore=ignore) for item in x)
    elif isinstance(x, dict):
        for key in x:
            x[key] = relocate_to_cpu(x[key], ignore=ignore)
        return x
    elif not ignore:
        raise TypeError('Unsupported type of data {}'.format(type(x)))
