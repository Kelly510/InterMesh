# Modified from DINOv2 (https://github.com/facebookresearch/dinov2)
import copy
import torch
from configs.paths import dinov2_vitb14_path, dinov2_vitl14_path
from models.encoders.dinov2.models.vision_transformer import vit_base, vit_large


def build_encoder(args):
    num_additional_blocks = 0
    if args.sat_cfg['use_sat'] and args.sat_cfg['use_additional_blocks']:
        num_additional_blocks = args.sat_cfg['get_map_layer']
    
    weights = None
    if args.encoder == 'vitb':
        model = vit_base(img_size=518, patch_size=14, init_values=1.0, ffn_layer="mlp", block_chunks=0, num_register_tokens=0, interpolate_antialias=False, interpolate_offset=0.1, num_additional_blocks=num_additional_blocks)
        if args.mode.lower() == 'train' and not args.pretrain:
            weights = torch.load(dinov2_vitb14_path, weights_only=False)
    elif args.encoder == 'vitl':
        model = vit_large(img_size=518, patch_size=14, init_values=1.0, ffn_layer="mlp", block_chunks=0, num_register_tokens=0, interpolate_antialias=False, interpolate_offset=0.1, num_additional_blocks=num_additional_blocks)
        if args.mode.lower() == 'train' and not args.pretrain:
            weights = torch.load(dinov2_vitl14_path, weights_only=False)
    else:
        raise NotImplementedError

    if weights is not None:
        if args.sat_cfg['use_sat'] and args.sat_cfg['use_additional_blocks']:
            add_blocks_weights(weights, args.sat_cfg['get_map_layer'])
        print('Loading pretrained DINOv2...')
        model.load_state_dict(weights, strict=True)

    return model


def add_blocks_weights(weights, num_layers):
    for k in list(weights.keys()):
        if k.startswith('blocks') and int(k.split('.')[1]) < num_layers:
            new_k = k.replace('blocks', 'additional_blocks')
            weights[new_k] = copy.deepcopy(weights[k])
