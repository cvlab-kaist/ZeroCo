# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub

from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed

import dust3r.utils.path_to_croco  # noqa: F401
from mast3r.dust3r.croco.models.croco import CroCoNet  # noqa
# from croco.mod import FeatureL2Norm, unnormalise_and_convert_mapping_to_flow

import numpy as np
import torch.nn as nn 
import torch.nn.functional as F

inf = float('inf')
hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), ("Outdated huggingface_hub version, "
                                                                     "please reinstall requirements.txt")

import cv2 
import os 
import numpy as np
t1=torch.rand(32).argsort()
t2=torch.rand(32).argsort()
rnd_pts=list(zip(t1,t2))
vis_rnd_pts=[(i.item(), j.item()) for (i,j) in rnd_pts]
num_vis=30

def vis_attn_map(attention_maps, img_target, img_source, count, save_path='./vis_ca_map'):
    
    ########################## VIS CROSS ATTN MAPS (START) ###############################
    
    b, _, H, W = img_target.shape 
    attn_maps = torch.stack(attention_maps, dim=1)  # b 12 196 196 (twelve layers of already head averaged attention maps)

    p_size=16
    pH=H//p_size  # num patch H
    pW=W//p_size  # num patch W     

    for batch in range(b):  
        img_t = img_target[batch] # 3 224 224 
        img_s = img_source[batch] 
        attn_map = attn_maps[batch] # 12 196 196

        attn_map = attn_map.mean(dim=0) # average all layers of attention maps 

        np_img_s = (img_s-img_s.min()) / (img_s.max()-img_s.min()) * 255.0 # [0,255]
        np_img_t = (img_t-img_t.min())/(img_t.max()-img_t.min()) * 255.0   # [0,255]
        np_img_s = np_img_s.squeeze().permute(1,2,0).detach().cpu().numpy() # 224 224 3 
        np_img_t = np_img_t.squeeze().permute(1,2,0).detach().cpu().numpy()

        # List to store all visualizations
        all_vis_imgs = []
        
        for points in vis_rnd_pts[:num_vis]:
            idx_h=points[0]     # to vis idx_h
            idx_w=points[1]     # to vis idx_w
            idx_n=idx_h*pW+idx_w  # to vis token idx
            
            # plot white pixel to vis tkn location
            vis_np_img_s = np_img_s.copy()  # same as clone()
            vis_np_img_s[idx_h*p_size:(idx_h+1)*p_size, idx_w*p_size:(idx_w+1)*p_size,:]=255    # color with white pixel
            
            # breakpoint()
            # generate attn heat map
            attn_msk=attn_map[idx_n]  # hw=14*14=196
            # attn_msk[0]=0
            # attn_msk=attn_msk.softmax(dim=-1)
            attn_msk=attn_msk.view(1,1,pH,pW)
            attn_msk=F.interpolate(attn_msk, size=(H,W), mode='bilinear', align_corners=True)   # 224 224
            attn_msk=(attn_msk-attn_msk.min())/(attn_msk.max()-attn_msk.min())  # [0,1]
            attn_msk=attn_msk.squeeze().detach().cpu().numpy()*255  # [0,255]
            heat_mask=cv2.applyColorMap(attn_msk.astype(np.uint8), cv2.COLORMAP_JET)
            
            # overlap heat_mask to source image
            img_t_attn_msked = np_img_t[...,::-1] + heat_mask
            img_t_attn_msked = (img_t_attn_msked-img_t_attn_msked.min())/(img_t_attn_msked.max()-img_t_attn_msked.min())*255.0
            
            # Concatenate source and target images horizontally for this point
            combined_img = np.concatenate([vis_np_img_s[:,:,[2,1,0]], img_t_attn_msked], axis=1)
            all_vis_imgs.append(combined_img)
        
        # Stack all visualizations vertically
        final_vis = np.concatenate(all_vis_imgs, axis=0)
        
        # Save the combined visualization
        log_img_path = save_path
        if not os.path.exists(log_img_path):
            os.makedirs(log_img_path)
        cv2.imwrite(f'{log_img_path}/count{count}_batch{batch}_all_points.jpg', final_vis)


def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricCroCo3DStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        self.count=0

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            try:
                model = super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)
            except TypeError as e:
                raise Exception(f'tried to load {pretrained_model_name_or_path} from huggingface, but failed')
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none': [],
            'mask': [self.mask_token],
            'encoder': [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim    
        f1 = self.decoder_embed(f1) # b 1024 768
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        camaps1=[]
        camaps2=[]
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _, camap1 = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _, camap2 = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))
            camaps1.append(camap1)
            camaps2.append(camap2)

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return final_output, camaps1, camaps2

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def softmax_with_temperature(self, x, beta, d = 1):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M # subtract maximum value for stability
        exp_x = torch.exp(x/beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    def soft_argmax(self, corr, beta=0.02):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        b,_,h,w = corr.size()
        corr = self.softmax_with_temperature(corr, beta=beta, d=1)
        corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
        x_normal = self.x_normal.expand(b,w)
        x_normal = x_normal.view(b,w,1,1)
        grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        
        grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
        y_normal = self.y_normal.expand(b,h)
        y_normal = y_normal.view(b,h,1,1)
        grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        return grid_x, grid_y
    
    def forward(self, view1, view2, output_mode='enc_feat'):
        B = view1.shape[0]
        img1 = view1    # b 3 512 512 
        img2 = view2 
        shape1 = torch.tensor(img1.shape[-2:])[None].repeat(B, 1)
        shape2 = torch.tensor(img2.shape[-2:])[None].repeat(B, 1) 

        # encode the two images --> B,S,D
        # (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)
        # feat1: b 1024 1024

        # combine all ref images into object-centric representation
        dec_feats, tgt_camap, src_camap = self._decoder(feat1, pos1, feat2, pos2)  # b 12 1024 1024


        if output_mode == 'enc_feat':
            # feat1: 1 196 1024 
            return (feat1, feat2)

        elif output_mode == 'dec_feat':
            # dec_feats[0][0]: 1 196 1024
            return dec_feats
        
        elif output_mode == 'ca_map':
            # tgt_camap[0]: b 12 196 196 
            return (tgt_camap, src_camap)

        else:
            pass 
        self.reciprocity = True
        if self.reciprocity:
            tgt_camap = [attn.mean(dim=1).detach() for attn in tgt_camap]
            src_camap = [attn.mean(dim=1).detach() for attn in src_camap]
            tgt_camap = [ (camap_t + camap_s.transpose(-1,-2))/2 for (camap_t, camap_s) in zip(tgt_camap, src_camap)]
            tgt_camap = [ camap.softmax(dim=-1) for camap in tgt_camap]
            for i in range(len(tgt_camap)):
                tgt_camap[i][:,:,0]= tgt_camap[i].min()   # b 196 196

        else:
            # average along head dim
            tgt_camap = [attn.mean(dim=1).detach() for attn in tgt_camap]   
            # heuristic attention refine
            for i in range(len(tgt_camap)):
                tgt_camap[i][:,:,0]= tgt_camap[i].min()   # b 196 196

        self.count+=1
        if self.count % 100 == 0:
            vis_attn_map(tgt_camap, img2, img1, self.count, save_path='./vis/camap/spair/REVERSE_eval512_mast3r512_recip_softargmax')

        tgt_attn_map = torch.stack(tgt_camap, dim=1).mean(dim=1)    # b 196 196

        self.feature_size = img1.shape[-2] // 16
        self.x_normal = np.linspace(-1,1,self.feature_size)
        self.x_normal = nn.Parameter(torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False)).cuda()
        self.y_normal = np.linspace(-1,1,self.feature_size)
        self.y_normal = nn.Parameter(torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False)).cuda()

        grid_x, grid_y = self.soft_argmax(tgt_attn_map.transpose(-1,-2).view(B, -1, self.feature_size, self.feature_size), beta=1e-4)
        self.grid_x = grid_x
        self.grid_y = grid_y

        coarse_flow = torch.cat((grid_x, grid_y), dim=1)
        coarse_flow = unnormalise_and_convert_mapping_to_flow(coarse_flow)  # b 2 14 14 = b 2 self.feature_size self.feature_size
        h, w = coarse_flow.shape[-2:]

        # output_shape = (self.img_size, self.img_size)
        # if self.output_flow_interp:
        #     coarse_flow = F.interpolate(coarse_flow, size=output_shape, mode='bilinear', align_corners=False)
        #     coarse_flow[:, 0] *= float(output_shape[1]) / float(w)
        #     coarse_flow[:, 1] *= float(output_shape[0]) / float(h)

        return coarse_flow 
