# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# CroCo model during pretraining
# --------------------------------------------------------

import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True # for gpu >= Ampere and pytorch >= 1.12
from functools import partial

from models.croco.blocks import Block, DecoderBlock, PatchEmbed
from models.croco.pos_embed import get_2d_sincos_pos_embed, RoPE2D 
from models.croco.masking import RandomMask

from torchvision import transforms
from utils_flow.pixel_wise_mapping import warp
import torch.nn.functional as F
from einops import rearrange

from models.croco.mod import FeatureL2Norm, unnormalise_and_convert_mapping_to_flow
import numpy as np

import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class CroCoNet(nn.Module):

    def __init__(self,
                 img_size=224,           # input image size
                 patch_size=16,          # patch_size 
                 mask_ratio=0.9,         # ratios of masked tokens 
                 enc_embed_dim=768,      # encoder feature dimension
                 enc_depth=12,           # encoder depth 
                 enc_num_heads=12,       # encoder number of heads in the transformer block 
                 dec_embed_dim=512,      # decoder feature dimension 
                 dec_depth=8,            # decoder depth 
                 dec_num_heads=16,       # decoder number of heads in the transformer block 
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_im2_in_dec=True,   # whether to apply normalization of the 'memory' = (second image) in the decoder 
                 pos_embed='cosine',     # positional embedding (either cosine or RoPE100)
                 args=None,
                ):
                
        super(CroCoNet, self).__init__()


        self.args = args 
        self.model = args.model 
        self.reciprocity = args.reciprocity
        self.output_ca_map = args.output_ca_map
        self.softmax_camap = args.softmax_camap
        self.output_flow_interp = args.output_flow_interp
        self.img_size = img_size

        self.feature_size = img_size[0] // 16
        self.l2norm = FeatureL2Norm()
        self.x_normal = np.linspace(-1,1,self.feature_size)
        self.x_normal = nn.Parameter(torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False))
        self.y_normal = np.linspace(-1,1,self.feature_size)
        self.y_normal = nn.Parameter(torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False))

                
        # patch embeddings  (with initialization done as in MAE)
        self._set_patch_embed(img_size, patch_size, enc_embed_dim)

        # mask generations
        self._set_mask_generator(self.patch_embed.num_patches, mask_ratio)

        self.pos_embed = pos_embed
        if pos_embed=='cosine':
            # positional embedding of the encoder 
            enc_pos_embed = get_2d_sincos_pos_embed(enc_embed_dim, self.patch_embed.grid_size, n_cls_token=0)
            self.register_buffer('enc_pos_embed', torch.from_numpy(enc_pos_embed).float())
            # positional embedding of the decoder  
            dec_pos_embed = get_2d_sincos_pos_embed(dec_embed_dim, self.patch_embed.grid_size, n_cls_token=0)
            self.register_buffer('dec_pos_embed', torch.from_numpy(dec_pos_embed).float())
            # pos embedding in each block
            self.rope = None # nothing for cosine 
        elif pos_embed.startswith('RoPE'): # eg RoPE100 
            self.enc_pos_embed = None # nothing to add in the encoder with RoPE
            self.dec_pos_embed = None # nothing to add in the decoder with RoPE
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(pos_embed[len('RoPE'):])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError('Unknown pos_embed '+pos_embed)

        # transformer for the encoder 
        self.enc_depth = enc_depth
        self.enc_embed_dim = enc_embed_dim
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)
        
        # masked tokens 
        self._set_mask_token(dec_embed_dim)

        # decoder 
        self._set_decoder(enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec)
        
        # prediction head 
        # self._set_prediction_head(dec_embed_dim, patch_size)
        
        # initializer weights
        self.initialize_weights()           

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, enc_embed_dim)

    def _set_mask_generator(self, num_patches, mask_ratio):
        self.mask_generator = RandomMask(num_patches, mask_ratio)
        
    def _set_mask_token(self, dec_embed_dim):
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        
    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder 
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder 
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope, softmax_camap=self.args.softmax_camap)
            for i in range(dec_depth)])
        # final norm layer 
        self.dec_norm = norm_layer(dec_embed_dim)
        
    def _set_prediction_head(self, dec_embed_dim, patch_size):
         self.prediction_head = nn.Linear(dec_embed_dim, patch_size**2 * 3, bias=True)
        
        
    def initialize_weights(self):
        # patch embed 
        self.patch_embed._init_weights()
        # mask tokens
        if self.mask_token is not None: torch.nn.init.normal_(self.mask_token, std=.02)
        # linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _encode_image(self, image, do_mask=False, return_all_blocks=False):
        """
        image has B x 3 x img_size x img_size 
        do_mask: whether to perform masking or not
        return_all_blocks: if True, return the features at the end of every block 
                           instead of just the features from the last block (eg for some prediction heads)
        """
        # embed the image into patches  (x has size B x Npatches x C) 
        # and get position if each return patch (pos has size B x Npatches x 2)
        x, pos = self.patch_embed(image)              
        # add positional embedding without cls token  
        if self.enc_pos_embed is not None: 
            x = x + self.enc_pos_embed[None,...]
        # apply masking 
        B,N,C = x.size()
        if do_mask:
            masks = self.mask_generator(x)
            x = x[~masks].view(B, -1, C)
            posvis = pos[~masks].view(B, -1, 2)
        else:
            B,N,C = x.size()
            masks = torch.zeros((B,N), dtype=bool)
            posvis = pos
        # now apply the transformer encoder and normalization        
        if return_all_blocks:
            out = []
            for blk in self.enc_blocks:
                x = blk(x, posvis)
                out.append(x)
            out[-1] = self.enc_norm(out[-1])
            return out, pos, masks
        else:
            for blk in self.enc_blocks:
                x = blk(x, posvis)
            x = self.enc_norm(x)
            return x, pos, masks
 
    def _decoder(self, feat1, pos1, masks1, feat2, pos2, return_all_blocks=False):
        """
        return_all_blocks: if True, return the features at the end of every block 
                           instead of just the features from the last block (eg for some prediction heads)
                           
        masks1 can be None => assume image1 fully visible 
        """
        # encoder to decoder layer 
        visf1 = self.decoder_embed(feat1)
        f2 = self.decoder_embed(feat2)
        # append masked tokens to the sequence
        B,Nenc,C = visf1.size()
        if masks1 is None: # downstreams
            f1_ = visf1
        else: # pretraining 
            Ntotal = masks1.size(1)
            f1_ = self.mask_token.repeat(B, Ntotal, 1).to(dtype=visf1.dtype)
            f1_[~masks1] = visf1.view(B * Nenc, C)
        # add positional embedding
        if self.dec_pos_embed is not None:
            f1_ = f1_ + self.dec_pos_embed
            f2 = f2 + self.dec_pos_embed
        # apply Transformer blocks
        out = f1_
        out2 = f2 
        attn_maps = []
        if return_all_blocks:
            _out, out = out, []
            for blk in self.dec_blocks:
                _out, out2, attn_map = blk(_out, out2, pos1, pos2)
                out.append(_out)
                attn_maps.append(attn_map)
            out[-1] = self.dec_norm(out[-1])
        else:
            for blk in self.dec_blocks:
                out, out2, attn_map = blk(out, out2, pos1, pos2)
                attn_maps.append(attn_map)
            out = self.dec_norm(out)
            
        if self.output_ca_map:
            return out, attn_maps
        return out, None

    def patchify(self, imgs):
        """
        imgs: (B, 3, H, W)
        x: (B, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        
        return x

    def unpatchify(self, x, channels=3):
        """
        x: (N, L, patch_size**2 *channels)
        imgs: (N, 3, H, W)
        """
        patch_size = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], channels, h * patch_size, h * patch_size))
        return imgs

    def resize_flow(self, flow, size):
        flow = flow.clone()
        h_o, w_o = flow.size()[-2:]
        h, w = size[-2:]
        
        ratio_h = float(h - 1)/float(h_o - 1)
        ratio_w = float(w - 1)/float(w_o - 1)
        flow[:, 0, :, :] *= ratio_w
        flow[:, 1, :, :] *= ratio_h

        flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=True)
        return flow
    
    def tile_image(self, img, tile_shape=(128, 128)):
        """
        Arguments:
            img: tensor shape of (3, 512, 512)
            tile_shape: tuple
        """
        _, H, W = img.shape
        tile = rearrange(img, 'C (T1 H) (T2 W) -> (T1 T2) C H W', H=tile_shape[0], W=tile_shape[1])
        return tile

    def tile_to_image(self, tile):
        T = int((tile.shape[0]) ** 0.5)
        return rearrange(tile, '(T1 T2) C H W -> C (T1 H) (T2 W)', T1=T, T2=T)

    def estimate_flow(self, source_img, target_img):
        if self.model == 'crocov2':
            flow_est = self.forward(target_img, source_img)
        return flow_est


    def zoom_in_batch(self, src_img, trg_img, zoom_ratio=(2,3), batch_size=1):
        flow_list = []
        uncertainty_list = []
        '''
            src_img: b 3 h w 일 때 -> tmp=src_img.split(1) -> b개의 3 h w 이미지가 나옴 -> 즉 len(tmp)=b
        '''
        for src, trg in zip(src_img.split(1), trg_img.split(1)):   
            flow, uncertainty = self.zoom_fix_multiscale(src, trg, zoom_ratio, batch_size=batch_size)
            flow_list.append(flow)
            uncertainty_list.append(uncertainty)
        return torch.cat(flow_list, dim=0), torch.cat(uncertainty_list, dim=0)
    
    def zoom_fix_multiscale(self, src_img, trg_img, zoom_ratio_list=(3, 4, 5), batch_size=1):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        src_img = src_img.to(device)
        trg_img = trg_img.to(device)

        src_img_resized = transforms.functional.resize(src_img, size=self.img_size)
        trg_img_resized = transforms.functional.resize(trg_img, size=self.img_size)
        with torch.no_grad():
            est_flow = self.estimate_flow(src_img_resized, trg_img_resized)
            est_flow_rev = self.estimate_flow(trg_img_resized, src_img_resized)

        est_flow = self.resize_flow(est_flow, trg_img.shape)
        est_flow = self.resize_flow(est_flow, trg_img.shape)
        est_flow_rev = self.resize_flow(est_flow_rev, src_img.shape)
        warped_src = warp(src_img.float(), est_flow)
        warped_trg = warp(trg_img.float(), est_flow_rev)

        final_flow_list = []
        final_flow_rev_list = []

        final_flow_list.append(est_flow)
        final_flow_rev_list.append(est_flow_rev)

        for zoom_ratio in zoom_ratio_list:
            warped_src_resized = F.interpolate(warped_src.float(), size=self.img_size[0] * zoom_ratio, mode='bilinear', align_corners=True)
            warped_src_tile = self.tile_image(warped_src_resized[0], tile_shape=(self.img_size[0], self.img_size[0]))

            resized_trg = F.interpolate(trg_img.float(), size=self.img_size[0] * zoom_ratio, mode='bilinear', align_corners=True)
            trg_tile = self.tile_image(resized_trg[0], tile_shape=(self.img_size[0], self.img_size[0]))


            est_flow_tile_list = []
            for s, t in zip(warped_src_tile.split(batch_size), trg_tile.split(batch_size)):
                with torch.no_grad():
                    est_flow_tile = self.estimate_flow(s.to('cuda'), t.to('cuda'))
                    est_flow_tile_list.append(est_flow_tile)

            est_flow_tile = torch.cat(est_flow_tile_list, dim=0)
            est_flow_zoom = self.tile_to_image(est_flow_tile)

            est_flow_zoom_origsize = self.resize_flow(est_flow_zoom[None], trg_img.shape)
            warped_flow = warp(est_flow, est_flow_zoom_origsize)
            final_flow = warped_flow + est_flow_zoom_origsize

            final_flow_list.append(final_flow)

        for zoom_ratio in zoom_ratio_list:
            warped_trg_resized = F.interpolate(warped_trg.float(), size=self.img_size[0] * zoom_ratio, mode='bilinear', align_corners=True)
            warped_trg_tile = self.tile_image(warped_trg_resized[0], tile_shape=(self.img_size[0], self.img_size[0]))

            resized_src = F.interpolate(src_img.float(), size=self.img_size[0] * zoom_ratio, mode='bilinear', align_corners=True)
            src_tile = self.tile_image(resized_src[0], tile_shape=(self.img_size[0], self.img_size[0]))


            est_flow_tile_list = []
            for t, s in zip(warped_trg_tile.split(batch_size), src_tile.split(batch_size)):
                with torch.no_grad():
                    est_flow_tile = self.estimate_flow(t.to('cuda'), s.to('cuda'))
                    est_flow_tile_list.append(est_flow_tile)

            est_flow_tile = torch.cat(est_flow_tile_list, dim=0)
            est_flow_zoom = self.tile_to_image(est_flow_tile)

            est_flow_zoom_origsize = self.resize_flow(est_flow_zoom[None], src_img.shape)
            warped_flow = warp(est_flow_rev, est_flow_zoom_origsize)
            final_flow = warped_flow + est_flow_zoom_origsize

            final_flow_rev_list.append(final_flow)

        final_flow_list = torch.cat(final_flow_list, dim=0)
        final_flow_rev_list = torch.cat(final_flow_rev_list, dim=0)

        final_confidence_list = torch.norm(final_flow_list + warp(final_flow_rev_list, final_flow_list), dim=1, p=2, keepdim=True)
        final_confidence_list_rev = torch.norm(final_flow_rev_list + warp(final_flow_list, final_flow_rev_list), dim=1, p=2, keepdim=True)

        final_flow = torch.gather(final_flow_list, dim=0, index=final_confidence_list.min(dim=0, keepdim=True)[1].repeat(1, 2, 1, 1))
        final_flow_rev = torch.gather(final_flow_rev_list, dim=0, index=final_confidence_list_rev.min(dim=0, keepdim=True)[1].repeat(1, 2, 1, 1))

        return final_flow, torch.norm(final_flow + warp(final_flow_rev, final_flow), dim=1, p=2, keepdim=True)

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
    
    @staticmethod
    def constrain_large_log_var_map(var_min, var_max, large_log_var_map):
        """
        Constrains variance parameter between var_min and var_max, returns log of the variance. Here large_log_var_map
        is the unconstrained variance, outputted by the network
        Args:
            var_min: min variance, corresponds to parameter beta_minus in paper
            var_max: max variance, corresponds to parameter beta_plus in paper
            large_log_var_map: value to constrain

        Returns:
            larger_log_var_map: log of variance parameter
        """
        if var_min > 0 and var_max > 0:
            large_log_var_map = torch.log(var_min +
                                          (var_max - var_min) * torch.sigmoid(large_log_var_map - torch.log(var_max)))
        elif var_max > 0:
            large_log_var_map = torch.log((var_max - var_min) * torch.sigmoid(large_log_var_map - torch.log(var_max)))
        elif var_min > 0:
            # large_log_var_map = torch.log(var_min + torch.exp(large_log_var_map))
            max_exp = large_log_var_map.detach().max() - 10.0
            large_log_var_map = torch.log(var_min / max_exp.exp() + torch.exp(large_log_var_map - max_exp)) + max_exp
        return large_log_var_map

    def forward(self, img_target, img_source, output_correlation='ca_map'):

        B,_,H_224,W_224 = img_target.size()
        feat_targets, pos_target, mask_target = self._encode_image(img_target, do_mask=False, return_all_blocks=True)
        feat_sources, pos_source, mask_source = self._encode_image(img_source, do_mask=False, return_all_blocks=True)

        feat_target = feat_targets[-1]
        feat_source = feat_sources[-1]

        # decoder
        decfeat, attn_map = self._decoder(feat_target, pos_target, mask_target, feat_source, pos_source, return_all_blocks=True)
        if self.reciprocity:
            decfeat_source, attn_map_source = self._decoder(feat_source, pos_source, mask_source, feat_target, pos_target, return_all_blocks=True)

        if output_correlation == 'enc_feat':
            feats1, feats2 = feat_targets, feat_sources
            # use the last encoder feature
            feat1, feat2 = feats1[-1], feats2[-1]
            l2norm = FeatureL2Norm()   
            # from (b, n, d) normalize along the d dimension.
            feat1 = l2norm(feat1.permute(0,2,1)).permute(0,2,1)    
            feat2 = l2norm(feat2.permute(0,2,1)).permute(0,2,1)
            corr = torch.einsum('bnd, bmd -> bnm', feat1, feat2)   

        elif output_correlation == 'dec_feat':
            dec_feats1, dec_feats2 = decfeat, decfeat_source
            # use the first decoder feature
            dec_feat1, dec_feat2 = dec_feats1[0], dec_feats2[0]
            l2norm = FeatureL2Norm()
            # from (b, n, d) normalize along the d dimension.
            dec_feat1 = l2norm(dec_feat1.permute(0,2,1)).permute(0,2,1)  
            dec_feat2 = l2norm(dec_feat2.permute(0,2,1)).permute(0,2,1)
            corr = torch.einsum('bnd, bmd -> bnm', dec_feat1, dec_feat2)    

        elif output_correlation == 'ca_map':

            camap1, camap2 = attn_map, attn_map_source
            camap1 = [attn.mean(dim=1) for attn in camap1]   
            camap2 = [attn.mean(dim=1) for attn in camap2]   

            if self.args.heuristic_attn_map_refine:
                for i in range(len(camap1)):
                    camap1[i][:,:,0]=camap1[i].min()
                    camap2[i][:,:,0]=camap2[i].min()

            camap1 = torch.stack(camap1, dim=1) 
            camap2 = torch.stack(camap2, dim=1)
            refined_layered_corr = (camap1 + camap2.transpose(-1,-2))/2. 
            refined_corr = (camap1.mean(dim=1) + camap2.mean(dim=1).transpose(-1,-2))/2.    

        else:
            assert False, 'output_mode not defined'

        beta=self.args.softargmax_beta  
        grid_x, grid_y = self.soft_argmax(refined_corr.transpose(-1,-2).view(B, -1, self.feature_size, self.feature_size), beta=beta)
        coarse_flow = torch.cat((grid_x, grid_y), dim=1)    
        flow_est = unnormalise_and_convert_mapping_to_flow(coarse_flow) 

        H_32, W_32 = 224, 224
        feature_size = H_32 // 16   
        flow_est = F.interpolate(flow_est, size=(H_32, W_32), mode='bilinear', align_corners=False) 
        flow_est[:,0,:,:] *= W_32/feature_size 
        flow_est[:,1,:,:] *= H_32/feature_size 
    
        return flow_est 
        
    