import numpy as np
import torch
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader

from validation.flow_evaluation.metrics_uncertainty import (compute_average_of_uncertainty_metrics, compute_aucs,
                                                            compute_uncertainty_per_image)
from datasets.geometric_matching_datasets.ETH3D_interval import ETHInterval
from validation.plot import plot_sparse_keypoints, plot_flow_and_uncertainty, plot_individual_images
from utils_flow.pixel_wise_mapping import warp

import torch.nn.functional as F
from torchvision.utils import save_image
import torch.nn as nn 
from models.modules.mod import unnormalise_and_convert_mapping_to_flow

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FeatureL2Norm(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature, dim=1):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), dim) + epsilon, 0.5).unsqueeze(dim).expand_as(feature)
        return torch.div(feature, norm)
    
def softmax_with_temperature(x, beta, d = 1):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    M, _ = x.max(dim=d, keepdim=True)
    x = x - M # subtract maximum value for stability
    exp_x = torch.exp(x/beta)
    exp_x_sum = exp_x.sum(dim=d, keepdim=True)
    return exp_x / exp_x_sum

def soft_argmax(corr, beta=0.02, x_normal=None, y_normal=None):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    b,_,h,w = corr.size()
    corr = softmax_with_temperature(corr, beta=beta, d=1)
    corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

    grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
    x_normal = x_normal.expand(b,w)
    x_normal = x_normal.view(b,w,1,1)
    grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
    
    grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
    y_normal = y_normal.expand(b,h)
    y_normal = y_normal.view(b,h,1,1)
    grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
    return grid_x, grid_y

def resize_images_to_min_resolution(min_size, img, x, y, stride_net=16):  # for consistency with RANSAC-Flow
    """
    Function that resizes the image according to the minsize, at the same time resize the x,y coordinate.
    Extracted from RANSAC-Flow (https://github.com/XiSHEN0220/RANSAC-Flow/blob/master/evaluation/evalCorr/getResults.py)
    We here use exactly the same function that they used, for fair comparison. Even through the index_valid could
    also theoretically include the lower bound x = 0 or y = 0.
    """
    # Is is source image resized
    # Xs contains the keypoint x coordinate in source image
    # Ys contains the keypoints y coordinate in source image
    # valids is bool on wheter the keypoint is contained in the source image
    x = np.array(list(map(float, x.split(';')))).astype(np.float32)  # contains all the x coordinate
    y = np.array(list(map(float, y.split(';')))).astype(np.float32)

    w, h = img.size
    ratio = min(w / float(min_size), h / float(min_size))
    new_w, new_h = round(w / ratio), round(h / ratio)
    new_w, new_h = new_w // stride_net * stride_net, new_h // stride_net * stride_net

    ratioW, ratioH = new_w / float(w), new_h / float(h)
    img = img.resize((new_w, new_h), resample=Image.LANCZOS)

    x, y = x * ratioW, y * ratioH  # put coordinate in proper size after resizing the images
    index_valid = (x > 0) * (x < new_w) * (y > 0) * (y < new_h)

    return img, x, y, index_valid


def run_evaluation_generic(network, test_dataloader, device, estimate_uncertainty=False, name_dataset=None, rate=None, curr_id=0, args=None):
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    mean_epe_list, epe_all_list, pck_1_list, pck_3_list, pck_5_list = [], [], [], [], []

    for i_batch, mini_batch in pbar:
        source_img = mini_batch['source_image']
        target_img = mini_batch['target_image']
        flow_gt = mini_batch['flow_map'].to(device)
        mask_valid = mini_batch['correspondence_mask'].to(device)
        mask_valid_orig = mask_valid.clone()

        b, _, H_orig, W_orig = source_img.shape  

        source_img_orig = source_img.clone() / 255.0
        target_img_orig = target_img.clone() / 255.0

        source_img_orig = source_img_orig.to(device)
        target_img_orig = target_img_orig.to(device)

        if args.dataset == 'eth3d' and args.eval_img_size is not None:
            eval_h, eval_w = args.eval_img_size
            # H_32, W_32 = (H//32)*32, (W//32)*32
            source_img = F.interpolate(source_img, size=(eval_h, eval_w), mode='bilinear', align_corners=True).to(device)
            target_img = F.interpolate(target_img, size=(eval_h, eval_w), mode='bilinear', align_corners=True).to(device)
            mask_valid = F.interpolate(mask_valid.float().unsqueeze(0), size=(eval_h, eval_w), mode='nearest').squeeze(0).bool().to(device)
            flow_gt_h,flow_gt_w = flow_gt.size(2),flow_gt.size(3)
            flow_gt = F.interpolate(flow_gt, size=(eval_h, eval_w), mode='bilinear', align_corners=True).to(device)
            flow_gt[:,0,:,:] *= eval_w/flow_gt_w
            flow_gt[:,1,:,:] *= eval_h/flow_gt_h   

        # crocoflow, croco_catseg, 
        if args.model == 'croco':
            source_img = source_img / 255.0
            target_img = target_img / 255.0

            # for crocoflow as it predicts uncertainty
            if estimate_uncertainty:
                output = network(target_img, source_img)
                flow_est = output[:,:-1,:,:]
                conf = output[:,-1,:,:]
            
        
        elif args.model == 'crocov2' or args.model == 'crocov1':

            H_224, W_224 = args.model_img_size

            source_img = source_img / 255.0
            target_img = target_img / 255.0

            in1k_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            in1k_std =  torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

            source_img = (source_img - in1k_mean) / in1k_std
            target_img = (target_img - in1k_mean) / in1k_std

            source_img = source_img.to(device)
            target_img = target_img.to(device)
          
            if args.dense_zoom_in:
                # flow_est, uncertainty_est = network.zoom_in_batch(target_img, source_img, zoom_ratio=args.dense_zoom_ratio, batch_size=b)
                
                flow_est, uncertainty_est = network.zoom_in_batch(source_img, target_img, zoom_ratio=args.dense_zoom_ratio, batch_size=b)
            else:
                source_img = F.interpolate(source_img, size=(H_224, W_224), mode='bilinear', align_corners=False)   # b 3 224 224
                target_img = F.interpolate(target_img, size=(H_224, W_224), mode='bilinear', align_corners=False)
                output_correlation = args.output_correlation    # correlation: enc_feat, dec_feat, camap
                flow_est = network(target_img, source_img, output_correlation=output_correlation) # b 2 14 14

                if args.uncertainty:
                    flow_est = flow_est['flow_estimates'][0]
                    
                flow_est = F.interpolate(flow_est, size=(H_orig, W_orig), mode='bilinear', align_corners=False)  
                # corr_size = H_224//16
                flow_est[:,0,:,:] *= W_orig/224
                flow_est[:,1,:,:] *= H_orig/224


            if args.log_warped_images:
                if args.dataset == 'eth3d':
                    save_path = f'{args.save_dir}/rate{rate}'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    
                    warped_source_gt = warp(source_img_orig, flow_gt)  
                    warped_source_est = warp(source_img_orig, flow_est)
                    save_image(source_img_orig, f'{save_path}/{name_dataset}_{i_batch}_img_src.jpg', normalize=True)
                    save_image(target_img_orig, f'{save_path}/{name_dataset}_{i_batch}_img_tgt.jpg', normalize=True)
                    save_image(warped_source_gt, f'{save_path}/{name_dataset}_{i_batch}_img_warped_src_gt.jpg', normalize=True)
                    save_image(warped_source_est, f'{save_path}/{name_dataset}_{i_batch}_img_warped_src_est.jpg', normalize=True)
            

                elif args.dataset == 'hp' and curr_id < 5:
                    save_path = f'{args.save_dir}/{curr_id}'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    warped_source_gt = warp(source_img_orig, flow_gt)  
                    warped_source_est = warp(source_img_orig, flow_est)
                    # Create a 2x2 grid of images
                    grid = torch.cat([
                        torch.cat([source_img_orig, target_img_orig], dim=3),
                        torch.cat([warped_source_gt*mask_valid, warped_source_est*mask_valid], dim=3)
                    ], dim=2)
                    
                    save_image(grid, f'{save_path}/{i_batch}_combined.jpg')

        # evaluation protocol
        # eval_img_size -> resize input img to 224 -> model's output_flow 224 -> 
        elif args.model == 'dust3r' or args.model == 'mast3r':

            H_32, W_32 = args.model_img_size
            in1k_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device)
            in1k_std =  torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device)
            source_img = source_img.float()/255.
            target_img = target_img.float()/255.
            source_img_orig = source_img.clone()
            target_img_orig = target_img.clone()
            source_img = (source_img - in1k_mean) / in1k_std
            target_img = (target_img - in1k_mean) / in1k_std
            source_img = F.interpolate(source_img, size=(H_32, W_32), mode='bilinear', align_corners=False)
            target_img = F.interpolate(target_img, size=(H_32, W_32), mode='bilinear', align_corners=False)
            
            output_mode = args.output_mode    # enc_feat, dec_feat, camap
            outputs = network(target_img, source_img, output_mode=output_mode)

            if output_mode == 'enc_feat':
                feats1, feats2 = outputs[0], outputs[1]
                feat1 = feats1[-1]  # b n d 
                feat2 = feats2[-1]
                feat1 = feat1.unsqueeze(0)
                feat2 = feat2.unsqueeze(0)

                l2norm = FeatureL2Norm()    # normalizes along the feature 
                ## from (b, n, d) normalize along the n dimension.

                # from (b, n, d) normalize along the d dimension.
                feat1 = l2norm(feat1.permute(0,2,1)).permute(0,2,1)    
                feat2 = l2norm(feat2.permute(0,2,1)).permute(0,2,1)
                corr = torch.einsum('bnd, bmd -> bnm', feat1, feat2)

            elif output_mode == 'dec_feat':
                dec_feats1 = [outputs[i][0] for i in range(len(outputs))]
                dec_feats2 = [outputs[i][1] for i in range(len(outputs))]
                dec_feats1.pop(0)
                dec_feats2.pop(0)

                # use the first decoder feature
                dec_feat1 = dec_feats1[0]
                dec_feat2 = dec_feats2[0]


                l2norm = FeatureL2Norm()
                ## from (b, n, d) normalize along the n dimension.
                dec_feat1 = l2norm(dec_feat1)
                dec_feat2 = l2norm(dec_feat2)

                # from (b, n, d) normalize along the d dimension.

                corr = torch.einsum('bnd, bmd -> bnm', dec_feat1, dec_feat2)    # b 196 196

            elif output_mode == 'ca_map':
                camap1, camap2 = outputs[0], outputs[1]   # b 12 196 196
                camap1 = [attn.mean(dim=1).detach() for attn in camap1]   # b 196 196
                camap2 = [attn.mean(dim=1).detach() for attn in camap2]   # avg heads



                for i in range(len(camap1)):
                    camap1[i][:,:,0]=camap1[i].min()
                for i in range(len(camap2)):
                    camap2[i][:,:,0]=camap2[i].min()



                camap1 = torch.stack(camap1, dim=1)
                camap2 = torch.stack(camap2, dim=1)
                corr = (camap1.mean(dim=1) + camap2.mean(dim=1).transpose(-1,-2))/2.



            else:
                pass

            feature_size = H_32 // 16
            x_normal = np.linspace(-1,1,feature_size)
            x_normal = nn.Parameter(torch.tensor(x_normal, dtype=torch.float, requires_grad=False)).cuda()
            y_normal = np.linspace(-1,1,feature_size)
            y_normal = nn.Parameter(torch.tensor(y_normal, dtype=torch.float, requires_grad=False)).cuda()

            grid_x, grid_y = soft_argmax(corr.transpose(-1,-2).view(b, -1, feature_size, feature_size), beta=1e-4, x_normal=x_normal, y_normal=y_normal)
            coarse_flow = torch.cat((grid_x, grid_y), dim=1)
            flow_est = unnormalise_and_convert_mapping_to_flow(coarse_flow)  # b 2 14 14 = b 2 self.feature_size self.feature_size
            flow_est = F.interpolate(flow_est, size=(H_orig, W_orig), mode='bilinear', align_corners=True)
            flow_est[:,0,:,:] *= W_orig/feature_size
            flow_est[:,1,:,:] *= H_orig/feature_size   
            
            save_path = f'./vis/suppl/hp/{args.model}/{curr_id}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            vis_size = (224, 224)
            source_img_orig = F.interpolate(source_img_orig, size=vis_size, mode='bilinear', align_corners=True)
            target_img_orig = F.interpolate(target_img_orig, size=vis_size, mode='bilinear', align_corners=True)
        
            mask_valid_orig = F.interpolate(mask_valid_orig.float().unsqueeze(0), size=vis_size, mode='bilinear', align_corners=True)
            save_image(source_img_orig, f'{save_path}/{i_batch}_img_src.jpg', normalize=True)
            save_image(target_img_orig, f'{save_path}/{i_batch}_img_tgt.jpg', normalize=True)
            # save_image(mask_valid.float(), f'{save_path}/{curr_id}/{i_batch}_img_mask.jpg', normalize=True)

            warped_source_gt = warp(source_img, flow_gt)  
            warped_source_est = warp(source_img, flow_est)
            warped_source_gt = F.interpolate(warped_source_gt, size=vis_size, mode='bilinear', align_corners=True)
            warped_source_est = F.interpolate(warped_source_est, size=vis_size, mode='bilinear', align_corners=True)
            save_image(warped_source_gt, f'{save_path}/{i_batch}_img_warped_src_gt.jpg', normalize=True)
            save_image(warped_source_est*mask_valid_orig, f'{save_path}/{i_batch}_img_warped_src_est.jpg', normalize=True)


        elif args.model == 'crocoflow':

            H_32, W_32 = args.model_img_size

            in1k_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device)
            in1k_std =  torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device)
            source_img = source_img.float()/255.
            target_img = target_img.float()/255.

            source_img = (source_img - in1k_mean) / in1k_std
            target_img = (target_img - in1k_mean) / in1k_std

            source_img = F.interpolate(source_img, size=(H_32, W_32), mode='bilinear', align_corners=False)
            target_img = F.interpolate(target_img, size=(H_32, W_32), mode='bilinear', align_corners=False)
            
            # breakpoint()
            output = network(target_img, source_img)
            flow_est = output[:,:-1,:,:]
            conf = output[:,-1,:,:]

            flow_est = F.interpolate(flow_est, size=(H_orig, W_orig), mode='bilinear', align_corners=True)
            flow_est[:,0,:,:] *= W_orig/224.
            flow_est[:,1,:,:] *= H_orig/224.   

            save_image(source_img, f'./suppl_crocoflow_img_src.jpg', normalize=True)
            save_image(target_img, f'./suppl_crocoflow_img_tgt.jpg', normalize=True)
            save_image(mask_valid.float(), f'./suppl_crocoflow_img_mask.jpg', normalize=True)
            warped_source_gt = warp(source_img, flow_gt)  
            warped_source_est = warp(source_img, flow_est)
            save_image(warped_source_gt, f'./suppl_crocoflow_img_warped_src_gt.jpg', normalize=True)
            save_image(warped_source_est*mask_valid.unsqueeze(1), f'./suppl_crocoflow_img_warped_src_est.jpg', normalize=True)

        else:
            if estimate_uncertainty:
                flow_est, uncertainty_est = network.estimate_flow_and_confidence_map(source_img, target_img)    # flow_est: 1 2 224 224, uncertainty_est: pdcnet 
            else:
                flow_est = network.estimate_flow(source_img, target_img)    

            save_path = f'./vis/tmp/hp/{args.model}/{curr_id}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            
            # vis_size = (H, W)
            save_image(source_img, f'{save_path}/{i_batch}_img_src.jpg')
            save_image(target_img, f'{save_path}/{i_batch}_img_tgt.jpg')

            warped_source_gt = warp(source_img, flow_gt)  
            warped_source_est = warp(source_img, flow_est)
            save_image(warped_source_gt, f'{save_path}/{i_batch}_img_warped_src_gt.jpg')
            save_image(warped_source_est*mask_valid_orig, f'{save_path}/{i_batch}_img_warped_src_est.jpg')

        flow_est = flow_est.permute(0, 2, 3, 1)[mask_valid]
        flow_gt = flow_gt.permute(0, 2, 3, 1)[mask_valid]

        epe = torch.sum((flow_est - flow_gt) ** 2, dim=1).sqrt()

        epe_all_list.append(epe.view(-1).cpu().numpy())
        mean_epe_list.append(epe.mean().item())
        pck_1_list.append(epe.le(1.0).float().mean().item())
        pck_3_list.append(epe.le(3.0).float().mean().item())
        pck_5_list.append(epe.le(5.0).float().mean().item())

    epe_all = np.concatenate(epe_all_list)
    pck1_dataset = np.mean(epe_all <= 1)
    pck3_dataset = np.mean(epe_all <= 3)
    pck5_dataset = np.mean(epe_all <= 5)
    output = {'AEPE': np.mean(mean_epe_list), 'PCK_1_per_image': np.mean(pck_1_list),
              'PCK_3_per_image': np.mean(pck_3_list), 'PCK_5_per_image': np.mean(pck_5_list),
              'PCK_1_per_dataset': pck1_dataset, 'PCK_3_per_dataset': pck3_dataset,
              'PCK_5_per_dataset': pck5_dataset, 'num_pixels_pck_1': np.sum(epe_all <= 1).astype(np.float64),
              'num_pixels_pck_3': np.sum(epe_all <= 3).astype(np.float64),
              'num_pixels_pck_5': np.sum(epe_all <= 5).astype(np.float64),
              'num_valid_corr': len(epe_all)
              }
    print("Validation EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (np.mean(mean_epe_list), pck1_dataset, pck3_dataset, pck5_dataset))

    return output


def run_evaluation_eth3d(network, data_dir, input_images_transform, gt_flow_transform, co_transform, device,
                         estimate_uncertainty, args=None):
    # ETH3D dataset information
    dataset_names = ['lakeside', 'sand_box', 'storage_room', 'storage_room_2', 'tunnel', 'delivery_area', 'electro',
                     'forest', 'playground', 'terrains']
    rates = list(range(3, 16, 2))
    dict_results = {}
    
    for rate in rates:
        print('Computing results for interval {}...'.format(rate))
        dict_results['rate_{}'.format(rate)] = {}
        list_of_outputs_per_rate = []
        num_pck_1 = 0.0
        num_pck_3 = 0.0
        num_pck_5 = 0.0
        num_valid_correspondences = 0.0
        for name_dataset in dataset_names:
            print('looking at dataset {}...'.format(name_dataset))

            test_set = ETHInterval(root=data_dir,
                                   path_list=os.path.join(data_dir, 'info_ETH3D_files',
                                                          '{}_every_5_rate_of_{}'.format(name_dataset, rate)),
                                   source_image_transform=input_images_transform,
                                   target_image_transform=input_images_transform,
                                   flow_transform=gt_flow_transform,
                                   co_transform=co_transform)  # only test
            test_dataloader = DataLoader(test_set,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=8)
            print(test_set.__len__())
            output = run_evaluation_generic(network, test_dataloader, device, estimate_uncertainty, name_dataset=name_dataset, rate=rate, args=args)
            # to save the intermediate results
            # dict_results['rate_{}'.format(rate)][name_dataset] = output
            list_of_outputs_per_rate.append(output)
            num_pck_1 += output['num_pixels_pck_1']
            num_pck_3 += output['num_pixels_pck_3']
            num_pck_5 += output['num_pixels_pck_5']
            num_valid_correspondences += output['num_valid_corr']

        # average over all datasets for this particular rate of interval
        avg = {'AEPE': np.mean([list_of_outputs_per_rate[i]['AEPE'] for i in range(len(dataset_names))]),
               'PCK_1_per_image': np.mean([list_of_outputs_per_rate[i]['PCK_1_per_image'] for i in
                                           range(len(dataset_names))]),
               'PCK_3_per_image': np.mean([list_of_outputs_per_rate[i]['PCK_3_per_image'] for i in
                                           range(len(dataset_names))]),
               'PCK_5_per_image': np.mean([list_of_outputs_per_rate[i]['PCK_5_per_image'] for i in
                                           range(len(dataset_names))]),
               'pck-1-per-rate': num_pck_1 / (num_valid_correspondences + 1e-6),
               'pck-3-per-rate': num_pck_3 / (num_valid_correspondences + 1e-6),
               'pck-5-per-rate': num_pck_5 / (num_valid_correspondences + 1e-6),
               'num_valid_corr': num_valid_correspondences
               }
        dict_results['rate_{}'.format(rate)] = avg

    avg_rates = {'AEPE': np.mean([dict_results['rate_{}'.format(rate)]['AEPE'] for rate in rates]),
                 'PCK_1_per_image': np.mean(
                     [dict_results['rate_{}'.format(rate)]['PCK_1_per_image'] for rate in rates]),
                 'PCK_3_per_image': np.mean(
                     [dict_results['rate_{}'.format(rate)]['PCK_3_per_image'] for rate in rates]),
                 'PCK_5_per_image': np.mean(
                     [dict_results['rate_{}'.format(rate)]['PCK_5_per_image'] for rate in rates]),
                 'pck-1-per-rate': np.mean([dict_results['rate_{}'.format(rate)]['pck-1-per-rate'] for rate in rates]),
                 'pck-3-per-rate': np.mean([dict_results['rate_{}'.format(rate)]['pck-3-per-rate'] for rate in rates]),
                 'pck-5-per-rate': np.mean([dict_results['rate_{}'.format(rate)]['pck-5-per-rate'] for rate in rates]),
                 }
    dict_results['avg'] = avg_rates
    print(f"eth3d - Validation EPE: {avg_rates['AEPE']}, rate_3_AEPE: {dict_results['rate_3']['AEPE']}, rate_5_AEPE: {dict_results['rate_5']['AEPE']}, rate_7_AEPE: {dict_results['rate_7']['AEPE']}, rate_9_AEPE: {dict_results['rate_9']['AEPE']}, rate_11_AEPE: {dict_results['rate_11']['AEPE']}, rate_13_AEPE: {dict_results['rate_13']['AEPE']}, rate_15_AEPE: {dict_results['rate_15']['AEPE']}")    
    return dict_results