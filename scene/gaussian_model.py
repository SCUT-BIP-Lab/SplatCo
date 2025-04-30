#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from functools import reduce
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from .embedding import Embedding
from .grids import PlaneGrid
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import cv2
from utils.loss_utils import ssim
import time
from sklearn.neighbors import NearestNeighbors

from utils.grid_utils import normalize_xyz, _grid_creater, _grid_encoder

from utils.entropy_models import  Entropy_gaussian


class Spatial_CTX(nn.Module):
    def __init__(self, reso_3D, off_3D, reso_2D, off_2D):
        super().__init__()
        self.reso_3D = reso_3D
        self.off_3D = off_3D
        self.reso_2D = reso_2D
        self.off_2D = off_2D
    def forward(self, xyz_for_creater, xyz_for_interp, feature, determ=False, return_all=False):
        assert xyz_for_creater.shape[0] == feature.shape[0]
        grid_3D = _grid_creater.apply(xyz_for_creater, feature, self.reso_3D, self.off_3D, determ)  # [offsets_list_3D[-1], 48]
        grid_xy = _grid_creater.apply(xyz_for_creater[:, 0:2], feature, self.reso_2D, self.off_2D, determ)  # [offsets_list[-1], 48]
        grid_xz = _grid_creater.apply(xyz_for_creater[:, 0::2], feature, self.reso_2D, self.off_2D, determ)  # [offsets_list[-1], 48]
        grid_yz = _grid_creater.apply(xyz_for_creater[:, 1:3], feature, self.reso_2D, self.off_2D, determ)  # [offsets_list[-1], 48]

        context_info_3D = _grid_encoder.apply(xyz_for_interp, grid_3D, self.off_3D, self.reso_3D)  # [N_choose, 48*n_levels]
        context_info_xy = _grid_encoder.apply(xyz_for_interp[:, 0:2], grid_xy, self.off_2D, self.reso_2D)  # [N_choose, 48*n_levels]
        context_info_xz = _grid_encoder.apply(xyz_for_interp[:, 0::2], grid_xz, self.off_2D, self.reso_2D)  # [N_choose, 48*n_levels]
        context_info_yz = _grid_encoder.apply(xyz_for_interp[:, 1:3], grid_yz, self.off_2D, self.reso_2D)  # [N_choose, 48*n_levels]

        context_info = torch.cat([context_info_3D, context_info_xy, context_info_xz, context_info_yz], dim=-1)  # [N_choose, 48*n_levels*4]
        if return_all:
            return context_info, (xyz_for_creater, xyz_for_interp, feature, grid_3D, grid_xy, grid_xz, grid_yz, context_info_3D, context_info_xy, context_info_xz, context_info_yz, self.reso_3D, self.off_3D)
        return context_info


class Conctractor(nn.Module):
    def __init__(self, xyz_min, xyz_max, enable = True):
        super().__init__()
        self.enable = enable
        if not self.enable:
            print('**Disable Contractor**')
        self.register_buffer('xyz_min', xyz_min)
        self.register_buffer('xyz_max', xyz_max)

    def decontracte(self, xyz): 
        if not self.enable:
            raise Exception("Not implement")

        mask = torch.abs(xyz) > 1.0
        res = xyz.clone()
        signs = (res <0) & (torch.abs(res)>1.0)
        res[mask] = 1.0/(1.0- (torch.abs(res[mask])-1)) 
        res[signs] *= -1
        res = res * (self.xyz_max-self.xyz_min) /2 + (self.xyz_max+self.xyz_min) /2

        return res
    
    def contracte(self, xyz):

        indnorm = (xyz-self.xyz_min)*2.0 / (self.xyz_max-self.xyz_min) -1
        if self.enable:
            mask = torch.abs(indnorm)>1.0
            signs = (indnorm <0) & (torch.abs(indnorm)>1.0)
            indnorm[mask] = (1.0- 1.0/torch.abs(indnorm[mask])) +1.0
            indnorm[signs] *=-1
        return indnorm

class FeaturePlanes(nn.Module):
    def __init__(self, world_size, xyz_min, xyz_max, feat_dim = 24, mlp_width = [168], out_dim=[32], subplane_multiplier=1, resolutions_list=[300, 400, 500],
                 resolutions_list_3D=[60, 80, 100]):
        super(FeaturePlanes, self).__init__()
        
        self.world_size, self.xyz_min, self.xyz_max = world_size, xyz_min, xyz_max

        self.activate_level = 0
        self.num_levels = 3
        self.level_factor = 0.5

        t_ws = torch.tensor(world_size)

        self.k0s =  torch.nn.ModuleList()

        for i in range(self.num_levels):
            cur_ws = (t_ws*self.level_factor**(self.num_levels-i-1)).cpu().int().numpy().tolist()
            if i == 0:
                print("TA activate!")
                self.k0s.append(PlaneGrid(feat_dim, cur_ws, xyz_min, xyz_max,config={'factor':1}, TAflag=True))
            self.k0s.append(PlaneGrid(feat_dim, cur_ws, xyz_min, xyz_max,config={'factor':1}))
            print('Create Planes @ ', cur_ws)

       
        self.resolutions_list, self.offsets_list = self.get_offsets(resolutions_list, dim=2)
        self.resolutions_list_3D, self.offsets_list_3D = self.get_offsets(resolutions_list_3D, dim=3)
        self.CTXs =  torch.nn.ModuleList()
        for i in range(self.num_levels):
            self.CTXs.append(Spatial_CTX(self.resolutions_list_3D[i],
            self.offsets_list_3D[i],
            self.resolutions_list[i],
            self.offsets_list[i]))

        self.models = torch.nn.ModuleList()
        self.CTX_models = torch.nn.ModuleList()

        mlp_width = [mlp_width[0],mlp_width[0],mlp_width[0]] 
        out_dim = [out_dim[0],out_dim[0],out_dim[0]]

        for i in range(self.num_levels):
            
            #tri_context
            self.models.append(nn.Sequential(
                                nn.BatchNorm1d(self.k0s[i].get_dim()),
                                nn.Linear(self.k0s[i].get_dim(), out_dim[i])
                                ))
            
            self.CTX_models.append(nn.Sequential(
                                nn.BatchNorm1d(71),
                                nn.Linear(71, out_dim[i])
                                ))

    def forward(self, x, g_fea, Q=0):
        # Pass the input through k0

        level_features = []

        for i in range(self.activate_level + 1):
            feat = self.k0s[i](x , Q)
            level_features.append(feat)

        res = []
        cnt =0
        for m,feat,CTXm in zip(self.models,level_features,self.CTX_models):
            rr = m(feat)
            rrr = CTXm(g_fea)
            res.append(torch.concat((rr, rrr), dim=1))
            cnt = cnt + 1
            if cnt>self.activate_level:
                break
        

        return sum(res)

    def get_offsets(self, resolutions_list, dim=3):
        offsets_list = [0]
        offsets = 0
        for resolution in resolutions_list:
            offset = resolution ** dim
            offsets_list.append(offsets + offset)
            offsets += offset
        offsets_list = torch.tensor(offsets_list, device='cuda', dtype=torch.int)
        resolutions_list = torch.tensor(resolutions_list, device='cuda', dtype=torch.int)
        return resolutions_list, offsets_list


        
class GaussianLearner(nn.Module):
    def __init__(self, model_params, xyz_min = [-2, -2, -2], xyz_max=[2, 2, 2] ):
        super(GaussianLearner, self).__init__()
        self.Q0 = 0.03
        self.xyz_min = torch.tensor(xyz_min).cuda()
        self.xyz_max = torch.tensor(xyz_max).cuda()

        self.world_size = [model_params.plane_size]*3
        self.max_step = 6
        self.current_step = 0

        self._feat = FeaturePlanes(world_size=self.world_size, xyz_min = self.xyz_min, xyz_max= self.xyz_max,
                                    feat_dim = model_params.num_channels, mlp_width = [model_params.mlp_dim], out_dim=[32], subplane_multiplier=model_params.subplane_multiplier )  # 27,4,3,1

        self.register_buffer('opacity_scale', torch.tensor(10))
        self.opacity_scale = self.opacity_scale.cuda()

        self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()


    def activate_plane_level(self):
        self._feat.activate_level +=1
        print('******* Plane Level to:', self._feat.activate_level)


    def inference(self, xyz, g_fea, Q0):
        inputs = xyz.cuda().detach()
        g_fea = g_fea

        geo_fea  = self._feat(inputs, g_fea, self.Q0)

        return geo_fea

    def tv_loss(self, w):
        for level in range(self._feat.activate_level+1):
            factor = 1.0
            self._feat.k0s[level].total_variation_add_grad(w*((0.5)**(2-level)))
            




class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
    
    def setup_contractor(self,center,length, contractor):
        center = torch.tensor(center)
        length = torch.tensor(length)
        self.contractor = Conctractor(xyz_min=center-length*self.bbox_scale/2, xyz_max=center+length*self.bbox_scale/2, enable = contractor)
        self.contractor = self.contractor.cuda()




    def __init__(self, 
                 feat_dim: int=32,
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 model_params =None):

        self.mv = 4 ############################

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)

        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)
                
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim+3+self.opacity_dist_dim+64, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim+3+self.cov_dist_dim+64, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim+3+self.color_dist_dim+self.appearance_dim+64, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        self.feat_planes = GaussianLearner(model_params).cuda()

        self.setup_functions()

        self.magic_k = False
        self.enable_net = False

        self.bbox_scale = model_params.bbox_scale

        self.setup_contractor(model_params.scene_center, model_params.scene_length, model_params.contractor )

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self.feat_planes.state_dict(),
            self.contractor.state_dict(),
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._anchor, 
        self._offset,
        #self._local,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    def scale_grid(self):
        if not self.feat_planes.scale_grid():
            self.training_setup(self.training_args)

    def get_apperance_embedding(self, idx):
        return self._appearance_embeddings[idx]

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        
        return data

    def build_properties(self, data, visible ):

        tmp = torch.ones([self._xyz.size(0)]+list(data.size()[1:]),device = data.device)*-5
        tmp[visible] = data
        return tmp

    def activate_plane_level(self, training_args):
        self.feat_planes.activate_plane_level()
        self.training_setup(training_args)

    def update_contractor(self):
        points = self.get_xyz.detach()

        center = torch.mean(points,dim=0)
        length = (torch.max(points,dim=0)[0] - torch.min(points,dim=0)[0])*1.1

        self.setup_contractor(center.cpu().tolist(),length.cpu().tolist(), False)
        print('scene_center:',center.cpu().tolist(),'scene_length',length.cpu().tolist())

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"}
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]

        for i in range(3):
            if i == self.feat_planes._feat.activate_level:
                l.append( {'params': self.feat_planes._feat.k0s[i].parameters(), 'lr': 0.01, 'name': 'feat_planes%d'%i})
                l.append( {'params': self.feat_planes._feat.models[i].parameters(), 'lr': 1e-4, 'name': 'fp_mlp_f%d'%i})
            else:
                l.append( {'params': self.feat_planes._feat.k0s[i].parameters(), 'lr': 0.001, 'name': 'feat_planes%d'%i})
                l.append( {'params': self.feat_planes._feat.models[i].parameters(), 'lr': 1e-5, 'name': 'fp_mlp_f%d'%i})
        


        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)
        self.planes_scheduler_args = get_expon_lr_func(lr_init=0.01,
                                                    lr_final=0.005,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.planesmlp_scheduler_args = get_expon_lr_func(lr_init=1e-4,
                                                    lr_final=5e-5,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  len(group["params"]) != 1 :
                continue
            if group["name"]=='Qs' or group["name"]=='sigmas' or 'feat_planes' in group['name']:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name'] or \
                'feat_planes' in group['name'] :
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

        # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        #grad_norm /= self.mv##############################
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name'] or \
                'feat_planes' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
            
        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def anchor_growing(self, grads, threshold, offset_mask):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }
                

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                


    def adjust_anchor(self, iteration, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        if iteration % 3000 == 0 or iteration == 1600:
        #if iteration % 3000 == 0:
            print("Curvature Densification!")
            knn_curvatures = self.compute_curvature(self.get_anchor)
            knn_curvatures = knn_curvatures.view(self.get_anchor.shape[0], -1)
            curvature_mask =(knn_curvatures <= 0.1).squeeze()
            #print(curvature_mask.shape)
            curvature_mask = torch.cat([curvature_mask] * self.n_offsets, dim=0)
            #print(curvature_mask.shape)
            offset_mask =  torch.logical_or(offset_mask, curvature_mask)
        
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 

        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def graph_downsampling(self, rate, mode= "random"):
        pts = self.get_xyz
        pts = pts.detach().cpu().numpy()
        num_pts = pts.shape[0]
        mask = torch.ones((num_pts), device="cuda", dtype=bool)
        t1 = time.time()
        print("Graph Downsampling Processing, points number before sampling: ", num_pts)
        if mode == "random":
            idxs = np.random.choice(num_pts, int(np.floor(num_pts * rate)), replace=False)
        idxs = torch.from_numpy(idxs).long().cuda()
        mask[idxs] = 0
        self.prune_points(mask)
        
        torch.cuda.empty_cache()
        print("Graph Downsampling Processed, points number after sampling: ", self.get_xyz.shape[0], "Time: ", time.time() - t1, "seconds")

    def save_mlp_checkpoints(self, path, mode = 'unite'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim+64).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim+64).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim+64).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

        elif mode == 'unite':
            if self.use_feat_bank:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_dim > 0:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            else:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'unite'):#split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError

    def compute_curvature(self, points, k=10): 
        nppoints = points.clone().detach()
        
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(nppoints.cpu().numpy())
        distances, indices = nbrs.kneighbors(nppoints.cpu().numpy())

        curvatures = torch.zeros(nppoints.shape[0], device=points.device)
        for i in range(len(nppoints)):
            neighbors = nppoints[indices[i, 1:]]  # 排除自己
            mean = torch.mean(neighbors, dim=0)
            centered_points = neighbors - mean
            covariance_matrix = torch.matmul(centered_points.t(), centered_points) / (centered_points.size(0) - 1)
        
            eigenvalues, _ = torch.linalg.eigh(covariance_matrix)
            eigenvalues, _ = torch.sort(eigenvalues)
            curvature = eigenvalues[0] / torch.sum(eigenvalues)
            curvatures[i] = curvature

        return curvatures
    
    def compute_fast_loss_with_key_points(
    self,
    real_img1: torch.Tensor, real_img2: torch.Tensor,
    gen_img1: torch.Tensor, gen_img2: torch.Tensor,
    K1: torch.Tensor, R1: torch.Tensor, t1: torch.Tensor,
    K2: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor,
    existing_point_cloud: torch.Tensor,
    distance_threshold: float = 0.01,
    overall_ssim_threshold: float = 0.6,
    sigma_threshold: float = 3.0,  # Threshold for removing outliers
    min_cam_distance: float = 0.5,  # Minimum distance to camera centers
    device: str = 'cuda', pts_flag: bool = True
) -> tuple:
        """
        Compute weighted L1 loss, cross loss (real image difference), and return a combined mask for filtering points.

        Returns:
            tuple:
                Weighted L1 loss, real image difference loss, intersecting points, and combined mask.
        """

        def align_images(img1, img2, img3, img4):
            """
            Align four images by cropping the larger one to match the smaller dimensions.
            """
            _, h1, w1 = img1.shape
            _, h2, w2 = img2.shape
            _, h3, w3 = img3.shape
            _, h4, w4 = img4.shape

            min_height = min(h1, h2, h3, h4)
            min_width = min(w1, w2, w3, w4)

            img1_aligned = img1[:, :min_height, :min_width]
            img2_aligned = img2[:, :min_height, :min_width]
            img3_aligned = img3[:, :min_height, :min_width]
            img4_aligned = img4[:, :min_height, :min_width]

            return img1_aligned, img2_aligned, img3_aligned, img4_aligned

        # Align images
        real_img1, real_img2, gen_img1, gen_img2 = align_images(real_img1, real_img2, gen_img1, gen_img2)

        # Move data to device
        real_img1, real_img2 = real_img1.to(device), real_img2.to(device)
        gen_img1, gen_img2 = gen_img1.to(device), gen_img2.to(device)
        K1, R1, t1 = K1.to(device), R1.to(device), t1.to(device)
        K2, R2, t2 = K2.to(device), R2.to(device), t2.to(device)

        # Compute overall SSIM
        overall_ssim = ssim(real_img1, real_img2)
        if overall_ssim < overall_ssim_threshold:
            return 0.0, 0.0, torch.empty((0, 3), device=device), torch.zeros(existing_point_cloud.size(0), dtype=torch.bool, device=device)

        # Simplify to global weight
        global_weight = overall_ssim

        # Compute weighted L1 loss for generated images
        gen_diff = torch.abs(gen_img1 - gen_img2)
        weighted_gen_l1_loss = (gen_diff * global_weight).mean()

        # Compute difference between real images
        real_diff = torch.abs(real_img1 - real_img2)
        weighted_cross_l1_loss = (real_diff * global_weight).mean()

        if not pts_flag:
            return weighted_gen_l1_loss, weighted_cross_l1_loss, torch.empty((0, 3), device=device), \
                torch.zeros(existing_point_cloud.size(0), dtype=torch.bool, device=device)

        # Compute ray intersections in batch
        ray_dir1 = t2.view(-1) - t1.view(-1)
        ray_dir2 = t1.view(-1) - t2.view(-1)
        ray_dir1 /= torch.norm(ray_dir1)
        ray_dir2 /= torch.norm(ray_dir2)

        cam_center1 = t1.view(-1)
        cam_center2 = t2.view(-1)

        # Batch compute distances to point cloud
        ray1_dots = torch.mm(existing_point_cloud - cam_center1, ray_dir1.unsqueeze(1))
        ray2_dots = torch.mm(existing_point_cloud - cam_center2, ray_dir2.unsqueeze(1))

        proj1 = cam_center1 + ray_dir1 * ray1_dots
        proj2 = cam_center2 + ray_dir2 * ray2_dots

        dist1 = torch.norm(existing_point_cloud - proj1, dim=1)
        dist2 = torch.norm(existing_point_cloud - proj2, dim=1)

        valid_mask = (dist1 < distance_threshold) & (dist2 < distance_threshold)

        # Apply camera distance filtering
        cam_dist1 = torch.norm(existing_point_cloud - cam_center1, dim=1)
        cam_dist2 = torch.norm(existing_point_cloud - cam_center2, dim=1)
        too_close_mask = (cam_dist1 < min_cam_distance) | (cam_dist2 < min_cam_distance)

        # Apply outlier filtering using mean and standard deviation
        cloud_mean = existing_point_cloud.mean(dim=0)
        cloud_std = existing_point_cloud.std(dim=0)
        outlier_mask = ~torch.all(torch.abs(existing_point_cloud - cloud_mean) < sigma_threshold * cloud_std, dim=1)

        # Combine masks
        combined_mask = valid_mask & ((too_close_mask) | (outlier_mask))
        combined_mask = combined_mask.detach()

        # Filter intersecting points
        intersecting_points = existing_point_cloud[combined_mask]

        return weighted_gen_l1_loss, weighted_cross_l1_loss, intersecting_points, combined_mask



