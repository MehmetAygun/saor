from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from toolz import valfilter
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from pytorch3d.loss import (mesh_edge_loss, mesh_laplacian_smoothing as laplacian_smoothing)
from pytorch3d.renderer import (TexturesVertex, look_at_view_transform,
                                TexturesUV, FoVPerspectiveCameras)
from pytorch3d.structures import Meshes


from .encoder import Encoder, TxtEncoder
from .field import ProgressiveField, Field
from .generator import ProgressiveGiraffeGenerator
from .loss import (get_loss, volume_loss, pose_reg_loss,
                   part_uniform_loss,
                   part_peak_loss,
                   depthloss,
                   part_share_loss, part_center_loss)

from .renderer import Renderer, save_mesh_as_gif
from .tools import create_mlp, init_rotations, convert_3d_to_uv_coordinates, safe_model_state_dict
from .tools import azim_to_rotation_matrix, elev_to_rotation_matrix, roll_to_rotation_matrix

from utils import path_mkdir, use_seed
from utils.image import convert_to_img
from utils.logger import print_warning
from utils.mesh import save_mesh_as_obj, repeat, get_icosphere, normal_consistency, normalize
from utils.metrics import MeshEvaluator
from utils.pytorch import torch_to

import matplotlib.cm as cm


Tensor = torch.cuda.FloatTensor

# SKELETON
N_JOINTS = 8

# POSE & SCALE DEFAULT
N_POSES = 6
N_ELEV_AZIM = [1, 6]
SCALE_ELLIPSE = [1, 0.7, 0.7]

PRIOR_TRANSLATION = [0., 0., 2.732]


class SAOR(nn.Module):
    name = 'saor'

    def __init__(self, img_size, **kwargs):

        super().__init__()
        self.init_kwargs = deepcopy(kwargs)
        self.init_kwargs['img_size'] = img_size

        self.device = kwargs.get('device', None)
        self.n_classes = kwargs.get('n_classes', -1)

        self._init_encoder(img_size, **kwargs.get('encoder', {}))

        self._init_meshes(**kwargs.get('mesh', {}))
        self.renderer = Renderer(img_size, **kwargs.get('renderer', {}))
        self._init_rend_predictors(**kwargs.get('rend_predictor', {}))
        self._init_background_model(img_size, **kwargs.get('background', {}))
        self._init_milestones(**kwargs.get('milestones', {}))
        self._init_loss(**kwargs.get('loss', {}))

        self._init_articulation(**kwargs.get('articulation', {}))

        self.prop_heads = torch.zeros(self.n_poses)
        self.cur_epoch, self.cur_iter = 0, 0
        self._debug = False
        self.pose_history = {}
        self.counter = 0
        self.def_frozen = False
        self.swap_pose_ranges = [[30,180], [40,165], [50,150], [60,135]]

    @property
    def n_features(self):
        return self.encoder.out_ch

    @property
    def tx_code_size(self):
        return self.txt_generator.current_code_size

    @property
    def sh_code_size(self):
        return self.deform_field.current_code_size

    def _init_encoder(self, img_size, **kwargs):

        self.shared_encoder = kwargs.pop('shared', True)
        if self.shared_encoder:
            self.encoder = Encoder(img_size, **kwargs)
        else:
            self.encoder = Encoder(img_size, **kwargs)
            self.encoder_tx = TxtEncoder(img_size, **kwargs)

    def _init_meshes(self, **kwargs):
        kwargs = deepcopy(kwargs)
        mesh_init = kwargs.pop('init', 'sphere')
        scale = kwargs.pop('scale', 1)

        self.symmetry = kwargs.pop('symmetry', False)

        if 'sphere' in mesh_init or 'ellipse' in mesh_init:
            mesh = get_icosphere(4 if 'hr' in mesh_init else 3)
            if 'ellipse' in mesh_init:
                scale = scale * torch.Tensor([SCALE_ELLIPSE])
        else:
            raise NotImplementedError
        self.mesh_src = mesh.scale_verts(scale)
        verts, faces = self.mesh_src.get_mesh_verts_faces(0)

        if self.symmetry:
            pos_side = torch.where(verts[:,2]>=0)[0]
            neg_side = torch.where(verts[:,2]<0)[0]

            diff = verts[neg_side, None, :2] - verts[None, pos_side,:2]
            diff = torch.pow(diff,2).sum(2)
            idxs = torch.argmin(diff, 1)

            not_remove_vers = []# verts that have a connection to other side
            for i in range(neg_side.shape[0]):
                pair = neg_side[i], pos_side[idxs[i]]
                cond = pair[1] == faces[torch.where(pair[0] == faces)[0]]
                if not torch.sum(cond)>0:
                    not_remove_vers.append(i)
            neg_side = neg_side[not_remove_vers]
            idxs = idxs[not_remove_vers]
            #
            self.register_buffer('pos_side', pos_side)
            self.register_buffer('neg_side', neg_side)
            self.register_buffer('neg_to_pos', pos_side[idxs])

        self.initial_volume = volume_loss(self.mesh_src, 0)

        
        self.register_buffer('uvs', convert_3d_to_uv_coordinates(self.mesh_src.get_mesh_verts_faces(0)[0])[None])
                
        self.use_mean_txt = kwargs.pop('use_mean_txt', kwargs.pop('use_mean_text', False))  # retro-compatibility
        dfield_kwargs = kwargs.pop('deform_fields', {})
        tgen_kwargs = kwargs.pop('texture_uv', {})
        tgen_kwargs['shared_encoder'] = self.shared_encoder
        assert len(kwargs) == 0

        self.avg_edge_lenght = mesh_edge_loss(mesh, 0.)

        self.deform_field = ProgressiveField(inp_dim=self.n_features, name='deformation', **dfield_kwargs)

        if self.shared_encoder:
            self.txt_generator = ProgressiveGiraffeGenerator(inp_dim=self.n_features, **tgen_kwargs)
        else:
            self.txt_generator = ProgressiveGiraffeGenerator(inp_dim=self.encoder_tx.out_ch, **tgen_kwargs)

    def _init_rend_predictors(self, **kwargs):

        kwargs = deepcopy(kwargs)
        self.n_poses = kwargs.pop('n_poses', N_POSES)
        n_elev, n_azim = kwargs.pop('n_elev_azim', N_ELEV_AZIM)
        assert self.n_poses == n_elev * n_azim
        self.alternate_optim = kwargs.pop('alternate_optim', True)
        self.pose_step = True
        self.pose_rep = kwargs.pop('pose_rep', 'axis')

        NF, NP = self.n_features, self.n_poses

        # Translation
        self.T_regressors = nn.ModuleList([create_mlp(NF, 3, zero_last_init=True) for _ in range(NP)])
        T_range = kwargs.pop('T_range', 1)
        T_range = [T_range] * 3 if isinstance(T_range, (int, float)) else T_range
        self.register_buffer('T_range', torch.Tensor(T_range))
        self.register_buffer('T_init', torch.zeros(3))
        self.register_buffer('T_cam', torch.Tensor(kwargs.pop('prior_translation', PRIOR_TRANSLATION)))


        # Rotation
        self.rot_regressors = nn.ModuleList([create_mlp(NF, 3, zero_last_init=True) for _ in range(NP)])
        a_range, e_range, r_range = kwargs.pop('azim_range'), kwargs.pop('elev_range'), kwargs.pop('roll_range')

        azim, elev, roll = [(e[1] - e[0]) / n for e, n in zip([a_range, e_range, r_range], [n_azim, n_elev, 1])]
        R_init = init_rotations('uniform', n_elev=n_elev, n_azim=n_azim, elev_range=e_range, azim_range=a_range)
        self.register_buffer('R_range', torch.Tensor([azim * 0.52, elev * 0.52, roll * 0.52]))
        self.register_buffer('R_init', R_init)
        self.azim_range, self.elev_range, self.roll_range = a_range, e_range, r_range
        self.register_buffer('R_cam', torch.eye(3))

        # Scale
        self.scale_regressor = create_mlp(NF, 3, zero_last_init=True)
        scale_range = kwargs.pop('scale_range', 0.5)
        scale_range = [scale_range] * 3 if isinstance(scale_range, (int, float)) else scale_range
        self.register_buffer('scale_range', torch.Tensor(scale_range))
        self.register_buffer('scale_init', torch.ones(3))

        fov_range = kwargs.pop('fov_range', 0)
        fov_init = kwargs.pop('fov_init', 30)

        self.register_buffer('fov_range', torch.Tensor([fov_range]))
        self.register_buffer('fov_init', torch.Tensor([fov_init]))
        self.fov_regressor = create_mlp(NF, 1, zero_last_init=True)

        # Pose probabilities
        self.proba_regressor = create_mlp(NF, NP)
        self.pose_temp = 1.

        assert len(kwargs) == 0, kwargs

    @property
    def n_candidates(self):
        # this is always one saor only use one pose prediction for loss calculation
        return 1 

    def _init_background_model(self, img_size, **kwargs):
        if len(kwargs) > 0:
            bkg_kwargs = deepcopy(kwargs)
            self.bkg_generator = ProgressiveGiraffeGenerator(inp_dim=self.n_features, img_size=img_size, **bkg_kwargs)

    def _init_milestones(self, **kwargs):
        kwargs = deepcopy(kwargs)
        self.milestones = {
            'constant_txt': kwargs.pop('constant_txt', kwargs.pop('constant_text', 0)),  # retro-compatibility
            'freeze_T_pred': kwargs.pop('freeze_T_predictor', 0),
            'freeze_R_pred': kwargs.pop('freeze_R_predictor', 0),
            'freeze_s_pred': kwargs.pop('freeze_scale_predictor', 0),
            'freeze_shape': kwargs.pop('freeze_shape', 0),
            'freeze_articulation': kwargs.pop('freeze_articulation', 0),
            'freeze_swap': kwargs.pop('freeze_swap', 0),
            'mean_txt': kwargs.pop('mean_txt', kwargs.pop('mean_text', self.use_mean_txt)),  # retro-compatibility
        }
        assert len(kwargs) == 0

    def _init_loss(self, **kwargs):
        kwargs = deepcopy(kwargs)
        loss_weights = {
            'rgb': kwargs.pop('rgb_weight', 0.0),
            'normal': kwargs.pop('normal_weight', 0),
            'laplacian': kwargs.pop('laplacian_weight', 0),
            'perceptual': kwargs.pop('perceptual_weight', 0),
            'edge': kwargs.pop('edge_weight', 0),
            'silhouette': kwargs.pop('silhouette_weight', 0),
            'volume': kwargs.pop('volume_weight', 0),
            'pose_score': kwargs.pop('pose_score_weight', 0),
            'part_reg': kwargs.pop('part_reg_weight', 0),#part uniformity
            'part_peak_reg': kwargs.pop('part_peak_reg_weight', 0), #part peakiness
            'articulation_reg': kwargs.pop('articulation_reg_weight', 0),
            'part_share_reg': kwargs.pop('part_share_reg_weight', 0), #part sharing
            'part_center_reg': kwargs.pop('part_center_reg_weight', 0), #part sharing
            'pose_reg': kwargs.pop('pose_reg_weight', 0),
            'scale_reg': kwargs.pop('scale_reg_weight', 0),
            'depth_reg': kwargs.pop('depth_reg_weight', 0),
            'swap': kwargs.pop('swap_weight', 0),
            'new_swap': kwargs.pop('new_swap_weight', 0),

        }

        self.prog_mask = kwargs.pop('prog_mask', False)

        name = kwargs.pop('name', 'mse')
        perceptual_kwargs = kwargs.pop('perceptual', {})
       
        assert len(kwargs) == 0, kwargs

        self.loss_weights = valfilter(lambda v: v > 0, loss_weights)
        self.loss_names = [f'loss_{n}' for n in list(self.loss_weights.keys()) + ['total']]
        self.criterion = get_loss(name)(reduction='none')

        self.silhouette_loss = get_loss('mse')(reduction='none')

        self.volume_loss = volume_loss
        self.pose_reg_loss = pose_reg_loss
        self.l1_loss = torch.nn.L1Loss(reduction='none')

        if 'perceptual' in self.loss_weights:
            self.perceptual_loss = get_loss('perceptual')(**perceptual_kwargs)

        if 'depth_reg' in self.loss_weights:
            self.depth_loss = depthloss

    def _init_articulation(self, **kwargs):

        kwargs = deepcopy(kwargs)
        self.articulation = kwargs.pop('apply', False)
        self.predict_parts = kwargs.pop('predict_parts', False)
        self.n_joints = kwargs.pop('n_joints', N_JOINTS)
        self.temperature = kwargs.pop('temperature', 10.)
        self.rigid_part = kwargs.pop('rigid_part', False)

        verts = self.mesh_src.verts_packed()
        n_verts = verts.shape[0]

        if self.predict_parts:
            n_j = self.n_joints+1 if self.rigid_part else self.n_joints
            self.part_field = Field(n_units=128, n_layers=3, latent_size=self.n_features, in_ch=3, out_ch=n_j, zero_last_init=False)
        else:
            if self.rigid_part:
                self.register_parameter('vertex_to_joint', torch.nn.Parameter(torch.rand(n_verts,self.n_joints+1)))
            else:
                self.register_parameter('vertex_to_joint', torch.nn.Parameter(torch.rand(n_verts,self.n_joints)))

        T_range = kwargs.pop('T_range', 1)
        self.register_buffer('articulation_T_range', torch.Tensor(T_range))
        self.register_buffer('articulation_T_init', torch.zeros(3))

        a_range, e_range, r_range = kwargs.pop('azim_range', [0,360]), kwargs.pop('elev_range', [0,360]), kwargs.pop('roll_range', [0,360])
        azim, elev, roll = [(e[1] - e[0])/2 for e in [a_range, e_range, r_range]]
        a_range, e_range, r_range = [(e[1] + e[0])/2 for e in [a_range, e_range, r_range]]

        self.register_buffer('articulation_R_init', torch.Tensor([a_range,e_range,r_range]))
        self.register_buffer('articulation_R_range', torch.Tensor([azim, elev, roll]))

        s_range = kwargs.pop('scale_range', 0.25)
        s_init = kwargs.pop('scale_init', 1.0)

        self.register_buffer('articulation_S_init', torch.Tensor([s_init]))
        self.register_buffer('articulation_S_range', torch.Tensor([s_range]))

        NF, NP = self.n_features, self.n_joints

        # Translation per Joint
        self.joint_T_regressors = nn.ModuleList([create_mlp(NF, 3, zero_last_init=True) for _ in range(NP)])
        # Rotation per Joint
        self.joint_rot_regressors = nn.ModuleList([create_mlp(NF, 3, zero_last_init=True) for _ in range(NP)])
        #Scale per joint
        self.joint_scale_regressors = nn.ModuleList([create_mlp(NF, 3, zero_last_init=True) for _ in range(NP)])


    @property
    def pred_background(self):
        return hasattr(self, 'bkg_generator')

    def is_live(self, name):
        milestone = self.milestones[name]
        if isinstance(milestone, bool):
            return milestone
        else:
            return True if self.cur_epoch < milestone else False

    def to(self, device):
        super().to(device)
        self.mesh_src = self.mesh_src.to(device)
        self.renderer = self.renderer.to(device)
        return self

    #@profile
    def forward(self, inp, labels=None, debug=False, return_meshes=False):
        # XXX pytorch3d objects are not well handled by DDP so we need to manually move them to GPU
        # self.mesh_src, self.renderer = [t.to(inp['imgs'].device) for t in [self.mesh_src, self.renderer]]

        self._debug = debug
        imgs, masks, depths, poses, K, B = inp['imgs'], inp['masks'], inp['depths'], inp['poses'], self.n_candidates, len(inp['imgs'])
        cats = inp.get('cats')
              
        features = self.encoder(imgs)
        if self.shared_encoder:
            meshes = self.predict_meshes(features)
        else:
            features_tx = self.encoder_tx(imgs)
            meshes = self.predict_meshes(features, features_tx)

        if self.articulation and (not self.is_live('freeze_articulation')):
            no_art_meshes = meshes
            meshes = self.articulate_meshes(meshes, features)
        else:
            no_art_meshes = meshes

        R, T = self.predict_poses(features)
        bkgs = self.predict_background(features) if self.pred_background else None
        self.bkgs = bkgs
        if self.alternate_optim:
            if self.pose_step:
                meshes, bkgs = meshes.detach(), bkgs.detach() if self.pred_background else None
            else:
                R, T = R.detach(), T.detach()

        posed_meshes, R_cam, T_cam = self.update_with_poses(meshes, R, T)

        fgs, alpha = self.renderer(posed_meshes, R_cam, T_cam, fov =self._fovs.squeeze()).split([3, 1], dim=1)  # (K*B)CHW
        rec = fgs * alpha + (1 - alpha) * bkgs if self.pred_background else fgs

        losses, select_idx = self.compute_losses(meshes, no_art_meshes, imgs, features, bkgs, rec, masks, depths, poses, alpha, R, T, labels, cats, inp)

        if debug:
            out = rec.view(K, B, *rec.shape[1:]) if K > 1 else rec[None]
        elif return_meshes:
            # we need to realign mesh to account for pose parametrization mismatch
            verts, faces = posed_meshes.verts_padded(), posed_meshes.faces_padded()
            R_gt, T_gt = map(lambda t: t.squeeze(2), inp['poses'].split([3, 1], dim=2))
            verts = (verts @ R_cam + T_cam[:, None] - T_gt[:, None]) @ R_gt.transpose(1, 2)
            meshes = Meshes(verts=verts, faces=faces, textures=meshes.textures)
            out = losses, meshes
        else:
            rec = rec.view(K, B, *rec.shape[1:])[select_idx, torch.arange(B)] if K > 1 else rec
            out = losses, rec

        self._debug = False
        return out

    def predict_meshes(self, features, features_tx=None):
        torch.autograd.set_detect_anomaly(True)
        verts, faces = self.mesh_src.get_mesh_verts_faces(0)
        meshes = self.mesh_src.extend(len(features))  # XXX does a copy

        if self.symmetry:
            B = len(features)
            delta_v = self.predict_disp_verts(verts, features).contiguous().view(B, -1, 3)
            delta_v[:, self.neg_side, :2] = delta_v[:,self.neg_to_pos,:2]
            delta_v[:, self.neg_side, 2] = -1* delta_v[:,self.neg_to_pos,2]
            delta_v = delta_v.tanh()
            meshes.offset_verts_(delta_v.view(-1, 3))
        else:
            delta_v = self.predict_disp_verts(verts, features)
            delta_v = delta_v.tanh()
            meshes.offset_verts_(delta_v)

        if self.shared_encoder:
            meshes.textures = self.predict_textures(faces, features)
        else:
            meshes.textures = self.predict_textures(faces, features_tx)

        meshes.scale_verts_(self.predict_scales(features))

        return meshes

    def articulate_meshes(self, meshes, features):

        device = features.device
        B = len(features)

        art_features = features
        
        j_R, j_T, j_S = self.predict_articulation_poses(art_features)
        
        j_S = j_S.view(-1, B, 3).permute(1,0,2)#BXJX3

        j_R = j_R.view(-1, B, 3, 3).permute(1,0,2,3) #BxJx3x3
        j_T = j_T.view(-1, B, 3, 1).permute(1,0,2,3)# BX1X3x1

        if self.rigid_part:
            no_R = torch.eye(3).repeat(B,1,1,1).to(device)
            no_T = torch.zeros(3,1).repeat(B,1,1,1).to(device)
            no_S = torch.ones(3).repeat(B,1,1).to(device)

            j_R = torch.cat((j_R, no_R),1)
            j_T = torch.cat((j_T, no_T),1)
            j_S = torch.cat((j_S, no_S),1)

        j_P = torch.cat((j_R, j_T),3)

        verts = meshes.verts_packed().view(B,-1,3)
        n_v= verts.shape[1]

        if self.predict_parts:
            vertex_to_parts = self.part_field(verts, art_features)
            weights = torch.softmax(vertex_to_parts*self.temperature, dim=2)
            self.vertex_to_joint = weights

        else:
            weights = torch.softmax(self.vertex_to_joint*self.temperature, dim=1) # VXJ
            weights = weights.repeat(B,1,1)

        n_j = weights.shape[2]

        part_weights = F.normalize(weights,p=1,dim=1)
        part_centers = torch.bmm(part_weights.permute(0,2,1),verts) # BxJx3

        self.part_centers = part_centers

        rel_pos = verts[:,:,None,:] - part_centers[:,None,:,:] # BxVxJx3
        self.rel_pos = rel_pos
        homo_cord = torch.ones(B,n_v,n_j,1).to(device)
        rel_pos = torch.cat((rel_pos,homo_cord),3)
        articulated_verts =  torch.zeros_like(verts)

        for j in range(n_j):
            new_verts = torch.bmm(j_P[:,j].view(B,3,4),rel_pos[:,:,j].permute(0,2,1))
            weigted_verts = new_verts.permute(0,2,1)*weights[:,:,j,None]
            articulated_verts = articulated_verts + weigted_verts


        vert_scales = torch.bmm(weights, j_S)
        articulated_verts = articulated_verts * vert_scales

        meshes = Meshes(articulated_verts, faces=meshes.faces_padded(), textures=meshes.textures)

        return meshes

    def predict_disp_verts(self, verts, features):

        disp_verts = self.deform_field(verts.view(-1,3), features)
        if self.is_live('freeze_shape'):
            disp_verts = disp_verts * 0
        return disp_verts.view(-1, 3)

    def predict_textures(self, faces, features):
        B = len(features)
        perturbed = self.training and np.random.binomial(1, p=0.2)

        maps = self.txt_generator(features)
        if self.symmetry:
            h_maps = torchvision.transforms.functional.hflip(maps)
            maps = torch.cat((maps,h_maps),dim=3)

        if self.is_live('constant_txt'):
            H, W = maps.shape[-2:]
            nb = int(H * W * 0.1)
            idxh, idxw = torch.randperm(H)[:nb], torch.randperm(W)[:nb]
            maps = maps[:, :, idxh, idxw].mean(2)[..., None, None].expand(-1, -1, H, W)
        elif self.training and perturbed and self.use_mean_txt and self.is_live('mean_txt'):
            H, W = maps.shape[-2:]
            nb = int(H * W * 0.1)
            idxh, idxw = torch.randperm(H)[:nb], torch.randperm(W)[:nb]
            maps = maps[:, :, idxh, idxw].mean(2)[..., None, None].expand(-1, -1, H, W)

        return TexturesUV(maps.permute(0, 2, 3, 1), faces[None].expand(B, -1, -1), self.uvs.expand(B, -1, -1))

    def predict_scales(self, features):
        s_pred = self.scale_regressor(features).tanh()
        if self.is_live('freeze_s_pred'):
            s_pred = s_pred * 0
        self._scales = s_pred * self.scale_range + self.scale_init
        return self._scales

    def predict_fov(self, features):
        fov_pred = self.fov_regressor(features).tanh()
        self._fovs = fov_pred * self.fov_range + self.fov_init

        return self._fovs

    def predict_poses(self, features, update_history=True):

        B = len(features)

        _ = self.predict_fov(features)

        T_pred = torch.stack([p(features) for p in self.T_regressors], dim=0).tanh()
        if self.is_live('freeze_T_pred'):
            T_pred = T_pred * 0
        T = (T_pred * self.T_range + self.T_init).view(-1, 3)

        R_pred = torch.stack([p(features) for p in self.rot_regressors], dim=0).tanh()  # KBC
        R_pred = R_pred[..., [1, 0, 2]]  # XXX for retro-compatibility
        if self.is_live('freeze_R_pred'):
            R_pred = R_pred * 0
        R_pred = (R_pred * self.R_range + self.R_init[:, None]).view(-1, 3)
        azim, elev, roll = map(lambda t: t.squeeze(1), R_pred.split([1, 1, 1], 1))

        R = azim_to_rotation_matrix(azim) @ elev_to_rotation_matrix(elev) @ roll_to_rotation_matrix(roll)
        if R.dim() == 2:
            R = R.unsqueeze(0)

        self.pose_scores = self.proba_regressor(features.view(B, -1)).permute(1,0)
        self._pose_proba = torch.softmax((-1.*self.pose_scores)/self.pose_temp, dim=0)

        self.azim_preds = azim
        self.elev_preds = elev
        self.roll_preds = roll


        #if (torch.rand(1) < 0.2 or self.cur_epoch < 50) and self.training:
        #p = 0.2 if self.cur_epoch < 200 else 0.2 - (self.cur_epoch/500.)*0.2
        p = 0.2
        # p = 1 - min(self.cur_epoch, 50) /50.*0.8
        #if (self.cur_epoch < 50 or torch.rand(1)) < p and self.training:
        if torch.rand(1) < p and self.training:
            indices = torch.rand_like(self._pose_proba).max(0)[1]
            select_fn = lambda t: t.view(self.n_poses, B, *t.shape[1:])[indices, torch.arange(B)]
            R, T = map(select_fn, [R, T])
            azim_max, elev_max = map(select_fn, [azim, elev])
            azim_max = azim_max.detach().cpu().numpy()
            elev_max = elev_max.detach().cpu().numpy()
        else:
            indices = self._pose_proba.max(0)[1]
            select_fn = lambda t: t.view(self.n_poses, B, *t.shape[1:])[indices, torch.arange(B)]
            R, T = map(select_fn, [R, T])
            azim_max, elev_max = map(select_fn, [azim, elev])
            azim_max = azim_max.detach().cpu().numpy()
            elev_max = elev_max.detach().cpu().numpy()

        self.pose_indices = indices
        self.pose_selected_probs = self._pose_proba[indices,torch.arange(B)]
        self.pose_scores_selected = self.pose_scores[indices,torch.arange(B)]

        if update_history:
            self.pose_history = {'R':R.detach().cpu().numpy(),
                                 'T':T.detach().cpu().numpy(),
                                 'azim':azim.detach().cpu().numpy(),
                                 'elev':elev.detach().cpu().numpy(),
                                 'roll':elev.detach().cpu().numpy(),
                                 'azim_max': azim_max,
                                 'elev_max': elev_max
                                 }

        return R, T

    def predict_articulation_poses(self, features):
        B = len(features)

        T_pred = torch.stack([p(features) for p in self.joint_T_regressors], dim=0).tanh()
        T = (T_pred * self.articulation_T_range + self.articulation_T_init).view(-1, 3)

        R_pred = torch.stack([p(features) for p in self.joint_rot_regressors], dim=0).tanh()  # KBC
        R_pred = R_pred[..., [1, 0, 2]]  # XXX for retro-compatibility

        R_pred = (R_pred * self.articulation_R_range + self.articulation_R_init).view(-1, 3)
        azim, elev, roll = map(lambda t: t.squeeze(1), R_pred.split([1, 1, 1], 1))
        R = azim_to_rotation_matrix(azim) @ elev_to_rotation_matrix(elev) @ roll_to_rotation_matrix(roll)

        S_pred =  torch.stack([p(features) for p in self.joint_scale_regressors], dim=0).tanh()
        S = (S_pred * self.articulation_S_range + self.articulation_S_init).view(-1, 3)

        return R,T,S

    def predict_background(self, features):
        res = self.bkg_generator(features)
        return res.repeat(self.n_candidates, 1, 1, 1) if self.n_candidates > 1 else res

    #@profile
    def update_with_poses(self, meshes, R, T):
        K, B = len(T) // len(meshes), len(meshes)
        meshes = repeat(meshes, K)  # XXX returns a copy of meshes
        Nv = meshes.num_verts_per_mesh()[0]
        meshes = Meshes(meshes.verts_padded() @ R, faces=meshes.faces_padded(), textures=meshes.textures)
        meshes.offset_verts_(T[:, None].expand(-1, Nv, -1).reshape(-1, 3))
        R, T = self.R_cam[None].expand(K * B, -1, -1), self.T_cam[None].expand(K * B, -1)
        return meshes, R, T

    #@profile
    def compute_losses(self, meshes, no_art_meshes, imgs, features, bkgs, rec, masks, depths, poses, alpha, R, T, labels, cats, inp):

        device = features.device
        K, B = self.n_candidates, len(imgs)
        if any(labels) is None:
            labels = torch.zeros(K).to(device)
        if K > 1:
            imgs = imgs.repeat(K, 1, 1, 1)
            if self.pred_background:
                bkgs = bkgs.repeat(K, 1, 1, 1)
            masks = masks.repeat(K, 1, 1, 1)
            depths = depths.repeat(K, 1, 1, 1)
            poses = poses.repeat(K, 1, 1)

        depths_c = inp.get('depths_c')
        if depths_c is not None:
            depths_c = depths_c.to(device)

        losses = {k: torch.tensor(0.0, device=imgs.device) for k in self.loss_weights}
        update_3d, update_pose = (not self.pose_step, self.pose_step) if self.alternate_optim else (True, True)

        # Standard reconstrution error on RGB values
        if 'rgb' in losses:# and not self.is_live('freeze_shape'):
            #masked rgb loss
            losses['rgb'] = self.loss_weights['rgb'] * (self.criterion(rec, imgs).sum(dim=1) * masks.squeeze()).flatten(1).mean(1)

        # Perceptual loss
        if 'perceptual' in losses:# and not self.is_live('freeze_shape'):# and (self.tx_code_size > 0 and self.sh_code_size > 0):

            target = imgs*masks if self.pred_background else imgs
            l0 = self.perceptual_loss(rec, target)
            # resize = torchvision.transforms.Resize(rec.shape[-1]//2)
            # l1 = self.perceptual_loss(resize(rec), resize(target))
            # resize = torchvision.transforms.Resize(rec.shape[-1]//4)
            # l2 = self.perceptual_loss(resize(rec), resize(target))
            losses['perceptual'] = self.loss_weights['perceptual'] * (l0)


        if 'silhouette' in losses:

            masks_means = torch.abs(torch.nn.functional.avg_pool2d(masks, (8,8)) - 0.5)
            weights = torch.exp(-1*masks_means)
            weights = nn.Upsample(scale_factor=(8,8), mode='nearest')(weights)
            losses['silhouette'] = self.loss_weights['silhouette'] * (self.silhouette_loss(alpha, masks)*weights).flatten(1).mean(1)


        if 'scale_reg' in losses:
            scale_loss = torch.abs(self._scales - self.scale_init).mean(1)
            losses['scale_reg'] = self.loss_weights['scale_reg'] * (scale_loss)


        if 'depth_reg' in losses:# and (self.tx_code_size > 0 and self.sh_code_size > 0):
            posed_meshes, R_cam, T_cam = self.update_with_poses(meshes, R, T)

            device = meshes.device
            cameras = FoVPerspectiveCameras(device=device, R=R_cam, T=T_cam, fov=self._fovs.squeeze())
            fragments = self.renderer.mesh_rasterizer(posed_meshes, R=R_cam, T=T_cam, cameras=cameras)

            loss_d = self.depth_loss(fragments.zbuf[:,:,:,0], alpha.squeeze(), depths.float(), masks)
            loss_d = loss_d * masks.squeeze()

            losses['depth_reg'] = self.loss_weights['depth_reg'] * (loss_d).flatten(1).mean(1)


        if 'pose_score' in losses:
            l_rec = 0.
            if 'silhouette' in losses:
                l_rec = l_rec + losses['silhouette']
            if 'rgb' in losses:# and not self.is_live('freeze_shape'):
                l_rec = l_rec + losses['rgb']
            if 'depth_reg' in losses:
                l_rec = l_rec + losses['depth_reg']

            loss =  torch.pow((self.pose_scores_selected - l_rec.detach()),2).mean()
            losses['pose_score'] = self.loss_weights['pose_score'] * loss


        if 'articulation_reg' in losses and (not self.is_live('freeze_articulation')):
            verts_after = meshes.verts_packed()
            verts_before = no_art_meshes.verts_packed()
            l = torch.pow((verts_after-verts_before),2).view(B,-1,3).flatten(1).mean(1)
            losses['articulation_reg'] =  self.loss_weights['articulation_reg'] * l


        # Mesh regularization
        if update_3d:
            if 'normal' in losses:
                losses['normal'] = self.loss_weights['normal'] * normal_consistency(meshes)
            if 'laplacian' in losses:
                losses['laplacian'] = self.loss_weights['laplacian'] * laplacian_smoothing(meshes, method='uniform')
            if 'edge' in losses:
                avg_length = torch.sqrt(mesh_edge_loss(meshes, 0.))
                losses['edge'] = self.loss_weights['edge'] * mesh_edge_loss(meshes, avg_length)
            if 'volume' in losses:
                losses['volume'] = self.loss_weights['volume'] * volume_loss(meshes, self.initial_volume.to(imgs.device))

        # Swap loss
        # XXX when latent spaces are small, codes are similar so there is no need to compute the swap loss
        if 'new_swap' in losses and (self.tx_code_size > 0 and self.sh_code_size > 0):
            B, dev = len(meshes), imgs.device
            if  cats is not None:# here we select swap pairs from same category
                u_clss = torch.unique(cats)
                src_idxs = []
                trg_idxs = []

                for c in u_clss:
                    cls_idxs = torch.arange(B)[cats==c]
                    if torch.sum(cats==c) > 1:
                        for s_i in cls_idxs:
                            src_idxs.append(s_i.item())
                            targets = cls_idxs[cls_idxs!=s_i]
                            t_i = targets[torch.randperm(targets.shape[0])[0]]
                            trg_idxs.append(t_i.item())

                src_idxs = torch.tensor(src_idxs)
                trg_idxs = torch.tensor(trg_idxs)
            else:
                src_idxs = torch.arange(B)
                trg_idxs = torch.randperm(B)

            if src_idxs.shape[0] !=0 : #check if cats matched
                faces = meshes.faces_padded()
                scales = self._scales[:, None]
                verts_no_art = no_art_meshes.verts_padded()
                faces, textures = meshes.faces_padded(), meshes.textures

                new_meshes = Meshes((verts_no_art[src_idxs] / scales[src_idxs]) * scales[trg_idxs], faces[trg_idxs], textures[trg_idxs])
                new_meshes = self.articulate_meshes(new_meshes, features[trg_idxs])

                posed_meshes, R_cam, T_cam = self.update_with_poses(new_meshes, R[trg_idxs], T[trg_idxs])
                rec_sw, alpha_sw = self.renderer(posed_meshes, R_cam, T_cam, fov = self._fovs.squeeze()[trg_idxs]).split([3, 1], dim=1)


                loss = 0.
                if 'rgb' in losses:
                    loss += self.loss_weights['rgb'] * (self.criterion(rec_sw, imgs[trg_idxs]).sum(dim=1) * masks[trg_idxs].squeeze()).flatten(1).mean(1)
                if 'silhouette' in losses:

                    weights = torch.nn.functional.avg_pool2d(masks[trg_idxs], (8,8)) - 0.5
                    weights = torch.exp(-1*torch.abs(weights))
                    weights = nn.Upsample(scale_factor=(8,8), mode='nearest')(weights)

                    loss += self.loss_weights['silhouette'] * (self.silhouette_loss(alpha_sw, masks[trg_idxs])*weights).flatten(1).mean(1)
        
                if src_idxs.shape[0] != B:
                    losses['new_swap'] = self.loss_weights['new_swap'] * loss.mean()
                else:
                    losses['new_swap'] = self.loss_weights['new_swap'] * loss #* (self.cur_epoch/100.)


        if update_pose and 'pose_reg' in losses:
            losses['pose_reg'] = self.loss_weights['pose_reg'] * self.pose_reg_loss(self.azim_preds)

        if 'part_center_reg' in losses and (not self.is_live('freeze_articulation')) and self.articulation:
            losses['part_center_reg'] = self.loss_weights['part_center_reg'] * part_center_loss(self.part_centers)

        if 'part_share_reg' in losses and (not self.is_live('freeze_articulation')) and self.articulation:
            losses['part_share_reg'] = self.loss_weights['part_share_reg'] * part_share_loss(self.vertex_to_joint)

        if 'part_peak_reg' in losses and (not self.is_live('freeze_articulation')) and self.articulation:
            losses['part_peak_reg'] = self.loss_weights['part_peak_reg'] * part_peak_loss(self.vertex_to_joint)

        if 'part_reg' in losses and (not self.is_live('freeze_articulation')) and self.articulation:
            losses['part_reg'] = self.loss_weights['part_reg'] * part_uniform_loss(self.vertex_to_joint)


        dist = sum(losses.values())


        if K > 1:
            dist, select_idx = dist.view(K, B), self._pose_proba.max(0)[1]
            dist = (self._pose_proba * dist).sum(0)
            for k, v in losses.items():
                if v.numel() != 1:
                    losses[k] = (self._pose_proba * v.view(K, B)).sum(0).mean()

            # For monitoring purpose only
            pose_proba_d = self._pose_proba.detach().cpu()
            self._prob_heads = pose_proba_d.mean(1).tolist()
            self._prob_max = pose_proba_d.max(0)[0].mean().item()
            self._prob_min = pose_proba_d.min(0)[0].mean().item()
            count = torch.zeros(K, B).scatter(0, select_idx[None].cpu(), 1).sum(1)
            self.prop_heads = count / B

        else:
            pose_proba_d = self._pose_proba.detach().cpu()
            self._prob_heads = pose_proba_d.mean(1).tolist()
            self._prob_max = pose_proba_d.max(0)[0].mean().item()
            self._prob_min = pose_proba_d.min(0)[0].mean().item()
            select_idx = torch.zeros(B).long()
            for k, v in losses.items():
                if v.numel() != 1:
                    losses[k] = (self.pose_selected_probs*v).mean()

        losses['total'] = dist.mean()
        return losses, select_idx

    def iter_step(self):
        self.cur_iter += 1
        if self.alternate_optim and self.cur_iter % self.alternate_optim == 0:
            self.pose_step = not self.pose_step

    def step(self):

        self.cur_epoch += 1
        self.deform_field.step()
        self.txt_generator.step()
        if self.pred_background:
            self.bkg_generator.step()

        swap_idx = (self.cur_epoch - 100) // 100
        swap_idx = min(max(swap_idx, 0), len(self.swap_pose_ranges)-1)
        self.swap_min_angle, self.swap_max_angle = self.swap_pose_ranges[swap_idx]

        # self.pose_temp = 1 - (min((self.cur_epoch / 500.), 0.99))
        self.pose_temp = 1 - (min((self.cur_epoch / self.n_epoches), 0.99))
        if self.cur_epoch> 1000:
            self.loss_weights['rgb'] = 0.1


    def set_cur_epoch(self, epoch):
        self.cur_epoch = epoch
        self.deform_field.set_cur_milestone(epoch)
        self.txt_generator.set_cur_milestone(epoch)
        if self.pred_background:
            self.bkg_generator.set_cur_milestone(epoch)

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        unloaded_params = []
        state = self.state_dict()
        for name, param in safe_model_state_dict(state_dict).items():
            if name in state:
                try:
                    state[name].copy_(param.data if isinstance(param, nn.Parameter) else param)
                except RuntimeError:
                    print_warning(f'Error load_state_dict param={name}: {list(param.shape)}, {list(state[name].shape)}')
            else:
                unloaded_params.append(name)
        if len(unloaded_params) > 0:
            print_warning(f'load_state_dict: {unloaded_params} not found')

    ########################
    # Visualization utils
    ########################
    def get_part_color_texture(self, k):

        device = self.mesh_src.device
        if self.predict_parts:
            weights = self.vertex_to_joint[k]
        else:
            weights = torch.softmax(self.vertex_to_joint*self.temperature, dim=1)

        weight_idxs = torch.argmax(weights,1).cpu().numpy()
        print (weight_idxs[0:200:10])

        colors = cm.get_cmap('tab20b')(np.linspace(0,1,weights.shape[1]))[::-1,:3]
        vert_colors = colors[weight_idxs]
        colors = torch.from_numpy(vert_colors).to(device).float()
        return TexturesVertex(verts_features=colors[None])


    def get_grey_textures(self):
        verts = self.mesh_src.verts_packed()
        colors = torch.ones(verts.shape, device=verts.device) * 0.95
        return TexturesVertex(verts_features=colors[None])

    def get_synthetic_textures(self, colored=False):
        verts = self.mesh_src.verts_packed()
        faces = self.mesh_src.faces_packed()
        uv_image  = np.asarray(Image.open('colorwheel.png').convert('RGB'))/255.
        uv_image = torch.from_numpy(uv_image).float().to(verts.device)
        return TexturesUV(uv_image[None], faces[None], self.uvs)

    def get_prototype(self):
        verts = self.mesh_src.get_mesh_verts_faces(0)[0]
        latent = torch.zeros(1, self.n_features, device=verts.device)
        meshes = self.mesh_src.offset_verts(self.deform_field(verts, latent).view(-1, 3))
        return meshes

    @use_seed()
    @torch.no_grad()
    def get_random_prototype_views(self, N=10):
        mesh = self.get_prototype()
        if mesh is None:
            return None

        mesh.textures = self.get_synthetic_textures(colored=True)
        azim = torch.randint(*self.azim_range, size=(N,))
        elev = torch.randint(*self.elev_range, size=(N,)) if np.diff(self.elev_range)[0] > 0 else self.elev_range[0]
        R, T = look_at_view_transform(dist=self.T_cam[-1], elev=elev, azim=azim, device=mesh.device)
        return self.renderer(mesh.extend(N), R, T).split([3, 1], dim=1)[0]

    @torch.no_grad()
    def save_prototype(self, path=None):
        mesh = self.get_prototype()
        if mesh is None:
            return None

        path = path_mkdir(path or Path('.'))
        d, elev = self.T_cam[-1], np.mean(self.elev_range)
        mesh.textures = self.get_synthetic_textures()
        save_mesh_as_obj(mesh, path / 'proto.obj')
        save_mesh_as_gif(mesh, path / 'proto_li.gif', dist=d, elev=elev, renderer=self.renderer, eye_light=True)
        mesh.textures = self.get_synthetic_textures(colored=True)
        save_mesh_as_gif(mesh, path / 'proto_uv.gif', dist=d, elev=elev, renderer=self.renderer)

    ########################
    # Evaluation utils
    ########################

    @torch.no_grad()
    def quantitative_eval(self, loader, device):
        self.eval()
        mesh_eval = MeshEvaluator()
        for inp, labels in loader:
            if isinstance(labels, torch.Tensor) and torch.all(labels == -1):
                break

            mesh_pred = self(torch_to(inp, device), return_meshes=True)[1]
            mesh_eval.update(mesh_pred, torch_to(labels, device))
        return OrderedDict(zip(mesh_eval.metrics.names, mesh_eval.metrics.values))

    @torch.no_grad()
    def qualitative_eval(self, loader, device, path=None, N=32):
        path = path or Path('.')
        self.eval()
        self.save_prototype(path / 'model')

        renderer = self.renderer
        n_zeros, NI = int(np.log10(N - 1)) + 1, max(N // loader.batch_size, 1)
        for j, (inp, _) in enumerate(loader):
            if j == NI:
                break
            imgs = inp['imgs'].to(device)
            features = self.encoder(imgs)
            if self.shared_encoder:
                meshes = self.predict_meshes(features)
            else:
                features_tx = self.encoder_tx(imgs)
                meshes = self.predict_meshes(features, features_tx)

            R, T = self.predict_poses(features)

            posed_meshes, R_new, T_new = self.update_with_poses(meshes, R, T)
            rec, alpha = renderer(posed_meshes, R_new, T_new, fov= self._fovs.squeeze()).split([3, 1], dim=1)  # (K*B)CHW
            if self.pred_background:
                bkgs = self.predict_background(features)
                rec = rec * alpha + (1 - alpha) * bkgs

            B, NV = len(imgs), 50
            d, e = self.T_cam[-1], np.mean(self.elev_range)
            for k in range(B):
                i = str(j*B+k).zfill(n_zeros)
                convert_to_img(imgs[k]).save(path / f'{i}_inpraw.png')
                convert_to_img(rec[k]).save(path / f'{i}_inprec_full.png')
                if self.pred_background:
                    convert_to_img(bkgs[k]).save(path / f'{i}_inprec_wbkg.png')

                mcenter = normalize(meshes[k], mode=None, center=True, use_center_mass=True)
                save_mesh_as_gif(mcenter, path / f'{i}_meshabs.gif', n_views=NV, dist=d, elev=e, renderer=renderer)
                save_mesh_as_obj(mcenter, path / f'{i}_mesh.obj')
                mcenter.textures = self.get_synthetic_textures(colored=True)
                save_mesh_as_obj(mcenter, path / f'{i}_meshuv.obj')
                save_mesh_as_gif(mcenter, path / f'{i}_meshuv_raw.gif', dist=d, elev=e, renderer=renderer)
