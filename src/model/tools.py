from collections import OrderedDict
from toolz import keymap

import numpy as np
from pytorch3d.transforms import random_rotations
import torch
from torch import nn
from torch.nn import functional as F
from utils.logger import print_log
import matplotlib.pyplot as plt


N_UNITS = 128
N_LAYERS = 3


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def safe_model_state_dict(state_dict):
    """Convert a state dict saved from a DataParallel module to normal module state_dict."""
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    return keymap(lambda s: s[7:], state_dict, factory=OrderedDict)  # remove 'module.'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def conv3x3(in_planes, out_planes, stride=1, padding=1, groups=1, dilation=1, zero_init=False):
    """3x3 convolution with padding"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)
    if zero_init:
        conv.weight.data.zero_()
    return conv


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def create_mlp(in_ch, out_ch, n_units=N_UNITS, n_layers=N_LAYERS, kaiming_init=True, zero_last_init=False):

    #nl = nn.ReLU(True)
    nl = nn.LeakyReLU(0.2, True)
    #nl = Sine()
    if n_layers > 0:
        seq = [nn.Linear(in_ch, n_units), nn.GroupNorm(32, n_units), nl]

        for _ in range(n_layers - 1):
            seq += [nn.Linear(n_units, n_units), nn.GroupNorm(32, n_units), nl]
        seq += [nn.Linear(n_units, out_ch)]
    else:
        seq = [nn.Linear(in_ch, out_ch)]
    mlp = nn.Sequential(*seq)

    if kaiming_init:
        mlp.apply(kaiming_weights_init)
    if zero_last_init:
        with torch.no_grad():
            if zero_last_init is True:
                mlp[-1].weight.zero_()
            else:
                mlp[-1].weight.normal_(mean=0, std=zero_last_init)
            mlp[-1].bias.zero_()
    return mlp


@torch.no_grad()
def kaiming_weights_init(m, nonlinearity='leaky_relu'):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def get_nb_out_channels(layer):
    return list(filter(lambda e: isinstance(e, nn.Conv2d), layer.modules()))[-1].out_channels


def get_output_size(in_channels, img_size, model, vit=True):

    x = torch.zeros(1, in_channels, *img_size)
    if vit:
        y = model.get_intermediate_layers(x)[0]
        return np.prod(y.shape)
    else:
        y= model(x)
        if y.shape[-1] != 1:
            return y.shape[1]
        return np.prod(y.shape)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


##########################################
# Generator utils
##########################################


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.Tensor([1, 2, 1])
        kernel = kernel[None, None, :] * kernel[None, :, None]
        kernel = kernel / kernel.norm(p=1)
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        kernel = self.kernel.unsqueeze(1).expand(-1, C, -1, -1)
        kernel = kernel.reshape(-1, 1, *kernel.shape[2:])
        x = x.view(-1, kernel.size(0), x.size(-2), x.size(-1))
        return F.conv2d(x, kernel, groups=kernel.size(0), padding=0, stride=1).view(B, C, H, W)


def create_upsample_layer(name):
    if name == 'nn':
        return nn.Upsample(scale_factor=2)
    elif name == 'bilinear':
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), Blur())
    else:
        raise NotImplementedError


##########################################
# Rendering / pose utils
##########################################


def init_rotations(init_type='uniform', N=None, n_elev=None, n_azim=None, elev_range=None, azim_range=None):
    if init_type == 'uniform':
        assert n_elev is not None and n_azim is not None
        assert N == n_elev * n_azim if N is not None else True
        eb, ee = elev_range if elev_range is not None else (-90, 90)
        ab, ae = azim_range if azim_range is not None else (-180, 180)
        er, ar = ee - eb, ae - ab
        elev = torch.Tensor([k*er/n_elev + eb - er/(2*n_elev) for k in range(1, n_elev + 1)])  # [-60, 0, 60]
        if ar == 360:
            azim = torch.Tensor([k*ar/n_azim + ab for k in range(n_azim)])  # e.g. [-180, -90, 0, 90]
        else:
            azim = torch.Tensor([k*ar/n_azim + ab - ar/(2*n_azim) for k in range(1, n_azim + 1)])  # [-60, 0, 60]
        elev, azim = map(lambda t: t.flatten(), torch.meshgrid(elev, azim))
        roll = torch.zeros(elev.shape)
        print_log(f'init_rotations: azim={azim.tolist()}, elev={elev.tolist()}, roll={roll.tolist()}')
        R_init = torch.stack([azim, elev, roll], dim=1)
    elif init_type.startswith('random'):
        R_init = random_rotations(N)
    else:
        raise NotImplementedError
    return R_init


def convert_3d_to_uv_coordinates_via_3dcube(X, eps=1e-7):

    abs_X = torch.abs(X)

    uc = torch.zeros((X.shape[0])).to(X.device)
    vc = torch.zeros((X.shape[0])).to(X.device)

    max_idxs = torch.argmax(abs_X,dim=1)

    #max is x
    uc[(max_idxs == 0) & (X[:,0] >= 0)] = -X[(max_idxs == 0) & (X[:,0] >= 0) ,2]
    uc[(max_idxs == 0) & (X[:,0] < 0)] = X[(max_idxs == 0) & (X[:,0] < 0) ,2]
    vc[max_idxs == 0] = X[max_idxs == 0,1]

    #max is y
    uc[max_idxs == 1] = X[max_idxs == 1,0]
    vc[(max_idxs == 1) & (X[:,1] >= 0)] = -X[(max_idxs == 1) & (X[:,1] >= 0) ,2]
    vc[(max_idxs == 1) & (X[:,1] < 0)] = X[(max_idxs == 1) & (X[:,1] > 0) ,2]

    #max is z
    uc[(max_idxs == 2) & (X[:,2] >= 0)] = X[(max_idxs == 2) & (X[:,2] >= 0) ,1]
    uc[(max_idxs == 2) & (X[:,2] < 0)] = -X[(max_idxs == 2) & (X[:,2] < 0) ,1]

    vc[max_idxs == 2] = X[max_idxs == 2,1]


    uu = 0.5 * (uc / torch.max(abs_X, dim=1)[0] + 1.0);
    vv = 0.5 * (vc / torch.max(abs_X, dim=1)[0] + 1.0);

    # ax = plt.subplot()
    # ax.scatter(uu.cpu().numpy(),vv.cpu().numpy())
    # plt.show()

    return torch.stack([uu, vv], dim=-1)
    # ax = plt.subplot()
    # ax.scatter(uu.numpy(),vv.numpy())
    # plt.show()


def detect_wrapped_uv_cords(UV, F):

    device = F.device
    homo_cords = torch.cat((UV[F], torch.zeros(F.shape[0],3,1).to(device)),dim=2)
    normals = torch.cross(homo_cords[:,1,:] - homo_cords[:,0,:], homo_cords[:,2,:] - homo_cords[:,0,:])
    return normals[:,2] < 0


def convert_3d_to_uv_coordinates_v2(V, F, eps=1e-5):

    """Resulting UV in [0, 1]"""

    r = torch.norm(V, dim=-1).clamp(min=eps)
    V = V / r[:,None]
    uu = 0.5 + torch.atan2(V[..., 0], V[..., 2])  / (2*np.pi)
    vv = 0.5 + torch.asin(V[..., 1]) / (np.pi)

    uv = torch.stack([uu, vv], dim=-1)
    wrapped_faces = detect_wrapped_uv_cords(uv, F)
    poor_uvs = torch.unique(F[wrapped_faces].view(-1,1), sorted=False)

    uv[poor_uvs,0] = uv[poor_uvs,0] + 1.

    colors = np.zeros((uv.shape[0],3))
    colors[poor_uvs.cpu().numpy()] = [0.5,1,1]

    ax = plt.subplot()
    ax.scatter(uv[:,0].cpu().numpy(),uv[:,1].cpu().numpy(), c=colors)
    plt.savefig('uv_plot.png')

    return uv#, poor_uvs
    return torch.stack([uu, vv], dim=-1)


def convert_3d_to_uv_coordinates(X, eps=1e-9):

    """Resulting UV in [0, 1]"""

    radius = torch.norm(X, dim=-1)
    theta = torch.acos((X[..., 1] / radius).clamp(min=-1 + eps, max=1 - eps))    # Inclination: Angle with +Y [0,pi]
    phi = torch.atan2(X[..., 0], X[..., 2])  # Azimuth: Angle with +Z [-pi,pi]
    vv = (theta / np.pi)
    uu = ((phi + np.pi) / (2*np.pi))



    # b= torch.where((vv>0.9) & (uu>0.95))
    # print (b)
    # colors = np.zeros((uu.shape[0],3))
    # print(X[b[0]])
    # colors[b[0]] = [0.5,0.5,0.5,]
    # ax = plt.subplot()
    # ax.scatter(uu.numpy(),vv.numpy())
    # ax.set_ylabel('vv')
    # ax.set_xlabel('uu')
    # plt.savefig('uvmap.png')
    # plt.show()
    return torch.stack([uu, vv], dim=-1)


def convert_uv_to_3d_coordinates(uv, radius=1, half_sphere=False):
    """input UV in [0, 1]"""
    phi = np.pi * (uv[..., 0] * 2 - 1)
    theta = np.pi * uv[..., 1]
    if half_sphere:
        theta = theta / 2
    z = torch.sin(theta) * torch.cos(phi)
    x = torch.sin(theta) * torch.sin(phi)
    y = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1) * radius


def convert_spherical_to_3d_coordinates(phi, theta, radius=1, as_degree=True):
    """input UV in [0, 1]"""
    if as_degree:
        phi, theta = np.pi * phi / 180, np.pi * theta / 180
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    points3d = torch.stack([x, y, z], dim=-1)
    return points3d * radius


def azim_to_rotation_matrix(azim, as_degree=True):
    """Angle with +X in XZ plane"""
    if isinstance(azim, (int, float)):
        azim = torch.Tensor([azim])
    azim_rad = azim * np.pi / 180 if as_degree else azim
    R = torch.eye(3, device=azim.device)[None].repeat(len(azim), 1, 1)
    cos, sin = torch.cos(azim_rad), torch.sin(azim_rad)
    zeros = torch.zeros(len(azim), device=azim.device)
    R[:, 0, :] = torch.stack([cos, zeros, sin], dim=-1)
    R[:, 2, :] = torch.stack([-sin, zeros, cos], dim=-1)
    return R.squeeze()


def elev_to_rotation_matrix(elev, as_degree=True):
    """Angle with +Z in YZ plane"""
    if isinstance(elev, (int, float)):
        elev = torch.Tensor([elev])
    elev_rad = elev * np.pi / 180 if as_degree else elev
    R = torch.eye(3, device=elev.device)[None].repeat(len(elev), 1, 1)
    cos, sin = torch.cos(-elev_rad), torch.sin(-elev_rad)
    R[:, 1, 1:] = torch.stack([cos, sin], dim=-1)
    R[:, 2, 1:] = torch.stack([-sin, cos], dim=-1)
    return R.squeeze()


def roll_to_rotation_matrix(roll, as_degree=True):
    """Angle with +X in XY plane"""
    if isinstance(roll, (int, float)):
        roll = torch.Tensor([roll])
    roll_rad = roll * np.pi / 180 if as_degree else roll
    R = torch.eye(3, device=roll.device)[None].repeat(len(roll), 1, 1)
    cos, sin = torch.cos(roll_rad), torch.sin(roll_rad)
    R[:, 0, :2] = torch.stack([cos, sin], dim=-1)
    R[:, 1, :2] = torch.stack([-sin, cos], dim=-1)
    return R.squeeze()


def cpu_angle_between(R1, R2, as_degree=True):

    eps= 1e-7
    #angle = ((torch.einsum('bii -> b', (R1.transpose(-2, -1) @ R2).view(-1, 3, 3)) - 1) / 2).acos()
    angle = ((torch.einsum('bii -> b', (R1.transpose(-2, -1) @ R2).view(-1, 3, 3)) - 1) / 2)
    angle = torch.acos(torch.clamp(angle, -1 + eps, 1 - eps))

    return (180 / np.pi) * angle if as_degree else angle
