from math import exp
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from torchvision import transforms

def get_loss(name):
    return {
        'bce': nn.BCEWithLogitsLoss,
        'mse': nn.MSELoss,
        'l2': nn.MSELoss,
        'l1': nn.L1Loss,
        'huber': nn.SmoothL1Loss,
        'cosine': nn.CosineSimilarity,
        'perceptual': PerceptualLoss,
        'ssim': SSIMLoss,
    }[name]


def norm_depth(AA):

    batch_size, height, width = AA.shape

    AA = AA.view(AA.size(0), -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= (AA.max(1, keepdim=True)[0] + 1e-7)
    AA = AA.view(batch_size, height, width)
    return AA


def compute_scale_and_shift(prediction, target, mask):

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def depth_to_normal(depth, eps=1e-7):

    blurrer = transforms.GaussianBlur(kernel_size=(3, 3), sigma=10.)

    #depth BxHXW
    #return normal Bx2xHxW
    depth = F.pad(depth.unsqueeze(0),(0,1,0,1), mode='reflect').squeeze(0)
    grad_x = blurrer(depth[:, :-1, 1:] - depth[:,:-1,:-1])
    grad_y = blurrer(depth[:, 1:, :-1] - depth[:,:-1,:-1])
    magnitude = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2)) + eps

    norm_x = grad_x / magnitude
    norm_y = grad_y / magnitude
    return torch.cat((norm_x[:,None,:],norm_y[:,None,]),dim=1)


def depthloss(rec_depths, alphas, depths, masks):

    depths = depths.squeeze()
    masks = masks.squeeze()

    scale, shift = compute_scale_and_shift(rec_depths, depths, masks)
    rec_depths_s = rec_depths* scale[:,None,None] + shift[:,None,None]

    return torch.abs(rec_depths_s - depths)

def dice_loss (pred, target):
    smooth = 1.
    p = 2.
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    num = torch.sum(torch.mul(pred, target), dim=1) + smooth
    den = torch.sum(pred.pow(p) + target.pow(p), dim=1) + smooth
    loss = 1 - num / den
    return loss


class SoftHistogram(nn.Module):
    #https://discuss.pytorch.org/t/differentiable-torch-histc/25865/35
    def __init__(self, bins, min, max, sigma, device):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)
        self.centers = self.centers.to(device)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta/2)) - torch.sigmoid(self.sigma * (x - self.delta/2))
        x = x.sum(dim=1)
        x = x / torch.sum(x)
        return x

def mesh_edge_loss_max(meshes, target_length: float = 0.0):

    #https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_edge_loss.html
    #modification of original loss to only penalize top %10 percent

    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    edge_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(E_n), )
    num_edges_per_mesh = meshes.num_edges_per_mesh()  # N

    # Determine the weight for each edge based on the number of edges in the
    # mesh it corresponds to.
    # TODO (nikhilar) Find a faster way of computing the weights for each edge
    # as this is currently a bottleneck for meshes with a large number of faces.

    # Mehmet: I dont need weights as all meshes contains same  number of faces
    # weights = num_edges_per_mesh.gather(0, edge_to_mesh_idx)
    # weights = 1.0 / weights.float()

    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)
    loss = ((v0 - v1).norm(dim=1, p=2) - target_length) ** 2.0
    avg_loss = loss.mean()
    loss = loss[loss>avg_loss*2]
    return loss.sum() / N


def part_spatial_loss(meshes, vertex_to_joint, part_centers):

    #we want to make part assignment to be spatially consistent
    #try to minimize variance of part centers?
    B, V, n_joints = vertex_to_joint.shape

    verts = meshes.verts_packed().view(B,-1,3)# BxVx3
    #part_centers : BxJx3

    #TODO: find variances of each part assignments
    n_v= verts.shape[1]

    return 0.


def part_center_loss(part_centers):
    #regularization for vertex to parts to be far away possible
    diff = part_centers[:,None,:,:] - part_centers[:,:,None,:]
    diff = torch.pow(diff,2).sum(-1)
    loss = torch.exp(-1*diff)
    mask = (1 - torch.eye(diff.shape[1])).unsqueeze(0).repeat(diff.shape[0],1,1).to(part_centers.device)
    loss = torch.mean(loss * mask)
    return loss


def part_share_loss(vertex_to_joint):
    #regularization for vertex to part consistency across instances
    if vertex_to_joint.dim() == 2:
        vertex_to_joint = vertex_to_joint.unsqueeze(0)
    B, V, n_joints = vertex_to_joint.shape
    mean_v_to_joint = vertex_to_joint.mean(0) #V x J
    diff = vertex_to_joint - mean_v_to_joint[None,:,:]
    return torch.mean(diff ** 2)


def part_peak_loss(vertex_to_joint):
    if vertex_to_joint.dim() == 2:
        vertex_to_joint = vertex_to_joint.unsqueeze(0)
    B, V, n_joints = vertex_to_joint.shape
    entropy = -1 * vertex_to_joint * torch.log(vertex_to_joint)
    return entropy.mean()


def part_uniform_loss(vertex_to_joint):
    if vertex_to_joint.dim() == 2:
        vertex_to_joint = vertex_to_joint.unsqueeze(0)

    B, V, n_joints = vertex_to_joint.shape
    diff = vertex_to_joint.sum(1) - (V/n_joints)
    #diff = (vertex_to_joint.view(-1,n_joints).mean(0) - 1/n_joints)
    return diff.abs().mean() / (V/n_joints)


def pose_reg_loss(azim):

    #hist = torch.histc(azim, bins=12, min=0, max=360)
    #print (hist)
    softhist = SoftHistogram(bins=12, min=0, max=360, sigma=15, device=azim.device)
    #softhist = SoftHistogram(bins=6, min=0, max=360, sigma=30., device=azim.device)#26 aug
    hist = softhist(azim)

    coverage_ratio = torch.sum(hist > 1/azim.shape[0]) / hist.shape[0]
    if coverage_ratio > 0.80:
        return torch.tensor([0.]).to(azim.device)
    with torch.no_grad():
        smooth_hist = torch.softmax(hist, dim=0)
    #soft_coverage_ratio = torch.sum(soft_hist > 1/hist.shape[0]) / hist.shape[0]
    loss = torch.mean(torch.abs(hist - smooth_hist))
    return  loss


def volume_loss(meshes, target=None):

    """
    Computes mesh volume regularization loss averaged across all meshes
    in a batch.
    https://stackoverflow.com/questions/1406029/how-to-calculate-the-volume-of-a-3d-mesh-object-the-surface-of-which-is-made-up

    Args:
        meshes: Meshes object with a batch of meshes.
        target: Target value for the mesh volume.

    Returns:
        loss: Average loss across the batch. Returns 0 if meshes contains
        no meshes or all empty meshes.
    """
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )
    N = len(meshes)

    faces_packed = meshes.faces_packed()  # (sum(E_n), 3)
    n_f = faces_packed.shape[0] // N
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    edges_packed = meshes.edges_packed()

    faces_verts = verts_packed[faces_packed]
    c = torch.cross(faces_verts[:,1,:],faces_verts[:,2,:],1)
    s_v = torch.sum(faces_verts[:,0,:]* c, 1) /6.

    volumes = torch.sum(torch.abs(s_v).view(N,-1),1)

    if target:
        return torch.sum(torch.pow(volumes - target,2))/N

    return volumes


class PerceptualLoss(nn.Module):
    def __init__(self, normalize_input=True, normalize_features=True, feature_levels=None, sum_channels=False,
                 requires_grad=False):
        super().__init__()
        self.normalize_input = normalize_input
        self.normalize_features = normalize_features
        self.sum_channels = sum_channels
        self.feature_levels = feature_levels if feature_levels is not None else [3]
        assert isinstance(self.feature_levels, (list, tuple))
        self.max_level = max(self.feature_levels)
        self.register_buffer('mean_rgb', torch.Tensor([0.485, 0.456, 0.406]))
        self.register_buffer('std_rgb', torch.Tensor([0.229, 0.224, 0.225]))

        layers = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = layers[:4]     # relu1_2
        self.slice2 = layers[4:9]    # relu2_2
        self.slice3 = layers[9:16]   # relu3_3
        self.slice4 = layers[16:23]  # relu4_3
        self.slice5 = layers[23:30]  # relu5_3
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, im1, im2, mask=None):

        inp = torch.cat([im1, im2], 0)
        if self.normalize_input:
            inp = (inp - self.mean_rgb.view(1, 3, 1, 1)) / self.std_rgb.view(1, 3, 1, 1)

        # if mask is not None:
            # mask = mask.repeat(2,3,1,1)

        feats = []
        for k in range(1, 6):
            if k > self.max_level:
                break
            inp = getattr(self, f'slice{k}')(inp)
            feats.append(torch.chunk(inp, 2, dim=0))

        losses = []
        for k, (f1, f2) in enumerate(feats, start=1):
            if k in self.feature_levels:
                if self.normalize_features:
                    f1, f2 = map(lambda t: t / (t.norm(dim=1, keepdim=True) + 1e-10), [f1, f2])
                loss = (f1 - f2) ** 2
                if mask is not None:
                    mask = torchvision.transforms.Resize(size=loss.shape[-1])(mask)
                    loss = loss * mask
                if self.sum_channels:
                    losses.append(loss.sum(1).flatten(2).mean(2))
                else:
                    losses.append(loss.flatten(1).mean(1))
        return sum(losses)

    def adhoc_self_correlation(self, im1, im2):
        inp = torch.cat([im1, im2], 0)
        b = inp.shape[0]
        inp = (inp - self.mean_rgb.view(1, 3, 1, 1)) / self.std_rgb.view(1, 3, 1, 1)
        feats = []
        for k in range(1, 6):
            if k > self.max_level:
                break
            inp = getattr(self, f'slice{k}')(inp)
            feats.append(torch.chunk(inp, 2, dim=0))

        for k, (f1, f2) in enumerate(feats, start=1):
            if f1.shape[1] == 128:
                f1 = f1 / (f1.norm(dim=1, keepdim=True) + 1e-10)
                f2 = f2 / (f1.norm(dim=1, keepdim=True) + 1e-10)
                B,C,H,W = f1.shape
                corr_1 = torch.bmm(f1.view(B,C,H*W).permute(0,2,1), f1.view(B,C,H*W))
                scorr_1 = torch.softmax(corr_1, dim=2)
                scorr_max_1 = torch.max(scorr_1, dim=2).values.view(B,H,W)

                corr_2 = torch.bmm(f2.view(B,C,H*W).permute(0,2,1), f1.view(B,C,H*W))
                scorr_2 = torch.softmax(corr_2, dim=2)
                scorr_max_2 = torch.max(scorr_2, dim=2).values.view(B,H,W)
                break
        loss = (scorr_max_1 - scorr_max_2) ** 2
        return torch.mean(loss)
                


    def adhoc_self_similarity(self, inp):
        with torch.no_grad():
            b = inp.shape[0]
            inp = (inp - self.mean_rgb.view(1, 3, 1, 1)) / self.std_rgb.view(1, 3, 1, 1)
            feats = []
            for k in range(1, 6):
                if k > self.max_level:
                    break
                inp = getattr(self, f'slice{k}')(inp)
                feats.append(inp)

            distances = []
            for k, f1 in enumerate(feats, start=1):
                if k in self.feature_levels:
                    f1 = f1 / (f1.norm(dim=1, keepdim=True) + 1e-10)
                    diff = (f1[:,None] - f1[None,:]) ** 2
                    diff = diff.view(b, b, -1).mean(2)
                    distances.append(diff)

        return distances
######################################################################
# SSIM original repo implem: https://github.com/Po-Hsun-Su/pytorch-ssim
######################################################################


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, channel=3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self.create_window(window_size, channel)

    def ssim(self, img1, img2):
        window_size, channel = self.window_size, self.channel
        window = self.window.to(img1.device)
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map

    @staticmethod
    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, inp, target):
        return self.ssim(inp, target).flatten(1).mean(1)
