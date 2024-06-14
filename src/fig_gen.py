# This script generate pdf figure of various visualistaion 
# of the 3D reconstruction like pose, parts and other view
# usage is python src/fig_gen.py --i input_folder --m model.pkl
# and it will create a input_folder_fig and dump the outputs to that fodler

import argparse
import warnings

import numpy as np
from torch.utils.data import DataLoader
import torch
from dataset import get_dataset
from model import load_model_from_path
from utils import path_mkdir
from utils.logger import print_log
from utils.mesh import  normalize
from utils.pytorch import get_torch_device
import matplotlib.pyplot as plt

from torchvision.transforms import Resize

from pytorch3d.renderer import (
    look_at_view_transform,
)


BATCH_SIZE = 1
N_WORKERS = 4
PRINT_ITER = 10
SAVE_GIF = True
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D reconstruction from single-view images in a folder')
    parser.add_argument('-m', '--model', nargs='?', type=str, required=True, help='Model name to use')
    parser.add_argument('-i', '--input', nargs='?', type=str, required=True, help='Input folder')
    parser.add_argument('-a', '--no_art', action='store_true', help='Remove articulation')

    args = parser.parse_args()
    assert args.model is not None and args.input is not None

    device = get_torch_device()
    m = load_model_from_path(args.model).to(device)
    if args.no_art:
        m.articulation = False
    m.eval()

    print_log(f"Model {args.model} loaded: input img_size is set to {m.init_kwargs['img_size']}")

    data = get_dataset(args.input)(split="train", img_size=m.init_kwargs['img_size'])
    data.padding_mode = 'constant'
    loader = DataLoader(data, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=False)
    print_log(f"Found {len(data)} images in the folder")

    print_log("Starting reconstruction...")
    out = path_mkdir(args.input + '_fig')
    n_zeros = int(np.log10(len(data) - 1)) + 1

    for j, (inp, label) in enumerate(loader):
        with torch.no_grad():
            imgs = inp['imgs'].to(device)

            features = m.encoder(imgs)

            if m.shared_encoder:
                meshes = m.predict_meshes(features)
            else:
                features_tx = m.encoder_tx(imgs)
                meshes = m.predict_meshes(features, features_tx)

            if m.articulation:
                meshes = m.articulate_meshes(meshes, features)
            R, T = m.predict_poses(features)

        posed_meshes, R_new, T_new = m.update_with_poses(meshes, R, T)
        rec, alpha = m.renderer(posed_meshes, R_new, T_new, viz_purpose='qual').split([3, 1], dim=1)

        if rec.shape[-1] != imgs.shape[-1]:
            imgs = Resize(rec.shape[-1])(imgs)

        meshes = posed_meshes
        B, d, e = len(imgs), m.T_cam[-1], np.mean(m.elev_range)
        for k in range(B):
            nb = j*B + k
            if nb % PRINT_ITER == 0:
                print_log(f"Reconstructed {nb} images...")
            i = str(nb).zfill(n_zeros)
            print (i)
            mcenter = normalize(posed_meshes[k], mode=None, center=True, use_center_mass=True)

            fig, axs = plt.subplots(1,5,figsize=(9,2),gridspec_kw = {'wspace':0.02, 'hspace':0.02})
            for ax in axs:
                ax.axis("off")

            axs[0].imshow(imgs[k].permute(1,2,0).detach().cpu().numpy())

            T_o = T_new.clone()
            T_o[:,2] += T[:,2]*2
            T_o[:,0] += 0.25
            axs[2].imshow(rec[k].permute(1,2,0).detach().cpu().numpy())

            R_view = look_at_view_transform(dist=T_o[:,2], elev=-5, azim=0, device=device)[0]
            rec_o2, _ = m.renderer(mcenter, R_view @ R_new, T_o , viz_purpose=True).split([3, 1], dim=1)
            axs[3].imshow(rec_o2[k].permute(1,2,0).detach().cpu().numpy())

            posed_meshes.textures = m.get_synthetic_textures(colored=True)
            rec_uv, alpha = m.renderer(posed_meshes, R_new, T_new, viz_purpose=True).split([3, 1], dim=1)
            uv_paste_image = (imgs[k] * (1-alpha[k])) + (alpha[k] * rec_uv[k] * 0.75) + (imgs[k] * alpha[k] * 0.25)
            axs[1].imshow(uv_paste_image.permute(1,2,0).detach().cpu().numpy())

            posed_meshes.textures = m.get_part_color_texture(k)
            rec_part, alpha = m.renderer(posed_meshes, R_new, T_new, viz_purpose='qual').split([3, 1], dim=1)
            axs[4].imshow(rec_part[k].permute(1,2,0).detach().cpu().numpy())

            plt.tight_layout()
            plt.savefig(out / f'{i}.png', bbox_inches="tight")
            plt.close()

    print_log("Done!")
    