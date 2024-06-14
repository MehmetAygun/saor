# This script generate 3D reconstruction of objects and produce various visualisations 
# like pose, parts and other view and if --gif flag given 360 degree videos
# usage is python src/reconstruct.py --i input_folder --m model.pkl --gif
# and it will create a input_folder_rec and dump the outputs to that fodler

import argparse
import warnings
import numpy as np
from torch.utils.data import DataLoader
import torch
from dataset import get_dataset
from model import load_model_from_path
from model.renderer import save_mesh_as_gif, Renderer
from utils import path_mkdir
from utils.path import MODELS_PATH
from utils.logger import print_log
from utils.mesh import save_mesh_as_obj, normalize
from utils.pytorch import get_torch_device
from utils.image import convert_to_img

import torchvision.transforms as Trf

BATCH_SIZE = 1
N_WORKERS = 4
PRINT_ITER = 10

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D reconstruction from single-view images in a folder')
    parser.add_argument('-m', '--model', nargs='?', type=str, required=True, help='Model name to use')
    parser.add_argument('-i', '--input', nargs='?', type=str, required=True, help='Input folder')
    parser.add_argument('-a', '--no_art', action='store_true', help='Remove articulation')
    parser.add_argument('-f', '--fast', action='store_true', help='fast')
    parser.add_argument('-g', '--gif', action='store_true', help='fast')

    args = parser.parse_args()
    assert args.model is not None and args.input is not None
    SAVE_GIF = args.gif

    n_view = 25 if args.fast else 50
    device = get_torch_device()
    m = load_model_from_path(args.model).to(device)
    if args.no_art:
        m.articulation = False
    m.eval()


    print_log(f"Model {args.model} loaded: input img_size is set to {m.init_kwargs['img_size']}")

    data = get_dataset(args.input)(split="train", img_size=m.init_kwargs['img_size'])

    loader = DataLoader(data, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=False)
    print_log(f"Found {len(data)} images in the folder")

    print_log("Starting reconstruction...")
    out = path_mkdir(args.input + '_rec')
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
        rec, alpha = m.renderer(posed_meshes, R_new, T_new, fov=m._fovs.squeeze(), viz_purpose='True').split([3, 1], dim=1)

        if m.pred_background:
            bkgs = m.predict_background(features)
            rec = rec * alpha + (1 - alpha) * bkgs

        faces = meshes.faces_padded()
        verts = meshes.verts_padded()
        B = len(imgs)

        meshes = posed_meshes
        B, d, e = len(imgs), m.T_cam[-1], np.mean(m.elev_range)
        for k in range(B):
            nb = j*B + k
            if nb % PRINT_ITER == 0:
                print_log(f"Reconstructed {nb} images...")
            i = str(nb).zfill(n_zeros)
            print (i, m._scales, m._fovs)
            mcenter = normalize(meshes[k], mode=None, center=True, use_center_mass=True)

            save_mesh_as_obj(mcenter, out / f'{i}_mesh.obj')
            convert_to_img(imgs[k]).save(out / f'{i}_inpraw.png')
            convert_to_img(rec[k]).save(out / f'{i}_rec.png')

            posed_meshes.textures = m.get_synthetic_textures(colored=True)
            rec_uv, alpha = m.renderer(posed_meshes, R_new, T_new, fov=m._fovs.squeeze(), viz_purpose='True').split([3, 1], dim=1)

            imgs = Trf.Resize(256)(imgs)
            uv_paste_image = (imgs[k] * (1-alpha[k])) + (alpha[k] * rec_uv[k] * 0.75) + (imgs[k] * alpha[k] * 0.25)
            convert_to_img(uv_paste_image).save(out / f'{i}_uvpaste.png')

            if m.articulation:
                posed_meshes.textures = m.get_part_color_texture(k)
                rec_part, alpha = m.renderer(posed_meshes, R_new, T_new, fov=m._fovs.squeeze(), viz_purpose='True').split([3, 1], dim=1)

                convert_to_img(rec_part[k]).save(out / f'{i}_part.png')

            if SAVE_GIF:
                save_mesh_as_gif(mcenter, out / f'{i}_mesh.gif', n_views=n_view, dist=d+1, elev=e, eye_light=True, fov=m._fovs.squeeze(), renderer=m.renderer)
                if m.articulation:
                    mcenter.textures = m.get_part_color_texture(k)
                    save_mesh_as_gif(mcenter, out / f'{i}_parts_raw.gif', n_views=n_view, dist=d+1, elev=e, fov=m._fovs.squeeze(), renderer=m.renderer)
                    mcenter.textures = m.get_part_color_texture(k)
                    save_mesh_as_obj(mcenter, out / f'{i}_part.obj')

    print_log("Done!")
