import argparse
import warnings
import os

import torch
from model import load_model_from_path
from utils import path_mkdir
from utils.path import MODELS_PATH
from utils.logger import print_log
from utils.pytorch import get_torch_device
from utils.image import convert_to_img
from torchvision.transforms import (ToTensor, Compose)

from PIL import Image
from dataset.torch_transforms import SquarePad, Resize as ResizeCust

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D reconstruction from single-view images in a folder')
    parser.add_argument('-m', '--model', nargs='?', type=str, required=True, help='Model name to use')
    parser.add_argument('-s', '--src', nargs='?', type=str, required=True, help='src image')
    parser.add_argument('-t', '--trg', nargs='?', type=str, required=True, help='trg image')
    parser.add_argument('-o', '--output_folder', nargs='?', type=str, required=False, help='output_folder')
    parser.add_argument('-n', '--n_frames', nargs='?', type=int, required=False, default=50)


    args = parser.parse_args()
    assert args.model is not None
    path_mkdir(args.output_folder)
    path_mkdir(f"{args.output_folder}/tmp")
    device = get_torch_device()
    m = load_model_from_path(args.model).to(device)
    m.eval()

    transform = Compose([ResizeCust(m.init_kwargs['img_size'], fit_inside=True), SquarePad(), ToTensor()])
    print_log(f"Model {args.model} loaded: input img_size is set to {m.init_kwargs['img_size']}")

    print_log("Starting reconstruction...")
    
    src_name = args.src.split("/")[-1].split(".")[-2]
    trg_name = args.trg.split("/")[-1].split(".")[-2]

    img_src = transform(Image.open(args.src).convert('RGB'))
    img_trg = transform(Image.open(args.trg).convert('RGB'))

    with torch.no_grad():
        imgs = torch.cat([img_src.unsqueeze(0),img_trg.unsqueeze(0)],0)
        imgs = imgs.to(device)
        features = m.encoder(imgs)

        if m.shared_encoder:
            meshes = m.predict_meshes(features)
        else:
            features_tx = m.encoder_tx(imgs)
            meshes = m.predict_meshes(features, features_tx)
        R, T = m.predict_poses(features)
        final_meshes = m.articulate_meshes(meshes, features)

    posed_meshes, R_new, T_new = m.update_with_poses(final_meshes, R, T)
    rec, alpha = m.renderer(posed_meshes, R_new, T_new, viz_purpose=True).split([3, 1], dim=1)

    src_rec_img = convert_to_img(img_src).resize((256,256))
    trg_rec_img = convert_to_img(img_trg).resize((256,256))

    src_rec_img.save('{}/src.png'.format(args.output_folder))
    trg_rec_img.save('{}/trg.png'.format(args.output_folder))

    src_f = features[0].clone()
    trg_f = features[1].clone()
    tmp_output_folder = f'{args.output_folder}/tmp'
    for i in range(args.n_frames):
        ld = 0.0 + (1./args.n_frames)*i
        int_feature = src_f * (1-ld) + trg_f * ld
        meshes_art = m.articulate_meshes(meshes[0], int_feature.unsqueeze(0))
        posed_meshes, R_new, T_new = m.update_with_poses(meshes_art, R[0,None], T[0,None])
        rec, alpha = m.renderer(posed_meshes, R_new, T_new, viz_purpose=True).split([3, 1], dim=1)
        transferred_rec_img = convert_to_img(rec[0])
        save_name = 'art_{0:03d}.png'.format(i)
        transferred_rec_img.save(f'{args.output_folder}/tmp/{save_name}')

    output_name = '{}_{}_art.mp4'.format(src_name, trg_name)

    os.system("/usr/bin/ffmpeg  -i {}/tmp/art_%03d.png -c:v h264 -vf fps=25 -pix_fmt yuv420p {}/{}".format(args.output_folder, args.output_folder, output_name))
    os.system("rm -rf {}/tmp".format(args.output_folder))
