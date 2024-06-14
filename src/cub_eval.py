import argparse
import warnings
from itertools import combinations
from random import sample

import numpy as np
from torch.utils.data import DataLoader
import torch
from dataset import get_dataset
from model import load_model_from_path
from model.renderer import save_mesh_as_gif
from utils import use_seed, path_exists, path_mkdir, load_yaml
from utils.path import MODELS_PATH
from utils.logger import print_log
from utils.mesh import save_mesh_as_obj, normalize
from utils.pytorch import get_torch_device
from utils.image import convert_to_img
from utils.path import CONFIGS_PATH, RUNS_PATH
from utils.metrics import calc_iou, calc_pck
from dataset import create_train_val_test_loader
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pytorch3d.ops.knn import knn_gather, knn_points
from model.renderer import Renderer
from pytorch3d.renderer import FoVPerspectiveCameras
from model.renderer import Renderer
from dataset.cub_200 import CUB200Dataset
import random
random.seed(69)
torch.random.manual_seed(69)

BATCH_SIZE = 64
N_WORKERS = 8
PRINT_ITER = 10
SAVE_GIF = True
N_PAIRS_PCK = 10000
EVAL_IMG_SIZE = (128,128)

import matplotlib
font = {'family' : "Times New Roman",
        'size'   : 16}
matplotlib.rc('font', **font)

warnings.filterwarnings("ignore")

def transfer_keypoints(model, kp_src, R_src, T_src, mesh_src, R_tgt, T_tgt, mesh_tgt):

    camera, size = model.renderer.cameras, EVAL_IMG_SIZE
    R_src, T_src, R_tgt, T_tgt = [t[None] if t.size(0) != 1 else t for t in [R_src, T_src, R_tgt, T_tgt]]

    # Consider visible vertices only
    verts_viz_src = model.renderer.compute_vertex_visibility(mesh_src, R_src, T_src).squeeze()
    verts_3d_src = mesh_src.verts_packed()[verts_viz_src]

    # Project vertex 3D positions from source mesh to image space
    verts_2d_src = camera.transform_points_screen(verts_3d_src[None], image_size=size, R=R_src, T=T_src)[:, :, :2]
    # Find index correspondence between keypoints and vertices
    kp2proj_idx = knn_points(kp_src[:, :2][None].float(), verts_2d_src).idx.long().squeeze()

    # Select vertices in target mesh associated to keypoints and project them to image space
    verts_3d_tgt = mesh_tgt.verts_packed()[verts_viz_src][kp2proj_idx]
    #verts_3d_tgt = mesh_tgt.verts_packed()[kp2proj_idx]

    kp_out = camera.transform_points_screen(verts_3d_tgt[None], image_size=size, R=R_tgt, T=T_tgt)[0, :, :2]
    return kp_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D reconstruction from single-view images in a folder')
    parser.add_argument('-m', '--model', nargs='?', type=str, required=True, help='Model name to use')
    parser.add_argument('-t', '--threshold', type=float, default=0.1)
    parser.add_argument('-is', '--img_size', type=int, default=None)
    parser.add_argument('-v', '--vis', action='store_true')
    args = parser.parse_args()

    device = get_torch_device()
    m = load_model_from_path(args.model).to(device)
    n_faces = m.mesh_src.faces_packed().shape[0]

    m.eval()
    print_log(f"Model {args.model} loaded: input img_size is set to {m.init_kwargs['img_size']}")

    ratio_eval = EVAL_IMG_SIZE[0] / m.init_kwargs['img_size'][0]

    test = CUB200Dataset(split="test", img_size=m.init_kwargs['img_size'])
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True, pin_memory=True)

    print_log(f"Found {len(test_loader)} batch of images")

    print_log("Starting Evaluation...")
    #out = path_mkdir(args.input + '_rec')
    n_zeros = int(np.log10(len(test_loader) - 1)) + 1
    #m.articulation = True

    n_pair_batch = N_PAIRS_PCK // len(test_loader)
    ious = torch.tensor([])
    pcks_new = []


    for j, (inp, _) in enumerate(test_loader):
        print ('{}/{}'.format(j, len(test_loader)))
        imgs = inp['imgs'].to(device)
        masks = inp['masks'].to(device)
        kps = inp['kps'].to(device)
        B, d, e = len(imgs), m.T_cam[-1], np.mean(m.elev_range)

        pair_list = list(combinations(range(0,imgs.shape[0]),2))
        if n_pair_batch >= len(pair_list):
            continue
        pair_idxs = sample(pair_list, n_pair_batch)

        with torch.no_grad():
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
            recs, alphas = m.renderer(posed_meshes, R_new, T_new, fov= m._fovs.squeeze()).split([3, 1], dim=1)  # (K*B)CHW
            if args.vis:
                recs_uvs = []
                for i_t in range(len(features)):
                    mesh = posed_meshes[i_t]
                    mesh.textures =  m.get_synthetic_textures(colored=True)
                    recs_uv, _ = m.renderer(mesh, R_new[i_t].unsqueeze(0), T_new[i_t].unsqueeze(0), fov = m._fovs.squeeze()[i_t]).split([3, 1], dim=1)
                    recs_uvs.append(recs_uv)

        ious = torch.cat((ious, calc_iou(alphas.cpu(), masks.cpu())))

        for src_idx, trg_idx in pair_idxs:

            kp_out = transfer_keypoints(m, kps[src_idx]*ratio_eval, R_new[src_idx], T_new[src_idx], posed_meshes[src_idx], R_new[trg_idx], T_new[trg_idx], posed_meshes[trg_idx])

            common_kps = (kps[src_idx][:,2] == 1) & (kps[trg_idx][:,2] == 1)
            if kp_out is not None:

                # pair_pck = calc_pck(pred.to(device), gt.to(device), img_size,  args.threshold)
                pair_pck_new = calc_pck(kp_out[common_kps].to(device), kps[trg_idx,common_kps,:2].to(device)*ratio_eval, EVAL_IMG_SIZE[0],  args.threshold)
                pcks_new.append(pair_pck_new)
                #print (pair_pck)
                #print (pair_pck_new)

                if args.vis and pair_pck_new < 0.20: #np.random.rand() > 1.9:
                    input_image = convert_to_img(imgs[src_idx])
                    target_image = convert_to_img(imgs[trg_idx])

                    input_uv_image = convert_to_img(recs_uvs[src_idx])
                    target_uv_image = convert_to_img(recs_uvs[trg_idx])

                    src_uv_paste_image = (imgs[src_idx] * (1-alphas[src_idx])) + (alphas[src_idx] * recs_uvs[src_idx] * 0.75) + (imgs[src_idx] * alphas[src_idx] * 0.25)
                    src_uv_paste_image =  convert_to_img(src_uv_paste_image)
                    trg_uv_paste_image = (imgs[trg_idx] * (1-alphas[trg_idx])) + (alphas[trg_idx] * recs_uvs[trg_idx] * 0.75) + (imgs[trg_idx] * alphas[trg_idx] * 0.25)
                    trg_uv_paste_image =  convert_to_img(trg_uv_paste_image)

                    predicted_mask = alphas.cpu()#.numpy()

                    input_image = np.asarray(convert_to_img(imgs[src_idx]))
                    target_image = convert_to_img(imgs[trg_idx])

                    alpha_src = predicted_mask[src_idx][0][:,:,None].cpu().numpy()
                    alpha_trg = predicted_mask[trg_idx][0][:,:,None].cpu().numpy()

                    fig, axs = plt.subplots(1,4, figsize=(10,3))

                    axs[0].set_title('Source')
                    axs[1].set_title('Source - UV')
                    axs[2].set_title('Target - UV')
                    axs[3].set_title('KP-Transfer')

                    axs[0].axis('off')
                    axs[1].axis('off')
                    axs[2].axis('off')
                    axs[3].axis('off')

                    src_kps = kps[src_idx,common_kps,:2].cpu().numpy()
                    trg_kps = kps[trg_idx,common_kps,:2].cpu().numpy()
                    kp_out = kp_out[common_kps].cpu().numpy()

                    axs[0].imshow(input_image)
                    axs[1].imshow(src_uv_paste_image)
                    axs[2].imshow(trg_uv_paste_image)
                    axs[3].imshow(target_image)

                    colors = cm.get_cmap('gist_rainbow')(np.linspace(0,1, src_kps.shape[0]))[:,:3]

                    axs[0].scatter(src_kps[:,0], src_kps[:,1],c=colors)
                    axs[3].scatter(kp_out[:,0], kp_out[:,1],c=colors, marker = 'x')
                    axs[3].scatter(trg_kps[:,0], trg_kps[:,1],c=colors)
                    plt.tight_layout()
                    plt.show()
                    plt.close()

    print ('Mean IoU:', torch.mean(ious))
    print ('Mean PCK New:', torch.mean(torch.tensor(pcks_new)))
