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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"#######################
import torch
import json
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, network_gui
import shutil, pathlib
import sys
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import wandb
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from utils.camera_utils import set_rays_od

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')

def calc_scene_bbox(scene):
    num_cam = len(scene.getTrainCameras())
    pos = []
    for i in scene.getTrainCameras():
        pos.append(i.camera_center)
    pos = torch.stack(pos)
    center = torch.mean(pos,dim=0)
    length = pos.max() - pos.min()
    length = length.repeat(3)
    return center, length

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

def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, model_params=dataset)
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)

    if  dataset.contractor:
        center, length = calc_scene_bbox(scene)
        dataset.scene_center = center.detach().cpu().numpy().tolist()
        dataset.scene_length = length.detach().cpu().numpy().tolist()
        gaussians.setup_contractor(center=dataset.scene_center,length=dataset.scene_length, contractor = dataset.contractor)
        
    else:
        center = dataset.scene_center
        length = dataset.scene_length

    print('center:',center,length)

    print('--------------------------')
    print('scene_center:',dataset.scene_center,'scene_length',dataset.scene_length)
    print(opt.graph_downsampling_iters)
    print('--------------------------')
    tb_writer = prepare_output_and_logger(dataset)
   

    gaussians.training_setup(opt)
    if checkpoint:
        print("loading ckpt", checkpoint)

        print("resolution: ", dataset.resolution, "MV:", pipe.mv)
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        first_iter = 0

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_loss_for_log_net = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    set_rays_od(scene.getTrainCameras())

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        total_loss = 0
        camera_t = []
        imgs =[]
        cams=[]
        for i in range(pipe.mv):
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            camera_t.append(viewpoint_cam.camera_center/torch.norm(viewpoint_cam.camera_center))
            cams.append(viewpoint_cam)
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe,background)
            retain_grad = (iteration < opt.update_until and iteration >= 0)
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
        
            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
           
            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            ssim_loss = (1.0 - ssim(image, gt_image))
            scaling_reg = scaling.prod(dim=1).mean()

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg
            
            total_loss += loss 
            imgs.append([image, gt_image])

            if iteration > opt.update_from and iteration < opt.update_until:
                if i == pipe.mv - 1:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    muiti_con_loss = 0
                    
                    for i in range(pipe.mv):
                        for j in range(i+1, pipe.mv):
                            gen_img1 = imgs[i][0]
                            real_img1 = imgs[i][1]
                            gen_img2 = imgs[j][0]
                            real_img2 = imgs[j][1]
                            gen_img1, gen_img2, real_img1, real_img2 = align_images(gen_img1, gen_img2, real_img1, real_img2)
                            cam1 = cams[i]
                            cam2 = cams[j]
                            if ssim(real_img1, real_img2) > 0.6:
                                loss_tmp = ssim(real_img1, real_img2) * torch.abs(l1_loss(real_img1 - real_img2, gen_img1 - gen_img2))
                            else:
                                loss_tmp = 0

                            if iteration % opt.update_interval ==0:
                                with torch.no_grad():
                                    #if i == 0 and j == 1:
                                        #print("MV_prune!")
                                    #cam1_K = torch.tensor(cam1.K, dtype=torch.float32, device=device)
                                    cam1_R = torch.tensor(cam1.R, dtype=torch.float32, device=device)
                                    cam1_T = torch.tensor(cam1.T, dtype=torch.float32, device=device)

                                    #cam2_K = torch.tensor(cam2.K, dtype=torch.float32, device=device)
                                    cam2_R = torch.tensor(cam2.R, dtype=torch.float32, device=device)
                                    cam2_T = torch.tensor(cam2.T, dtype=torch.float32, device=device)
                                    _, _, _, intersection_mask = gaussians.compute_fast_loss_with_key_points(real_img1, real_img2, gen_img1, gen_img2, 
                                                                                        cam1.K, cam1_R, cam1_T, cam2.K, cam2_R, cam2_T, 
                                                                                        gaussians.get_anchor,
                                                                                        distance_threshold = gaussians.voxel_size,
                                                                                        overall_ssim_threshold = 0.6)
                                    gaussians.prune_anchor(intersection_mask)
                            muiti_con_loss += loss_tmp

                    total_loss += 0.05 * muiti_con_loss
        total_loss.backward()
        
        if  gaussians.enable_net and iteration %4 ==0 and not args.no_regularization:
            gaussians.feat_planes.tv_loss(opt.tv_weight_a)
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "#points": f"{gaussians._anchor.size(0)}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)

                if iteration > opt.update_from and iteration % opt.update_interval == 0:

                    diffs = []
                    for i in range(pipe.mv):
                        cam_t1 = camera_t[i]
                        for j in range(i+1, pipe.mv):
                            cam_t2 = camera_t[j]
                            diff = torch.sqrt(torch.sum((cam_t1 - cam_t2)**2))
                            diffs.append(diff)
                    diffs = torch.stack(diffs)
                    if torch.any(diffs > 1):
                        densify_t = opt.densify_grad_threshold * 0.5
                    else:
                        densify_t = opt.densify_grad_threshold
                    gaussians.adjust_anchor(iteration=iteration, check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=densify_t, min_opacity=opt.min_opacity)
            
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()

            #if iteration>16000:
            if iteration>30000:
                gaussians.magic_k = True

            if iteration in opt.graph_downsampling_iters:
                gaussians.graph_downsampling(opt.pc_downsamplerate) 
                opt.densify_grad_threshold = opt.densify_grad_threshold*1.2

            #if iteration==1201: #slow
            if iteration==1:   #fast
                if not dataset.contractor:
                    gaussians.update_contractor()
                gaussians.enable_net = True


            if iteration in [12000, 21000] and not args.no_multilevel: #slow
            #if iteration in [15100, 15200] and not args.no_multilevel:  #fast
                gaussians.activate_plane_level(opt)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                print('number of points:', gaussians._anchor.size(0))
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(gaussians.capture(), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000, 7_000, 12_000, 17_000, 22_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_downsample", action="store_true")
    parser.add_argument("--no_multilevel", action="store_true")
    parser.add_argument("--no_regularization", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    

    if args.no_downsample:
        args.graph_downsampling_iters = []
        print('$$$$ no downsample $$$$')

    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    training(lp.extract(args), op.extract(args), pp.extract(args), dataset, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args, wandb=None, logger=logger)

    # All done
    print("\nTraining complete.")
