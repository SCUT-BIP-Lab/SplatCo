"""
 Copyright (C) 2024, Graz University of Technology
 This code is licensed under the MIT license (see LICENSE.txt in this folder for details)
"""

import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import json
from tqdm import tqdm
from popping_utils.flip import LDRFLIPLoss
from popping_utils.occlusion_utils import detect_occlusion
import imageio.v3 as iio

from utils import frame_utils
import uuid

from raft import RAFT
from utils.utils import InputPadder

METRICS = ['MSE', 'FLIP']
EPSILON = 0.000000001
ENABLE_FLIP_MIN = True
WARPED_FOLDER = 'warped'

@torch.no_grad()
def compute_flow(model, image1, image2, iters=32):
	"""
	compute optical flow from image1 to image2
	"""
	flow_prev = None
	image1_gt = torch.from_numpy(image1).permute(2, 0, 1).float()
	image2_gt = torch.from_numpy(image2).permute(2, 0, 1).float()

	padder = InputPadder(image1_gt.shape)
	image1_gt_pad, image2_gt_pad = padder.pad(image1_gt[None].cuda(), image2_gt[None].cuda())

	# flow forward
	_, flow_pr = model(image2_gt_pad, image1_gt_pad, iters=iters, flow_init=flow_prev, test_mode=True)
	flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

	# flow backward
	_, flow_pr = model(image1_gt_pad, image2_gt_pad, iters=iters, flow_init=flow_prev, test_mode=True)
	flow_bw = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

	# occlusion detection for more stable predictions
	flow_clone = np.copy(flow)
	occ = detect_occlusion(fw_flow=flow_clone, bw_flow=flow_bw)

	flow = torch.from_numpy(flow).permute(2, 0, 1).float()
	flow[0, :, :] += torch.arange(flow.shape[2])
	flow[1, :, :] += torch.arange(flow.shape[1])[:, None]
	
	# return flow, occlusion mask and image1/image2 in [0,1] (float)
	return flow, occ, image1_gt / 255., image2_gt / 255.
	

def plot_figs(metrics, results, names, dir, step):
	"""
	plots curves for MSE and FLIP over images
	"""
	# preprocess names (effectively removes .png from the filename)
	colors = ['red', 'green', 'blue']
	arangement = np.array([int(i[:-4]) for i in list(results[metrics[0]][names[0]])])
	
	fig, axes = plt.subplots(1, len(metrics), figsize=(16, 9))
	fig.suptitle(f'Metrics, RAFT, step = {step}')
	
	# plot: gt in green, ours in red
	for idx_m, m in enumerate(metrics):
		axes[idx_m].set_title(m)
		for idx_n, n in enumerate(names):
			axes[idx_m].plot(arangement, np.array(list(results[m][n].values())), color=colors[idx_n], label=n, alpha=0.75)
	
	for ax in axes:
		ax.legend()
 
	# this functionality is for subplots of a specific size, to make it easier to spot outliers in the plots
	plt.savefig(os.path.join(dir, f'test_all_step_{step}.png'), bbox_inches='tight')
	plt.close()
	

def colormap_magma(x):
	"""
	custom implementation of the magma colormap, numpy friendly
	"""
	c0 = np.array((-0.002136485053939582, -0.000749655052795221, -0.005386127855323933))
	c1 = np.array((0.2516605407371642, 0.6775232436837668, 2.494026599312351))
	c2 = np.array((8.353717279216625, -3.577719514958484, 0.3144679030132573))
	c3 = np.array((-27.66873308576866, 14.26473078096533, -13.64921318813922))
	c4 = np.array((52.17613981234068, -27.94360607168351, 12.94416944238394))
	c5 = np.array((-50.76852536473588, 29.04658282127291, 4.23415299384598))
	c6 = np.array((18.65570506591883, -11.48977351997711, -5.601961508734096))
	x = np.clip(x, 0, 1)
	res = (c0+x*(c1+x*(c2+x*(c3+x*(c4+x*(c5+c6*x))))))
	return np.clip(res, 0, 1)


def with_alpha(x: np.ndarray):
	"""
	adds alpha channel to RGB images (useful for remapping)
	"""
	return np.concatenate((x, np.ones_like(x)[..., 0][..., None]),axis=-1)

@torch.no_grad()
def validate_popping(model, frame_directories, step=1, iters: int = 32, write_images: bool = False, output_dir: str = None, write_warped: bool = False):
	"""
	Method to evaluate popping artefacts

	model: model to use
	frame_directories: directories or videos
	step: timestep during evaluation (short-range = 1, long-range = 7 were used)
	write_images: whether to write images or not
	output_dir: where to store the files
	"""
	# check for video input (.mp4) in both filenames
	is_video_input = all(['.mp4' in f[-4:] for f in frame_directories])

	if is_video_input:
		names = [f.split('/')[-1][:-4] for f in frame_directories]
	else:
		# sanity check
		assert(all(['.mp4' not in f[-4:] for f in frame_directories]))
		# workaround to support directories ending with '/'
		names = [f.split('/')[-1] if f[-1] != '/' else f[:-1].split('/')[-1] for f in frame_directories]

	# generate top level directory
	# random if not defined
	if output_dir is None:
		output_dir = 'output/' + str(uuid.uuid4())[:10]
	tld = os.path.join(output_dir)
	os.makedirs(tld, exist_ok=True)

	# dict to store outputs
	metrics = {
		f'{m}': {
			f'{z}': {} for z in names
		} for m in METRICS
	}
	
	# define flip
	flip = LDRFLIPLoss()
	
 	# load filenames
	frames = {}

	# setup directories for later (where we render individual frames to)
	for n in names:
		if write_images:
			for m in METRICS:
				os.makedirs(os.path.join(tld, n, m), exist_ok=True)
		if write_warped:
			os.makedirs(os.path.join(tld, n, WARPED_FOLDER), exist_ok=True)

	# setup mask: we do not consider the outermost 20 pixels to handle translations
	mask = None

	if is_video_input:
		# if video input: read frames directly
		for n_idx, name in enumerate(names):
			vid = frame_directories[n_idx]
			fs = []
			for _, frame in enumerate(iio.imiter(vid)):
				fs.append(frame)
			frames[name] = fs
		ex_img = frames[names[0]][0]
		h, w = frames[names[0]][0].shape[:2]
	else:
		# otherwise: load file names and load images on demand
		frames = {
			f'{fd.split("/")[-1]}': sorted([os.path.join(fd,i) for i in os.listdir(fd) if 'png' in i]) for fd in frame_directories
		}
		ex_img = frame_utils.read_gen(frames[names[0]][0])
		w, h = ex_img.width, ex_img.height

	# setting up the mask
	mask = np.zeros((h, w))
	num_px_ignored = 20
	mask[num_px_ignored:-num_px_ignored, num_px_ignored:-num_px_ignored] = 1.
 
	# sanity check:
	# len must be equal for all entries in the dict
	assert all([l == len(frames[names[0]]) for l in [len(frames[f]) for f in frames]]), "not all directories contain an equal number of frames"

	# iterate over all available images
	for test_id in tqdm(range(len(frames[names[0]]) - step)):

		img_name = f'{test_id:05d}.png'
		outs_this_iter = {}
		occ = np.copy(mask)

		# iterate over all methods
		for n in names:
			image1, image2 = None, None
			if not is_video_input:
				image1 = frame_utils.read_gen(frames[n][test_id])
				image2 = frame_utils.read_gen(frames[n][test_id + step])
				image1 = np.array(image1).astype(np.uint8)[..., :3]
				image2 = np.array(image2).astype(np.uint8)[..., :3]
			else:
				image1 = frames[n][test_id]
				image2 = frames[n][test_id + step]
    
			# compute optical flow and occlusion mask
			flow, occ_, image1, image2 = compute_flow(model=model, image1=image1, image2=image2, iters=iters)
			# remap with flow
			warped_result = cv2.remap(with_alpha((image1).permute(1,2,0).cpu().numpy()), flow.permute(1,2,0).cpu().numpy(), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
   
			outs_this_iter[n] = {
				'image1': image1,
				'image2': image2,
				'warped_result': warped_result,
				'occ': occ_,
			}
			# accumulate occupancy map
			occ *= occ_ * warped_result[..., -1]

		occ_cuda = torch.from_numpy(occ).cuda()

		for n in names:
			warped_result = outs_this_iter[n]['warped_result']
			image2 = outs_this_iter[n]['image2']

			# compute MSE and FLIP
			outs_this_iter[n]['MSE'] = ((torch.from_numpy(warped_result[..., :3]).permute(2,0,1) - image2)**2).sum(0) * occ
			outs_this_iter[n]['FLIP'] = flip(test=torch.from_numpy(warped_result[..., :3]).permute(-1, 0, 1).cuda()[None, ...], reference=image2[None, ...].cuda(), mask=occ_cuda)

		# subtract the minimum flip error for more stable predictions
		# can be disabled by setting ENABLE_FLIP_MIN = 0 
		min_flip = torch.min(torch.stack([outs_this_iter[n]['FLIP'] for n in names], dim=-1), dim=-1).values

		for n in names:
			if ENABLE_FLIP_MIN:
				outs_this_iter[n]['FLIP'] = outs_this_iter[n]['FLIP'] - min_flip

			if write_warped:
				image = (outs_this_iter[n]['warped_result'] * 255).astype(np.float32)
				cv2.imwrite(os.path.join(tld, n, WARPED_FOLDER, img_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

			for m in METRICS:
				metrics[m][n][img_name] = ((outs_this_iter[n][m].sum() / (occ.sum() + EPSILON)).item())
				# write output images
				if write_images:
					image = (colormap_magma(outs_this_iter[n][m][..., None].cpu().numpy()) * 255).astype(np.float32)
					cv2.imwrite(os.path.join(tld, n, m, img_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
	
	# setup full dict (averaged metrics)
	full_dict = {
		f'{m}': {
			f'{n}': np.array(list(metrics[m][n].values())).mean() for n in names
		} for m in METRICS
  	}
  
	# final plot
	plot_figs(metrics=METRICS, results=metrics, names=names, dir=tld, step=step)

	# save per-view results as well as full results
	with open(os.path.join(tld, f'per_view.json'), 'w') as outfile:
		json.dump(metrics, outfile, indent=2)
	with open(os.path.join(tld, f'results.json'), 'w') as outfile:
		json.dump(full_dict, outfile, indent=2)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', '-m', help="restore checkpoint")
	parser.add_argument('--step', type=int, default=1, help='Timestep/Offset')
	parser.add_argument('--frame_directories', "-f", required=True, nargs="+", type=str, default=[], help='Input Files/Directories')
	parser.add_argument('--all_images', action='store_true', help='Whether to store all outputs')
	parser.add_argument('--output_dir', action='store', type=str, required=False, help='Where to store the outputs', default=None)
	parser.add_argument('--warped', action='store_true', help='Whether to write warped frames')
	args = parser.parse_args()

	# custom args
	args.small = False
	args.mixed_precision = False

	model = torch.nn.DataParallel(RAFT(args))
	model.load_state_dict(torch.load(args.model))

	model.cuda()
	model.eval()

	with torch.no_grad():
		validate_popping(model.module, frame_directories=args.frame_directories, step=args.step, write_images=args.all_images, output_dir=args.output_dir, write_warped=args.warped)
