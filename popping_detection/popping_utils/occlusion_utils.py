#
# The contents of the this file are based on publicly available code, which falls under the MIT license. 

# Title: fast_blind_video_consistency
# Project code: https://github.com/phoenix104104/fast_blind_video_consistency
# Copyright (c) 2018 UC Merced Vision and Learning Lab
# License: https://github.com/phoenix104104/fast_blind_video_consistency/blob/master/LICENSE (MIT)
#

import numpy as np
import torch
import cv2

@torch.no_grad()
def compute_flow_magnitude(flow):
	flow_mag = flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2
	return flow_mag

@torch.no_grad()
def compute_flow_gradients(flow):
	H = flow.shape[0]
	W = flow.shape[1]

	flow_x_du = np.zeros((H, W))
	flow_x_dv = np.zeros((H, W))
	flow_y_du = np.zeros((H, W))
	flow_y_dv = np.zeros((H, W))

	flow_x = flow[:, :, 0]
	flow_y = flow[:, :, 1]

	flow_x_du[:, :-1] = flow_x[:, :-1] - flow_x[:, 1:]
	flow_x_dv[:-1, :] = flow_x[:-1, :] - flow_x[1:, :]
	flow_y_du[:, :-1] = flow_y[:, :-1] - flow_y[:, 1:]
	flow_y_dv[:-1, :] = flow_y[:-1, :] - flow_y[1:, :]

	return flow_x_du, flow_x_dv, flow_y_du, flow_y_dv

@torch.no_grad()
def detect_occlusion(fw_flow, bw_flow):
	"""
	occlusion detection method from Ruder et. al [GCPR'16]
	"""
	# fw-flow: img1 => img2
	# bw-flow: img2 => img1

	bw_flow_t = torch.from_numpy(bw_flow).cuda()

	# warp fw-flow to img2
	flow = torch.from_numpy(fw_flow).permute(2, 0, 1).float()
	flow[0, :, :] += torch.arange(flow.shape[2])
	flow[1, :, :] += torch.arange(flow.shape[1])[:, None]
	fw_flow_w = cv2.remap((bw_flow_t).cpu().numpy(), flow.permute(1, 2, 0).cpu().numpy(), None, cv2.INTER_LINEAR)

	# convert to numpy array
	fw_flow_w = (fw_flow_w)

	# occlusion
	fb_flow_sum = fw_flow_w + bw_flow
	fb_flow_mag = compute_flow_magnitude(fb_flow_sum)
	fw_flow_w_mag = compute_flow_magnitude(fw_flow_w)
	bw_flow_mag = compute_flow_magnitude(bw_flow)

	mask1 = fb_flow_mag > 0.01 * (fw_flow_w_mag + bw_flow_mag) + 0.5

	# motion boundary
	fx_du, fx_dv, fy_du, fy_dv = compute_flow_gradients(bw_flow)
	fx_mag = fx_du ** 2 + fx_dv ** 2
	fy_mag = fy_du ** 2 + fy_dv ** 2

	mask2 = (fx_mag + fy_mag) > 0.01 * bw_flow_mag + 0.002

	# combine mask
	mask = np.logical_or(mask1, mask2)
	occlusion = np.zeros((fw_flow.shape[0], fw_flow.shape[1]))
	occlusion[mask == 1] = 1

	return occlusion