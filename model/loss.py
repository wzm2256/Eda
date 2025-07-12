import pdb

import torch
import numpy as np
from Utils import so3_utils


def isotropic_R_error(r1, r2):
	'''
	Calculate isotropic rotation degree error between r1 and r2.
	:param r1: shape=(B, 3, 3), pred
	:param r2: shape=(B, 3, 3), gt
	:return:
	'''
	r2_inv = r2.permute(0, 2, 1).contiguous()
	r1r2 = torch.matmul(r2_inv, r1)
	tr = torch.einsum("...ii->...", r1r2)
	rads = torch.acos(torch.clamp((tr - 1) / 2, -1, 1))
	degrees = rads / np.pi * 180
	return degrees

def g_error(g, g_gt):
	"""
	Calculate SE3 error between g and g_gt
	:param g: shape=(B, 4, 4), pred
	:param g_gt: shape=(B, 4, 4), gt
	:return:
	"""
	r_error = isotropic_R_error(g[:, :3, :3], g_gt[:, :3, :3])
	t_error = (g[:, :3, 3] - g_gt[:, :3, 3]).norm(dim=-1)
	return r_error, t_error

def non_fixed_g_error(predict_g1:torch.Tensor, predict_g2:torch.Tensor, true_g1:torch.Tensor, true_g2:torch.Tensor):
	""" Calculate SE3 error between (g1, g2) and (g1_gt, g2_gt) """
	transformed_g1 = torch.matmul(torch.matmul(true_g2, so3_utils.inv_SE3(predict_g2)), predict_g1)
	return g_error(transformed_g1, true_g1)


def regress_loss(deg2: torch.Tensor, deg1: torch.Tensor, gt_deg2: torch.Tensor, gt_deg1: torch.Tensor,
                 distance_weight: float = 1.0):
	"""Calculate MSE loss for regression task."""
	r_loss = ((deg2 - gt_deg2) ** 2).mean()
	t_loss = ((deg1 - gt_deg1) ** 2).mean()
	loss = r_loss + t_loss * distance_weight
	return loss, r_loss, t_loss



def opt_loss3(g_gt_all: torch.Tensor, g_pred_all: torch.Tensor, transform_batch: torch.Tensor, return_all=False) -> \
		(torch.Tensor, torch.Tensor):
	"""Calcuate the averaged pairwise SE3 error within a batch."""
	num_piece = g_gt_all.shape[0]

	# Select all pairs that share the same batch index
	All_index = torch.arange(num_piece, device=g_gt_all.device)
	All_edges = torch.stack(torch.meshgrid(All_index, All_index, indexing='xy')).flatten(start_dim=1)
	Select_index = torch.logical_and(transform_batch[All_edges[0]] == transform_batch[All_edges[1]], All_edges[0] != All_edges[1])
	All_edges = All_edges[:, Select_index]

	predict_g1 = g_pred_all.index_select(0, All_edges[0])
	predict_g2 = g_pred_all.index_select(0, All_edges[1])
	true_g1 = g_gt_all.index_select(0, All_edges[0])
	true_g2 = g_gt_all.index_select(0, All_edges[1])

	error = non_fixed_g_error(predict_g1, predict_g2, true_g1, true_g2)
	if return_all:
		return error[0], error[1]
	return torch.mean(error[0]), torch.mean(error[1])






# if __name__ == '__main__':
# 	from Utils.so3_utils import random_SE3
# 	# mean = torch.randn(5,3)
# 	# g_gt = random_SE3(5)
# 	# # g_pre = random_SE3(5)
	#
	# g_diff = random_SE3(1).repeat((5,1,1))
	#
	# r_error, t_error = opt_loss(g_gt, g_diff @ g_gt, mean)
	#
	# print(f'r: {r_error} t:{t_error}')

	# g_gt = random_SE3(2)
	# g_pred = random_SE3(2)

	# r_loss, t_loss = opt_loss2(g_gt, g_pred)



# def skip(*args, **kwargs):
# 	return (torch.tensor([0.], device=predict_g1.device), torch.tensor([0.], device=predict_g1.device)), 1
#
#
#
# @ torch.compile(dynamic=True, fullgraph=True, mode='max-autotune')
# def opt_loss1(g_gt_all: torch.Tensor, g_pred_all: torch.Tensor, transform_batch: torch.Tensor, piece_num: int) -> (torch.Tensor, torch.Tensor):
#
# 	r_list = torch.zeros((piece_num, piece_num), device=g_gt_all.device)
# 	t_list = torch.zeros((piece_num, piece_num), device=g_gt_all.device)
# 	Mask   = torch.zeros((piece_num, piece_num), device=g_gt_all.device)
#
# 	for i in range(piece_num):
# 		for j in range(piece_num):
# 			# if i == j or transform_batch[i] != transform_batch[j]:
# 			# 	continue
# 			# skip =
# 			# skip = False
# 			# skip = (transform_batch[i] != transform_batch[j])
# 			# skip = (i == j)
# 			r_error, t_error, skip_flag = torch.cond((transform_batch[i] != transform_batch[j]) or (i == j), non_fixed_g_error, skip, (g_pred_all[j:j+1], g_pred_all[i:i+1], g_gt_all[j:j+1], g_gt_all[i:i+1]))
#
# 			# r_error, t_error = non_fixed_g_error(g_pred_all[j:j+1], g_pred_all[i:i+1], g_gt_all[j:j+1], g_gt_all[i:i+1],
# 			#                                      skip=skip)
#
# 			r_list[i, j] = r_error
# 			t_list[i, j] = t_error
# 			Mask[i, j] = skip_flag
# 			# pdb.set_trace()
# 	# pdb.set_trace()
# 	Total_element = torch.sum(Mask)
# 	r_mean = torch.sum(r_list) / Total_element
# 	t_mean = torch.sum(t_list) / Total_element
# 	return r_mean, t_mean

# @ torch.compile(dynamic=True, fullgraph=True, mode='max-autotune')
# def opt_loss2(g_gt_all: torch.Tensor, g_pred_all: torch.Tensor, transform_batch: torch.Tensor) -> (torch.Tensor, torch.Tensor):
#
# 	r_list = []
# 	t_list = []
#
# 	for b in torch.arange(torch.max(transform_batch) + 1):
# 		g_gt = g_gt_all[transform_batch == b]
# 		g_pred = g_pred_all[transform_batch == b]
#
# 		num_piece = g_gt.shape[0]
#
# 		g_global = g_gt @ so3_utils.inv_SE3(g_pred)
# 		predict = einops.rearrange(einops.einsum(g_global, g_pred, 'i j k, s k l -> s i j l'), 's i j l -> (s i) j l')
# 		true = einops.repeat(g_gt, 's i j -> (s r) i j', r=num_piece)
# 		r_error, t_error = g_error(predict, true)
# 		r_list.append(r_error.sum() / (num_piece * (num_piece - 1)))
# 		t_list.append(t_error.sum() / (num_piece * (num_piece - 1)))
#
# 	# pdb.set_trace()
# 	return torch.tensor(r_list).mean(), torch.tensor(t_list).mean()
