import einops
import torch
import pdb
import matplotlib.pyplot as plt
from Utils import so3_utils
from model.module_util import  scatter_mean_keepsize
from model.FastSo3 import so3


def weighted_procrustes_r(
		src_points,
		ref_points,
		batch_index=None,
):
	r"""Only compute r

	Args:
		src_points: torch.Tensor (N, 3)
		ref_points: torch.Tensor  (N, 3)
		weights: torch.Tensor (N,) (default: None)
		eps: float (default: 1e-5)

	Returns:
		R: torch.Tensor  (1, 3, 3)
	"""

	All = einops.einsum(src_points, ref_points, 'b c, b d -> b c d')
	batch_4 = einops.repeat(batch_index, 'b -> (b n)', n=4)
	H = scatter_mean_keepsize(All, batch_4, dim=0, sum=True, keepsize=False)
	batch_size = H.shape[0]

	U, Sigma, V = torch.svd(H)  # H = USV^T
	Ut, V = U.transpose(1, 2), V
	eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(V.device)
	eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
	R = V @ eye @ Ut

	transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(V.device)
	transform[:, :3, :3] = R

	return transform



def logitnorm(size, m, s):
	n = s * torch.randn(size) + m
	out = 1 / (1 + torch.exp(-n))
	return out


def sample_t(size, m=None, s=None, t_type='uniform', eps=1e-4, coe_f_exp=-5, coe_f_poly=2, total_step=1000):
	"""Sample time points for training."""

	if t_type == 'uniform':
		return torch.rand(size) * (1 - eps)
	if t_type == 'logitnorm':
		return torch.clip(logitnorm(size, m=m, s=s), 0, 1 - eps)
	# uniform_discrete, logitnorm_discrete, poly_discrete, exp_discrete, cos_discrete
	if t_type == 'uniform_discrete':
		return torch.randint(0, total_step, size) / total_step
	if t_type == 'logitnorm_discrete':
		return torch.floor(logitnorm(size, m=m, s=s) * total_step) / total_step
	if t_type == 'exp_discrete':
		# pdb.set_trace()
		return 1 - torch.exp(coe_f_exp * torch.randint(0, total_step, size) / total_step)
	if t_type == 'poly_discrete':
		return 1 - torch.tensor(torch.randint(0, total_step, size) / total_step) ** coe_f_poly

	raise NotImplementedError


class t_Sampler:
	"""Sample time points for training and test."""
	def __init__(self, args):
		self.t_type = args.t_type
		self.t_m = args.t_m
		self.t_s = args.t_s
		self.training_step = args.training_step
		self.coe_f_poly = args.coe_f_poly
		self.coe_f_exp = args.coe_f_exp

		self.test_step = args.test_step
		# uniform, poly, exp, cos
		self.t_test_type = args.t_test_type

	def get_t(self, batch_size=1, training=True, step=None):
		if training:
			return sample_t(size=(batch_size,), m=self.t_m, s=self.t_s, t_type=self.t_type, total_step=self.training_step,
			                coe_f_exp=self.coe_f_exp, coe_f_poly=self.coe_f_poly)
		else:
			if self.t_test_type == 'linear':
				return torch.tensor(step / self.test_step), 1 / self.test_step
			if self.t_test_type == 'poly':
				t_now = 1 - torch.tensor(1 - step / self.test_step) ** self.coe_f_poly
				t_next = 1 - torch.tensor(1 - (step + 1) / self.test_step) ** self.coe_f_poly
				return t_now, t_next - t_now
			if self.t_test_type == 'exp':
				t_now = 1 - torch.exp(torch.tensor(self.coe_f_exp * (step / self.test_step)))
				t_next = 1 - torch.exp(torch.tensor(self.coe_f_exp * ((step + 1) / self.test_step)))
				return t_now, t_next - t_now
			else:
				raise NotImplementedError


def get_global_r(g0=None, g1=None,  distance_weight=1.0, seed_noise=None, r_type='random', only_translation=0,
                 batch_index=None):
	"""Compute rotation correction."""
	if only_translation == 1 or r_type == 'fixed':
		return torch.eye(4).unsqueeze(0).to(g0.device)

	if r_type == 'random':
		global_r = so3_utils.random_SE3(1, sigma=0., zero_mean=True, seed=seed_noise)
		return global_r
	if r_type == 'matching':
		scale_t = torch.tensor([1,1,1, distance_weight]).unsqueeze(0).to(g0.device)
		src = (g1 * scale_t)[:, :3, :]
		ref = (g0 * scale_t)[:, :3, :]
		global_r = weighted_procrustes_r(einops.rearrange(src, 'b i j -> (b j) i'),
		                                einops.rearrange(ref, 'b i j -> (b j) i'), batch_index=batch_index)
		global_r_all = torch.index_select(global_r, 0, batch_index)
		return global_r_all

	raise NotImplementedError


def rotate_list(graph_list, bi_graph_list, g_t, L):
	"""Rotate all graphs in the list. Compute edge vector and rotation matrix."""
	bi_graph_list_rotated = []

	for bi in bi_graph_list:
		bi_rotate = bi.clone()
		g_full_u = torch.index_select(g_t, 0, bi.u_piece)
		gu_cor = (g_full_u[:, :3, :3] @ bi.u_cor.unsqueeze(-1) + g_full_u[:, :3, 3:]).squeeze(-1)

		g_full_v = torch.index_select(g_t, 0, bi.v_piece)
		gv_cor = (g_full_v[:, :3, :3] @ bi.v_cor.unsqueeze(-1) + g_full_v[:, :3, 3:]).squeeze(-1)

		bi_edge_vec = gu_cor[bi.e[0]] - gv_cor[bi.e[1]]

		bi_rotate.uv_rotation = so3.RotationToWignerDMatrix(so3.init_edge_rot_mat(bi_edge_vec, norm=bi.edge_length), L)
		bi_rotate.u_cor = gu_cor
		bi_rotate.v_cor = gv_cor
		bi_rotate.edge_vec = bi_edge_vec
		bi_graph_list_rotated.append(bi_rotate)


	g_graph_list_rotated = []
	for g in graph_list:

		g_rotate = g.clone()
		g_full = torch.index_select(g_t, 0, g.piece_index)

		g_rotate.x_cor = (g_full[:, :3, :3] @ g.x_cor.unsqueeze(-1) + g_full[:, :3, 3:]).squeeze(-1)
		g_rotate.edge_vec = g_rotate.x_cor.index_select(0, g.e[0]) - g_rotate.x_cor.index_select(0, g.e[1])
		g_rotate.e_rotation = so3.RotationToWignerDMatrix(so3.init_edge_rot_mat(g_rotate.edge_vec, norm=g_rotate.edge_length), L)

		g_graph_list_rotated.append(g_rotate)
	return g_graph_list_rotated, bi_graph_list_rotated


def get_vector(g_t, PC, model, t, path_type=1, path_coe=1.0, graph_list=None, bi_graph_list=None):
	"""Compute the vector at a certain time t and position g_t"""
	PC_t = so3_utils.SE3_action(g_t, PC)

	g_graph_list_rotated, bi_graph_list_rotated = rotate_list(graph_list, bi_graph_list, g_t, PC.x_f.shape[-2])

	# torch.compiler.cudagraph_mark_step_begin()
	deg1, deg2 = model(PC_t, t, graph_list=g_graph_list_rotated, bi_graph_list=bi_graph_list_rotated)

	predict_vec = torch.cat([deg2, deg1], -1)
	return predict_vec



def Step(t, g_t, h, model, PC, order=1, alpha=1, path_type=1, path_coe=1.0,
         graph_list=None,  bi_graph_list=None, tol=1e-7):
	"""
	One Runge-Kutta step.
	"""
	F_0 = get_vector(g_t, PC, model, t, path_type=path_type, path_coe=path_coe, graph_list=graph_list, bi_graph_list=bi_graph_list)

	if order == 1:
		## Euler
		return so3_utils.exp_se3(h * F_0, tol=tol) @ g_t

	if order == 2:
		## Heun. Not used
		if t[0].item() + alpha * h > 1.0:
			return so3_utils.exp_se3(h * F_0) @ g_t

		v_1 = so3_utils.exp_se3(alpha * h * F_0) @ g_t
		F_1 = get_vector(v_1, PC, model, t + alpha * h,  path_type=path_type, path_coe=path_coe,  graph_list=graph_list, bi_graph_list=bi_graph_list)
		return so3_utils.exp_se3(h / (2 * alpha) * F_1) @ so3_utils.exp_se3(h * (1 - 1 / (2 * alpha)) * F_0) @ g_t
	if order == 4:
		## Runge–Kutta 4 with with 1/6-rule
		F_1 = get_vector(so3_utils.exp_se3(h / 2 * F_0) @ g_t, PC, model, t + h / 2,  path_type=path_type, path_coe=path_coe,  graph_list=graph_list, bi_graph_list=bi_graph_list)
		F_2 = get_vector(so3_utils.exp_se3(h / 2 * F_1) @ g_t, PC, model, t + h / 2,  path_type=path_type, path_coe=path_coe,  graph_list=graph_list, bi_graph_list=bi_graph_list)
		F_3 = get_vector(so3_utils.exp_se3(h * F_2) @ g_t, PC, model, t + h,  path_type=path_type, path_coe=path_coe,  graph_list=graph_list, bi_graph_list=bi_graph_list)
		return so3_utils.exp_se3(h / 6 * F_3) @ so3_utils.exp_se3(h / 3 * F_2) @ so3_utils.exp_se3(h / 3 * F_1) \
			@ so3_utils.exp_se3(h / 6 * F_0) @ g_t

	if order == 41:
		## Runge–Kutta 4 with 3/8-rule
		F_1 = get_vector(so3_utils.exp_se3(h / 3 * F_0) @ g_t, PC, model, t + h / 3,  path_type=path_type, path_coe=path_coe,  graph_list=graph_list, bi_graph_list=bi_graph_list)
		F_2 = get_vector(so3_utils.exp_se3(h     * F_1) @ so3_utils.exp_se3(-h / 3 * F_0) @ g_t, PC, model, t + 2 * h / 3,  path_type=path_type, path_coe=path_coe,  graph_list=graph_list, bi_graph_list=bi_graph_list)
		F_3 = get_vector(so3_utils.exp_se3(h * F_2) @ so3_utils.exp_se3(-h * F_1) @ so3_utils.exp_se3(h * F_0) @ g_t, PC, model, t + h,  path_type=path_type, path_coe=path_coe,  graph_list=graph_list, bi_graph_list=bi_graph_list)

		return so3_utils.exp_se3(h / 8 * F_3) @ so3_utils.exp_se3(3 *h / 8 * F_2) @ so3_utils.exp_se3(3 * h / 8 * F_1) \
			@ so3_utils.exp_se3(h / 8 * F_0) @ g_t

	raise NotImplementedError






if __name__ == '__main__':
	# out = logitnorm((100000), 0, 0.5)
	# plt.hist(out.numpy())
	# plt.show()
	# pdb.set_trace()

	# g0 = so3_utils.random_SE3(piece_num=3, zero_mean=True, sigma=1.)
	# r = so3_utils.random_SE3(piece_num=1, zero_mean=True, sigma=0.)
	# g1 = r@ g0
	#
	# global_r = get_global_r(g0=g0, g1=g1, distance_weight=1.0, seed_noise=None, r_type='matching')
	# print(global_r @ r)
	#
	# pdb.set_trace()

	A = sample_t(size=(10000,), m=0, s=1, t_type='poly_discrete', eps=1e-4, coe_f_exp=-5, coe_f_poly=2, total_step=1000)
	plt.hist(A.numpy(), bins=50)
	plt.show()
	pdb.set_trace()
# def weighted_procrustes(
# 		src_points,
# 		ref_points,
# 		weights=None,
# 		weight_thresh=0.0,
# 		eps=1e-5,
# 		return_transform=False,
# ):
# 	r"""Compute rigid transformation from `src_points` to `ref_points` using weighted SVD.
#
# 	Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).
#
# 	Args:
# 		src_points: torch.Tensor (B, N, 3) or (N, 3)
# 		ref_points: torch.Tensor (B, N, 3) or (N, 3)
# 		weights: torch.Tensor (B, N) or (N,) (default: None)
# 		weight_thresh: float (default: 0.)
# 		eps: float (default: 1e-5)
# 		return_transform: bool (default: False)
#
# 	Returns:
# 		R: torch.Tensor (B, 3, 3) or (3, 3)
# 		t: torch.Tensor (B, 3) or (3,)
# 		transform: torch.Tensor (B, 4, 4) or (4, 4)
# 	"""
# 	if src_points.ndim == 2:
# 		src_points = src_points.unsqueeze(0)
# 		ref_points = ref_points.unsqueeze(0)
# 		if weights is not None:
# 			weights = weights.unsqueeze(0)
# 		squeeze_first = True
# 	else:
# 		squeeze_first = False
#
# 	batch_size = src_points.shape[0]
# 	if weights is None:
# 		weights = torch.ones_like(src_points[:, :, 0])
# 	weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
# 	weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
# 	weights = weights.unsqueeze(2)  # (B, N, 1)
#
# 	src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
# 	ref_centroid = torch.sum(ref_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
# 	src_points_centered = src_points - src_centroid  # (B, N, 3)
# 	ref_points_centered = ref_points - ref_centroid  # (B, N, 3)
#
# 	H = src_points_centered.permute(0, 2, 1) @ (weights * ref_points_centered)
# 	U, _, V = torch.svd(H)  # H = USV^T
# 	Ut, V = U.transpose(1, 2), V
# 	eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(V.device)
# 	eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
# 	R = V @ eye @ Ut
#
# 	# pdb.set_trace()
# 	t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
# 	t = t.squeeze(2)
#
# 	# pdb.set_trace()
# 	if return_transform:
# 		transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(V.device)
# 		transform[:, :3, :3] = R
# 		transform[:, :3, 3] = t
# 		if squeeze_first:
# 			transform = transform.squeeze(0)
# 		return transform
# 	else:
# 		if squeeze_first:
# 			R = R.squeeze(0)
# 			t = t.squeeze(0)
# 		return R, t

	# if path_type == 1:
	# 	predict_se01 = predict / (1 - t.item())
	# elif path_type == 3:
	# 	# pdb.set_trace()
	# 	predict_se01 = predict * path_coe * torch.exp(- path_coe * t) / (1 - t.item())
	# elif path_type == 4:
	# 	predict_se01 = predict * 2.
	# else:
	# 	raise NotImplementedError
	# return predict_se01


# @torch.autocast(device_type='cpu', dtype=torch.float16)
# def deg1_step(F_0, h, g_t):
# 	# print('This function will proceu nan value due to the difference in GPU and CPU behavior. ')
# 	# raise NotImplementedError
# 	F_0_cpu, g_t_cpu, h_cpu = F_0.to('cpu', dtype=torch.float), g_t.to('cpu', dtype=torch.float), torch.tensor(h)
# 	g_t_cpu1 = so3_utils.exp_se3(h_cpu * F_0_cpu) @ g_t_cpu
# 	return g_t_cpu1.to(F_0.device)
#
#
# # @torch.autocast(device_type='cpu', dtype=torch.float16)
# def deg4_step(F_0, F_1, F_2, F_3, h, g_t):
#
# 	# print('This function will proceu nan value due to the difference in GPU and CPU behavior. ')
# 	raise NotImplementedError
# 	# F_0_cpu, F_1_cpu, F_2_cpu, F_3_cpu, g_t_cpu, h_cpu = F_0.to('cpu'), F_1.to('cpu'), F_2.to('cpu'), F_3.to('cpu'), g_t.to('cpu'), torch.tensor(h)
# 	# g_t_cpu1 = so3_utils.exp_se3(h_cpu / 6 * F_3_cpu) @ so3_utils.exp_se3(h_cpu / 3 * F_2_cpu) @ so3_utils.exp_se3(h_cpu / 3 * F_1_cpu) @ so3_utils.exp_se3(h_cpu / 6 * F_0_cpu) @ g_t_cpu
# 	# return g_t_cpu1.to(F_0.device)
#
# # @torch.autocast(device_type='cpu', dtype=torch.float16)
# def deg41_step(F_0, F_1, F_2, F_3, h, g_t):
# 	F_0_cpu, F_1_cpu, F_2_cpu, F_3_cpu, g_t_cpu, h_cpu = F_0.to('cpu'), F_1.to('cpu'), F_2.to('cpu'), F_3.to('cpu'), g_t.to('cpu'), torch.tensor(h)
#
# 	g_t_cpu1 = so3_utils.exp_se3(h_cpu / 8 * F_3_cpu) @ so3_utils.exp_se3(3 * h_cpu / 8 * F_2_cpu) @ so3_utils.exp_se3(3 * h_cpu / 8 * F_1_cpu) @ so3_utils.exp_se3(h_cpu / 8 * F_0_cpu) @ g_t_cpu
# 	return g_t_cpu1.to(F_0.device)
#


