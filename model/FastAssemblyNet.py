from typing import List, Tuple, Optional, Union, Iterable, NamedTuple, Any, Sequence, Dict, Callable
import pdb

import torch
import torch.nn as nn

from .FastSo3 import so3, so2

from model.module_util import SinusoidalPositionEmbeddings, TimestepEmbedder, scatter_mean_keepsize, get_graph_scale, scatter_softmax_coo# , scatter_softmax_ori # ,regress_se3 compute_vector
from model.Module.Linear import Linear_p, LastLayer
from model.Module.Block import EquiformerBlock
from torch_geometric.data import Data


def propagate_bi_g(bi_g, u_f, u_f_0):
	'''
	Copy the feature of node u to node v
	:param bi_g: bi_graph
	'''
	if hasattr(bi_g, 'v_idx'):
		return u_f[bi_g.v_idx], u_f_0[bi_g.v_idx]
	raise NotImplementedError


def get_degree(x: torch.Tensor, d: int):
	'''
	Get a specific degree d in full tensor x
	:param x: Full tensor (..., L, m)
	:param d: needed degree
	:return:
	'''
	L = x.shape[-2]
	return x[..., d-1, L-d : L+ d + 1]



class TransitionDown(nn.Module):
	'''
	Down-sampling
	'''
	def __init__(self, emb_channel: List[int], L: int, num_heads: int, fc_neurons: List[int], forward_mid_scale: int = 3,
	             qk_norm = False, norm_type: str = 'n111', nonlinear_type = 'silu', max_len: float=1.0,
	             max_len_bi: float=1.0):
		super().__init__()
		self.Layers = torch.nn.ModuleDict()
		self.Layers['eddge_embedding'] = SinusoidalPositionEmbeddings(dim=fc_neurons[0])

		self.Layers['down_gnn'] = EquiformerBlock(emb_channel, L, num_heads, fc_neurons,
		        forward_mid_scale=forward_mid_scale,
	             qk_norm=qk_norm,
	             norm_type=norm_type,
	             nonlinear_type=nonlinear_type,
		        )

		self.Layers['gnn'] = EquiformerBlock(emb_channel, L, num_heads, fc_neurons,
		        forward_mid_scale=forward_mid_scale,
	             qk_norm=qk_norm,
	             norm_type=norm_type,
	             nonlinear_type=nonlinear_type,
				)

		self.max_len = max_len
		self.max_len_bi = max_len_bi


	def forward(self, bi_g: Data, g_coarse: Data, t_code: torch.Tensor, x: torch.Tensor, x_0: torch.Tensor):
		'''
		:param bi_g: the bigraph connecting g and g_coarse
		:param g: the graph at this scale
		:param g_coarse: the graph at the coarser scale
		bi_g.u_cor is g.x_cor, and g.x_f is not None
		bi_g.v_cor is g_coarse.x_cor
		This function popagate the feature of g to feature of g_coarse through bi_g,
		it updates bi_g with bi_g.v_f and bi_g.u_f, updated g_coarse with g_coarse.x_f
		'''

		# get preliminary features fow low resolution graph
		pre_v_f, pre_v_f_0 = propagate_bi_g(bi_g, x, x_0)

		bi_g.edge_scalar = self.Layers['eddge_embedding'](bi_g.edge_length, max_val=self.max_len_bi)
		g_coarse.edge_scalar = self.Layers['eddge_embedding'](g_coarse.edge_length, max_val=self.max_len)

		x_down, x_down_0 = self.Layers['down_gnn'](x_source=x, x_target=pre_v_f, x_0_source=x_0,
		                                   x_0_target=pre_v_f_0, edge=bi_g.e, batch_src=bi_g.u_batch,
		                                   batch_dst=bi_g.v_batch, piece_src=bi_g.u_piece, piece_dst=bi_g.v_piece,
											edge_scalar=bi_g.edge_scalar, t_code=t_code,
		                                   Rotation_W=bi_g.uv_rotation)


		y, y_0 = self.Layers['gnn'](x_source=x_down, x_target=x_down, x_0_source=x_down_0,
		                                   x_0_target=x_down_0, edge=g_coarse.e, batch_src=g_coarse.x_cor_batch,
		                                   batch_dst=g_coarse.x_cor_batch, piece_src=g_coarse.piece_index,
		                                   piece_dst=g_coarse.piece_index, edge_scalar=g_coarse.edge_scalar, t_code=t_code,
		                                   Rotation_W=g_coarse.e_rotation)
		return y, y_0


class DiTr(nn.Module):
	def __init__(self, emb_channel: List[int], L: int, num_heads: int, fc_neurons: List[int], forward_mid_scale: int = 3,
	             qk_norm = False, norm_type: str = 'n111', nonlinear_type = 'silu', eps=1e-3):
		super().__init__()

		self.Layers = torch.nn.ModuleDict()
		self.Layers['self_gnn'] = EquiformerBlock(emb_channel, L, num_heads, fc_neurons,
									        forward_mid_scale=forward_mid_scale,
								             qk_norm=qk_norm,
								             norm_type=norm_type,
								             nonlinear_type=nonlinear_type,
		                                     )
		self.Layers['cross_gnn'] = EquiformerBlock(emb_channel, L, num_heads, fc_neurons,
									        forward_mid_scale=forward_mid_scale,
								             qk_norm=qk_norm,
								             norm_type=norm_type,
								             nonlinear_type=nonlinear_type,
		                                     )
		self.Layers['eddge_embedding'] = SinusoidalPositionEmbeddings(dim=fc_neurons[0])
		self.L = L
		self.eps = eps

	def forward(self, g, t_code, x: torch.Tensor, x_0: torch.Tensor):

		# If cross edges have not been computed yet
		if not hasattr(g, 'cross_e'):
			All_index = torch.arange(g.x_cor.shape[0], device=g.x_cor.device)
			All_edges = torch.stack(torch.meshgrid(All_index, All_index, indexing='xy')).flatten(start_dim=1)
			Select_index = torch.logical_and(g.x_cor_batch[All_edges[0]] == g.x_cor_batch[All_edges[1]],
			                                 g.piece_index[All_edges[0]] != g.piece_index[All_edges[1]])
			All_edges = All_edges[:, Select_index]

			cross_edge_vec = g.x_cor.index_select(0, All_edges[0]) - g.x_cor.index_select(0, All_edges[1])
			cross_edge_norm = cross_edge_vec.norm(dim=1, p=2)

			# remove edges that are too small to avoid numerical issues
			Mask = cross_edge_norm > self.eps
			All_edges = All_edges[:, Mask]
			cross_edge_norm = cross_edge_norm[Mask]
			cross_edge_vec = cross_edge_vec[Mask]

			# add edge features
			g.cross_e = All_edges
			g.cross_edge_scalar = self.Layers['eddge_embedding'](cross_edge_norm, max_val=1.0)
			g.cross_e_rotation = so3.RotationToWignerDMatrix(so3.init_edge_rot_mat(cross_edge_vec), self.L)

		x_f_corss, x_f_corss_0 = self.Layers['cross_gnn'](x_source=x, x_target=x, x_0_source=x_0,
		                                   x_0_target=x_0, edge=g.cross_e, batch_src=g.x_cor_batch,
		                                   batch_dst=g.x_cor_batch, piece_src=g.piece_index,
		                                   piece_dst=g.piece_index, edge_scalar=g.cross_edge_scalar, t_code=t_code,
		                                   Rotation_W=g.cross_e_rotation)

		y, y_0 = self.Layers['self_gnn'](x_source=x_f_corss, x_target=x_f_corss, x_0_source=x_f_corss_0,
		                                   x_0_target=x_f_corss_0, edge=g.e, batch_src=g.x_cor_batch,
		                                   batch_dst=g.x_cor_batch, piece_src=g.piece_index,
		                                   piece_dst=g.piece_index, edge_scalar=g.edge_scalar, t_code=t_code,
		                                   Rotation_W=g.e_rotation)

		return y, y_0


class Eda(torch.nn.Module):
	"""The network of Eda """
	def __init__(self, settings):
		super().__init__()

		self.emb_channel = [int(i) for i in settings.emb_channel]
		self.n_layers = settings.n_layers
		self.n_scales = settings.n_scales

		self.input_emb = Linear_p(settings.input_channel, self.emb_channel, L=settings.L)
		self.final_layer = LastLayer(self.emb_channel, [2,2], settings.L, settings.norm_type)
		self.time_embedder = TimestepEmbedder(hidden_size=self.emb_channel[1] + self.emb_channel[0] * settings.L)

		self.down_blocks = torch.nn.ModuleList()
		self.up_blocks = torch.nn.ModuleList()

		for n in range(self.n_scales):
			self.down_blocks.append(TransitionDown(emb_channel=settings.emb_channel, L=settings.L,
			                                       num_heads=settings.num_heads, fc_neurons=settings.fc_neurons,
			                                       forward_mid_scale=settings.irreps_mlp_mid, qk_norm=settings.qk_norm,
			                                       norm_type=settings.norm_type, nonlinear_type=settings.nonlinear_type,
			                                       max_len=settings.Max_len[0][n+1],
			                                       max_len_bi=settings.Max_len[1][n],
			                                       ))

		self.mid_block = torch.nn.ModuleList()
		for i in range(self.n_layers):
			self.mid_block.append(DiTr(emb_channel=settings.emb_channel, L=settings.L,
			                                       num_heads=settings.num_heads, fc_neurons=settings.fc_neurons,
			                                       forward_mid_scale=settings.irreps_mlp_mid, qk_norm=settings.qk_norm,
			                                       norm_type=settings.norm_type, nonlinear_type=settings.nonlinear_type,
			                                    ))

		self.zero_init = settings.zero_init
		if settings.zero_init > 0:
			self._init_weights()


	def _init_weights(self):
		# Initialize timestep embedding MLP:
		# nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
		# nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

		# pdb.set_trace()
		# Zero-out adaLN modulation layers
		for block in self.down_blocks:
			nn.init.constant_(block.Layers['down_gnn'].adaLN_modulation[-1].weight, 0)
			nn.init.constant_(block.Layers['down_gnn'].adaLN_modulation[-1].bias, 0)
			nn.init.constant_(block.Layers['gnn'].adaLN_modulation[-1].weight, 0)
			nn.init.constant_(block.Layers['gnn'].adaLN_modulation[-1].bias, 0)

		for block in self.mid_block:
			nn.init.constant_(block.Layers['self_gnn'].adaLN_modulation[-1].weight, 0)
			nn.init.constant_(block.Layers['self_gnn'].adaLN_modulation[-1].bias, 0)
			nn.init.constant_(block.Layers['cross_gnn'].adaLN_modulation[-1].weight, 0)
			nn.init.constant_(block.Layers['cross_gnn'].adaLN_modulation[-1].bias, 0)


		# Zero-out output layers:
		nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
		nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
		if self.zero_init == 2:
			nn.init.constant_(self.final_layer.linear.weights, 0)
			nn.init.constant_(self.final_layer.linear.linear_0.bias, 0)
			nn.init.constant_(self.final_layer.linear.linear_0.weight, 0)


	def forward(self, g, t, graph_list=None, bi_graph_list=None):

		x_cor = g.x_cor  # (N, 3)
		x = g.x_f  # (N, F_in, L, m)
		x_0 = g.x_f_0 # N, F_in_0

		assert x.ndim == 4, 'Feature shape must be (N, C, L, m)'
		assert x_cor.ndim == 2, 'Node shape must be (N, 3)'
		assert x_0.ndim == 2, 'Feature 0 shape must be (N, C)'
		assert len(x) == len(x_cor), 'Number of nodes must be consistent'

		y, y_0 = self.input_emb(x, x_0)  # (N, F_embedding, L, m)
		t_code = self.time_embedder(t)

		########### Down sample Block #############
		for n in range(self.n_scales):
			y, y_0 = self.down_blocks[n](bi_graph_list[n], graph_list[n], t_code, y, y_0)


		########### Middle Block #############
		for n in range(self.n_layers):
			y, y_0 = self.mid_block[n](graph_list[-1], t_code, y, y_0)


		out, out_0 = self.final_layer(y, y_0, graph_list[-1].x_cor_batch, graph_list[-1].piece_index, t_code)
		weight = scatter_softmax_coo(out_0, graph_list[-1].piece_index, 0)

		out_deg1 = get_degree(out, 1)
		vec = scatter_mean_keepsize(out_deg1 * weight.unsqueeze(-1), graph_list[-1].piece_index, 0, keepsize=False, sum=True)

		return vec[:, 0, :], vec[:, 1, :]


# if __name__ == '__main__':
# 	import yaml
# 	from torch_geometric.data import Data
# 	import argparse
# 	from e3nn.util.test import assert_equivariant
#
#
#
# 	with open('config.yaml', 'r') as f:
# 		cfg = yaml.safe_load(f)
#
# 	parser = argparse.ArgumentParser()
# 	for k,v in cfg.items():
# 		parser.add_argument(f'--{k}', type=type(v), default=v)
# 	args = parser.parse_args()
#
#
# 	# pdb.set_trace()
#
# 	f = torch.randn((1000, 3))
# 	x = torch.randn((1000, 3))
# 	batch = torch.cat([torch.zeros(200, dtype=torch.int64), torch.ones(800, dtype=torch.int64)])
#
# 	unet = Unet(args)
#
# 	# input_pc = Data(x_cor=x, e=None, batch=batch, x_f=f)
# 	# out = unet(input_pc)
# 	# pdb.set_trace()
#
# 	def wrapper(x, batch, f):
# 		return unet(Data(x_cor=x, batch=batch, x_f=f))
#
#
# 	# `assert_equivariant` uses logging to print a summary of the equivariance error,
# 	# so we enable logging
# 	assert_equivariant(
# 		wrapper,
# 		# We provide the original data that `assert_equivariant` will transform...
# 		args_in=[x, batch, f],
# 		# ...in accordance with these irreps...
# 		irreps_in=[
# 			"1x1e",  # pos has vector 1o irreps, but is also translation equivariant
# 			None,  # pos has vector 1o irreps, but is also translation equivariant
# 			'3x0e',  # `None` indicates invariant, possibly non-floating-point data
# 		],
# 		# ...and confirm that the outputs transform correspondingly for these irreps:
# 		# irreps_out=['1x1e'],
# 		irreps_out=['16x0e+16x1e'],
#
# 	)
#
# 	pdb.set_trace()

# from model.diffusion_edf.edf_interface.data import PointCloud, SE3, DemoDataset, TargetPoseDemo
# # preprocess
# from model.diffusion_edf.gnn_data import FeaturedPoints
#
#
# def pcd_to_featured_points(pcd: PointCloud, batch_idx: int = 0) -> FeaturedPoints:
# 	return FeaturedPoints(x=pcd.points, f=pcd.colors,
# 	                      b=torch.empty_like(pcd.points[..., 0], dtype=torch.long).fill_(batch_idx))
#
#
# # import torch
# # from torch_geometric.data import Data
# #
# # x = torch.randn(100, 3)
# # batch = torch.zeros(100, dtype=torch.int64)
# #
# # g = Data(x_cor=x, batch=batch, type=None)
# #
# # pooling = FpsPool(ratio=0.5, k=4)
# # graph_list = []
# # graph_down_list = []
# #
# # for i in range(2):
# # 	bi_graph = pooling(g)
# # 	g = knn_graph_g(bi_graph.v_cor, k=2, batch=bi_graph.v_batch, loop=False)
# # 	graph_down_list.append(bi_graph)
# # 	graph_list.append(g)
# #
# # pdb.set_trace()
#
# with open('model/diffusion_edf/score_model_configs.yaml', 'r') as f:
# 	train_configs = yaml.load(f, Loader=yaml.FullLoader)
# # pdb.set_trace()
# key_feature_extractor_kwargs = train_configs['model_kwargs']['key_kwargs']['feature_extractor_kwargs']
# # pdb.set_trace()
# # key_model = UnetFeatureExtractor(**(key_feature_extractor_kwargs))


#
	# with open('model/diffusion_edf/preprocess.yaml') as f:
	# 	preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
	# 	preprocess_config = preprocess_config['preprocess_config']
	#
	# demo_idx = 0
	# testset = DemoDataset(dataset_dir='model/diffusion_edf/demo/panda_bowl_on_dish_test')
	# task_type = "place"
	#
	# demo: TargetPoseDemo = testset[demo_idx][
	# 	0 if task_type == 'pick' else 1 if task_type == 'place' else "task_type must be either 'pick' or 'place'"]
	# scene_pcd: PointCloud = demo.scene_pcd
	#
	# small_scene = PointCloud(scene_pcd.points[:1000], scene_pcd.colors[:1000], '', unit_length=scene_pcd.unit_length)
	#
	# scene_input: FeaturedPoints = pcd_to_featured_points(small_scene)
	#
	# pdb.set_trace()
	# out = key_model(scene_input)
	#


# class TransitionUp(nn.Module):
# 	'''
# 	Up-sampling
# 	'''
# 	def __init__(self, emb_channel: List[int], L: int, num_heads: int, fc_neurons: List[int], forward_mid_scale: int = 3,
# 	             qk_norm = False, norm_type: str = 'n111', nonlinear_type = 'silu', skip_last=False):
#
# 		super().__init__()
#
# 		self.skip_last = skip_last
# 		self.Layers = torch.nn.ModuleDict()
# 		self.Layers['up_gnn'] = EquiformerBlock(emb_channel, L, num_heads, fc_neurons,
# 									        forward_mid_scale=forward_mid_scale,
# 								             qk_norm=qk_norm,
# 								             norm_type=norm_type,
# 								             nonlinear_type=nonlinear_type,
# 		                                        att_type=0
# 		                                    )
#
# 		self.Layers['gnn'] = EquiformerBlock(emb_channel, L, num_heads, fc_neurons,
# 									        forward_mid_scale=forward_mid_scale,
# 								             qk_norm=qk_norm,
# 								             norm_type=norm_type,
# 								             nonlinear_type=nonlinear_type,
# 								             att_type=0
# 		                                     )
#
# 	def forward(self, bi_g, g, t_code, x: torch.Tensor, x_0: torch.Tensor, x_higher: torch.Tensor, x_higher_0: torch.Tensor):
# 		x_up, x_up_0 = self.Layers['up_gnn'](x_source=x, x_target=x_higher, x_0_source=x_0,
# 											x_0_target=x_higher_0, edge=torch.flip(bi_g.e, [0]), batch_src=bi_g.v_batch,
# 											batch_dst=bi_g.u_batch, piece_src = bi_g.v_piece, piece_dst=bi_g.u_piece,
# 											edge_scalar=bi_g.edge_scalar, t_code=t_code,
# 											Rotation_W=bi_g.uv_rotation)
#
#
# 		y, y_0 = self.Layers['gnn'](x_source=x_up, x_target=x_up, x_0_source=x_up_0,
# 		                                x_0_target=x_up_0, edge=g.e, batch_src=g.x_cor_batch,
# 		                                batch_dst=g.x_cor_batch, edge_scalar=g.edge_scalar,
# 		                                piece_src=g.piece_index, piece_dst=g.piece_index,
# 		                                t_code=t_code, Rotation_W=g.e_rotation)
#
# 		return y, y_0
