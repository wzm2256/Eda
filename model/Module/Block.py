import pdb
from typing import List, Optional, Union, Tuple

import torch
from .Linear import Linear_p
from .Nonlinear import Nonlinear_p
import torch.nn as nn
from .Norm import LayerNorm, modulate, gate, get_size
from .Attention import MultiHeadAttention


class FeedForward(torch.nn.Module):
	"""Linear + nonlinear + linear"""
	def __init__(self, input_channel: List[int], output_channel: List[int], middle_scale: int,
	             L: int, nonlinear_type='gelu'):
		super().__init__()

		middle_channel = [middle_scale * i for i in input_channel]
		self.Linear_1 = Linear_p(input_channel, middle_channel, L)
		self.Linear_2 = Linear_p(middle_channel, output_channel, L)
		self.L = L
		self.Nonlinear = Nonlinear_p(middle_channel, L, type=nonlinear_type)

	def forward(self, x: torch.Tensor, x_0: torch.Tensor) -> (torch.Tensor, torch.Tensor):
		y, y_0 = self.Linear_2(*self.Nonlinear(*self.Linear_1(x, x_0)))
		return y, y_0


class EquiformerBlock(torch.nn.Module):

	"""An equivariant transformer layer.
	emb_channel: [C1, C2] C1: the channel of equivariant feature. C2: the channel of invariant feature
	L: the highest equivariant degree.
	fc_neurons: the channels of fully connected layer used for edge feature.
	forward_mid_scale: control the hidden channels of ffn
	qk_norm: whether to use qk_norm to stablize training.
	normalization type: See Norm.py for details.
	nonlinear_type: equivariant nonlinear type, 'gelu' or 'silu'.
	"""
	def __init__(self,
	             emb_channel: List[int],
	             L: int,
	             num_heads: int,
	             fc_neurons: List[int],
	             forward_mid_scale: int = 3,
	             qk_norm=False,
	             norm_type: str = 'n111',
	             nonlinear_type='silu',
	             att_type=1,
	             ):

		super().__init__()
		self.emb_channel = emb_channel
		self.L  = L
		self.norm_type = norm_type
		self.norm_1 = LayerNorm(emb_channel, L, type=norm_type, use_affine=False)
		self.norm_2 = LayerNorm(emb_channel, L, type=norm_type, use_affine=False)
		self.norm_3 = LayerNorm(emb_channel, L, type=norm_type, use_affine=False)
		self.att = MultiHeadAttention(emb_channel, L, fc_neurons, num_heads, qk_norm, norm_type, att_type=att_type)
		self.ffn = FeedForward(emb_channel, emb_channel, forward_mid_scale, L, nonlinear_type)

		self.size1, self.size0 = get_size(emb_channel, L, norm_type)
		self.adaLN_modulation = nn.Sequential(
			nn.SiLU(),
			nn.Linear(emb_channel[0] + emb_channel[1] * L, self.size1 * 4 + self.size0 * 6)
		)


	def forward(self, x_source: torch.Tensor, x_target: torch.Tensor, x_0_source: torch.Tensor, x_0_target: torch.Tensor,
	            edge: torch.Tensor, batch_src: torch.Tensor, batch_dst: torch.Tensor,
	            piece_src: torch.Tensor, piece_dst: torch.Tensor,
	            edge_scalar: torch.Tensor, t_code: torch.Tensor, Rotation_W: torch.Tensor) -> (torch.Tensor, torch.Tensor):

		######
		adaLN_weight = self.adaLN_modulation(t_code)

		scale_mlp,  gate_mlp, gate_msa, scale_msa = adaLN_weight[:, :self.size1 * 4].chunk(4, dim=1)
		scale_mlp_0,  gate_mlp_0, gate_msa_0, scale_msa_0, shift_msa_0, shift_mlp_0 = adaLN_weight[:, self.size1 * 4:].chunk(6, dim=1)

		x_src, x_src_0 = modulate(*self.norm_1(x_source, x_0_source, batch_src, piece_src), scale_msa[batch_src], scale_msa_0[batch_src], shift_msa_0[batch_src], type=self.norm_type)
		x_tar, x_tar_0 = modulate(*self.norm_2(x_target, x_0_target, batch_dst, piece_dst), scale_msa[batch_dst], scale_msa_0[batch_dst], shift_msa_0[batch_dst], type=self.norm_type)


		x_tar_msa, x_tar_0_msa = gate(*self.att(x_src, x_tar, x_src_0, x_tar_0,
		                                        Rotation_W, edge_scalar, edge, batch_dst, piece_dst,
		                                        outpu_size=x_target.shape[0]),
		                              gate_msa[batch_dst], gate_msa_0[batch_dst], type=self.norm_type)


		x_att =  x_target + x_tar_msa
		x_att_0 = x_0_target + x_tar_0_msa

		x_tar_ffn, x_tar_0_ffn = gate(*self.ffn(*modulate(*self.norm_3(x_att, x_att_0, batch_dst, piece_dst), scale_mlp[batch_dst], scale_mlp_0[batch_dst], shift_mlp_0[batch_dst], type=self.norm_type)),
		                              gate_mlp[batch_dst], gate_mlp_0[batch_dst], type=self.norm_type)

		y = x_tar_ffn + x_att
		y_0 = x_att_0 + x_tar_0_ffn

		return y, y_0


if __name__ == '__main__':
	# from .torch_cluster import knn_graph, knn
	# input_Channel: List[int], L: int, radius_channel: int, head: int
	from ..FastSo3 import so2, so3
	from .Connectivity import knn_bi_graph
	def get_message(x_source, x_target, x_0_source, x_0_target, x_cor_src, x_cor_tar, edge, batch_src, batch_dst,
	                multiatt, L, edge_scalar, t_code):

		edge_vec = x_cor_src[edge[0]] - x_cor_tar[edge[1]]
		rotation = so3.init_edge_rot_mat(edge_vec)
		W = so3.RotationToWignerDMatrix(rotation, L)

		message, message_L0  = multiatt(x_source, x_target, x_0_source, x_0_target, edge, batch_src, batch_dst,
		                                edge_scalar, t_code, W)

		return message, message_L0


	input_channel = [16, 16]
	# output_channel_list = [1, 1]
	L = 1
	head = 4
	P = [20, 30]
	P2 = [10, 20]

	radius_channel = [8, 16]
	k = 2
	batch = torch.cat([torch.ones(P[i]) for i in range(len(P))])
	batch1 = torch.cat([torch.ones(P2[i]) for i in range(len(P2))])

	pc1 = torch.randn((P[0], 3))
	pc2 = torch.randn((P[1], 3))
	x_cor_src = torch.cat([pc1, pc2], 0)

	pc3 = torch.randn((P2[0], 3))
	pc4 = torch.randn((P2[1], 3))
	x_cor_tar = torch.cat([pc3, pc4], 0)

	edge = knn_bi_graph(x_cor_src, x_cor_tar, k, batch, batch1)
	edge_feature = torch.randn((edge.shape[1], radius_channel[0]))

	x_src = torch.randn((P[0] + P[1], input_channel[0], L, 2 * L + 1))
	x_0_src = torch.randn((P[0] + P[1], input_channel[1]))

	x_tar = torch.randn((P2[0] + P2[1], input_channel[0], L, 2 * L + 1))
	x_0_tar = torch.randn((P2[0] + P2[1], input_channel[1]))

	multiatt = MultiHeadAttention(input_channel, L, radius_channel, head)

	block = EquiformerBlock(input_channel, L, head, radius_channel, norm_type='n111', nonlinear_type='gelu')

	t_code = torch.randn((1, input_channel[0] + input_channel[1]))

	message, message_L0 = get_message(x_src, x_tar, x_0_src, x_0_tar, x_cor_src, x_cor_tar, edge, batch, batch1,
	                block, L, edge_feature, t_code)

	# pdb.set_trace()
	from scipy.spatial.transform import Rotation
	R = torch.tensor(Rotation.random(num=1).as_matrix()).type(torch.float32)
	R_w = so3.RotationToWignerDMatrix(R, end_lmax=L)

	x_src_R = so3.rotate(R_w, x_src)
	x_tar_R = so3.rotate(R_w, x_tar)

	x_cor_src_R = (R @ x_cor_src.unsqueeze(-1)).squeeze(-1)
	x_cor_tar_R = (R @ x_cor_tar.unsqueeze(-1)).squeeze(-1)

	message_R, message_L0 = get_message(x_src_R, x_tar_R, x_0_src, x_0_tar, x_cor_src_R, x_cor_tar_R, edge, batch, batch1,
	                block, L, edge_feature, t_code)

	message_Rotate = so3.rotate(R_w, message)

	# pdb.set_trace()

	print((message_Rotate - message_R).norm())

