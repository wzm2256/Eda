import pdb

import einops
import torch
from torch_scatter import segment_coo
from typing import Tuple, List, Union, Optional, NamedTuple
import torch.nn as nn
from ..FastSo3 import so2, so3
from .Linear import Linear_p
from .Norm import LayerNorm, modulate, gate, get_size, LayerNorm_0
from ..module_util import scatter_softmax_coo

class RadialProfile(nn.Module):
	"""
	Radial function.
	"""
	def __init__(self, channel_list: List[int], norm_type='0'):
		super().__init__()

		self.net = nn.ModuleList([
			nn.Linear(channel_list[0], channel_list[1]),
			LayerNorm_0(channel_list[1], type=norm_type),
			torch.nn.SiLU(),
			nn.Linear(channel_list[1], channel_list[1]),
			LayerNorm_0(channel_list[1], type=norm_type),
			torch.nn.SiLU(),
			nn.Linear(channel_list[1], channel_list[2])
		])

	def forward(self, e: torch.Tensor, batch: torch.Tensor, piece: torch.Tensor) -> torch.Tensor:
		'''
		:param e: e c
		:return: e c1
		'''
		for layer in self.net:
			if type(layer) == LayerNorm_0:
				e = layer(e, batch, piece)
			else:
				e = layer(e)
		return e



class MultiHeadAttention(nn.Module):
	def __init__(self, input_Channel: List[int], L: int, radius_channel: List[int], head: int, qk_norm: bool = False,
	             norm_type: str = 'n111',  att_type=1):
		'''

		:param input_Channel:
		:param L:
		:param radius_channel: [input channel, hidden_channel]
		'''
		super().__init__()
		# c_in, c_out, c_hidden, c_hidden_0, L, c_L0_in, c_L0_out

		self.att_type = att_type
		if att_type == 0:
			self.kv_embedding = so2.SO2_conv_e(input_Channel[0], 2 * input_Channel[0], 2 * input_Channel[0], 2 * input_Channel[1], L,
			                                   input_Channel[1], 2 * input_Channel[1])
			self.q_embedding = so2.SO2_conv_e(input_Channel[0], input_Channel[0], input_Channel[0], input_Channel[1], L,
			                                  input_Channel[1], input_Channel[1])

			self.w_size = input_Channel[0] * 2 * L
			self.w_0_size = input_Channel[1]
			self.require_size = 3 * self.w_size + 3 * self.w_0_size
		elif att_type == 1:
			# Do channel mixing, then depth wise convolution for key and value.
			self.q_embedding = so2.SO2_mix_c(input_Channel[0], input_Channel[0], L, input_Channel[1], input_Channel[1], bias=True)
			self.kv_1 = so2.SO2_mix_c(input_Channel[0], 2 * input_Channel[0], L, input_Channel[1], 2 * input_Channel[1], bias=True)
			self.kv_2 = so2.SO2_conv_c(2 * input_Channel[0], L, 2 * input_Channel[1], bias=False)
			self.w_size = input_Channel[0] * 2 * L * L
			self.w_0_size = input_Channel[1] + input_Channel[0] * L
			self.require_size = 2 * self.w_size + 2 * self.w_0_size

		### generate weight from invariant edge features
		# w:    B, c_hidden, 2 * L
		# w_L0: B, c_hidden_0
		if len(norm_type) <= 4:
			norm_type0 = '0'
		else:
			norm_type0 = norm_type[4]
		self.radius = RadialProfile(radius_channel + [self.require_size], norm_type=norm_type0)
		self.Channel = input_Channel
		self.L = L
		self.head = head

		self.proj = Linear_p(input_Channel, input_Channel, L)

		self.q_norm = LayerNorm(input_Channel, L, type=norm_type) if qk_norm else LayerNorm(input_Channel, L, type='n001')
		self.k_norm = LayerNorm(input_Channel, L, type=norm_type) if qk_norm else LayerNorm(input_Channel, L, type='n001')



	def forward(self, x_src: torch.Tensor, x_tar: torch.Tensor, x_src_0: torch.Tensor, x_tar_0: torch.Tensor,
	            Rotation_W: torch.Tensor, edge_feature: torch.Tensor,
	             edge: torch.Tensor,
	            batch_dst: torch.Tensor, piece_dst: torch.Tensor, outpu_size: int =None) \
			-> (torch.Tensor, torch.Tensor):
		'''

		:param x_source: B, c_in, L, 2L+1
		:param x_target: B, c_in, L, 2L+1
		:param x_0_source: B, c_L0_in
		:param x_0_target: B, c_L0_in
		:param batch: B
		:param Rotation_W:
		:param edge_feature:
		:return:
		'''

		w_all = self.radius(edge_feature, batch_dst[edge[1]], piece_dst[edge[1]])

		kv_w, kv_w0 = torch.split(w_all, [2 * self.w_size, 2 * self.w_0_size], -1)
		q, q_0 = self.q_embedding(so3.rotate(Rotation_W, x_tar[edge[1]]), x_tar_0[edge[1]])

		kv_mix, kv_mix_0 = self.kv_1(so3.rotate(Rotation_W, x_src[edge[0]]), x_src_0[edge[0]])
		kv, kv_0 = self.kv_2(kv_mix, kv_mix_0, kv_w.reshape([kv_w.shape[0], -1, self.L, self.L * 2]), kv_w0)
		k, v = torch.split(kv, [self.Channel[0], self.Channel[0]], 1)
		k_0, v_0 = torch.split(kv_0, [self.Channel[1], self.Channel[1]], 1)

		batch_edge = batch_dst[edge[1]]
		piece_edge = piece_dst[edge[1]]
		edge_dst = edge[1]
		edge_src = edge[0]

		q, q_0 = self.q_norm(q, q_0, batch_edge, piece_edge)
		k, k_0 = self.k_norm(k, k_0, batch_edge, piece_edge)

		d = k.shape[1] // self.head
		d_0 = v_0.shape[1] // self.head

		k_multi_all = torch.cat([einops.rearrange(k, 'b (h c) l m -> b h (c l m)', h=self.head),
		                         einops.rearrange(k_0, 'b (h c) -> b h c', h=self.head)], -1)
		q_multi_all = torch.cat([einops.rearrange(q, 'b (h c) l m -> b h (c l m)', h=self.head),
		                         einops.rearrange(q_0, 'b (h c) -> b h c', h=self.head)], -1)
		v_multi_all = torch.cat([einops.rearrange(v, 'b (h c) l m -> b h (c l m)', h=self.head),
		                         einops.rearrange(v_0, 'b (h c) -> b h c', h=self.head)], -1)

		alpha = scatter_softmax_coo(einops.einsum(q_multi_all, k_multi_all, 'b h c, b h c  -> b h') / (d ** 0.5), edge_dst, 0)

		message_all_new = einops.einsum(alpha, v_multi_all, 'b h, b h c -> b h c')
		message_0 = einops.rearrange(message_all_new[:, :, -d_0:], ' b h c -> b (h c)')
		v_weighted = einops.rearrange(message_all_new[:, :, :-d_0], ' b h (c l m) -> b (h c) l m', l=self.L,
		                              m=self.L * 2 + 1)

		message = so3.rotate(Rotation_W.mT, v_weighted)

		att = segment_coo(message, index=edge_dst, dim_size=outpu_size, reduce='sum')
		att_0 = segment_coo(message_0, index=edge_dst, dim_size=outpu_size, reduce='sum')

		y, y_0 = self.proj(att, att_0)
		return y, y_0



if __name__ == '__main__':
	from torch_cluster import knn_graph, knn
	# input_Channel: List[int], L: int, radius_channel: int, head: int

	def get_message(x, x_0, x_cor, edge, multiatt, L, edge_feature):
		x_source = x[edge[0]]
		x_target = x[edge[1]]
		x_0_source = x_0[edge[0]]
		x_0_target = x_0[edge[1]]

		edge_vec = x_cor[edge[0]] - x_cor[edge[1]]
		rotation = so3.init_edge_rot_mat(edge_vec)
		W = so3.RotationToWignerDMatrix(rotation, L)
		# pdb.set_trace()
		# x_W = so3.rotate(W, x)
		# message_so2, message_L0 = conv(x_W, x_L0, w, w_L0)
		message, message_L0  = multiatt(x_source, x_target, x_0_source, x_0_target, W, edge_feature, edge[1])

		# message = so3.rotate(W.mT, message_so2)
		return message, message_L0


	input_channel = [16, 16]
	# output_channel_list = [1, 1]
	L = 1
	head = 4
	P = [20, 30]
	radius_channel = [8, 16]
	k = 2
	batch = torch.cat([torch.ones(P[i]) for i in range(len(P))])

	pc1 = torch.randn((P[0], 3))
	pc2 = torch.randn((P[1], 3))
	x_cor = torch.cat([pc1, pc2], 0)
	edge = knn_graph(x_cor, k=k, batch=batch, loop=False)


	edge_feature = torch.randn((edge.shape[1], radius_channel[0]))

	x = torch.randn((P[0] + P[1], input_channel[0], L, 2 * L + 1))
	x_0 = torch.randn((P[0] + P[1], input_channel[1]))
	multiatt = MultiHeadAttention(input_channel, L, radius_channel, head)

	message, message_L0 = get_message(x, x_0, x_cor, edge, multiatt, L, edge_feature)

	from scipy.spatial.transform import Rotation
	R = torch.tensor(Rotation.random(num=1).as_matrix()).type(torch.float32)
	R_w = so3.RotationToWignerDMatrix(R, end_lmax=L)

	x_R = so3.rotate(R_w, x)
	x_cor_R = (R @ x_cor.unsqueeze(-1)).squeeze(-1)

	message_R, message_L0 = get_message(x_R, x_0, x_cor_R, edge, multiatt, L, edge_feature)

	message_Rotate = so3.rotate(R_w, message)

	# pdb.set_trace()

	print((message_Rotate - message_R).norm())


# pdb.set_trace()
	# # x_source: torch.Tensor, x_target: torch.Tensor, x_0_source: torch.Tensor, x_0_target: torch.Tensor,
	# # Rotation_W: torch.Tensor, edge_feature: torch.Tensor, edge_dst: torch.Tensor
	#
	#
	# out = multiatt(x_source, x_target, x_0_source, x_0_target, W, edge_feature, edge[1])
	# pdb.set_trace()
	# x_source = x[edge[0]]
	# x_target = x[edge[1]]
	# x_0_source = x_0[edge[0]]
	# x_0_target = x_0[edge[1]]
	# rotation_e = so3.init_edge_rot_mat(edge_vec)
	# W = so3.RotationToWignerDMatrix(rotation_e, L)
	# pdb.set_trace()
