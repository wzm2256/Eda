import pdb

import torch.nn as nn
import torch
import einops
from typing import Optional, List
import math


class Nonlinear_p(nn.Module):
	'''
	An extension of the classic Silu layer.
	Channel-wise linear layer.
	F * sigma (<F, V>), where sigma is the point-wise non-linear function,
	V is a learned direction.
	'''
	def __init__(self, input_channel_list: List[int], L: int, eps=1e-4, type='silu'):
		super().__init__()
		self.weights  = nn.Parameter(torch.empty((input_channel_list[0], input_channel_list[0], L)))
		torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
		if type == 'silu':
			self.nonlinear_0 = torch.nn.SiLU()
		elif type == 'gelu':
			self.nonlinear_0 = torch.nn.GELU(approximate="tanh")
		else:
			raise NotImplementedError
		self.eps = eps

	def forward(self, x: torch.Tensor, x_0: torch.Tensor) -> (torch.Tensor, torch.Tensor):
		# x:   B, c_in, L, 2L + 1
		# x_0: B, c_L0_in
		d = einops.einsum(self.weights, x, 'c d l, p c l m -> p d l m') # B C L m
		d_norm_sq = d / (d.norm(dim=-1, keepdim=True) + self.eps)  # B C L 1

		y = x * self.nonlinear_0((d_norm_sq * x).sum(-1, keepdims=True))
		y_0 = self.nonlinear_0(x_0)

		return y, y_0


if __name__ == '__main__':
	from ..FastSo3 import so2, so3

	L = 1
	c_in = 10
	c_in_0 = 10
	nl = Nonlinear_p([c_in, c_in_0], L=L)

	x = torch.randn((100, c_in, L, 2 * L + 1))
	x_0 = torch.randn((100, c_in_0))

	y, y_0 = nl(x, x_0)


	from scipy.spatial.transform import Rotation
	R = torch.tensor(Rotation.random(num=1).as_matrix()).type(torch.float32)
	R_w = so3.RotationToWignerDMatrix(R, end_lmax=L)

	x_R = so3.rotate(R_w, x)

	y_R, y_0_R = nl(x_R, x_0)

	y_Rotate = so3.rotate(R_w, y)

	print((y_Rotate - y_R).norm())

# pdb.set_trace()
	#
	#
	# input_channel = [16, 16]
	# L = 1
	# P = 20
	#
	#
	# x = torch.randn((P, input_channel[0], L, 2 * L + 1))
	# x_0 = torch.randn((P, input_channel[1]))
	#
	#
	#
	# # multiatt = MultiHeadAttention(input_channel, L, radius_channel, head)
	#
	# message, message_L0 = get_message(x, x_0, x_cor, edge, multiatt, L, edge_feature)
	#
	#
	# x_R = so3.rotate(R_w, x)
	# x_cor_R = (R @ x_cor.unsqueeze(-1)).squeeze(-1)
	#
	# message_R, message_L0 = get_message(x_R, x_0, x_cor_R, edge, multiatt, L, edge_feature)
	#
	# message_Rotate = so3.rotate(R_w, message)
	#
	# # pdb.set_trace()
	#
	# print((message_Rotate - message_R).norm())
