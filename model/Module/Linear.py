import pdb

import torch.nn as nn
import torch
import math
import einops
from typing import Optional, List
from .Norm import LayerNorm, modulate, gate, get_size

class Linear_p(nn.Module):
	'''
	Channel-wise linear layer.
	Features of different degrees are not mixed
	'''
	def __init__(self, input_channel_list: List[int], output_channel_list: List[int], L: int =1, bias=True):
		super().__init__()
		self.weights  = nn.Parameter(torch.empty((input_channel_list[0], output_channel_list[0], L)))
		self.linear_0 = nn.Linear(input_channel_list[1], output_channel_list[1], bias=bias)
		torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

	def forward(self, x: torch.Tensor, x_0: torch.Tensor) -> (torch.Tensor, torch.Tensor):
		# x:   B, c_in, L, 2L + 1
		# x_0: B, c_L0_in
		y = einops.einsum(self.weights, x, 'c d l, p c l m -> p d l m')
		y_0 = self.linear_0(x_0)
		return y, y_0


class LastLayer(nn.Module):
	"""Normalization + Linear layer"""
	def __init__(self, emb_channel: List[int], output_channel_list: List[int], L: int, norm_type: str):
		super().__init__()

		self.emb_channel = emb_channel
		self.L = L
		self.norm_type = norm_type
		self.norm = LayerNorm(emb_channel, L, type=norm_type, use_affine=False)
		self.size1, self.size0 = get_size(emb_channel, L, norm_type)
		self.adaLN_modulation = nn.Sequential(
			nn.SiLU(),
			nn.Linear(emb_channel[0] + emb_channel[1] * L, self.size0 * 2 + self.size1)
		)
		self.linear = Linear_p(emb_channel, output_channel_list=output_channel_list, L=L)

	def forward(self, x: torch.Tensor, x_0: torch.Tensor, batch:torch.Tensor, piece: torch.Tensor, t_code: torch.Tensor) -> (torch.Tensor, torch.Tensor):
		adaLN_weight = self.adaLN_modulation(t_code)
		scale = adaLN_weight[:, :self.size1]
		scale_0, shift_0 = adaLN_weight[:, self.size1:].chunk(2, dim=1)
		y, y_0 = modulate(*self.norm(x, x_0, batch, piece), scale[batch], scale_0[batch], shift_0[batch], type=self.norm_type)
		out, out_0 = self.linear(y, y_0)
		return out, out_0


if __name__ == '__main__':
	input_channel_list = [0, 1]
	output_channel_list = [1, 1]
	L = 1
	l = Linear_p(input_channel_list, output_channel_list, L)
	x = torch.randn((10, 0, 1, 3))
	x_0 = torch.randn((10, 1))
	y, y_0= l(x, x_0)
	pdb.set_trace()
